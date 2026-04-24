"""
visualize_agnn.py — AGNN Visualization (Grad-CAM + Graph + t-SNE)
=================================================================
Dataset format:
  images_root/
    class_a/  img1.jpg ...
    class_b/  img1.jpg ...
  split.json  {"train": [...], "val": [...]}

Script tự động sample 1 episode few-shot từ val classes,
rồi tạo 4 loại visualization (không cần GPU).

Usage:
  python visualize_agnn.py \\
    --config  config/5way_5shot_resnet50_custom.py \\
    --checkpoint  checkpoints/.../model_best.pth.tar \\
    --images_root  C:/path/to/images \\
    --split_json   C:/path/to/full_split.json \\
    --output_dir   visualizations/ \\
    --num_ways 5 --num_shots 5 --num_queries 3 --seed 42
"""

import os, sys, argparse, importlib.util, json, random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from backbone import ResNet12, ConvNet, ResNet50Pretrained
from agnn import AGNN
from utils import (allocate_tensors, initialize_nodes_edges,
                   backbone_two_stage_initialization, one_hot_encode)

# ── Transforms ────────────────────────────────────────────────

def get_transform(image_size):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    box  = int(image_size * 1.15)
    return transforms.Compose([
        transforms.Resize((box, box), interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        norm,
    ])

# ── Episode Sampler ───────────────────────────────────────────

def sample_episode(images_root, split_json, num_ways, num_shots, num_queries,
                   seed, partition='val'):
    """
    Đọc split.json, lấy các class trong `partition`,
    rồi sample num_ways class, mỗi class lấy (num_shots + num_queries) ảnh.
    Trả về:
      class_names  list[str]  độ dài num_ways
      support_paths list[list[str]]  [num_ways][num_shots]
      query_paths   list[list[str]]  [num_ways][num_queries]
    """
    rng = random.Random(seed)

    with open(split_json, 'r') as f:
        split = json.load(f)

    avail_classes = split[partition]

    # Lọc chỉ lấy class có đủ ảnh
    valid_classes = []
    for cls in avail_classes:
        cls_dir = os.path.join(images_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [p for p in os.listdir(cls_dir)
                if p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]
        if len(imgs) >= num_shots + num_queries:
            valid_classes.append(cls)

    if len(valid_classes) < num_ways:
        raise ValueError(
            f"Chỉ có {len(valid_classes)} class đủ ảnh trong '{partition}' split "
            f"(cần {num_ways}). Hãy giảm --num_ways hoặc --num_shots/--num_queries."
        )

    chosen = rng.sample(valid_classes, num_ways)
    support_paths, query_paths = [], []

    for cls in chosen:
        cls_dir = os.path.join(images_root, cls)
        imgs = sorted([os.path.join(cls_dir, p) for p in os.listdir(cls_dir)
                       if p.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))])
        rng.shuffle(imgs)
        support_paths.append(imgs[:num_shots])
        query_paths.append(imgs[num_shots: num_shots + num_queries])

    return chosen, support_paths, query_paths


def load_image(path, transform):
    pil = Image.open(path).convert('RGB')
    return transform(pil), pil

# ── Model Loading ─────────────────────────────────────────────

def load_models(args, config, num_supports, num_queries):
    bname = config['backbone']
    if bname == 'resnet12':
        enc = ResNet12(emb_size=config['emb_size'])
    elif bname == 'resnet50':
        enc = ResNet50Pretrained(emb_size=config['emb_size'])
    elif bname == 'convnet':
        enc = ConvNet(emb_size=config['emb_size'])
    else:
        raise ValueError(f"Unsupported backbone: {bname}")

    gnn = AGNN(
        config['num_generation'],
        config['train_config']['dropout'],
        num_supports,
        num_supports + num_queries,
        config['train_config']['loss_indicator'],
        config['point_distance_metric'],
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    def _clean(sd):
        return {(k[7:] if k.startswith('module.') else k): v for k,v in sd.items()}

    enc.load_state_dict(_clean(ckpt['enc_module_state_dict']))
    gnn.load_state_dict(_clean(ckpt['gnn_module_state_dict']))
    enc.eval(); gnn.eval()
    return enc, gnn

# ── Grad-CAM (pure torch hooks) ───────────────────────────────

class GradCAM:
    def __init__(self, encoder, backbone_name):
        self.encoder = encoder
        self._act = None
        self._grad = None
        self._hooks = []
        layer = self._target_layer(backbone_name)
        self._hooks.append(layer.register_forward_hook(
            lambda m,i,o: setattr(self, '_act', o.detach())))
        self._hooks.append(layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self, '_grad', go[0].detach())))

    def _target_layer(self, name):
        if name in ('resnet50', 'resnet12'):
            return self.encoder.layer4
        elif name == 'convnet':
            return self.encoder.conv_4
        raise ValueError(name)

    def remove(self):
        for h in self._hooks: h.remove()

    def compute(self, img_t, sup_embeds):
        """img_t: [1,3,H,W]. Returns numpy [H',W'] in [0,1]."""
        self.encoder.zero_grad()
        img_req = img_t.clone().detach().requires_grad_(True)
        emb = self.encoder(img_req)[0]                      # [1,128]
        proto = sup_embeds.mean(0)
        score = F.cosine_similarity(emb, proto.unsqueeze(0))
        score.backward()

        weights = self._grad.mean(dim=(2,3), keepdim=True)
        cam = F.relu((weights * self._act).sum(1).squeeze())
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.numpy()


def overlay(pil_img, cam_np, size, alpha=0.45):
    cam_r = np.array(Image.fromarray((cam_np*255).astype(np.uint8))
                     .resize((size,size), Image.BILINEAR)) / 255.0
    heat  = (plt.cm.jet(cam_r)[:,:,:3] * 255).astype(np.uint8)
    base  = np.array(pil_img.resize((size,size), Image.BILINEAR)).astype(np.float32)
    return ((1-alpha)*base + alpha*heat).clip(0,255).astype(np.uint8)

# ── Main visualization runner ─────────────────────────────────

def run_all(args, config):
    os.makedirs(args.output_dir, exist_ok=True)

    num_ways    = args.num_ways
    num_shots   = args.num_shots
    num_queries = args.num_queries
    img_size    = config.get('image_size', 84)
    tf          = get_transform(img_size)

    # 1. Sample episode
    print(f"Sampling {num_ways}-way {num_shots}-shot episode (seed={args.seed}) from val split...")
    class_names, support_paths, query_paths = sample_episode(
        args.images_root, args.split_json,
        num_ways, num_shots, num_queries, args.seed)

    for i, cls in enumerate(class_names):
        print(f"  Class {i}: {cls}  (support={len(support_paths[i])}, query={len(query_paths[i])})")

    # 2. Load images
    sup_tensors, sup_pils, sup_labels = [], [], []
    for ci, paths in enumerate(support_paths):
        for p in paths:
            t, pil = load_image(p, tf)
            sup_tensors.append(t); sup_pils.append(pil)
            sup_labels.append(ci)

    qry_tensors, qry_pils, qry_fnames = [], [], []
    for ci, paths in enumerate(query_paths):
        for p in paths:
            t, pil = load_image(p, tf)
            qry_tensors.append(t); qry_pils.append(pil)
            qry_fnames.append(f"{class_names[ci]}/{os.path.basename(p)}")

    num_supports = len(sup_tensors)
    num_q_total  = len(qry_tensors)

    support_data  = torch.stack(sup_tensors).unsqueeze(0)           # [1,K,3,H,W]
    support_label = torch.tensor(sup_labels, dtype=torch.long).unsqueeze(0) # [1,K]
    query_data    = torch.stack(qry_tensors).unsqueeze(0)           # [1,Q,3,H,W]
    query_label   = torch.zeros(1, num_q_total, dtype=torch.long)

    # 3. Load model
    print(f"\nLoading checkpoint from {args.checkpoint} (CPU)...")
    enc, gnn = load_models(args, config, num_supports, num_q_total)

    # 4. Forward pass
    tensors = allocate_tensors()
    batch = (support_data.unsqueeze(0), support_label.unsqueeze(0),
             query_data.unsqueeze(0),   query_label.unsqueeze(0))

    _, sup_lbl_node, _, _, all_data, _, node_feat_gd, edge_feat_gp = \
        initialize_nodes_edges(batch, num_supports, tensors, 1, num_q_total, 1, 'cpu')

    with torch.no_grad():
        last_emb, second_emb = backbone_two_stage_initialization(all_data, enc)
        point_sims, _ = gnn(second_emb, last_emb, node_feat_gd, edge_feat_gp, sup_lbl_node)

    # Predictions
    final_sim  = point_sims[-1]                                     # [1,N,N]
    qry_sim    = final_sim[:, num_supports:, :num_supports]         # [1,Q,K]
    one_hot_s  = one_hot_encode(num_ways, sup_lbl_node.long(), 'cpu')
    pred_scores= torch.bmm(qry_sim, one_hot_s).squeeze(0)          # [Q,num_ways]
    pred_labels= torch.argmax(pred_scores, dim=-1)                  # [Q]

    print("\nPredictions:")
    correct = 0
    gt_labels = []
    for ci, paths in enumerate(query_paths):
        for _ in paths:
            gt_labels.append(ci)
    for qi in range(num_q_total):
        pred = pred_labels[qi].item()
        gt   = gt_labels[qi]
        ok   = "✓" if pred == gt else "✗"
        correct += int(pred == gt)
        print(f"  {ok} [{qi+1:02d}] {qry_fnames[qi]} → pred:{class_names[pred]} gt:{class_names[gt]}")
    print(f"  Accuracy: {correct}/{num_q_total} = {correct/num_q_total:.1%}")

    sup_embeds = last_emb[0, :num_supports]  # [K,128]

    # ─────────────────────────────────────────────────
    # VIZ 1: Grad-CAM
    # ─────────────────────────────────────────────────
    print(f"\n[1/4] Grad-CAM heatmaps...")
    gcam = GradCAM(enc, config['backbone'])

    n_cols = min(num_q_total, 5)
    n_rows = (num_q_total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4.5*n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for qi in range(num_q_total):
        img_t = all_data[0, num_supports+qi].unsqueeze(0)
        pred_cls = pred_labels[qi].item()
        gt_cls   = gt_labels[qi]
        cls_mask = (sup_lbl_node[0] == pred_cls)
        cam_np   = gcam.compute(img_t, sup_embeds[cls_mask])
        ov       = overlay(qry_pils[qi], cam_np, img_size)

        r, c = divmod(qi, n_cols)
        ax = axes[r, c]
        ax.imshow(ov)
        color = 'green' if pred_cls == gt_cls else 'red'
        ax.set_title(
            f"Pred: {class_names[pred_cls]}\nGT: {class_names[gt_cls]}",
            fontsize=7, color=color, pad=2)
        ax.axis('off')

    for qi in range(num_q_total, n_rows*n_cols):
        r, c = divmod(qi, n_cols)
        axes[r, c].axis('off')

    fig.suptitle("Grad-CAM: Vùng Backbone Chú Ý (xanh=đúng, đỏ=sai)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(args.output_dir, 'gradcam_queries.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gcam.remove()
    print(f"  Saved → {p}")

    # ─────────────────────────────────────────────────
    # VIZ 2: Graph Edge Heatmap per generation
    # ─────────────────────────────────────────────────
    print(f"\n[2/4] Graph edge heatmaps...")
    N = num_supports + num_q_total
    num_gen = len(point_sims)

    # Node labels (ngắn gọn)
    node_lbl = []
    for ci in range(num_ways):
        short = class_names[ci][:6]
        for _ in range(num_shots):
            node_lbl.append(f"S{ci}:{short}")
    for qi in range(num_q_total):
        node_lbl.append(f"Q{qi+1}")

    fig, axes = plt.subplots(1, num_gen, figsize=(6*num_gen, 6))
    if num_gen == 1: axes = [axes]

    for gi, ps in enumerate(point_sims):
        mat = ps[0].detach().numpy()
        ax  = axes[gi]
        im  = ax.imshow(mat, cmap='viridis', vmin=0, vmax=mat.max())
        ax.set_title(f"Generation {gi+1}", fontsize=11, fontweight='bold')
        ax.set_xticks(range(N)); ax.set_xticklabels(node_lbl, rotation=90, fontsize=5)
        ax.set_yticks(range(N)); ax.set_yticklabels(node_lbl, fontsize=5)
        ax.axvline(num_supports-0.5, color='red', lw=1.5, ls='--')
        ax.axhline(num_supports-0.5, color='red', lw=1.5, ls='--')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    legend = [mpatches.Patch(color='red', label='Support | Query boundary')]
    fig.legend(handles=legend, loc='lower center', fontsize=9, bbox_to_anchor=(0.5,-0.03))
    fig.suptitle("AGNN Graph Edge Weights — Từng Generation\n"
                 "(Block sáng trên đường chéo = AGNN nhóm đúng cùng class)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(args.output_dir, 'graph_edge_weights.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {p}")

    # ─────────────────────────────────────────────────
    # VIZ 3: Intra vs Inter edge evolution bar chart
    # ─────────────────────────────────────────────────
    print(f"\n[3/4] Graph evolution chart...")
    sup_lbl_np = sup_lbl_node[0].numpy()
    intra_vals, inter_vals = [], []

    for ps in point_sims:
        mat = ps[0].detach().numpy()[:num_supports, :num_supports]
        intra, inter = [], []
        for i in range(num_supports):
            for j in range(num_supports):
                if i == j: continue
                (intra if sup_lbl_np[i]==sup_lbl_np[j] else inter).append(mat[i,j])
        intra_vals.append(np.mean(intra) if intra else 0)
        inter_vals.append(np.mean(inter) if inter else 0)

    x     = np.arange(num_gen)
    w     = 0.35
    fig, ax = plt.subplots(figsize=(max(5, 2.5*num_gen), 4.5))
    b1 = ax.bar(x-w/2, intra_vals, w, label='Same class (intra↑)', color='#2ecc71', edgecolor='k')
    b2 = ax.bar(x+w/2, inter_vals, w, label='Diff class (inter↓)',  color='#e74c3c', edgecolor='k')
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"Gen {i+1}" for i in range(num_gen)])
    ax.set_ylabel("Mean Edge Weight"); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(intra_vals), max(inter_vals))*1.25)
    ax.set_title("AGNN Graph Evolution: Intra vs Inter Class Edge Weights\n"
                 "(Intra↑ + Inter↓ qua các generation = AGNN học phân biệt đúng)",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(args.output_dir, 'graph_evolution.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {p}")

    # ─────────────────────────────────────────────────
    # VIZ 4: t-SNE
    # ─────────────────────────────────────────────────
    print(f"\n[4/4] t-SNE feature plot...")
    try:
        from sklearn.manifold import TSNE
        feats = last_emb[0].detach().numpy()          # [N, 128]
        lbl_all = np.array(
            list(sup_lbl_np) + gt_labels              # [N]
        )
        is_sup = np.arange(N) < num_supports

        perp = min(30, max(5, N // 3))
        feats_2d = TSNE(n_components=2, perplexity=perp,
                        n_iter=1000, random_state=42).fit_transform(feats)

        colors = plt.cm.tab10(np.linspace(0, 1, num_ways))
        fig, ax = plt.subplots(figsize=(8, 6))

        for ci, cname in enumerate(class_names):
            ms = is_sup  & (lbl_all==ci)
            mq = ~is_sup & (lbl_all==ci)
            if ms.any():
                ax.scatter(feats_2d[ms,0], feats_2d[ms,1],
                           c=[colors[ci]], marker='o', s=120,
                           edgecolors='k', lw=0.8, label=f"{cname} (support)", zorder=3)
            if mq.any():
                ax.scatter(feats_2d[mq,0], feats_2d[mq,1],
                           c=[colors[ci]], marker='*', s=250,
                           edgecolors='k', lw=0.8, label=f"{cname} (query)", zorder=4)

        ax.set_title("t-SNE: Không Gian Feature Sau Backbone\n"
                     "(●=support  ★=query — màu theo class)",
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=7, framealpha=0.8)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = os.path.join(args.output_dir, 'tsne_features.png')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {p}")
    except ImportError:
        print("  [SKIP] scikit-learn không có — bỏ qua t-SNE.")

    print(f"\n{'='*55}")
    print(f"  ✅ Done! Output saved to: {args.output_dir}")
    print(f"     gradcam_queries.png    — Grad-CAM heatmap")
    print(f"     graph_edge_weights.png — Edge matrix per generation")
    print(f"     graph_evolution.png    — Intra vs Inter evolution")
    print(f"     tsne_features.png      — t-SNE feature space")
    print(f"{'='*55}")


# ── Entry point ───────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='AGNN Visualization')
    p.add_argument('--config',       required=True)
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--images_root',  required=True,
                   help='Root folder chứa các sub-folder theo class')
    p.add_argument('--split_json',   required=True,
                   help='Path tới full_split.json')
    p.add_argument('--output_dir',   default='visualizations')
    p.add_argument('--num_ways',     type=int, default=5)
    p.add_argument('--num_shots',    type=int, default=5)
    p.add_argument('--num_queries',  type=int, default=3,
                   help='Số query ảnh MỖI CLASS (nhỏ lại để nhanh hơn)')
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--partition',    default='val',
                   help='train hoặc val — lấy class từ split nào')
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location('cfg', args.config)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = mod.config

    print(f"Backbone : {config['backbone']}")
    print(f"Episode  : {args.num_ways}-way {args.num_shots}-shot, "
          f"{args.num_queries} queries/class, seed={args.seed}")
    print(f"Partition: {args.partition}")
    run_all(args, config)

if __name__ == '__main__':
    main()
