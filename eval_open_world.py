"""
eval_open_world.py — Open World Few-Shot Evaluation (Paper Source 6 aligned)

Metrics  : HM (Harmonic Mean), S (Seen Acc), U (Unseen Acc), AUC (S-U curve), Top-1
Threshold: Calibration factor γ swept automatically to maximise HM
Plot     : Seen vs. Unseen Accuracy curve + Confidence distribution
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, random, argparse, importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backbone import ResNet12, ConvNet, ResNet50Pretrained, LaStViTBackbone
from agnn import AGNN
from utils import backbone_two_stage_initialization, one_hot_encode
from main_gnn import load_flexible

try:
    from dataloader import MiniImagenet, TieredImagenet, CUB200, Flowers, CustomImageFolder
except ImportError:
    from dataloader import CustomImageFolder


# ─────────────────────────────────────────────
# Episode sampling
# ─────────────────────────────────────────────

def _get_tensor(dataset, index):
    """Return a float tensor [C,H,W] for a single sample, handling both cached tensor and PIL paths."""
    item = dataset._get_pil(index)
    if getattr(dataset, '_cache_is_tensor', False):
        return item.float()
    return dataset.aug_transform(item).float()


def sample_episode(dataset, num_ways, num_shots, num_known_q,
                   num_unknown_ways, num_unknown_q):
    """
    Sample one Open-World episode.

    Returns
    -------
    support_data   : Tensor [1, num_ways*num_shots, C, H, W]
    support_label  : Tensor [1, num_ways*num_shots]          (0..num_ways-1)
    query_data     : Tensor [1, total_queries, C, H, W]
    query_label    : Tensor [1, total_queries]               (-1 = unknown)
    is_unknown     : bool list of length total_queries
    """
    all_cls = dataset.full_class_list
    l2i = dataset.label2ind

    # Adaptive class selection
    actual_unknown_ways = min(num_unknown_ways, len(all_cls) - num_ways)
    
    if actual_unknown_ways <= 0:
        raise ValueError(
            f"Dataset only has {len(all_cls)} classes; need at least {num_ways + 1} "
            f"for Open World evaluation.")

    if actual_unknown_ways < num_unknown_ways:
        print(f"  [Warning] Dataset has only {len(all_cls)} classes. "
              f"Adjusting Unknown ways to {actual_unknown_ways}.")

    chosen = random.sample(all_cls, num_ways + actual_unknown_ways)
    known_cls   = chosen[:num_ways]
    unknown_cls = chosen[num_ways:]

    C, H, W = dataset.data_size
    n_sup   = num_ways * num_shots
    n_knq   = num_ways * num_known_q
    n_unq   = actual_unknown_ways * num_unknown_q
    n_q     = n_knq + n_unq

    sup_data  = torch.empty(1, n_sup, C, H, W)
    sup_label = torch.empty(1, n_sup, dtype=torch.float32)
    q_data    = torch.empty(1, n_q,   C, H, W)
    q_label   = torch.full((1, n_q), -1, dtype=torch.float32)

    # Support + known queries
    for ci, cls in enumerate(known_cls):
        pool = l2i[cls]
        idx  = random.sample(pool, num_shots + num_known_q)
        for k, ii in enumerate(idx[:num_shots]):
            sup_data[0, ci*num_shots + k]  = _get_tensor(dataset, ii)
            sup_label[0, ci*num_shots + k] = ci
        for k, ii in enumerate(idx[num_shots:]):
            pos = ci*num_known_q + k
            q_data[0, pos]  = _get_tensor(dataset, ii)
            q_label[0, pos] = ci

    # Unknown queries
    for ui, cls in enumerate(unknown_cls):
        pool = l2i[cls]
        idx  = random.sample(pool, num_unknown_q)
        for k, ii in enumerate(idx):
            pos = n_knq + ui*num_unknown_q + k
            q_data[0, pos] = _get_tensor(dataset, ii)
            # q_label stays -1

    is_unknown = [False]*n_knq + [True]*n_unq
    return sup_data, sup_label, q_data, q_label, is_unknown


# ─────────────────────────────────────────────
# Single episode inference
# ─────────────────────────────────────────────

def run_episode(enc, gnn, sup_data, sup_label, q_data, num_ways, device):
    """
    Run backbone + AGNN for one episode.

    Returns
    -------
    confidence : np.ndarray [total_queries]   max class score per query
    pred_class : np.ndarray [total_queries]   argmax class index
    """
    num_supports = sup_data.size(1)
    num_queries  = q_data.size(1)
    num_samples  = num_supports + num_queries

    all_data = torch.cat([sup_data, q_data], dim=1).to(device)   # [1, N, C, H, W]
    sup_lbl  = sup_label.to(device)

    with torch.no_grad():
        last, second = backbone_two_stage_initialization(all_data, enc)

        # Initialise edge (uniform for queries)
        edge = torch.zeros(1, num_samples, num_samples, device=device)
        edge[:, :num_supports, :num_supports] = 1.0 / num_supports
        edge[:, num_supports:, :num_supports] = 1.0 / num_supports
        edge[:, num_supports:, num_supports:] = 0.0
        for i in range(num_queries):
            edge[:, num_supports+i, num_supports+i] = 1.0

        # GNN forward
        point_sims, _ = gnn(second, last,
                            torch.zeros(1, num_samples, num_ways, device=device),
                            edge,
                            sup_lbl.long())

        sim_last = point_sims[-1]  # [1, num_samples, num_samples]

        # Query-to-support scores → class scores
        q2s = sim_last[:, num_supports:, :num_supports]     # [1, nq, ns]
        oh  = one_hot_encode(num_ways, sup_lbl.long(), device)  # [1, ns, nw]
        cls_scores = torch.bmm(q2s, oh).squeeze(0).cpu().numpy()  # [nq, nw]

    confidence = cls_scores.max(axis=1)   # [nq]
    pred_class = cls_scores.argmax(axis=1)
    return confidence, pred_class


# ─────────────────────────────────────────────
# Compute S, U, HM at a given γ
# ─────────────────────────────────────────────

def compute_su(confidences, pred_classes, true_labels, is_unknown, gamma):
    """
    Parameters
    ----------
    confidences, pred_classes, true_labels, is_unknown : 1-D arrays (all queries)
    gamma : float, calibration factor (rejection threshold)

    Returns
    -------
    S   : Seen accuracy   (known queries correctly classified AND not rejected)
    U   : Unseen accuracy (unknown queries correctly rejected)
    HM  : Harmonic mean of S and U
    """
    confidences  = np.array(confidences)
    pred_classes = np.array(pred_classes)
    true_labels  = np.array(true_labels)
    is_unknown   = np.array(is_unknown)

    known_mask   = ~is_unknown
    unknown_mask =  is_unknown

    # Seen accuracy: not rejected AND correct prediction
    if known_mask.sum() == 0:
        S = 0.0
    else:
        not_rejected = confidences > gamma
        correct      = (pred_classes == true_labels)
        S = float((not_rejected & correct & known_mask).sum()) / known_mask.sum()

    # Unseen accuracy: correctly rejected
    if unknown_mask.sum() == 0:
        U = 1.0
    else:
        rejected = confidences <= gamma
        U = float((rejected & unknown_mask).sum()) / unknown_mask.sum()

    HM = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0
    return S, U, HM


# ─────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────

def evaluate(enc, gnn, dataset, args):
    """Run all episodes, sweep γ, compute metrics + plot."""
    num_ways         = args.num_ways
    num_shots        = args.num_shots
    num_known_q      = args.num_queries
    num_unknown_ways = args.num_unknown_ways
    num_unknown_q    = args.num_unknown_queries

    all_conf   = []
    all_pred   = []
    all_true   = []
    all_is_unk = []

    print(f"\n[Open World] Running {args.num_episodes} episodes …")
    for ep in range(args.num_episodes):
        if (ep+1) % 100 == 0:
            print(f"  Episode {ep+1}/{args.num_episodes}")

        sup_d, sup_l, q_d, q_l, is_unk = sample_episode(
            dataset, num_ways, num_shots, num_known_q,
            num_unknown_ways, num_unknown_q)

        conf, pred = run_episode(enc, gnn, sup_d, sup_l, q_d,
                                  num_ways, args.device)
        true = q_l.squeeze(0).numpy()

        all_conf.extend(conf.tolist())
        all_pred.extend(pred.tolist())
        all_true.extend(true.tolist())
        all_is_unk.extend(is_unk)

    all_conf   = np.array(all_conf)
    all_pred   = np.array(all_pred)
    all_true   = np.array(all_true)
    all_is_unk = np.array(all_is_unk)

    # ── Sweep calibration factor γ ──────────────────────────────────────
    gammas    = np.linspace(0.0, 1.0, 201)
    S_list, U_list, HM_list = [], [], []

    for g in gammas:
        S, U, HM = compute_su(all_conf, all_pred, all_true, all_is_unk, g)
        S_list.append(S)
        U_list.append(U)
        HM_list.append(HM)

    S_arr  = np.array(S_list)
    U_arr  = np.array(U_list)
    HM_arr = np.array(HM_list)

    best_idx = HM_arr.argmax()
    opt_gamma = gammas[best_idx]
    best_S  = S_arr[best_idx]
    best_U  = U_arr[best_idx]
    best_HM = HM_arr[best_idx]

    # AUC under Seen-Unseen curve (trapezoidal, normalised to [0,1])
    sort_idx = np.argsort(S_arr)
    auc = float(np.trapz(U_arr[sort_idx], S_arr[sort_idx]))
    # AUC normalised: divide by the range of S covered
    s_range = S_arr.max() - S_arr.min()
    auc_norm = (auc / s_range) if s_range > 1e-6 else 0.0

    # Top-1 accuracy (known queries only, at optimal γ)
    known_mask   = ~all_is_unk
    not_rejected = all_conf > opt_gamma
    top1 = float((known_mask & not_rejected & (all_pred == all_true)).sum()) / \
           max(known_mask.sum(), 1)

    # ── Print results ───────────────────────────────────────────────────
    print("\n" + "="*52)
    print("  OPEN WORLD EVALUATION RESULTS")
    print(f"  Episodes : {args.num_episodes}  |  {num_ways}-way {num_shots}-shot")
    print(f"  Unknown  : {num_unknown_ways} class(es) × {num_unknown_q} queries/class")
    print("="*52)
    print(f"  Optimal γ (calibration factor) : {opt_gamma:.3f}")
    print(f"  Seen Accuracy        (S)       : {best_S*100:.2f}%")
    print(f"  Unseen Accuracy      (U)       : {best_U*100:.2f}%")
    print(f"  Harmonic Mean        (HM)      : {best_HM*100:.2f}%")
    print(f"  AUC (S-U curve, norm)          : {auc_norm:.4f}")
    print(f"  Top-1 Accuracy (known, @γ*)    : {top1*100:.2f}%")
    print("="*52)

    # ── Plot ─────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    _plot_su_curve(S_arr, U_arr, best_S, best_U, opt_gamma, auc_norm, args.save_dir)
    _plot_conf_dist(all_conf, all_is_unk, opt_gamma, args.save_dir)

    return {
        'S': best_S, 'U': best_U, 'HM': best_HM,
        'AUC': auc_norm, 'Top1': top1, 'gamma': opt_gamma,
    }


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def _plot_su_curve(S, U, best_S, best_U, gamma, auc, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sort_idx = np.argsort(S)
    ax.plot(S[sort_idx], U[sort_idx], color='steelblue', lw=2, label=f'S-U curve (AUC={auc:.3f})')
    ax.scatter([best_S], [best_U], color='red', zorder=5,
               label=f'Optimal γ={gamma:.3f}  HM={2*best_S*best_U/(best_S+best_U+1e-9)*100:.1f}%')
    ax.set_xlabel('Seen Accuracy (S)', fontsize=12)
    ax.set_ylabel('Unseen Accuracy (U)', fontsize=12)
    ax.set_title('Open World: Seen vs. Unseen Accuracy Curve', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    path = os.path.join(save_dir, 'su_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved: {path}")


def _plot_conf_dist(confidences, is_unknown, gamma, save_dir):
    conf_known   = confidences[~is_unknown]
    conf_unknown = confidences[ is_unknown]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf_known,   bins=50, alpha=0.65, color='steelblue', label='Known (Seen)')
    ax.hist(conf_unknown, bins=50, alpha=0.65, color='tomato',    label='Unknown (Unseen)')
    ax.axvline(gamma, color='black', linestyle='--', lw=1.5, label=f'γ* = {gamma:.3f}')
    ax.set_xlabel('Confidence (max class score)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution: Known vs. Unknown', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confidence_dist.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved: {path}")


# ─────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────

def build_backbone(config):
    backbone = config['backbone']
    emb_size = config['emb_size']
    if backbone == 'resnet12':
        return ResNet12(emb_size=emb_size)
    elif backbone == 'resnet50':
        return ResNet50Pretrained(emb_size=emb_size)
    elif backbone == 'last_vit':
        return LaStViTBackbone(emb_size=emb_size, pretrained=False)
    elif backbone == 'convnet':
        return ConvNet(emb_size=emb_size)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


def build_gnn(config):
    train_opt = config['train_config']
    # Use the same dropout as training to ensure model architecture (layer count) matches checkpoint
    dropout = train_opt.get('dropout', 0.0)
    
    return AGNN(
        in_c=config['emb_size'],
        num_generations=config['num_generation'],
        dropout=dropout,
        num_support_sample=train_opt['num_ways'] * train_opt['num_shots'],
        num_sample=train_opt['num_ways'] * (train_opt['num_shots'] + train_opt['num_queries']),
        loss_indicator=train_opt['loss_indicator'],
        point_metric=config['point_distance_metric'],
        ablation_mode=config.get('ablation_mode', 'full'),
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Open World Few-Shot Evaluation (Source 6 aligned)')
    parser.add_argument('--config',     type=str, required=True,  help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True,  help='Checkpoint path (model_best.pth.tar)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--device',     type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_episodes', type=int, default=600,  help='Number of evaluation episodes')
    parser.add_argument('--num_ways',     type=int, default=5,    help='Known classes per episode')
    parser.add_argument('--num_shots',    type=int, default=5,    help='Support images per known class')
    parser.add_argument('--num_queries',  type=int, default=15,   help='Known queries per class')
    parser.add_argument('--num_unknown_ways',    type=int, default=5,  help='Unknown (distractor) classes per episode')
    parser.add_argument('--num_unknown_queries', type=int, default=15, help='Unknown queries per distractor class')
    parser.add_argument('--partition',  type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--save_dir',   type=str, default='./ow_results', help='Output directory for plots/logs')
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location("cfg", args.config)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config    = mod.config
    train_opt = config['train_config']

    # ── Build models ─────────────────────────────────────────────────────
    enc = build_backbone(config).to(args.device)
    gnn = build_gnn(config).to(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    load_flexible(enc, ckpt['enc_module_state_dict'])
    load_flexible(gnn, ckpt['gnn_module_state_dict'])
    enc.eval(); gnn.eval()

    # ── Load dataset ─────────────────────────────────────────────────────
    dataset_name = config['dataset_name']
    img_size     = config.get('image_size', 84)
    split_path   = config.get('split_path', None)

    if dataset_name == 'custom':
        dataset = CustomImageFolder(
            root=args.dataset_root,
            partition=args.partition,
            image_size=img_size,
            split_path=split_path,
        )
    elif dataset_name == 'mini-imagenet':
        dataset = MiniImagenet(root=args.dataset_root, partition=args.partition)
    elif dataset_name == 'tiered-imagenet':
        dataset = TieredImagenet(root=args.dataset_root, partition=args.partition)
    elif dataset_name == 'cub-200-2011':
        dataset = CUB200(root=args.dataset_root, partition=args.partition)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if hasattr(dataset, 'cache_to_memory'):
        dataset.cache_to_memory()

    min_cls = args.num_ways + args.num_unknown_ways
    if len(dataset.full_class_list) < min_cls:
        raise RuntimeError(
            f"Partition '{args.partition}' only has {len(dataset.full_class_list)} classes, "
            f"but num_ways({args.num_ways}) + num_unknown_ways({args.num_unknown_ways}) = {min_cls} required.")

    # ── Evaluate ─────────────────────────────────────────────────────────
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    evaluate(enc, gnn, dataset, args)


if __name__ == '__main__':
    main()
