import os
import sys
import argparse
import importlib.util
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Thêm đường dẫn gốc vào sys.path để import các module local
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backbone import ResNet12, ConvNet, ResNet50Pretrained
from agnn import AGNN
from utils import (allocate_tensors, initialize_nodes_edges,
                   backbone_two_stage_initialization, one_hot_encode)

def get_transform(image_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        norm,
    ])

def sample_episode(images_root, split_json, num_ways, num_shots, num_queries, seed, partition='val'):
    rng = random.Random(seed)
    with open(split_json, 'r') as f:
        split = json.load(f)
    avail_classes = split[partition]
    
    valid_classes = []
    for cls in avail_classes:
        cls_dir = os.path.join(images_root, cls)
        if not os.path.isdir(cls_dir): continue
        imgs = [p for p in os.listdir(cls_dir) if p.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) >= num_shots + num_queries:
            valid_classes.append(cls)

    chosen = rng.sample(valid_classes, num_ways)
    support_paths, query_paths = [], []
    for cls in chosen:
        cls_dir = os.path.join(images_root, cls)
        imgs = sorted([os.path.join(cls_dir, p) for p in os.listdir(cls_dir) if p.lower().endswith(('.jpg','.jpeg','.png'))])
        rng.shuffle(imgs)
        support_paths.append(imgs[:num_shots])
        query_paths.append(imgs[num_shots: num_shots + num_queries])
    return chosen, support_paths, query_paths

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
        config['emb_size'],
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

def run_tsne(args, config):
    os.makedirs(args.output_dir, exist_ok=True)
    num_ways, num_shots, num_queries = args.num_ways, args.num_shots, args.num_queries
    tf = get_transform(config.get('image_size', 84))

    print(f"Sampling {num_ways}-way {num_shots}-shot episode...")
    class_names, support_paths, query_paths = sample_episode(
        args.images_root, args.split_json, num_ways, num_shots, num_queries, args.seed)

    sup_tensors, sup_labels = [], []
    for ci, paths in enumerate(support_paths):
        for p in paths:
            sup_tensors.append(get_transform(config.get('image_size', 84))(Image.open(p).convert('RGB')))
            sup_labels.append(ci)

    qry_tensors, gt_labels = [], []
    for ci, paths in enumerate(query_paths):
        for p in paths:
            qry_tensors.append(get_transform(config.get('image_size', 84))(Image.open(p).convert('RGB')))
            gt_labels.append(ci)

    num_supports, num_q_total = len(sup_tensors), len(qry_tensors)
    N = num_supports + num_q_total
    
    # Model forward
    enc, gnn = load_models(args, config, num_supports, num_q_total)
    
    all_data = torch.stack(sup_tensors + qry_tensors).unsqueeze(0) # [1, N, 3, H, W]
    support_label = torch.tensor(sup_labels, dtype=torch.long).unsqueeze(0)
    query_label = torch.zeros(1, num_q_total, dtype=torch.long)
    
    tensors = allocate_tensors()
    batch = (all_data[:, :num_supports].unsqueeze(0), support_label.unsqueeze(0),
             all_data[:, num_supports:].unsqueeze(0), query_label.unsqueeze(0))

    _, sup_lbl_node, _, _, all_data_v, _, node_feat_gd, edge_feat_gp = \
        initialize_nodes_edges(batch, num_supports, tensors, 1, num_q_total, 1, 'cpu')

    with torch.no_grad():
        last_emb, second_emb = backbone_two_stage_initialization(all_data_v, enc)
        _, _, agnn_feats = gnn(second_emb, last_emb, node_feat_gd, edge_feat_gp, sup_lbl_node)

    # Preparing for t-SNE
    print("Computing t-SNE...")
    feats_b = last_emb[0].numpy()
    feats_a = agnn_feats[0].numpy()
    lbl_all = np.array(sup_labels + gt_labels)
    is_sup = np.arange(N) < num_supports
    
    perp = min(30, max(5, N // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed)
    
    low_b = tsne.fit_transform(feats_b)
    low_a = tsne.fit_transform(feats_a)
    
    # Plotting
    colors = plt.cm.tab10(np.linspace(0, 1, num_ways))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    titles = ["Backbone Features (Initial)", f"AGNN Refined Features ({config['num_generation']} Layers)"]
    data_2d = [low_b, low_a]
    
    for idx in range(2):
        ax = axes[idx]
        curr_2d = data_2d[idx]
        for ci, cname in enumerate(class_names):
            ms = is_sup & (lbl_all == ci)
            mq = ~is_sup & (lbl_all == ci)
            ax.scatter(curr_2d[ms, 0], curr_2d[ms, 1], c=[colors[ci]], marker='o', s=100, edgecolors='k', label=f"{cname} (S)")
            ax.scatter(curr_2d[mq, 0], curr_2d[mq, 1], c=[colors[ci]], marker='*', s=200, edgecolors='k', label=f"{cname} (Q)")
        ax.set_title(titles[idx], fontweight='bold')
        ax.grid(True, alpha=0.2)
        if idx == 1: ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"tsne_layer_{config['num_generation']}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images_root', required=True)
    parser.add_argument('--split_json', required=True)
    parser.add_argument('--output_dir', default='visualizations/tsne')
    parser.add_argument('--num_ways', type=int, default=5)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--num_queries', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('cfg', args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    run_tsne(args, config.config)
