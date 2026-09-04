import torch
import torch.nn as nn
import os
import argparse
import importlib.util
from PIL import Image
import numpy as np
from backbone import ResNet12, ConvNet, ResNet50Pretrained
from agnn import AGNN
from utils import allocate_tensors, backbone_two_stage_initialization, one_hot_encode, label2edge

def get_transform(image_size):
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    box_size = int(image_size * 1.15) if image_size > 0 else 96

    def transform(img):
        img = img.convert("RGB")
        bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
        img = img.resize((box_size, box_size), resample=bicubic)

        left = max(0, (box_size - image_size) // 2)
        top = max(0, (box_size - image_size) // 2)
        img = img.crop((left, top, left + image_size, top + image_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(mean_pix, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(std_pix, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    return transform

def load_images_from_folder(folder, transform):
    images = []
    filenames = []
    img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    
    for fname in sorted(os.listdir(folder)):
        if any(fname.lower().endswith(ext) for ext in img_extensions):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
            filenames.append(fname)
    return torch.stack(images) if images else torch.empty(0), filenames

def main():
    parser = argparse.ArgumentParser(description='Inference script for AGNN (Supports Hybrid Open/Closed-World)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (model_best.pth.tar)')
    parser.add_argument('--support_dir', type=str, default=None, help='Path to support set directory (optional)')
    parser.add_argument('--base_prototypes', type=str, default=None, help='Path to pre-computed base prototypes .pth file (optional)')
    parser.add_argument('--query_dir', type=str, required=True, help='Path to query images directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
    
    args = parser.parse_args()

    if args.support_dir is None and args.base_prototypes is None:
        parser.error("At least one of --support_dir or --base_prototypes must be specified.")

    # 1. Load Config
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    image_size = config.get('image_size', 84)
    transform = get_transform(image_size)

    # 2. Initialize Models
    cifar_flag = (image_size <= 32)
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
    elif config['backbone'] == 'resnet50':
        enc_module = ResNet50Pretrained(emb_size=config['emb_size'])
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")

    # 3. Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    enc_module.load_state_dict(clean_state_dict(checkpoint['enc_module_state_dict']))
    enc_module.to(args.device).eval()

    # 4. Load Support Features & Prototypes
    base_class_names = []
    base_features_last = []
    base_features_second = []
    base_labels = []
    
    if args.base_prototypes:
        print(f"Loading base prototypes from: {args.base_prototypes}")
        proto_data = torch.load(args.base_prototypes, map_location=args.device, weights_only=False)
        base_class_names = proto_data['class_names']
        base_features_last = proto_data['prototypes_last'].to(args.device) # [num_base, emb_size]
        base_features_second = proto_data['prototypes_second'].to(args.device) # [num_base, emb_size]
        num_base = len(base_class_names)
        base_labels = torch.arange(num_base, dtype=torch.long, device=args.device)
        print(f" - Loaded {num_base} base classes: {base_class_names}")

    novel_class_names = []
    novel_features_last = []
    novel_features_second = []
    novel_labels = []
    
    if args.support_dir:
        novel_class_names = sorted([d for d in os.listdir(args.support_dir) if os.path.isdir(os.path.join(args.support_dir, d))])
        print(f"Detected {len(novel_class_names)} novel classes from support directory.")
        
        for idx, cname in enumerate(novel_class_names):
            cpath = os.path.join(args.support_dir, cname)
            imgs, _ = load_images_from_folder(cpath, transform)
            num_shots = imgs.size(0)
            if num_shots == 0:
                continue
            print(f" - Class '{cname}': {num_shots} images")
            
            with torch.no_grad():
                imgs = imgs.to(args.device)
                last, second = backbone_two_stage_initialization(imgs.unsqueeze(0), enc_module)
                novel_features_last.append(last.squeeze(0))
                novel_features_second.append(second.squeeze(0))
                
                # Novel labels start after base labels
                start_label = len(base_class_names)
                novel_labels.append(torch.full((num_shots,), start_label + idx, dtype=torch.long, device=args.device))

    # Combine support features and labels
    class_names = base_class_names + novel_class_names
    num_ways = len(class_names)
    
    combined_last = []
    combined_second = []
    combined_labels = []
    
    if len(base_class_names) > 0:
        combined_last.append(base_features_last)
        combined_second.append(base_features_second)
        combined_labels.append(base_labels)
        
    if len(novel_class_names) > 0:
        combined_last.append(torch.cat(novel_features_last, dim=0))
        combined_second.append(torch.cat(novel_features_second, dim=0))
        combined_labels.append(torch.cat(novel_labels, dim=0))
        
    support_features_last = torch.cat(combined_last, dim=0) # [num_total_supports, emb_size]
    support_features_second = torch.cat(combined_second, dim=0) # [num_total_supports, emb_size]
    support_label = torch.cat(combined_labels, dim=0) # [num_total_supports]
    num_total_supports = support_features_last.size(0)

    # 5. Load Query Set
    query_imgs, query_filenames = load_images_from_folder(args.query_dir, transform)
    num_queries = query_imgs.size(0)
    if num_queries == 0:
        raise ValueError(f"No query images found in: {args.query_dir}")
    print(f"Loaded {num_queries} query images for prediction.")

    # Extract Query Features
    with torch.no_grad():
        query_imgs = query_imgs.to(args.device)
        query_last, query_second = backbone_two_stage_initialization(query_imgs.unsqueeze(0), enc_module)
        # query_last is [1, num_queries, emb_size]
        # query_second is [1, num_queries, emb_size]

    # Initialize GNN Module dynamically based on total support count and total nodes
    gnn_module = AGNN(in_c=config['emb_size'],
                      num_generations=config['num_generation'],
                      dropout=config['train_config']['dropout'],
                      num_support_sample=num_total_supports,
                      num_sample=num_total_supports + num_queries,
                      loss_indicator=config['train_config']['loss_indicator'],
                      point_metric=config['point_distance_metric'],
                      ablation_mode=config.get('ablation_mode', 'full'))

    gnn_module.load_state_dict(clean_state_dict(checkpoint['gnn_module_state_dict']))
    gnn_module.to(args.device).eval()

    # 6. GNN Inference Setup
    last_layer_data = torch.cat([support_features_last.unsqueeze(0), query_last], dim=1) # [1, total_nodes, emb_size]
    second_last_layer_data = torch.cat([support_features_second.unsqueeze(0), query_second], dim=1) # [1, total_nodes, emb_size]
    
    support_label_exp = support_label.unsqueeze(0) # [1, num_total_supports]

    with torch.no_grad():
        # Build node_feature_gd
        node_gd_init_support = label2edge(support_label_exp, args.device)
        node_gd_init_query = (torch.ones([1, num_queries, num_total_supports]) * torch.tensor(1. / num_total_supports)).to(args.device)
        node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)

        # Build edge_feature_gp
        num_total_nodes = num_total_supports + num_queries
        edge_feature_gp = torch.zeros(1, num_total_nodes, num_total_nodes, device=args.device)
        edge_feature_gp[:, :num_total_supports, :num_total_supports] = node_gd_init_support
        edge_feature_gp[:, num_total_supports:, :num_total_supports] = 1. / num_total_supports
        edge_feature_gp[:, :num_total_supports, num_total_supports:] = 1. / num_total_supports
        for i in range(num_queries):
            edge_feature_gp[:, num_total_supports + i, num_total_supports + i] = 1.0

        # GNN Forward Pass
        point_similarities, _ = gnn_module(second_last_layer_data,
                                           last_layer_data,
                                           node_feature_gd,
                                           edge_feature_gp,
                                           support_label_exp)

        # Predict labels using the last layer similarities
        point_similarity = point_similarities[-1]
        query_sim = point_similarity[:, num_total_supports:, :num_total_supports] # [1, num_queries, num_total_supports]
        
        # One-hot encoded support labels
        one_hot_support = one_hot_encode(num_ways, support_label_exp.long(), args.device) # [1, num_total_supports, num_ways]
        
        # Class logits
        query_node_pred = torch.bmm(query_sim, one_hot_support) # [1, num_queries, num_ways]
        
        pred_labels = torch.argmax(query_node_pred, dim=-1).squeeze(0) # [num_queries]
        confidences = torch.max(torch.softmax(query_node_pred, dim=-1), dim=-1)[0].squeeze(0) # [num_queries]

    print("\n" + "="*50)
    print("                INFERENCE RESULTS")
    print("="*50)
    
    # Handle single query case (ensure iterable)
    if num_queries == 1:
        pred_labels = [pred_labels]
        confidences = [confidences]

    for i, fname in enumerate(query_filenames):
        pred_idx = pred_labels[i].item()
        conf = confidences[i].item()
        print(f"[{i+1:02d}] {fname:25s} -> Prediction: {class_names[pred_idx]:20s} (Conf: {conf:.2%})")
    print("="*50)

if __name__ == '__main__':
    main()
