import torch
import torch.nn as nn
import os
import argparse
import importlib.util
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from backbone import ResNet12, ConvNet, ResNet50Pretrained
from agnn import AGNN
from utils import allocate_tensors, initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode

def get_transform(image_size):
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
    
    # Matching the val/test transform from dataloader.py
    box_size = int(image_size * 1.15) if image_size > 0 else 96
    return transforms.Compose([
        transforms.Resize((box_size, box_size), interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        lambda x: np.asarray(x),
        transforms.ToTensor(),
        normalize
    ])

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
    return torch.stack(images), filenames

def main():
    parser = argparse.ArgumentParser(description='Inference script for AGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (model_best.pth.tar)')
    parser.add_argument('--support_dir', type=str, required=True, help='Path to support set directory (organized by class folders)')
    parser.add_argument('--query_dir', type=str, required=True, help='Path to query images directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
    
    args = parser.parse_args()

    # 1. Load Config
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    image_size = config.get('image_size', 84)
    transform = get_transform(image_size)

    # 2. Load Support Set
    class_names = sorted([d for d in os.listdir(args.support_dir) if os.path.isdir(os.path.join(args.support_dir, d))])
    num_ways = len(class_names)
    
    all_support_data = []
    all_support_labels = []
    
    print(f"Detected {num_ways} classes from support directory.")
    
    for idx, cname in enumerate(class_names):
        cpath = os.path.join(args.support_dir, cname)
        imgs, _ = load_images_from_folder(cpath, transform)
        num_shots = imgs.size(0)
        print(f" - Class '{cname}': {num_shots} images")
        all_support_data.append(imgs)
        all_support_labels.append(torch.full((num_shots,), idx, dtype=torch.long))

    support_data = torch.cat(all_support_data, dim=0).unsqueeze(0) # [1, total_support, 3, H, W]
    support_label = torch.cat(all_support_labels, dim=0).unsqueeze(0) # [1, total_support]
    num_total_supports = support_data.size(1)

    # 3. Load Query Set
    query_data, query_filenames = load_images_from_folder(args.query_dir, transform)
    num_queries = query_data.size(0)
    query_data = query_data.unsqueeze(0) # [1, num_queries, 3, H, W]
    
    # Dummy query labels (not used for prediction but needed for initialize_nodes_edges)
    query_label = torch.zeros((1, num_queries), dtype=torch.long)

    print(f"Loaded {num_queries} query images for prediction.")

    # 4. Initialize Models
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'])
    elif config['backbone'] == 'resnet50':
        enc_module = ResNet50Pretrained(emb_size=config['emb_size'])
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'])
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")

    gnn_module = AGNN(config['num_generation'],
                      config['train_config']['dropout'],
                      num_total_supports,
                      num_total_supports + num_queries,
                      config['train_config']['loss_indicator'],
                      config['point_distance_metric'])

    # 5. Load Weights
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # Handle DataParallel prefix if necessary
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

    enc_module.load_state_dict(clean_state_dict(checkpoint['enc_module_state_dict']))
    gnn_module.load_state_dict(clean_state_dict(checkpoint['gnn_module_state_dict']))
    
    enc_module.to(args.device).eval()
    gnn_module.to(args.device).eval()

    # 6. Inference
    tensors = allocate_tensors()
    # Mock batch for initialize_nodes_edges
    # IMPORTANT: We use unsqueeze(0) to add a dummy "tnt-batch" dimension 
    # because initialize_nodes_edges in utils.py calls .squeeze(0) immediately.
    batch = (support_data.unsqueeze(0), 
             support_label.unsqueeze(0), 
             query_data.unsqueeze(0), 
             query_label.unsqueeze(0))
    
    with torch.no_grad():
        _, support_label_node, _, _, all_data, _, node_feature_gd, edge_feature_gp = \
            initialize_nodes_edges(batch, num_total_supports, tensors, 1, num_queries, num_ways, args.device)
        
        all_data = all_data.to(args.device)
        node_feature_gd = node_feature_gd.to(args.device)
        edge_feature_gp = edge_feature_gp.to(args.device)
        support_label_node = support_label_node.to(args.device)

        last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, enc_module)

        point_similarities, _ = gnn_module(second_last_layer_data,
                                          last_layer_data,
                                          node_feature_gd,
                                          edge_feature_gp,
                                          support_label_node)

        # Get prediction from the last generation
        point_similarity = point_similarities[-1]
        
        # Predictions: BMM of similarity and one-hot encoded support labels
        # point_similarity: [batch, total_nodes, total_nodes]
        # We want similarity between query nodes and support nodes
        # query_nodes are from num_total_supports to end
        query_sim = point_similarity[:, num_total_supports:, :num_total_supports]
        
        # one_hot_encode expects (num_classes, class_idx, device)
        support_label_long = support_label_node.long()
        one_hot_support = one_hot_encode(num_ways, support_label_long, args.device)
        
        # query_node_pred: [batch, num_queries, num_ways]
        query_node_pred = torch.bmm(query_sim, one_hot_support)
        
        # Get final labels
        pred_labels = torch.argmax(query_node_pred, dim=-1).squeeze(0)
        confidences = torch.max(torch.softmax(query_node_pred, dim=-1), dim=-1)[0].squeeze(0)

    print("\n" + "="*30)
    print("      INFERENCE RESULTS")
    print("="*30)
    for i, fname in enumerate(query_filenames):
        pred_idx = pred_labels[i].item()
        conf = confidences[i].item()
        print(f"[{i+1:02d}] {fname:25s} -> Prediction: {class_names[pred_idx]:15s} (Conf: {conf:.2%})")
    print("="*30)

if __name__ == '__main__':
    main()
