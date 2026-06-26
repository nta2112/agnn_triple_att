import torch
import torch.nn as nn
import os
import argparse
import importlib.util
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader
from backbone import ResNet12, ConvNet, ResNet50Pretrained
from utils import backbone_two_stage_initialization

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

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

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size=84):
        self.root = root
        self.image_size = image_size
        self.transform = get_transform(image_size)
        
        self.class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.data = []
        self.labels = []
        
        img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        for lb, cname in enumerate(self.class_names):
            cpath = os.path.join(root, cname)
            for fname in sorted(os.listdir(cpath)):
                if any(fname.lower().endswith(ext) for ext in img_extensions):
                    self.data.append(os.path.join(cpath, fname))
                    self.labels.append(lb)
                    
        print(f"Loaded {len(self.class_names)} classes with {len(self.data)} total images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

def main():
    parser = argparse.ArgumentParser(description='Compute base class prototypes for AGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (model_best.pth.tar)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root folder')
    parser.add_argument('--output', type=str, default='app/base_prototypes.pth', help='Path to save base prototypes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loader')
    
    args = parser.parse_args()

    # 1. Load Config
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    image_size = config.get('image_size', 84)

    print(f"Loading dataset from: {args.dataset_root}")
    dataset = SimpleDataset(
        root=args.dataset_root,
        image_size=image_size
    )
    
    data_loader = TorchDataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2. Initialize Backbone Model
    cifar_flag = (image_size <= 32)
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
    elif config['backbone'] == 'resnet50':
        enc_module = ResNet50Pretrained(emb_size=config['emb_size'])
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")

    # 3. Load Checkpoint Weights
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    if 'enc_module_state_dict' in checkpoint:
        state_dict = clean_state_dict(checkpoint['enc_module_state_dict'])
    elif 'backbone_state_dict' in checkpoint:
        state_dict = clean_state_dict(checkpoint['backbone_state_dict'])
    elif 'state_dict' in checkpoint:
        state_dict = clean_state_dict(checkpoint['state_dict'])
    else:
        state_dict = clean_state_dict(checkpoint)
        
    enc_module.load_state_dict(state_dict, strict=False)
    enc_module.to(args.device).eval()

    # 4. Extract Features
    print("Extracting features from dataset...")
    
    num_classes = len(dataset.class_names)
    class_features_last = [[] for _ in range(num_classes)]
    class_features_second = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(args.device)
            imgs_expanded = imgs.unsqueeze(1)
            
            last_feat, second_last_feat = backbone_two_stage_initialization(imgs_expanded, enc_module)
            
            last_feat = last_feat.squeeze(1).cpu() 
            second_last_feat = second_last_feat.squeeze(1).cpu() 
            
            for i, label in enumerate(labels):
                class_idx = label.item()
                class_features_last[class_idx].append(last_feat[i])
                class_features_second[class_idx].append(second_last_feat[i])

    # 5. Compute Class Prototypes
    print("Computing class prototypes...")
    prototypes_last = []
    prototypes_second = []
    
    for class_idx in range(num_classes):
        feats_last = torch.stack(class_features_last[class_idx])
        feats_second = torch.stack(class_features_second[class_idx])
        
        mean_last = torch.mean(feats_last, dim=0)
        mean_second = torch.mean(feats_second, dim=0)
        
        prototypes_last.append(mean_last)
        prototypes_second.append(mean_second)
        print(f"Class '{dataset.class_names[class_idx]}' (index {class_idx}): {len(feats_last)} samples cached.")

    prototypes_last_tensor = torch.stack(prototypes_last) 
    prototypes_second_tensor = torch.stack(prototypes_second) 

    # 6. Save prototypes
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save({
        'class_names': dataset.class_names,
        'prototypes_last': prototypes_last_tensor,
        'prototypes_second': prototypes_second_tensor,
        'emb_size': config['emb_size']
    }, args.output)
    
    print(f"OK - Saved prototypes to: {args.output}")

if __name__ == '__main__':
    main()
