"""
pretrain.py — Pretrain ResNet12 backbone trên tập dữ liệu phân loại thông thường.

Chiến lược đảm bảo tương thích hoàn toàn với AGNN:
- BackboneClassifier wrap ResNet12 + linear head NGOÀI backbone.
- Chỉ lưu `backbone.state_dict()` (keys khớp 100% với ResNet12 trong main_gnn.py).
- Khi load vào AGNN dùng strict=True → không có lỗi key mismatch.

Cách dùng:
    python pretrain.py \\
        --dataset_root ./data \\
        --split_path    ./data/split.json \\
        --checkpoint_dir ./pretrain_checkpoints \\
        --log_dir        ./pretrain_logs \\
        --num_epochs 100 --lr 1e-3 --batch_size 64
"""

import os
import json
import random
import logging
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torchvision.transforms as transforms

# Import ResNet12 trực tiếp (không qua backbone.py module-level vì backbone.py có
# import LaStViT có thể lỗi nếu không install torchvision đủ version mới nhất).
import importlib.util as _iutil
import sys as _sys

def _load_resnet12():
    """Load ResNet12 từ backbone.py an toàn bằng cách monkey-patch import lỗi."""
    # Fake module thế chỗ temp_last_vit.last_vit_model nếu chưa install
    if 'temp_last_vit' not in _sys.modules:
        import types
        fake_tvit = types.ModuleType('temp_last_vit')
        fake_model = types.ModuleType('temp_last_vit.last_vit_model')
        fake_model.build_last_vit_b16 = None
        fake_tvit.last_vit_model = fake_model
        _sys.modules['temp_last_vit'] = fake_tvit
        _sys.modules['temp_last_vit.last_vit_model'] = fake_model

    spec = _iutil.spec_from_file_location('backbone', 'backbone.py')
    mod  = _iutil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ResNet12

ResNet12 = _load_resnet12()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Standard image classification dataset cho backbone pretraining.

    Cấu trúc thư mục:
        dataset_root/
            classA/  img1.jpg  img2.jpg  ...
            classB/  ...
            ...

    Phân chia train/val được điều khiển bởi split.json:
        {
            "train": ["classA", "classB", ...],
            "val":   ["classC", "classD", ...]
        }
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm',
                      '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root: str, split_path: str,
                 partition: str = 'train', image_size: int = 84):
        self.root       = root
        self.partition  = partition
        self.image_size = image_size

        # ── Transforms ────────────────────────────────────────────────────────
        mean      = [0.485, 0.456, 0.406]
        std       = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        box_size  = int(image_size * 1.15)

        if partition == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(box_size),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=.1, contrast=.1,
                                       saturation=.1, hue=.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(box_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        # ── Đọc split.json ────────────────────────────────────────────────────
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split.json không tồn tại: {split_path}")

        with open(split_path, 'r') as f:
            splits = json.load(f)

        class_names = sorted(splits.get(partition, []))
        if not class_names:
            raise ValueError(
                f"Partition '{partition}' trống trong {split_path}. "
                f"Hãy kiểm tra lại file split.json.")

        # ── Xây dựng danh sách ảnh ───────────────────────────────────────────
        self.data   = []   # list of absolute image paths
        self.labels = []   # list of int labels (0 .. num_classes-1)

        for lb, cname in enumerate(class_names):
            cpath = os.path.join(root, cname)
            if not os.path.isdir(cpath):
                raise FileNotFoundError(
                    f"Không tìm thấy thư mục class: {cpath}")

            imgs_in_class = [
                fname for fname in sorted(os.listdir(cpath))
                if any(fname.lower().endswith(ext)
                       for ext in self.IMG_EXTENSIONS)
            ]
            if not imgs_in_class:
                print(f"  WARNING: class '{cname}' không có ảnh nào!")

            for fname in imgs_in_class:
                self.data.append(os.path.join(cpath, fname))
                self.labels.append(lb)

        self.num_classes = len(class_names)
        print(f"[PretrainDataset] partition={partition!r:6s}  "
              f"classes={self.num_classes}  images={len(self.data)}")

        if len(self.data) == 0:
            raise ValueError(
                f"Không tìm thấy ảnh nào cho partition '{partition}'! "
                f"Kiểm tra lại dataset_root và split.json.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img   = Image.open(self.data[idx]).convert('RGB')
        img   = self.transform(img)
        label = self.labels[idx]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BackboneClassifier(nn.Module):
    """
    ResNet12 backbone + linear classification head.

    Thiết kế để đảm bảo tương thích hoàn toàn với AGNN:
    - `self.backbone` là instance ResNet12 hoàn chỉnh (giống hệt trong main_gnn.py).
    - `self.classifier` là nn.Linear NGOÀI backbone → không nằm trong backbone.state_dict().
    - Khi lưu checkpoint, chỉ lưu `self.backbone.state_dict()`.
    - Khi load vào enc_module (ResNet12) trong main_gnn.py, dùng strict=True.
    - Không bao giờ xảy ra lỗi key mismatch.
    """

    def __init__(self, emb_size: int, num_classes: int, cifar_flag: bool = False):
        super().__init__()
        # ResNet12 hoàn chỉnh — giống hệt đối tượng được khởi tạo trong main_gnn.py
        self.backbone   = ResNet12(emb_size=emb_size, cifar_flag=cifar_flag)
        
        # Classifier heads cho cả 2 nhánh (Layer Last và Layer Second)
        # Việc train cả 2 nhánh giúp backbone học được cả đặc trưng ngữ nghĩa và cấu trúc
        self.classifier_last   = nn.Linear(emb_size, num_classes)
        self.classifier_second = nn.Linear(emb_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        # backbone trả về [layer_last_feat, layer_second_feat]
        feats = self.backbone(x)
        
        logits_last   = self.classifier_last(feats[0])
        logits_second = self.classifier_second(feats[1])
        
        return logits_last, logits_second

    def get_backbone_state_dict(self) -> dict:
        """
        Trả về state_dict của backbone (ResNet12 thuần).
        Keys khớp 100% với ResNet12 trong main_gnn.py → load strict=True an toàn.
        """
        return self.backbone.state_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PretrainTrainer:
    def __init__(self, model: BackboneClassifier,
                 train_loader: TorchDataLoader,
                 val_loader: TorchDataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 device: torch.device,
                 checkpoint_dir: str,
                 log: logging.Logger):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.log            = log
        self.criterion      = nn.CrossEntropyLoss()
        self.best_val_acc   = 0.0

        os.makedirs(checkpoint_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for imgs, labels in self.train_loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits_last, logits_second = self.model(imgs)
            
            # Tính loss trên cả 2 nhánh
            loss_last   = self.criterion(logits_last, labels)
            loss_second = self.criterion(logits_second, labels)
            loss        = loss_last + loss_second
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            # Dùng nhánh chính (last) để tính accuracy báo cáo
            correct    += (logits_last.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for imgs, labels in self.val_loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits_last, logits_second = self.model(imgs)
            loss_last   = self.criterion(logits_last, labels)
            loss_second = self.criterion(logits_second, labels)
            loss        = loss_last + loss_second

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits_last.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    # ──────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        """
        Lưu checkpoint chứa backbone_state_dict (không có classifier head).

        Keys trong file:
            epoch               : int
            val_acc             : float
            backbone_state_dict : dict  ← trực tiếp load vào ResNet12 (strict=True)
            emb_size            : int   ← để kiểm tra tính tương thích khi load
        """
        ckpt = {
            'epoch':               epoch,
            'val_acc':             val_acc,
            'backbone_state_dict': self.model.get_backbone_state_dict(),
            'emb_size':            self.model.backbone.emb_size,
        }

        last_path = os.path.join(self.checkpoint_dir, 'backbone_last.pth')
        torch.save(ckpt, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'backbone_best.pth')
            torch.save(ckpt, best_path)
            self.log.info(f'  ★ New best  val_acc={val_acc:.4f} '
                          f'→ saved {best_path}')

    # ──────────────────────────────────────────────────────────────────────────

    def run(self, num_epochs: int):
        self.log.info('=' * 65)
        self.log.info('  Bắt đầu pretrain backbone ResNet12')
        self.log.info('=' * 65)

        for epoch in range(1, num_epochs + 1):
            tr_loss, tr_acc = self._train_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            va_loss, va_acc = self._val_epoch()

            is_best = va_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = va_acc

            self.log.info(
                f'Epoch [{epoch:4d}/{num_epochs}]  '
                f'train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  '
                f'val_loss={va_loss:.4f}  val_acc={va_acc:.4f}'
                + ('  ★BEST' if is_best else '')
            )

            self._save_checkpoint(epoch, va_acc, is_best)

        self.log.info('=' * 65)
        self.log.info(f'  Pretrain hoàn tất. Best val_acc = {self.best_val_acc:.4f}')
        self.log.info('=' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    """Cấu hình logging ghi vào file và console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pretrain.log')

    logger = logging.getLogger('pretrain')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'Log file: {log_file}')
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pretrain ResNet12 backbone (classification task)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Thư mục gốc của dataset (mỗi class = 1 subfolder)')
    parser.add_argument('--split_path', type=str, required=True,
                        help='Đường dẫn đến file split.json '
                             '(phải có key "train" và "val")')

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--emb_size', type=int, default=128,
                        help='Embedding size (phải khớp với emb_size trong AGNN config)')
    parser.add_argument('--image_size', type=int, default=84,
                        help='Kích thước ảnh đầu vào (phải khớp với AGNN config)')

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate ban đầu (SGD)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80',
                        help='Epoch giảm LR, cách nhau bằng dấu phẩy')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                        help='Hệ số giảm LR tại mỗi milestone')

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./pretrain_checkpoints',
                        help='Thư mục lưu file backbone_best.pth và backbone_last.pth')
    parser.add_argument('--log_dir', type=str, default='./pretrain_logs',
                        help='Thư mục lưu file log huấn luyện')

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Thiết bị (cuda:0 hoặc cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số worker của DataLoader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ── Logging ───────────────────────────────────────────────────────────────
    log = setup_logging(args.log_dir)
    log.info('Arguments: ' + str(vars(args)))

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PretrainDataset(
        root=args.dataset_root, split_path=args.split_path,
        partition='train', image_size=args.image_size)

    val_ds = PretrainDataset(
        root=args.dataset_root, split_path=args.split_path,
        partition='val', image_size=args.image_size)

    num_classes = train_ds.num_classes
    log.info(f'Train classes: {num_classes}  |  '
             f'Val classes: {val_ds.num_classes}')

    train_loader = TorchDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = TorchDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    cifar_flag = (args.image_size <= 32)
    model = BackboneClassifier(
        emb_size=args.emb_size,
        num_classes=num_classes,
        cifar_flag=cifar_flag)

    log.info(f'Model: ResNet12(emb_size={args.emb_size}, cifar_flag={cifar_flag}) '
             f'+ Dual Classifier Heads({args.emb_size} → {num_classes})')

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    milestones = [int(e.strip()) for e in args.lr_decay_epochs.split(',')
                  if e.strip()]
    scheduler  = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_decay_factor)

    log.info(f'Optimizer: SGD  lr={args.lr}  milestones={milestones}  '
             f'gamma={args.lr_decay_factor}')

    # ── Xác nhận cấu trúc key của backbone trước khi train ───────────────────
    # Đây là bước kiểm tra cuối cùng để đảm bảo keys sẽ khớp với ResNet12.
    ref_keys = set(ResNet12(emb_size=args.emb_size,
                            cifar_flag=cifar_flag).state_dict().keys())
    backbone_keys = set(model.get_backbone_state_dict().keys())

    if ref_keys == backbone_keys:
        log.info('✓ Key verification PASSED: backbone keys khớp hoàn toàn '
                 f'với ResNet12 ({len(backbone_keys)} keys). '
                 'Load strict=True sẽ an toàn.')
    else:
        missing  = ref_keys - backbone_keys
        extra    = backbone_keys - ref_keys
        log.error('✗ Key verification FAILED!')
        if missing:
            log.error(f'  Keys thiếu (có trong ResNet12 nhưng không có trong backbone): {missing}')
        if extra:
            log.error(f'  Keys thừa (có trong backbone nhưng không có trong ResNet12): {extra}')
        raise RuntimeError('Key mismatch detected trước khi train. '
                           'Hãy kiểm tra lại backbone.py.')

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = PretrainTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        checkpoint_dir=args.checkpoint_dir, log=log)

    trainer.run(args.num_epochs)


if __name__ == '__main__':
    main()
