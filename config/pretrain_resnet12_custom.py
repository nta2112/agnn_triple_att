# pretrain_resnet12_custom.py
# Config mẫu để chạy pretrain.py cho ResNet12 trên custom dataset
#
# Cách dùng:
#   python pretrain.py \
#       --dataset_root  /path/to/your/dataset \
#       --split_path    /path/to/your/split.json \
#       --checkpoint_dir ./pretrain_checkpoints \
#       --log_dir        ./pretrain_logs \
#       --emb_size       128 \
#       --image_size     84 \
#       --num_epochs     100 \
#       --batch_size     64 \
#       --lr             1e-3 \
#       --lr_decay_epochs 60,80 \
#       --lr_decay_factor 0.1 \
#       --device         cuda:0
#
# Sau khi pretrain xong, chạy AGNN với:
#   python main_gnn.py \
#       --config config/5way_5shot_resnet12_custom.py \
#       --pretrain_path ./pretrain_checkpoints/backbone_best.pth
#
# ─────────────────────────────────────────────────────────────────────────────
# Lưu ý quan trọng:
# - emb_size ở đây PHẢI khớp với emb_size trong config AGNN (ví dụ: 5way_5shot_resnet12_custom.py)
# - image_size nên khớp với image_size khi train AGNN
# ─────────────────────────────────────────────────────────────────────────────

# Đây là file hướng dẫn, không phải file config Python cho main_gnn.py.
# Các tham số được truyền thẳng qua command line khi chạy pretrain.py.
