from collections import OrderedDict

# ============================================================
# AGNN Triple Attention — FULL MODEL (Cấu hình chuẩn)
#
# Tất cả 3 cơ chế chú ý đều được bật (paper IEEE 10004644):
#   [✔] self_att   – Node self-attention (MultiHeadAttention + Fusion)
#                    → Khởi tạo đồ thị ban đầu qua fuse C^x + softmax(C^y)
#   [✔] neigh_att  – Neighbor attention (top-k sparsity)
#                    → ratio = 0.1 × layer; layer 0 không sparse,
#                       layer 1 giữ 90%, layer 2 giữ 80%
#   [✔] mem_att    – Layer memory attention (dense connection)
#                    → Concat node features qua các GNN layer
#
# Lý do chọn num_generation=3:
#   - Paper báo cáo optimal ở 5 layers trên miniImageNet/ResNet-12
#   - ResNet50 (feature dim 2048→128) + 224×224 tốn VRAM nhiều hơn
#   - Với batch_size=2 và Colab/Kaggle T4/P100, 3 layers là an toàn
#   - Layer 0: không sparse (ratio=0) — khởi động
#   - Layer 1: giữ 90% neighbors
#   - Layer 2: giữ 80% neighbors → Neighbor Att thực sự có tác dụng
#
# Loss: Cross-Entropy thuần (paper Eq. 13) — ĐÃ bỏ BCE edge loss
# ============================================================

config = OrderedDict()

config['dataset_name'] = 'custom'
config['split_path']   = '/content/data/slpit_thuNghiem_V02.json'
config['image_size']   = 224

# ── GNN topology ────────────────────────────────────────────
config['num_generation']      = 3       # Paper optimal: ~5; điều chỉnh cho ResNet50 VRAM
config['num_loss_generation']  = 3       # Supervise tại cả 3 layer (paper Eq. 13)
config['generation_weight']    = 0.5    # Layer 0,1: weight 0.5; layer cuối: weight 1.0

# ── Distance metric ─────────────────────────────────────────
config['point_distance_metric']        = 'l1'
config['distribution_distance_metric'] = 'l1'

# ── Embedding ───────────────────────────────────────────────
config['emb_size'] = 128
config['backbone'] = 'resnet50'         # ResNet50 Pretrained ImageNet

# ── Lưu trữ ─────────────────────────────────────────────────
config['save_root'] = '/content/drive/MyDrive/Do_an_Data'
config['exp_name']  = 'agnn_full_triple_att'
config['log_step']  = 200

# ── Ablation mode ────────────────────────────────────────────
config['ablation_mode'] = 'full'        # Tất cả attention đều được bật

# ── Training ─────────────────────────────────────────────────
train_opt = OrderedDict()
train_opt['num_ways']    = 5
train_opt['num_shots']   = 5
train_opt['num_queries'] = 5            # 5 queries/class (~25 query mỗi task)
train_opt['batch_size']  = 2            # 2 tasks/batch — giới hạn VRAM ResNet50

# Episodes: paper dùng 60,000 (ResNet12). ResNet50 nặng hơn,
# chọn 20,000 đủ để hội tụ trong ~4–6 giờ Colab
train_opt['iteration']   = 20000

# Learning rate: paper dùng 1e-3 với ResNet12 train-from-scratch.
# ResNet50 pretrained cần lr nhỏ hơn để không phá vỡ pretrained weights.
# Dùng 1e-4 là hợp lý (50× lớn hơn 2e-6 cũ, nhưng vẫn conservative).
train_opt['lr']          = 1e-4

train_opt['weight_decay'] = 5e-4       # Tăng nhẹ so với 1e-4 cũ → regularize tốt hơn

# Multi-step LR decay tại milestone 10000 và 16000
# (thay vì periodic decay cứng mỗi N steps)
train_opt['dec_lr']      = [10000, 16000]
train_opt['lr_adj_base'] = 0.5         # Giảm lr xuống 50% tại mỗi milestone

train_opt['dropout']     = 0.1         # Giảm từ 0.2 → 0.1: dataset custom nhỏ hơn paper

# loss_indicator: [edge_loss, node_l2_loss, dist_loss]
# Sau khi đã sửa loss = CE only, chỉ node_l2 còn được ref trong agnn.py
# Đặt [1, 0.1, 0] để khớp chính xác tỷ lệ của bản gốc (BCE + 0.1 * CE)
train_opt['loss_indicator'] = [1, 0.1, 0]

# ── Evaluation ───────────────────────────────────────────────
eval_opt = OrderedDict()
eval_opt['num_ways']    = 5
eval_opt['num_shots']   = 5
eval_opt['num_queries'] = 15           # 15 queries/class — chuẩn few-shot benchmark
eval_opt['batch_size']  = 1            # 1 task/batch khi eval để tránh tràn RAM
eval_opt['iteration']   = 600          # 600 eval episodes → CI95 ổn định
eval_opt['interval']    = 500          # Eval sau mỗi 500 training steps

config['train_config'] = train_opt
config['eval_config']  = eval_opt
