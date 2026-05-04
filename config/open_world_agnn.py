from collections import OrderedDict

config = OrderedDict()

# ── Cấu hình Dataset ─────────────────────────────────────────
config['dataset_name'] = 'custom'
config['split_path']   = '/content/data/slpit_thuNghiem_V02.json' # Thay đổi nếu cần
config['image_size']   = 224

# ── Tham số Open World (Mới - Paper Source 6) ───────────────
# lambda_feasibility: Trọng số cho Feasibility Margin Loss
# Giúp đẩy xa ranh giới giữa lớp thật và lớp lạ.
config['lambda_feasibility'] = 0.1 

# ── GNN Topology ────────────────────────────────────────────
config['num_generation']      = 3
config['num_loss_generation']  = 3
config['generation_weight']    = 0.5

# ── Embedding & Backbone ─────────────────────────────────────
config['emb_size'] = 128
config['backbone'] = 'resnet50' # Hoặc resnet12, convnet
config['loss_margin'] = 0.1    # Margin dùng cho cả CE và Feasibility
config['logit_scale'] = 10.0

# ── Training Option ──────────────────────────────────────────
train_opt = OrderedDict()
train_opt['num_ways']    = 5
train_opt['num_shots']   = 5
train_opt['num_queries'] = 5
train_opt['batch_size']  = 2
train_opt['iteration']   = 20000
train_opt['lr']          = 1e-4
train_opt['weight_decay'] = 5e-4
train_opt['dec_lr']      = [10000, 16000]
train_opt['lr_adj_base'] = 0.5
train_opt['dropout']     = 0.1
train_opt['loss_indicator'] = [1, 1, 0]

# ── Evaluation Option (Dùng cho cả eval Closed và Open World HM) ──
eval_opt = OrderedDict()
eval_opt['num_ways']    = 5
eval_opt['num_shots']   = 5
eval_opt['num_queries'] = 15
eval_opt['batch_size']  = 1
eval_opt['iteration']   = 600   # Số episode cho eval thường
eval_opt['interval']    = 500   # Kiểm tra HM sau mỗi 500 step

config['train_config'] = train_opt
config['eval_config']  = eval_opt
