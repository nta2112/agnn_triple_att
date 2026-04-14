from collections import OrderedDict

# ============================================================
# Ablation Study – WITHOUT Layer Memory Attention
# Loại trừ: mem_att
#   [✔] self_att   – Node self-attention (MultiHeadAttention + Fusion)
#   [✔] neigh_att  – Neighbor attention (top-k sparsity)
#   [✘] mem_att    – Bỏ dense connection giữa các GNN layer
#                    → mỗi layer chỉ dùng output của layer hiện tại
#                       (không concat với feature của các layer trước)
#
# Lưu ý: với num_generation=1, ablation này không ảnh hưởng kết quả
# (vì chỉ có 1 layer GNN). Tăng num_generation > 1 để thấy tác động.
# ============================================================

config = OrderedDict()

config['dataset_name'] = 'custom'
config['split_path'] = '/content/data/slpit_thuNghiem_V02.json'
config['image_size'] = 224
config['num_generation'] = 1      # Tăng lên >1 để ablation mem_att có tác động
config['num_loss_generation'] = 1
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'

config['emb_size'] = 128
config['backbone'] = 'resnet50'

config['save_root'] = '/content/drive/MyDrive/Do_an_Data'
config['exp_name'] = 'ablation_no_mem_att'
config['log_step'] = 100

# Ablation mode: bỏ layer memory attention (dense connection)
config['ablation_mode'] = 'no_mem_att'

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 5
train_opt['num_queries'] = 5
train_opt['batch_size'] = 2
train_opt['iteration'] = 3000
train_opt['lr'] = 0.00002
train_opt['weight_decay'] = 1e-4
train_opt['dec_lr'] = 1500
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.2
train_opt['loss_indicator'] = [1, 1, 0]

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['num_queries'] = 5
eval_opt['batch_size'] = 1
eval_opt['iteration'] = 50
eval_opt['interval'] = 100

config['train_config'] = train_opt
config['eval_config'] = eval_opt
