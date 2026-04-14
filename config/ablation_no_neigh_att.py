from collections import OrderedDict

# ============================================================
# Ablation Study – WITHOUT Neighbor Attention
# Loại trừ: neigh_att
#   [✔] self_att   – Node self-attention (MultiHeadAttention + Fusion)
#   [✘] neigh_att  – Bỏ top-k sparsity trong PointSimilarity2
#                    → dùng fully connected adjacency matrix
#   [✔] mem_att    – Layer memory attention (dense connection)
# ============================================================

config = OrderedDict()

config['dataset_name'] = 'custom'
config['split_path'] = '/content/data/slpit_thuNghiem_V02.json'
config['image_size'] = 224
config['num_generation'] = 1
config['num_loss_generation'] = 1
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'

config['emb_size'] = 128
config['backbone'] = 'resnet50'

config['save_root'] = '/content/drive/MyDrive/Do_an_Data'
config['exp_name'] = 'ablation_no_neigh_att'
config['log_step'] = 100

# Ablation mode: bỏ neighbor attention (top-k sparsity), dùng fully connected
config['ablation_mode'] = 'no_neigh_att'

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
