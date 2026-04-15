from collections import OrderedDict

config = OrderedDict()

# Cấu hình LaSt-ViT Backbone cho AGNN
config['dataset_name'] = 'custom'
# Đường dẫn tệp tin chia nhãn (Nếu chạy trên Kaggle bạn nên đổi thành /kaggle/input/...)
config['split_path'] = ''
# ViT hoạt động tốt nhất ở 224, nhưng backbone.py tự resize nên 84 vẫn được
config['image_size'] = 224 
config['num_generation'] = 3
config['num_loss_generation'] = 1
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'

config['emb_size'] = 128
config['backbone'] = 'last_vit' # Lựa chọn backbone mới

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 5
train_opt['num_queries'] = 5 
train_opt['batch_size'] = 1 # RẤT QUAN TRỌNG: ViT rất nặng, nên để 1 task mỗi batch
train_opt['iteration'] = 3000
train_opt['lr'] = 1e-5     # LR cực nhỏ cho Transformer
train_opt['weight_decay'] = 0.3 # WD cao cho Transformer
train_opt['dec_lr'] = 1500
train_opt['lr_adj_base'] = 0.1
train_opt['dropout'] = 0.1
train_opt['loss_indicator'] = [1, 0, 0]

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['num_queries'] = 5
eval_opt['batch_size'] = 1
eval_opt['iteration'] = 50
eval_opt['interval'] = 100 

config['train_config'] = train_opt
config['eval_config'] = eval_opt
