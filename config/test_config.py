from collections import OrderedDict

config = OrderedDict()

config['dataset_name'] = 'custom'
# Đường dẫn tệp tin chia nhãn (Nếu để trống '' thì dùng cấu trúc thư mục)
config['split_path'] = '/content/data/slpit_thuNghiem_V02.json'
# Bạn có thể thay đổi kích thước ảnh ở đây. Mô hình sẽ tự động crop và resize về kích thước này
config['image_size'] = 224
config['num_generation'] = 1
#Q>1
config['num_loss_generation'] = 1
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'

config['emb_size'] = 128
config['backbone'] = 'resnet50'

# Cấu hình lưu trữ (Tùy chọn)
config['save_root'] = '' # Thư mục gốc để lưu logs và checkpoints
config['exp_name'] = '' # Tên thí nghiệm cụ thể (nếu muốn đặt tên riêng)
config['log_step'] = 100 # In thông tin huấn luyện sau mỗi 100 steps

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 5
train_opt['num_queries'] = 5 
train_opt['batch_size'] = 4 
train_opt['iteration'] = 10000
train_opt['lr'] = 0.00002
train_opt['weight_decay'] = 1e-3
train_opt['dec_lr'] = [4000, 7000]
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.3
train_opt['label_smoothing'] = 0.1
train_opt['loss_indicator'] = [1, 1, 0]

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['num_queries'] = 15 
eval_opt['batch_size'] = 2 
eval_opt['iteration'] = 600
eval_opt['interval'] = 200 

config['train_config'] = train_opt
config['eval_config']  = eval_opt




