from collections import OrderedDict

config = OrderedDict()

config['dataset_name'] = 'custom'
# Bạn có thể thay đổi kích thước ảnh ở đây. Mô hình sẽ tự động crop và resize về kích thước này
config['image_size'] = 84 
config['num_generation'] = 3
#Q=1
config['num_loss_generation'] = 1
#Q>1
# config['num_loss_generation'] = 5
config['generation_weight'] = 0.5
config['point_distance_metric'] = 'l1'
config['distribution_distance_metric'] = 'l1'

config['emb_size'] = 128
config['backbone'] = 'resnet12'

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 5
train_opt['batch_size'] = 2
train_opt['iteration'] = 2000
train_opt['lr'] = 1e-4
train_opt['weight_decay'] = 5e-4
#Q=1
train_opt['dec_lr'] = 1000
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.2
train_opt['loss_indicator'] = [1, 0, 0]

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['batch_size'] = 2
eval_opt['iteration'] = 50
eval_opt['interval'] = 100 # Test mô hình sau mỗi 400 steps

config['train_config'] = train_opt
config['eval_config'] = eval_opt
