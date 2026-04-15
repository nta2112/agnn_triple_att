import os

config = {
    'exp_name': 'last_vit_proto_5way_1shot',
    'dataset_name': 'custom',
    'split_path': 'split.json', # Thêm dòng này để trỏ tới file split của bạn
    'image_size': 224,           # Kích thước gốc ảnh trước khi resize lên 224
    # Dataset / Loader configurations
    'train_config': {
        'num_ways': 5,
        'num_shots': 5,
        'num_queries': 5,
        'batch_size': 5,       # Few-shot Episode Batch Size (Tasks) - reduced for ViT Memory Load!
        'iteration': 3000,
        
        # ViT Learning configuration (Need very small LR for ViT)
        'lr': 1e-5,               
        'weight_decay': 0.3,   # Large weight decay for ViT 
        'dec_lr': 1500,        # lr decay step
        'lr_adj_base': 0.1     # lr scale factor
    },
    
    'eval_config': {
        'num_ways': 5,
        'num_shots': 5,
        'num_queries': 5,
        'batch_size': 5,
        'iteration': 200,      # Tasks for evaluation
        'interval': 200        # Evaluate every 200 steps
    },
    
    # Path configuration
    'log_dir': os.path.join('.', 'temp_last_vit', 'logs'),
    'checkpoint_dir': os.path.join('.', 'temp_last_vit', 'checkpoints'),
    'log_step': 200
}
