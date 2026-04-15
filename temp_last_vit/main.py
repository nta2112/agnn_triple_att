import sys
import os

# Append parent dir to path to import AGNN modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging
import argparse

# AGNN Modules
from dataloader import MiniImagenet, TieredImagenet, CUB200, DataLoader, Flowers, CustomImageFolder
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing

# LaSt-ViT Modules
from temp_last_vit.proto_fewshot import PrototypicalLaStViT
from temp_last_vit.config import config


class ProtoViTTrainer(object):
    def __init__(self, model, data_loader, log, arg, cfg, best_step):
        self.arg = arg
        self.config = cfg
        self.train_opt = cfg['train_config']
        self.eval_opt = cfg['eval_config']

        self.model = model.to(arg.device)
        self.log = log
        self.data_loader = data_loader

        # Optimizer for ViT
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.train_opt['lr'],
            weight_decay=self.train_opt['weight_decay'])

        self.loss_fn = nn.CrossEntropyLoss()

        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        # We need to process batches. We will resize images to 224x224.
        resize_dim = (224, 224)
        n_way = self.train_opt['num_ways']
        k_shot = self.train_opt['num_shots']
        n_query = self.train_opt['num_queries']

        for iteration, batch in enumerate(self.data_loader['train']()):
            self.optimizer.zero_grad()
            self.global_step += 1
            self.model.train()

            # The AGNN DataLoader with torchnet.parallel(batch_size=1) returns:
            # batch[0]: support_data [1, num_tasks, n_support, 3, H, W]
            # batch[1]: support_label [1, num_tasks, n_support]
            # batch[2]: query_data [1, num_tasks, n_query, 3, H, W]
            # batch[3]: query_label [1, num_tasks, n_query]
            
            # Squeeze the first dimension (torchnet batch size 1)
            data_support_all = batch[0].squeeze(0).to(self.arg.device)
            data_query_all = batch[2].squeeze(0).to(self.arg.device)
            
            p = self.train_opt['batch_size'] # num_tasks
            
            task_losses = []
            task_accs = []
            
            for task_idx in range(p):
                support_x = data_support_all[task_idx] # [n_way * k_shot, 3, H, W]
                query_x = data_query_all[task_idx]     # [n_way * n_query, 3, H, W]
                
                # Resize only if necessary, ensuring 4D input [N, C, H, W]
                if support_x.shape[-2:] != resize_dim:
                    support_x = F.interpolate(support_x, size=resize_dim, mode='bilinear', align_corners=False)
                    query_x = F.interpolate(query_x, size=resize_dim, mode='bilinear', align_corners=False)
                
                # Recreate query_y (0 to n_way-1)
                query_y = torch.arange(n_way).repeat_interleave(n_query).to(self.arg.device)
                
                # Forward Pass
                logits = self.model(support_x, query_x, n_way, k_shot) # [num_queries, n_way]
                
                # Loss
                loss = self.loss_fn(logits, query_y)
                task_losses.append(loss)
                
                # Accuracy
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_y).float().mean()
                task_accs.append(acc)
            
            # Backprop task average
            total_loss = torch.mean(torch.stack(task_losses))
            total_loss.backward()
            self.optimizer.step()
            
            # Adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],
                                 lr=self.train_opt['lr'],
                                 iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'],
                                 lr_adj_base=self.train_opt['lr_adj_base'])

            if self.global_step % self.arg.log_step == 0:
                mean_acc = torch.mean(torch.stack(task_accs)).item()
                self.log.info(f'step: {self.global_step} | loss: {total_loss.item():.4f} | train_acc: {mean_acc:.4f}')

            # Eval
            if self.global_step > 0 and self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                val_acc = self.eval(partition='val')
                torch.cuda.empty_cache()
                
                if val_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = val_acc
                    self.best_step = self.global_step

                self.log.info(f'val_acc : {val_acc:.4f}         step : {self.global_step}')
                self.log.info(f'best_val_acc : {self.test_acc:.4f}    step : {self.best_step}')

                save_checkpoint({
                    'iteration': self.global_step,
                    'proto_vit_state_dict': self.model.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, self.arg.checkpoint_dir)

    def eval(self, partition='val'):
        n_way = self.eval_opt['num_ways']
        k_shot = self.eval_opt['num_shots']
        n_query = self.eval_opt['num_queries']
        resize_dim = (224, 224)

        self.model.eval()
        all_accs = []
        
        with torch.no_grad():
            for current_iteration, batch in enumerate(self.data_loader[partition]()):
                # Squeeze the first dimension (torchnet batch size 1)
                data_support_all = batch[0].squeeze(0).to(self.arg.device)
                data_query_all = batch[2].squeeze(0).to(self.arg.device)
                
                p = self.eval_opt['batch_size'] # num_tasks
                
                for task_idx in range(p):
                    support_x = data_support_all[task_idx]
                    query_x = data_query_all[task_idx]
                    
                    if support_x.shape[-2:] != resize_dim:
                        support_x = F.interpolate(support_x, size=resize_dim, mode='bilinear')
                        query_x = F.interpolate(query_x, size=resize_dim, mode='bilinear')
                    
                    query_y = torch.arange(n_way).repeat_interleave(n_query).to(self.arg.device)
                    
                    logits = self.model(support_x, query_x, n_way, k_shot)
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == query_y).float().mean()
                    all_accs.append(acc.item())
                    
        mean_acc = np.mean(all_accs)
        std_acc = np.std(all_accs)
        ci95 = 1.96 * std_acc / np.sqrt(len(all_accs))
        
        self.log.info(f'------------------------------------')
        self.log.info(f'{partition} accuracy: mean={mean_acc*100:.2f}%, std={std_acc*100:.2f}%, ci95={ci95*100:.2f}%')
        self.log.info(f'------------------------------------')
        
        return mean_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset_root', type=str, default='../data')
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--mode', type=str, default='train')
    args_opt = parser.parse_args()

    args_opt.log_dir = config['log_dir']
    args_opt.checkpoint_dir = config['checkpoint_dir']
    args_opt.log_step = config['log_step']

    if not os.path.exists(args_opt.log_dir):
        os.makedirs(args_opt.log_dir)
        
    set_logging_config(os.path.join(args_opt.log_dir, config['exp_name']))
    logger = logging.getLogger('main')

    # Seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)

    # Dataset
    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
    elif config['dataset_name'] == 'custom':
        dataset = CustomImageFolder
    else:
        logger.info('Invalid dataset.')
        exit()

    if config['dataset_name'] == 'custom':
        img_size = config.get('image_size', 84)
        split_path = config.get('split_path', None)
        dataset_train = dataset(root=args_opt.dataset_root, partition='train', image_size=img_size, split_path=split_path)
        dataset_valid = dataset(root=args_opt.dataset_root, partition='val', image_size=img_size, split_path=split_path)
    else:
        dataset_train = dataset(root=args_opt.dataset_root, partition='train')
        dataset_valid = dataset(root=args_opt.dataset_root, partition='val')

    train_loader = DataLoader(dataset_train,
                              num_tasks=config['train_config']['batch_size'],
                              num_ways=config['train_config']['num_ways'],
                              num_shots=config['train_config']['num_shots'],
                              num_queries=config['train_config']['num_queries'],
                              epoch_size=config['train_config']['iteration'])
                              
    valid_loader = DataLoader(dataset_valid,
                              num_tasks=config['eval_config']['batch_size'],
                              num_ways=config['eval_config']['num_ways'],
                              num_shots=config['eval_config']['num_shots'],
                              num_queries=config['eval_config']['num_queries'],
                              epoch_size=config['eval_config']['iteration'])

    data_loader = {'train': train_loader, 'val': valid_loader}

    # Model
    model = PrototypicalLaStViT(pretrained=True)

    if not os.path.exists(args_opt.checkpoint_dir):
        os.makedirs(args_opt.checkpoint_dir)
        best_step = 0
    else:
        best_step = 0 # add load support if needed

    trainer = ProtoViTTrainer(model, data_loader, logger, args_opt, config, best_step)

    if args_opt.mode == 'train':
        trainer.train()
    elif args_opt.mode == 'eval':
        trainer.eval()

if __name__ == '__main__':
    main()
