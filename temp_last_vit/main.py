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

        # Auto GPU Detection & DataParallel
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            log.info(f'Detected {num_gpu} GPUs. Using DataParallel.')
            self.model = nn.DataParallel(model).to(arg.device)
        else:
            log.info(f'Using single GPU: {arg.device}')
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

            # batch[0]: support_data [1, num_tasks, n_support, 3, H, W]
            # batch[2]: query_data [1, num_tasks, n_query, 3, H, W]
            
            # Squeeze and prepare 5D batch
            data_support_all = batch[0].squeeze(0).to(self.arg.device) # [p, n_s, 3, H, W]
            data_query_all = batch[2].squeeze(0).to(self.arg.device)   # [p, n_q, 3, H, W]
            
            # Resize whole batch if needed
            if data_support_all.shape[-1] != 224:
                p, ns, c, h, w = data_support_all.shape
                _, nq, _, _, _ = data_query_all.shape
                data_support_all = F.interpolate(data_support_all.view(-1, c, h, w), size=resize_dim, mode='bilinear', align_corners=False).view(p, ns, c, 224, 224)
                data_query_all = F.interpolate(data_query_all.view(-1, c, h, w), size=resize_dim, mode='bilinear', align_corners=False).view(p, nq, c, 224, 224)
            
            # Recreate query_y per task [p, n_q_total]
            p = data_support_all.shape[0]
            # query_y_task: [n_q_total]
            query_y_task = torch.arange(n_way).repeat_interleave(n_query).to(self.arg.device)
            # query_y: [p, n_q_total]
            query_y = query_y_task.expand(p, -1)
            
            # Forward Pass (Vectorized over Batch of Tasks)
            logits = self.model(data_support_all, data_query_all, n_way, k_shot) # [p, n_q_total, n_way]
            
            # Loss calculation for the batch
            logits_flat = logits.view(-1, n_way)
            query_y_flat = query_y.reshape(-1)
            
            loss = self.loss_fn(logits_flat, query_y_flat)
            loss.backward()
            self.optimizer.step()
            
            # Accuracy
            preds = torch.argmax(logits, dim=2)
            acc = (preds == query_y).float().mean()

            # Adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],
                                 lr=self.train_opt['lr'],
                                 iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'],
                                 lr_adj_base=self.train_opt['lr_adj_base'])

            if self.global_step % self.arg.log_step == 0:
                self.log.info(f'step: {self.global_step} | loss: {loss.item():.4f} | train_acc: {acc.item():.4f}')

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
                    'proto_vit_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
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
                # Squeeze and prepare
                data_support_all = batch[0].squeeze(0).to(self.arg.device)
                data_query_all = batch[2].squeeze(0).to(self.arg.device)
                
                if data_support_all.shape[-1] != 224:
                    p, ns, c, h, w = data_support_all.shape
                    _, nq, _, _, _ = data_query_all.shape
                    data_support_all = F.interpolate(data_support_all.view(-1, c, h, w), size=resize_dim, mode='bilinear').view(p, ns, c, 224, 224)
                    data_query_all = F.interpolate(data_query_all.view(-1, c, h, w), size=resize_dim, mode='bilinear').view(p, nq, c, 224, 224)
                
                p = data_support_all.shape[0]
                query_y_task = torch.arange(n_way).repeat_interleave(n_query).to(self.arg.device)
                query_y = query_y_task.expand(p, -1)
                
                logits = self.model(data_support_all, data_query_all, n_way, k_shot)
                preds = torch.argmax(logits, dim=2)
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
    logger.info(f"Experiment: {config['exp_name']}")

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
    elif args_opt.mode == 'test' or args_opt.mode == 'eval':
        trainer.eval(partition='val')

if __name__ == '__main__':
    main()
