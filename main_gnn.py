from backbone import ResNet12, ConvNet, ResNet50Pretrained, LaStViTBackbone
from agnn import AGNN
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode
from dataloader import DataLoader, CustomImageFolder
try:
    from dataloader import MiniImagenet, TieredImagenet, CUB200, Flowers
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import logging
import argparse
import importlib.util
import shutil


class AGNNTrainer(object):
    def __init__(self, enc_module, gnn_module, data_loader, log, arg, config, best_step):
        """
        The Trainer of AGNN model
        :param enc_module: backbone network (Conv4, ResNet12, ResNet18, WRN)
        :param gnn_module: AGNN model
        :param data_loader: data loader
        :param log: logger
        :param arg: command line arguments
        :param config: model configurations
        :param best_step: starting step (step at best eval acc or 0 if starts from scratch)
        """

        self.arg = arg
        self.config = config
        self.train_opt = config['train_config']
        self.eval_opt = config['eval_config']

        # initialize variables
        self.tensors = allocate_tensors()
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.arg.device)

        # set backbone and AGNN
        self.enc_module = enc_module.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)

        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # set parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(
            params=self.module_params,
            lr=self.train_opt['lr'],
            weight_decay=self.train_opt['weight_decay'])

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        """
        train function
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],
                          self.train_opt['num_shots'],
                          self.train_opt['num_queries'],
                          self.train_opt['batch_size'],
                          self.arg.device)

        # main training loop, batch size is the number of tasks
        for iteration, batch in enumerate(self.data_loader['train']()):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step += 1

            # initialize nodes and edges for dual graph model
            support_data, support_label, query_data, query_label, all_data, all_label_in_edge, node_feature_gd, \
            edge_feature_gp = initialize_nodes_edges(batch,
                                                     num_supports,
                                                     self.tensors,
                                                     self.train_opt['batch_size'],
                                                     self.train_opt['num_queries'],
                                                     self.train_opt['num_ways'],
                                                     self.arg.device)

            # set as train mode
            self.enc_module.train()
            self.gnn_module.train()
            
            # use backbone encode image
            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            # run the AGNN model
            point_similarity, node_similarity_l2 = self.gnn_module(second_last_layer_data,
                                                                   last_layer_data,
                                                                   node_feature_gd,
                                                                   edge_feature_gp, support_label)

            # compute loss
            total_loss, query_node_cls_acc_generations, query_edge_loss_generations = \
                self.compute_train_loss_pred(all_label_in_edge,
                                             point_similarity,
                                             node_similarity_l2,
                                             query_edge_mask,
                                             evaluation_mask,
                                             num_supports,
                                             support_label,
                                             query_label)

            # back propagation & update
            total_loss.backward()
            self.optimizer.step()

            # adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],
                                 lr=self.train_opt['lr'],
                                 iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'],
                                 lr_adj_base =self.train_opt['lr_adj_base'])

            # log training info
            if self.global_step % self.arg.log_step == 0:
                self.log.info('step : {}  train_edge_loss : {}  node_acc : {}'.format(
                    self.global_step,
                    query_edge_loss_generations[-1],
                    query_node_cls_acc_generations[-1]))

            # evaluation
            if self.global_step > 0 and self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                val_acc = self.eval(partition='val')
                torch.cuda.empty_cache() # Clear cache after evaluation to free up VRAM
                if val_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = val_acc
                    self.best_step = self.global_step

                # log evaluation info
                self.log.info('val_acc : {}         step : {} '.format(val_acc, self.global_step))
                self.log.info('best_val_acc : {}    step : {}'.format( self.test_acc, self.best_step))

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, self.arg.checkpoint_dir)


    def get_features(self, partition='test', log_flag=True):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        query_edge_loss_generations = []
        query_node_cls_acc_generations = []
        # set as eval mode
        self.enc_module.eval()
        self.gnn_module.eval()

        with torch.no_grad():
            # main training loop, batch size is the number of tasks
            for current_iteration, batch in enumerate(self.data_loader[partition]()):

                if current_iteration == 0:
                    # initialize nodes and edges for dual graph model
                    support_data, support_label, query_data, query_label, all_data, all_label_in_edge, node_feature_gd, \
                    edge_feature_gp = initialize_nodes_edges(batch,
                                                            num_supports,
                                                            self.tensors,
                                                            self.eval_opt['batch_size'],
                                                            self.eval_opt['num_queries'],
                                                            self.eval_opt['num_ways'],
                                                            self.arg.device)

                    last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)


                    point_similarity, _ , node_features = self.gnn_module(second_last_layer_data,
                                                    last_layer_data,
                                                    node_feature_gd,
                                                    edge_feature_gp,
                                                    support_label)
                    query_node_cls_acc_generations, query_edge_loss_generations = \
                    self.compute_eval_loss_pred(query_edge_loss_generations,
                                                query_node_cls_acc_generations,
                                                all_label_in_edge,
                                                point_similarity,
                                                query_edge_mask,
                                                evaluation_mask,
                                                num_supports,
                                                support_label,
                                                query_label)
                    break

        return point_similarity, node_features, support_label, query_label


    def eval(self, partition='test', log_flag=True):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        query_ce_loss_generations = []
        query_node_cls_acc_generations = []
        # set as eval mode
        self.enc_module.eval()
        self.gnn_module.eval()

        with torch.no_grad():
            # main training loop, batch size is the number of tasks
            for current_iteration, batch in enumerate(self.data_loader[partition]()):

                # initialize nodes and edges for dual graph model
                support_data, support_label, query_data, query_label, all_data, all_label_in_edge, node_feature_gd, \
                edge_feature_gp = initialize_nodes_edges(batch,
                                                        num_supports,
                                                        self.tensors,
                                                        self.eval_opt['batch_size'],
                                                        self.eval_opt['num_queries'],
                                                        self.eval_opt['num_ways'],
                                                        self.arg.device)

                last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

                # run the AGNN model
                point_similarity, _  = self.gnn_module(second_last_layer_data,
                                                last_layer_data,
                                                node_feature_gd,
                                                edge_feature_gp,
                                                support_label)

                query_node_cls_acc_generations, query_ce_loss_generations = \
                    self.compute_eval_loss_pred(query_ce_loss_generations,
                                                query_node_cls_acc_generations,
                                                all_label_in_edge,
                                                point_similarity,
                                                query_edge_mask,
                                                evaluation_mask,
                                                num_supports,
                                                support_label,
                                                query_label)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_ce_loss : {}  {}_node_acc : {}'.format(
                self.global_step, partition,
                np.array(query_ce_loss_generations).mean(),
                partition,
                np.array(query_node_cls_acc_generations).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_node_cls_acc_generations).mean() * 100,
                           np.array(query_node_cls_acc_generations).std() * 100,
                           1.96 * np.array(query_node_cls_acc_generations).std()
                           / np.sqrt(float(len(np.array(query_node_cls_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_node_cls_acc_generations).mean()

    def compute_train_loss_pred(self,
                                all_label_in_edge,
                                point_similarities,
                                node_similarities_l2,
                                query_edge_mask,
                                evaluation_mask,
                                num_supports,
                                support_label,
                                query_label):
        """
        Compute total loss, query classification accuracy, and per-generation CE loss.
        Loss = Cross-Entropy tại mỗi GNN layer — paper Eq. 13.
        BCE edge loss (từ DPGN) đã được loại bỏ để khớp đúng với paper AGNN Triple Attention.
        """

        # ── Dự đoán nhãn query (để tính Accuracy) từ point_similarity ─────────
        query_node_pred_generations = [
            torch.bmm(
                point_similarity[:, num_supports:, :num_supports],
                one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device)
            )
            for point_similarity in point_similarities
        ]

        # ── Dự đoán nhãn query (để tính Loss CE) từ node_similarity_l2 ────────
        query_node_pred_generations_l2 = [
            torch.bmm(
                node_similarity_l2[:, num_supports:, :num_supports],
                one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device)
            )
            for node_similarity_l2 in node_similarities_l2
        ]

        # ── Cross-Entropy loss (Node) dựa trên L2 Similarity ──────────────────
        query_node_ce_loss = []
        for query_node_pred_l2 in query_node_pred_generations_l2:
            pred_flat  = query_node_pred_l2.contiguous().view(-1, query_node_pred_l2.shape[-1])
            label_flat = query_label.long().contiguous().view(-1)
            query_node_ce_loss.append(self.pred_loss(pred_flat, label_flat).mean())

        # ── Balanced BCE Loss (Edge) - Đúng chuẩn bản gốc ────────────────────
        query_edge_bce_loss = []
        for point_similarity in point_similarities:
            # 1. Tính BCE loss (đảo ngược xác suất theo bản gốc)
            bce_loss = self.edge_loss(1.0 - point_similarity, 1.0 - all_label_in_edge)
            
            # 2. Tách Loss cho Positive Edges (Cùng lớp)
            pos_loss = torch.sum(bce_loss * query_edge_mask * all_label_in_edge * evaluation_mask) \
                       / (torch.sum(query_edge_mask * all_label_in_edge * evaluation_mask) + 1e-6)
            
            # 3. Tách Loss cho Negative Edges (Khác lớp)
            neg_loss = torch.sum(bce_loss * query_edge_mask * (1.0 - all_label_in_edge) * evaluation_mask) \
                       / (torch.sum(query_edge_mask * (1.0 - all_label_in_edge) * evaluation_mask) + 1e-6)
            
            # 4. Cộng lại (Cân bằng)
            query_edge_bce_loss.append(pos_loss + neg_loss)

        # ── Train accuracy ────────────────────────────────────────────────────
        query_node_acc_generations = [
            torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()
            for query_node_pred in query_node_pred_generations
        ]

        # ── Multi-layer weighted sum (CE + BCE) ───────────────────────────────
        # Kết hợp hai loại loss dựa trên loss_indicator: [BCE_weight, CE_weight, ...]
        w_bce = self.train_opt['loss_indicator'][0]
        w_ce  = self.train_opt['loss_indicator'][1]
        
        total_loss = []
        num_loss = self.config['num_loss_generation']
        for l in range(num_loss):
            gen_w = self.config['generation_weight'] if l < num_loss - 1 else 1.0
            
            # Cộng dồn loss của generation thứ l
            l_ce  = query_node_ce_loss[l] * w_ce
            l_bce = query_edge_bce_loss[l] * w_bce
            
            total_loss += [(l_ce + l_bce).view(-1) * gen_w]
            
        total_loss = torch.mean(torch.cat(total_loss, 0))

        return total_loss, query_node_acc_generations, query_node_ce_loss

    def compute_eval_loss_pred(self,
                               query_ce_losses,
                               query_node_accs,
                               all_label_in_edge,
                               point_similarities,
                               query_edge_mask,
                               evaluation_mask,
                               num_supports,
                               support_label,
                               query_label):
        """
        Compute query classification CE loss and accuracy during evaluation.
        Dùng Cross-Entropy thay BCE edge loss để nhất quán với train loss (paper Eq. 13).
        """
        # Lấy edge weights của generation cuối cùng
        point_similarity = point_similarities[-1]

        # Dự đoán nhãn query từ edge weights
        query_node_pred = torch.bmm(
            point_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device)
        )

        # Cross-Entropy loss
        pred_flat  = query_node_pred.contiguous().view(-1, query_node_pred.shape[-1])
        label_flat = query_label.long().contiguous().view(-1)
        query_ce_loss = self.pred_loss(pred_flat, label_flat).mean()

        # Accuracy
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

        query_ce_losses  += [query_ce_loss.item()]
        query_node_accs  += [query_node_acc.item()]

        return query_node_accs, query_ce_losses


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='gpu device number of using')

    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_1shot_resnet12_mini-imagenet.py'),
                        help='config file with parameters of the experiment. '
                             'It is assumed that the config file is placed under the directory ./config')

    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='path that checkpoint will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')

    parser.add_argument('--num_gpu', type=int, default=torch.cuda.device_count(),
                        help='number of gpu')

    parser.add_argument('--display_step', type=int, default=200,
                        help='display training information in how many step')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for dataloader')

    parser.add_argument('--log_step', type=int, default=200,
                        help='log information in how many steps')

    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'logs'),
                        help='path that log will be saved. '
                             'It is assumed that the checkpoint file is placed under the directory ./logs')

    parser.add_argument('--dataset_root', type=str, default='./data',
                        help='root directory of dataset')

    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')

    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')

    parser.add_argument('--tag', type=str, default='debug',
                        help='description')

    args_opt = parser.parse_args()

    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    train_opt = config['train_config']
    eval_opt = config['eval_config']

    # Define default experiment name
    default_exp_name = '{}way_{}shot_{}_{}'.format(train_opt['num_ways'],
                                                    train_opt['num_shots'],
                                                    config['backbone'],
                                                    config['dataset_name'])
    default_exp_name = default_exp_name + args_opt.tag

    # Handle flexible storage paths
    args_opt.exp_name = config.get('exp_name', default_exp_name)
    save_root = config.get('save_root', None)
    
    if save_root:
        # Unified structure: {save_root}/{exp_name}/logs and {save_root}/{exp_name}/checkpoints
        args_opt.log_dir = os.path.join(save_root, args_opt.exp_name, 'logs')
        args_opt.checkpoint_dir = os.path.join(save_root, args_opt.exp_name, 'checkpoints')
        log_path = args_opt.log_dir # logs are stored directly in this dir
    else:
        # Fallback to original behavior or config-overridden dirs
        args_opt.log_dir = config.get('log_dir', args_opt.log_dir)
        args_opt.checkpoint_dir = config.get('checkpoint_dir', args_opt.checkpoint_dir)
        log_path = os.path.join(args_opt.log_dir, args_opt.exp_name)
        args_opt.checkpoint_dir = os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)

    # Always override log steps from config if present, regardless of save_root
    args_opt.log_step = config.get('log_step', args_opt.log_step)
    args_opt.display_step = config.get('display_step', args_opt.display_step)

    set_logging_config(log_path)
    
    # Copy config file to log directory for reproducibility
    try:
        shutil.copy(config_file, os.path.join(log_path, os.path.basename(config_file)))
    except Exception as e:
        print(f"Warning: Could not copy config file: {e}")
        
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(log_path))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config['dataset_name'] == 'mini-imagenet':
        dataset = MiniImagenet
        print('Dataset: MiniImagenet')
    elif config['dataset_name'] == 'tiered-imagenet':
        dataset = TieredImagenet
        print('Dataset: TieredImagenet')
    elif config['dataset_name'] == 'cub-200-2011':
        dataset = CUB200
        print('Dataset: CUB200')
    elif config['dataset_name'] == 'flowers':
        dataset = Flowers
        print('Dataset: Flowers')
    elif config['dataset_name'] == 'custom':
        dataset = CustomImageFolder
        print('Dataset: Custom Dataset')
    else:
        logger.info('Invalid dataset: {}, please specify a dataset from '
                    'mini-imagenet, tiered-imagenet, cifar-fs, cub-200-2011 and flowers.'.format(config['dataset_name']))
        exit()

    cifar_flag = True if args_opt.exp_name.__contains__('cifar') else False
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ResNet12')
    elif config['backbone'] == 'resnet50':
        enc_module = ResNet50Pretrained(emb_size=config['emb_size'])
        print('Backbone: ResNet50 Pretrained ImageNet')
    elif config['backbone'] == 'last_vit':
        enc_module = LaStViTBackbone(emb_size=config['emb_size'], pretrained=True)
        print('Backbone: LaSt-ViT (DenseViT)')
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'], cifar_flag=cifar_flag)
        print('Backbone: ConvNet')
    else:
        logger.info('Invalid backbone: {}, please specify a backbone model from '
                    'convnet or resnet12.'.format(config['backbone']))
        exit()

    ablation_mode = config.get('ablation_mode', 'full')
    logger.info('Ablation mode: {}'.format(ablation_mode))
    gnn_module = AGNN(config['num_generation'],
                      train_opt['dropout'],
                      train_opt['num_ways'] * train_opt['num_shots'],
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'] * train_opt['num_queries'],
                      train_opt['loss_indicator'],
                      config['point_distance_metric'],
                      ablation_mode=ablation_mode)


    # num_params = 0
    # for param in gnn_module.parameters():
    #     num_params += param.numel()
    # print(num_params / 1e6)

    # multi-gpu configuration
    [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in range(args_opt.num_gpu)]

    if args_opt.num_gpu > 1:
        print('Construct multi-gpu model ...')
        enc_module = nn.DataParallel(enc_module, device_ids=range(args_opt.num_gpu), dim=0)
        gnn_module = nn.DataParallel(gnn_module, device_ids=range(args_opt.num_gpu), dim=0)
        print('done!\n')

    if not os.path.exists(args_opt.checkpoint_dir):
        os.makedirs(args_opt.checkpoint_dir)
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(
            args_opt.exp_name,
            args_opt.checkpoint_dir))
        best_step = 0
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar')):
            best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(
                args_opt.checkpoint_dir))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, 'model_best.pth.tar'), 
                                         map_location=args_opt.device, 
                                         weights_only=False)
 
            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
            logger.info('current best test accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))

    if config['dataset_name'] == 'custom':
        img_size = config.get('image_size', 84)
        split_path = config.get('split_path', None)
        dataset_train = dataset(root=args_opt.dataset_root, partition='train', image_size=img_size, split_path=split_path)
        dataset_valid = dataset(root=args_opt.dataset_root, partition='val', image_size=img_size, split_path=split_path)
        
        # Chỉ load tập test nếu ở chế độ eval
        dataset_test = None
        if args_opt.mode == 'eval':
            dataset_test = dataset(root=args_opt.dataset_root, partition='test', image_size=img_size, split_path=split_path)
    else:
        dataset_train = dataset(root=args_opt.dataset_root, partition='train')
        dataset_valid = dataset(root=args_opt.dataset_root, partition='val')
        
        dataset_test = None
        if args_opt.mode == 'eval':
            dataset_test = dataset(root=args_opt.dataset_root, partition='test')

    # ── Pre-load ảnh vào RAM để tối ưu tốc độ DataLoader ────────────────────
    # cache_to_memory() load PIL images vào một list trong main process.
    # Trên Colab/Kaggle (Linux, fork-based workers), các worker chia sẻ
    # bộ nhớ qua copy-on-write → không tốn RAM gấp đôi.
    # Nếu dataset quá lớn (> 8GB RAM), comment 2 dòng dưới để bỏ cache.
    if hasattr(dataset_train, 'cache_to_memory'):
        dataset_train.cache_to_memory()
        dataset_valid.cache_to_memory()
        if dataset_test is not None:
            dataset_test.cache_to_memory()

    train_loader = DataLoader(dataset_train,
                              num_tasks=train_opt['batch_size'],
                              num_ways=train_opt['num_ways'],
                              num_shots=train_opt['num_shots'],
                              num_queries=train_opt['num_queries'],
                              epoch_size=train_opt['iteration'],
                              num_workers=args_opt.num_workers)
    valid_loader = DataLoader(dataset_valid,
                              num_tasks=eval_opt['batch_size'],
                              num_ways=eval_opt['num_ways'],
                              num_shots=eval_opt['num_shots'],
                              num_queries=eval_opt['num_queries'],
                              epoch_size=eval_opt['iteration'],
                              num_workers=args_opt.num_workers)
    
    test_loader = None
    if dataset_test is not None:
        test_loader  = DataLoader(dataset_test,
                                  num_tasks=eval_opt['batch_size'],
                                  num_ways=eval_opt['num_ways'],
                                  num_shots=eval_opt['num_shots'],
                                  num_queries=eval_opt['num_queries'],
                                  epoch_size=eval_opt['iteration'],
                                  num_workers=args_opt.num_workers)

    data_loader = {'train': train_loader,
                   'val': valid_loader,
                   'test': test_loader}

    # create trainer
    trainer = AGNNTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           log=logger,
                           arg=args_opt,
                           config=config,
                           best_step=best_step)

    if args_opt.mode == 'train':
        trainer.train()
    elif args_opt.mode == 'eval':
        trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()
