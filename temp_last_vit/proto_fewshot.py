import torch
import torch.nn as nn
from temp_last_vit.last_vit_model import build_last_vit_b16

class PrototypicalLaStViT(nn.Module):
    def __init__(self, pretrained=True):
        super(PrototypicalLaStViT, self).__init__()
        # Load the LaSt-ViT model
        self.encoder = build_last_vit_b16(pretrained=pretrained)
        
    def forward(self, support_x, query_x, n_way, k_shot):
        """
        Forward pass for a Batch of Few-Shot Episodes (to support Multi-GPU DataParallel)
        :param support_x: [p, n_way * k_shot, 3, 224, 224] tensor (p is tasks per GPU)
        :param query_x: [p, n_query_total, 3, 224, 224] tensor
        :param n_way: number of classes
        :param k_shot: number of support samples per class
        :return: logits [p, n_query_total, n_way]
        """
        # Case 1: Simple 4D input (backward compatibility or single task)
        # We manually add a batch dimension of 1
        is_5d = len(support_x.shape) == 5
        if not is_5d:
            support_x = support_x.unsqueeze(0)
            query_x = query_x.unsqueeze(0)
            
        p, n_s, c, h, w = support_x.shape
        _, n_q, _, _, _ = query_x.shape
        
        # Flatten all images into a single Batch [p * N, 3, H, W] for the Encoder
        support_x_flat = support_x.view(-1, c, h, w)
        query_x_flat = query_x.view(-1, c, h, w)
        
        # Encode
        support_cls, _ = self.encoder(support_x_flat) # [p * n_s, embed_dim]
        query_cls, _ = self.encoder(query_x_flat)     # [p * n_q, embed_dim]
        
        embed_dim = support_cls.size(-1)
        
        # Reshape back to [p, n_s, embed_dim]
        support_cls = support_cls.view(p, n_s, embed_dim)
        query_cls = query_cls.view(p, n_q, embed_dim)
        
        # Calculate Prototypes per task
        # Reshape support to [p, n_way, k_shot, embed_dim]
        support_cls_reshaped = support_cls.view(p, n_way, k_shot, embed_dim)
        prototypes = support_cls_reshaped.mean(dim=2) # [p, n_way, embed_dim]
        
        # Calculate Euclidean distance [p, n_q, n_way]
        # torch.cdist handles batches if input is 3D!
        dists = torch.cdist(query_cls, prototypes, p=2.0) # [p, n_q, n_way]
        
        logits = -dists
        
        # If input was 4D, return 2D logits [num_queries, n_way]
        if not is_5d:
            return logits.squeeze(0)
        
        return logits
