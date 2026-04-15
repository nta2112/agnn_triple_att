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
        Forward pass for a Few-Shot Episode
        :param support_x: [n_way * k_shot, 3, 224, 224] tensor
        :param query_x: [num_queries, 3, 224, 224] tensor
        :param n_way: number of classes
        :param k_shot: number of support samples per class
        :return: logits [num_queries, n_way]
        """
        # Encode support and query images
        # LaSt-ViT dense_vit returns (cls_token, x_detach)
        support_cls, _ = self.encoder(support_x) 
        query_cls, _ = self.encoder(query_x)
        
        # Calculate Prototypes
        # Reshape to [n_way, k_shot, embed_dim]
        # Then mean over the k_shot dimension
        embed_dim = support_cls.size(-1)
        support_cls_reshaped = support_cls.view(n_way, k_shot, embed_dim)
        prototypes = support_cls_reshaped.mean(dim=1) # [n_way, embed_dim]
        
        # Calculate Euclidean distance between Query and Prototypes
        # cdist computes pair-wise distance between query_cls and prototypes
        # query_cls: [num_queries, embed_dim]
        # prototypes: [n_way, embed_dim]
        # dists: [num_queries, n_way]
        dists = torch.cdist(query_cls, prototypes, p=2.0)
        
        # We use negative distances as logits (the smaller the distance, the higher the raw score)
        logits = -dists
        
        return logits
