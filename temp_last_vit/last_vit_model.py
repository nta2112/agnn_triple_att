import torch
from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights

class DenseViT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kernel = None

    def gaussian_kernel_1d(self, kernel_size, sigma):
        kernel = torch.exp(-0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2)
        kernel = kernel / torch.max(kernel)
        return kernel

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        
        if self.cached_kernel is None:
            self.cached_kernel = self.gaussian_kernel_1d(768, 768 ** 0.5).to(x.device).unsqueeze(0).unsqueeze(0)
        
        x_detach = x[:, 1:]
        x_fft = torch.fft.fft(x[:, 1:], dim=-1)
        gs_k = self.cached_kernel.to(x.device)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * gs_k
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_recovered = torch.fft.ifft(x_fft, dim=-1).real
        
        # Calculate difference and select best representative patches
        diff = x_detach / (torch.abs(x_recovered - x_detach) + 1e-8)  # added small epsilon for stability
        _, indices = torch.topk(diff, k=1, dim=1, largest=True)
        sel_p = torch.gather(x_detach, 1, indices)
        
        # Mean over selected patches acts as the robust cls_token
        cls_token = torch.mean(sel_p, dim=1)
        
        return cls_token, x_detach

def build_last_vit_b16(pretrained=True):
    """
    Build LaSt-ViT (ViT-B/16 based) model.
    """
    model = DenseViT(
        image_size=224,
        patch_size=16, 
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    if pretrained:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        state_dict = weights.get_state_dict(progress=True)
        # Load weights, strict=False because DenseViT has the same parameters as VisionTransformer
        # the forward behavior is just different
        # Ignore classifier head since we use cls_token directly
        if 'heads.head.weight' in state_dict:
            del state_dict['heads.head.weight']
            del state_dict['heads.head.bias']
        model.load_state_dict(state_dict, strict=False)
        print("Loaded ImageNet1k V1 Pretrained weights for LaSt-ViT")
        
    return model
