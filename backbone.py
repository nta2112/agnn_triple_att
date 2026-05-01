import torch
import torch.nn as nn


class ResNet12Block(nn.Module):
    """
    ResNet Block
    """
    def __init__(self, inplanes, planes):
        super(ResNet12Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class ResNet12(nn.Module):
    """
    ResNet12 Backbone
    """
    def __init__(self, emb_size, block=ResNet12Block, cifar_flag=False):
        super(ResNet12, self).__init__()
        cfg = [64, 128, 256, 512]
        # layers = [1, 1, 1, 1]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        self.layer1 = self._make_layer(block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3])
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AvgPool2d(6)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 -> 64
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 64 -> 64
        x = self.layer1(x)
        # 64 -> 128
        x = self.layer2(x)
        # 128 -> 256
        inter = self.layer3(x)
        # 256 -> 512
        # print(inter.shape)
        x = self.layer4(inter)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # 512 -> 128
        x = self.layer_last(x)
        # print(x.shape)
        inter = self.maxpool(inter)
        # 256 * 5 * 5
        inter = inter.view(inter.size(0), -1)
        # 256 * 5 * 5 -> 128
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


class ConvNet(nn.Module):
    """
    Conv4 Backbone
    """
    def __init__(self, emb_size, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 25 if not cifar_flag else self.hidden
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2,
                                          out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out = []
        out.append(self.layer_last(output_data.view(output_data.size(0), -1)))
        out.append(self.layer_second(output_data0.view(output_data0.size(0), -1)))
        return out


class ResNet50Pretrained(nn.Module):
    """
    ResNet50 Backbone with pretrained ImageNet weights
    """
    def __init__(self, emb_size):
        super(ResNet50Pretrained, self).__init__()
        import torchvision.models as models
        # Load the pretrained ResNet50
        resnet = models.resnet50(weights=None)
        # Stem and layer1, 2
        self.features_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((5, 5))
        
        # AGNN required FC layers
        self.emb_size = emb_size
        self.layer_second = nn.Sequential(
            nn.Linear(in_features=1024 * 5 * 5, out_features=self.emb_size, bias=True),
            nn.LayerNorm(self.emb_size)
        )
        self.layer_last = nn.Sequential(
            nn.Linear(in_features=2048, out_features=self.emb_size, bias=True),
            nn.LayerNorm(self.emb_size)
        )

    def train(self, mode=True):
        super().train(mode)
        # Khóa toàn bộ BatchNorm2d của ResNet để không làm hỏng trọng số ImageNet vì Batch Size = 1
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, input_data):
        stem = self.features_stem(input_data)
        inter = self.layer3(stem)  # B, 1024, ~6x6
        x_out = self.layer4(inter) # B, 2048, ~3x3
        
        inter_pool = self.maxpool(inter)
        inter_flat = inter_pool.view(inter_pool.size(0), -1)
        inter_embed = self.layer_second(inter_flat)
        
        x_pool = self.avgpool(x_out)
        x_flat = x_pool.view(x_pool.size(0), -1)
        x_embed = self.layer_last(x_flat)
        
        out = []
        out.append(x_embed)
        out.append(inter_embed)
        return out


import torch.nn.functional as F
from temp_last_vit.last_vit_model import build_last_vit_b16

class LaStViTBackbone(nn.Module):
    """
    LaSt-ViT Backbone for AGNN (ViT-B/16 based)
    """
    def __init__(self, emb_size, pretrained=True):
        super(LaStViTBackbone, self).__init__()
        self.emb_size = emb_size
        
        # Load DenseViT (The backbone itself) 
        # We need to access individual blocks for intermediate features
        self.vit = build_last_vit_b16(pretrained=pretrained)
        
        self.image_size = 224 # ViT requires 224x224
        self.hidden_dim = 768 # ViT-B hidden dimension
        
        # Projection heads for AGNN (Stage Last and Stage Second-Last)
        self.layer_last = nn.Sequential(
            nn.Linear(self.hidden_dim, self.emb_size),
            nn.LayerNorm(self.emb_size)
        )
        self.layer_second = nn.Sequential(
            nn.Linear(self.hidden_dim, self.emb_size),
            nn.LayerNorm(self.emb_size)
        )

    def forward(self, x):
        # Resize if input is not 224x224
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Manually perform forward to get intermediate outputs
        # We process patch embedding and class token first
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Encoder processing: 12 Blocks
        # We'll take the output of 11th block as 'inter' and 12th as 'last'
        # VisionTransformer.encoder is a Sequential(Dropout, layers: Sequential)
        # torchvision version: vit.encoder has .layers
        for i, block in enumerate(self.vit.encoder.layers):
            x = block(x)
            if i == 10: # 11th block (index 10)
                stage_second = x
        
        stage_last = x # 12th block output 
        
        # For ViT, we typically use the [CLS] token (0-th token) as the representative feature
        # but LaSt-ViT provides a more robust 'cls_token' calculation logic from DenseViT 
        # We will use that logic on the final hidden states
        
        # Stage Last: Use LaSt-ViT robust cls_token calculation on the final states
        def get_robust_cls(h_states):
            if self.vit.cached_kernel is None:
                self.vit.cached_kernel = self.vit.gaussian_kernel_1d(768, 768 ** 0.5).to(h_states.device).unsqueeze(0).unsqueeze(0)
            
            x_detach = h_states[:, 1:]
            x_fft = torch.fft.fft(h_states[:, 1:], dim=-1)
            x_fft = torch.fft.fftshift(x_fft, dim=-1)
            x_fft = x_fft * self.vit.cached_kernel.to(h_states.device)
            x_fft = torch.fft.ifftshift(x_fft, dim=-1)
            x_recovered = torch.fft.ifft(x_fft, dim=-1).real
            
            diff = x_detach / (torch.abs(x_recovered - x_detach) + 1e-8)
            _, indices = torch.topk(diff, k=1, dim=1, largest=True)
            sel_p = torch.gather(x_detach, 1, indices)
            return torch.mean(sel_p, dim=1)

        last_cls = get_robust_cls(stage_last)
        second_cls = get_robust_cls(stage_second)
        
        # Project to AGNN emb_size
        x_embed = self.layer_last(last_cls)
        inter_embed = self.layer_second(second_cls)
        
        return [x_embed, inter_embed]
