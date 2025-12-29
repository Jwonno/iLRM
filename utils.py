import torch
from einops import rearrange

def compute_rays(fxfycxcy, c2w, h, w):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d

def compute_rays_resolution(fxfycxcy, c2w, h, w, factor=2):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    w_factor = w // factor
    h_factor = h // factor
    fxfycxcy_factor = fxfycxcy.clone() / factor

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w_factor, device=c2w.device)[None, :].expand(h_factor, -1)
    idx_y = torch.arange(h_factor, device=c2w.device)[:, None].expand(-1, w_factor)

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)
    idx_y = idx_y.flatten().expand(b * v, -1)

    fxfycxcy_factor = fxfycxcy_factor.reshape(b * v, 4)
    c2w = c2w.reshape(b * v, 4, 4)

    x = (idx_x + 0.5 - fxfycxcy_factor[:, 2:3]) / fxfycxcy_factor[:, 0:1]
    y = (idx_y + 0.5 - fxfycxcy_factor[:, 3:4]) / fxfycxcy_factor[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=1)
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h_factor*w_factor)

    ray_o = ray_o.reshape(b, v, 3, h_factor, w_factor)
    ray_d = ray_d.reshape(b, v, 3, h_factor, w_factor)

    return ray_o, ray_d

def compute_rays_resolution_offset(fxfycxcy, c2w, h, w, offset, factor=2):
    """Transform target before computing loss
    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    w_factor = w // factor
    h_factor = h // factor
    fxfycxcy_factor = fxfycxcy.clone() / factor

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w_factor, device=c2w.device)[None, :].expand(h_factor, -1)
    idx_y = torch.arange(h_factor, device=c2w.device)[:, None].expand(-1, w_factor)

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)
    idx_y = idx_y.flatten().expand(b * v, -1)

    fxfycxcy_factor = fxfycxcy_factor.reshape(b * v, 4)
    c2w = c2w.reshape(b * v, 4, 4)

    x = (idx_x + 0.5 + offset[..., 0] - fxfycxcy_factor[:, 2:3]) / fxfycxcy_factor[:, 0:1]
    y = (idx_y + 0.5 + offset[..., 1] - fxfycxcy_factor[:, 3:4]) / fxfycxcy_factor[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=1)
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h_factor*w_factor)

    ray_o = ray_o.reshape(b, v, 3, h_factor, w_factor)
    ray_d = ray_d.reshape(b, v, 3, h_factor, w_factor)

    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device, config):
        super(PerceptualLoss, self).__init__()
        vgg_weigths = config.train.get("perceptual_vgg_weights", "default")
        if vgg_weigths == "default":
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg19()
            vgg.load_state_dict(torch.load(vgg_weigths, map_location="cpu", weights_only=True))
        #print(vgg.features)
        # replace the maxpool layer with avgpool
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, nn.MaxPool2d):
                vgg.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList()
        out_idx = config.train.perceptual_out_idx
        out_idx = [0] + out_idx
        self.layer_weights = config.train.perceptual_out_weights
        assert len(self.layer_weights) == len(out_idx) - 1
        self.feature_scale = config.train.perceptual_feature_scale
        for i in range(len(out_idx)-1):
            self.blocks.append(nn.Sequential(vgg.features[out_idx[i]:out_idx[i+1]]).to(device).eval())
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        #self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
 
    def forward(self, pred, target):
        """
        pred, target: [B, 3, H, W] in range [0, 1]
        """
        weights = self.layer_weights
        scale = self.feature_scale
        pred = (pred - self.mean) * scale
        target = (target - self.mean) * scale
        loss = torch.mean(torch.abs(pred - target))
        for i_b, block in enumerate(self.blocks):
            pred = block(pred)
            target = block(target)
            loss += torch.mean(torch.abs(pred - target)) * weights[i_b]
        loss = loss / scale
        return loss