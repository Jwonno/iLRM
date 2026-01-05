import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

import cv2

import matplotlib.pyplot as plt
import os
import numpy as np

try:
    import xformers.ops as xops
except ImportError:
    xops = None


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4., mlp_bias=False, 
                 out_features=None, act_layer=nn.GELU, norm_layer=None):
        super().__init__()
        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(in_features, bias=False)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        if self.norm_exists:
            x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, qk_norm=True, 
                 norm_layer=None):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(dim, bias=False)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(2*dim, dim, bias=False)

    def forward(self, x, support, V):
        """
        x: (B, L, D)
        support: (B, C, D)
        Returns: same shape as input 
        """
        # B, N, C = x.shape
        support = rearrange(support, "b (v l) d -> (b v) l d", v=V)
        x = rearrange(x, 'b (v l) d -> (b v) l d', v=V)
        B_s, N_s, C_s = support.shape
        B_x, N_x, C_x = x.shape
        if self.norm_exists:
            x = self.norm(x)

        # q = self.q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads)
        q = self.q(x)
        q = rearrange(q, "b_v l (nh nd s) -> b_v (l s) nh nd", s=2, nh=self.num_heads, nd=C_x // self.num_heads)

        k = self.k(support).reshape(B_s, N_s, self.num_heads, C_s // self.num_heads)
        v = self.v(support).reshape(B_s, N_s, self.num_heads, C_s // self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        x = xops.memory_efficient_attention(q, k, v, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp))
        # x = rearrange(x, "(b v) l nh dh -> b (v l) (nh dh)", v=V)
        x = rearrange(x, "(b v) (l s) nh dh -> b (v l) (nh dh s)", v=V, s=2)
        x = self.proj(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, qk_norm=True, 
                 norm_layer=None, use_flashatt_v2=True, block_number=0):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_exists = norm_layer is not None
        if self.norm_exists:
            self.norm = norm_layer(dim, bias=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.block_number = block_number

    def forward(self, x, input_frames=None):
        """
        x: (B, L, D)
        input_frames: list of frame indices to visualize attention maps (B, V, C, H, W) [0.0~1.0]
        Returns: same shape as input 
        """

        if self.norm_exists:
            x = self.norm(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 1, 3, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, N, H, C) 

        q, k = self.q_norm(q), self.k_norm(k)

        x = xops.memory_efficient_attention(q, k, v, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp))
        x = rearrange(x, "b l nh dh -> b l (nh dh)")

        x = self.proj(x)

        # --- 시각화 로직 시작 ---
        if input_frames is not None:
            # 1. 설정 및 타겟 인덱스 선정
            # 배치 처리를 위해 input_frames는 첫 번째 배치만 사용한다고 가정 (기존 로직 유지)
            curr_frames = input_frames[0]  # (V, C, H, W)
            V, C_img, H_img, W_img = curr_frames.shape
            
            frame_token_num = N // V
            
            # 시각화할 토큰 인덱스 목록 생성
            target_indices = np.arange(
                0, 
                N, 
                frame_token_num // 2, 
            ).tolist()
            
            # 중복 제거 및 유효성 검사
            target_indices = sorted(list(set([idx for idx in target_indices if idx < N])))
            
            # 2. 핵심 최적화: 필요한 Query만 선택 (Slicing)
            # q: (B, N, H, C_head) -> (B, H, N, C_head)
            q_perm = q.permute(0, 2, 1, 3) 
            k_perm = k.permute(0, 2, 1, 3)
            
            # (B, H, Num_Targets, C_head)
            q_selected = q_perm[:, :, target_indices, :] 
            
            # 3. 어텐션 스코어 계산 (Selected Query vs All Keys)
            # (B, H, Num_Targets, C) @ (B, H, C, N) -> (B, H, Num_Targets, N)
            # N^2 연산이 (Num_Targets * N) 연산으로 줄어듦. 순식간에 계산됨.
            attn_score = q_selected @ k_perm.transpose(-2, -1)
            
            # Head 평균 (B, Num_Targets, N)
            attn_avg = attn_score.mean(dim=1) 
            
            # 첫 번째 배치만 시각화 (기존 로직 따름)
            attn_avg = attn_avg[0] # (Num_Targets, N)

            # 4. 시각화 맵 생성 (Batch Processing)
            # N = V * (H_feat * W_feat) 라고 가정
            feat_h, feat_w = H_img // 16, W_img // 16
            
            # (Num_Targets, V, H_feat, W_feat) 형태로 변형
            attn_maps = attn_avg.reshape(len(target_indices), V, feat_h, feat_w)
            
            # Interpolation을 위해 차원 병합: (Num_Targets * V, 1, H_feat, W_feat)
            attn_maps_resized = attn_maps.view(-1, 1, feat_h, feat_w)
            
            # 한 번에 Upsampling 수행
            attn_maps_up = F.interpolate(
                attn_maps_resized, 
                size=(H_img, W_img), 
                mode="bilinear", 
                align_corners=False
            ).squeeze() # (Num_Targets * V, H_img, W_img)
            
            # Min-Max Normalization (Batch-wise)
            # 각 맵별로 min, max 계산
            mins = attn_maps_up.flatten(1).min(dim=1)[0].view(-1, 1, 1)
            maxs = attn_maps_up.flatten(1).max(dim=1)[0].view(-1, 1, 1)
            attn_norm = (attn_maps_up - mins) / (maxs - mins + 1e-6) # 0~1 range

            # 5. 이미지 저장 (OpenCV 사용)
            attn_norm_np = attn_norm.detach().cpu().numpy() # (Total_Maps, H, W)
            frames_np = curr_frames.permute(0, 2, 3, 1).cpu().numpy() # (V, H, W, C)
            
            output_base_dir = f"experiments/attention_map/block_{self.block_number}_headavg"

            patch_size = 16
            
            for i, target_idx in enumerate(target_indices):
                token_dir = os.path.join(output_base_dir, f"token_{target_idx}")
                os.makedirs(token_dir, exist_ok=True)
                
                target_frame_idx = target_idx // frame_token_num
                local_token_idx = target_idx % frame_token_num

                grid_y = local_token_idx // (feat_w)    # Row
                grid_x = local_token_idx % (feat_w)   # Column

                center_x = int((grid_x + 0.5) * patch_size)
                center_y = int((grid_y + 0.5) * patch_size)

                for v in range(V):
                    # 해당 프레임의 어텐션 맵 가져오기
                    map_idx = i * V + v
                    heatmap = attn_norm_np[map_idx] # (H, W)
                    
                    # Colorize (OpenCV는 BGR 기준이므로 주의)
                    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    
                    # Overlay
                    # input_frames는 0~1 float 가정, OpenCV는 0~255 uint8 필요
                    bg_img = (frames_np[v] * 255).astype(np.uint8)
                    # RGB -> BGR 변환 (OpenCV 저장을 위해)
                    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR)
                    
                    overlay = cv2.addWeighted(bg_img, 0.5, colored_map, 0.5, 0)
                    
                    if v == target_frame_idx:
                        cv2.drawMarker(
                            overlay,
                            (center_x, center_y),
                            (0, 255, 0), # Green
                            markerType=cv2.MARKER_CROSS,
                            markerSize=20,
                            thickness=2
                        )

                    cv2.imwrite(os.path.join(token_dir, f"frame_{v}.png"), overlay)
                    
            print(f"Saved attention maps for block {id(self)}")

        return x


class ReadBlock(nn.Module):
    def __init__(
        self, dim, head_dim, mlp_ratio=4., 
        mlp_bias=False, qkv_bias=False, qk_scale=None, 
        qk_norm=True, act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm):
        super().__init__()

        self.read_cross = CrossAttention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            qk_norm=qk_norm, norm_layer=norm_layer)
        self.read_ff = Mlp(
            in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, 
            act_layer=act_layer, norm_layer=norm_layer)
        

    def forward(self, x, support_tokens, V):
        """
        x: (B, L, D)
        image_tokens: (B, C, D)
        Returns: same shape as input
        """
        x = x + self.read_cross(x, support_tokens, V)
        x = x + self.read_ff(x)
        return x
    

class SelfAttnBlock(nn.Module):
    def __init__(
        self, dim, head_dim, mlp_ratio=4., 
        mlp_bias=False, qkv_bias=False, qk_scale=None, 
        qk_norm=True, act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm,
        block_number=0):
        super().__init__()

        self.attn = SelfAttention(dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            qk_norm=qk_norm, norm_layer=norm_layer, block_number=block_number)
        self.mlp = Mlp(in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, 
            act_layer=act_layer, norm_layer=norm_layer)
        
    def forward(self, x, input_frames=None):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        x = x + self.attn(x, input_frames=input_frames)
        x = x + self.mlp(x)
        return x