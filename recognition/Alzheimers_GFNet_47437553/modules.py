"""
GFNet Modules for Alzheimer's Disease Classification

This module implements the Global Filter Network (GFNet) components for classifying
Alzheimer's disease from brain MRI scans. The model uses Fourier domain filtering
to capture global spatial relationships in medical images.

Components:
- Mlp: Multi-layer perceptron with GELU activation
- GlobalFilter: Fourier-based global filtering layer
- Block: Transformer-like block combining GlobalFilter and MLP
- PatchEmbed: Image to patch embedding conversion
- GFNet: Complete model for binary classification (Normal vs AD)

Reference:
Y. Rao, W. Zhao, Z. Zhu, J. Zhou, and J. Lu, "GFNet: Global Filter Networks 
for Visual Recognition," IEEE TPAMI, 2023.
"""

import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class LayerScale(nn.Module):
    """Applies learnable per-channel scaling to features.
    Initialized with small values to stabilize training."""
    
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma


class ChannelAttention(nn.Module):
    """Applies channel-wise attention using both average and max pooling.
    Helps the model focus on important feature channels."""
    
    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction_ratio, 4), bias=False),
            nn.GELU(),
            nn.Linear(max(dim // reduction_ratio, 4), dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, N, C = x.shape
        x_t = x.transpose(1, 2)
        avg_out = self.avg_pool(x_t).view(B, C)
        max_out = self.max_pool(x_t).view(B, C)
        channel_attention = self.fc(avg_out) + self.fc(max_out)
        return x * channel_attention.unsqueeze(1)


class EnhancedMlp(nn.Module):
    """Enhanced MLP with residual connections and layer normalization.
    Provides more stable training compared to standard MLP."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0., use_residual=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_residual = use_residual and (in_features == out_features)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden_features)
    
    def forward(self, x):
        shortcut = x if self.use_residual else 0
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x + shortcut * 0.1


class ImprovedGlobalFilter(nn.Module):
    """Global filter layer operating in Fourier domain with learnable frequency weights.
    Captures global spatial relationships through frequency domain processing."""
    
    def __init__(self, dim, h=14, w=8, init_scale=0.02):
        super().__init__()
        # Complex weights stored as [real, imag] in last dimension
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * init_scale
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        self.freq_dropout = nn.Dropout2d(0.1)
        self.w = w
        self.h = h
    
    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        
        # Determine spatial dimensions
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        
        x = x.view(B, a, b, C).to(torch.float32)
        
        # Transform to frequency domain
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # Apply learnable frequency filter
        weight = torch.view_as_complex(self.complex_weight) * self.temperature
        
        try:
            x_fft = x_fft * weight
        except Exception:
            # Fallback: reshape weight for broadcasting
            w_shape = (1, min(a, weight.shape[0]), 
                      min(x_fft.shape[2], weight.shape[1]), 
                      min(C, weight.shape[2]))
            weight_c = weight.reshape((1,) + weight.shape)
            x_fft = x_fft * weight_c
        
        # Transform back to spatial domain
        x = torch.fft.irfft2(x_fft, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class EnhancedBlock(nn.Module):
    """Transformer-like block combining global filtering with MLP.
    Includes optional channel attention and layer scaling for improved performance."""
    
    def __init__(self, dim, mlp_ratio=4., drop=0.1, drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8, 
                 use_channel_attention=True, use_layer_scale=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = ImprovedGlobalFilter(dim, h=h, w=w)
        
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(dim)
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = LayerScale(dim)
            self.layer_scale_2 = LayerScale(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EnhancedMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                               act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        # Global filter branch
        filtered = self.filter(self.norm1(x))
        
        if self.use_channel_attention:
            filtered = self.channel_attn(filtered)
        
        if self.use_layer_scale:
            filtered = self.layer_scale_1(filtered)
        
        x = x + self.drop_path(filtered)
        
        # MLP branch
        mlp_out = self.mlp(self.norm2(x))
        
        if self.use_layer_scale:
            mlp_out = self.layer_scale_2(mlp_out)
        
        x = x + self.drop_path(mlp_out)
        return x


class ImprovedPatchEmbed(nn.Module):
    """Converts image into patch embeddings using convolution.
    Supports optional overlapping patches for better feature extraction."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, use_overlap=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        if use_overlap:
            stride = (patch_size[0] // 2, patch_size[1] // 2)
            padding = patch_size[0] // 4
        else:
            stride = patch_size
            padding = 0
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate number of patches
        if use_overlap:
            num_patches = (
                ((img_size[0] + stride[0] - patch_size[0]) // stride[0] + 1) * 
                ((img_size[1] + stride[1] - patch_size[1]) // stride[1] + 1)
            )
        else:
            num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                             stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class EnhancedGFNet(nn.Module):
    """Enhanced Global Filter Network for Alzheimer's disease classification.
    Uses Fourier-based filtering with multi-scale features and auxiliary supervision."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, 
                 embed_dim=768, depth=12, mlp_ratio=4., drop_rate=0.1, 
                 drop_path_rate=0.2, norm_layer=None, use_channel_attention=True, 
                 use_layer_scale=True, use_overlap_patch=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Patch embedding
        self.patch_embed = ImprovedPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, use_overlap=use_overlap_patch
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Calculate frequency domain dimensions
        h = int(math.sqrt(num_patches))
        w = h // 2 + 1
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Build transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedBlock(
                dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, 
                use_channel_attention=use_channel_attention and (i % 2 == 0), 
                use_layer_scale=use_layer_scale
            )
            for i in range(depth)
        ])
        
        # Multi-scale feature extraction layers
        self.feature_layers = [depth // 3, 2 * depth // 3, depth - 1]
        
        self.norm = norm_layer(embed_dim)
        
        # Classification head with progressive dimensionality reduction
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Dropout(max(drop_rate / 2, 0.05)),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # Auxiliary head for additional supervision
        self.aux_head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights using appropriate strategies for each layer type."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        """Extract features through patch embedding and transformer blocks.
        Aggregates multi-scale features for improved representation."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.feature_layers:
                features.append(x.mean(1))
        
        x = self.norm(x).mean(1)
        
        # Combine multi-scale features
        if len(features) > 0:
            multi_scale_feature = sum(features) / len(features)
            x = x + multi_scale_feature * 0.1
        
        return x
    
    def forward(self, x, return_aux=False):
        """Forward pass through the network.
        Optionally returns auxiliary logits for multi-task training."""
        x = self.forward_features(x)
        logits = self.head(x)
        
        if return_aux:
            aux_logits = self.aux_head(x)
            return logits, aux_logits
        
        return logits


def create_enhanced_gfnet_small(num_classes=2, img_size=224, **kwargs):
    """Creates a small Enhanced GFNet model with 512 embedding dimension.
    Suitable for smaller datasets or resource-constrained scenarios."""
    model = EnhancedGFNet(
        img_size=img_size, patch_size=16, embed_dim=512, depth=12, 
        mlp_ratio=4, num_classes=num_classes, drop_rate=0.1, 
        drop_path_rate=0.15, use_channel_attention=True, 
        use_layer_scale=True, **kwargs
    )
    return model


def create_enhanced_gfnet_base(num_classes=2, img_size=224, **kwargs):
    """Creates a base Enhanced GFNet model with 768 embedding dimension.
    Balanced performance and computational cost for most applications."""
    model = EnhancedGFNet(
        img_size=img_size, patch_size=16, embed_dim=768, depth=14, 
        mlp_ratio=4, num_classes=num_classes, drop_rate=0.08, 
        drop_path_rate=0.18, use_channel_attention=True, 
        use_layer_scale=True, use_overlap_patch=False, **kwargs
    )
    return model


def create_enhanced_gfnet_large(num_classes=2, img_size=224, **kwargs):
    """Creates a large Enhanced GFNet model with 1024 embedding dimension.
    Highest capacity model with overlapping patches for maximum performance."""
    model = EnhancedGFNet(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=18, 
        mlp_ratio=4, num_classes=num_classes, drop_rate=0.2, 
        drop_path_rate=0.25, use_channel_attention=True, 
        use_layer_scale=True, use_overlap_patch=True, **kwargs
    )
    return model