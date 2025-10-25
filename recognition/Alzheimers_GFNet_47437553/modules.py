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
from functools import partial
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with two linear layers and GELU activation.
    
    This feedforward network is used after the GlobalFilter layer to process
    the filtered features. Includes dropout for regularization.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden layer features. 
                                         Defaults to in_features if None
        out_features (int, optional): Number of output features. 
                                      Defaults to in_features if None
        act_layer (nn.Module): Activation function class. Default: nn.GELU
        drop (float): Dropout probability. Default: 0.0
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # First linear layer expands to hidden dimension
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # Second linear layer projects back to output dimension
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    """
    Global Filter layer using 2D Fourier Transform for spatial mixing.
    
    This layer applies learnable filters in the frequency domain, enabling
    efficient global spatial interactions. Particularly effective for medical
    imaging where global context is important for diagnosis.
    
    The operation:
    1. Converts spatial features to frequency domain via FFT
    2. Applies learnable complex-valued filters
    3. Converts back to spatial domain via inverse FFT
    
    Args:
        dim (int): Number of input channels
        h (int): Height of the spatial feature map. Default: 14
        w (int): Width for real FFT (typically h//2 + 1). Default: 8
    """
    
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # Learnable complex weights for frequency domain filtering
        # Shape: (h, w, dim, 2) where last dimension is [real, imaginary]
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        """
        Apply global filtering in frequency domain.
        
        Args:
            x (torch.Tensor): Input features of shape (B, N, C) where
                             N = H * W (flattened spatial dimensions)
            spatial_size (tuple, optional): (H, W) spatial dimensions.
                                           If None, assumes square (sqrt(N), sqrt(N))
        
        Returns:
            torch.Tensor: Filtered features of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Determine spatial dimensions
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        
        # Reshape to 2D spatial format: (B, H, W, C)
        x = x.view(B, a, b, C)
        
        # Ensure float32 for FFT compatibility
        x = x.to(torch.float32)
        
        # Apply 2D Real FFT (efficient for real-valued inputs)
        # Output shape: (B, H, W//2+1, C) as complex tensor
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # Convert learned weights to complex format
        weight = torch.view_as_complex(self.complex_weight)
        
        # Element-wise multiplication in frequency domain
        x = x * weight
        
        # Apply inverse FFT to return to spatial domain
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        
        # Reshape back to sequence format: (B, N, C)
        x = x.reshape(B, N, C)
        
        return x


class Block(nn.Module):
    """
    GFNet Transformer Block.
    
    Combines layer normalization, global filtering, and MLP in a residual
    configuration. Similar to Vision Transformer blocks but uses GlobalFilter
    instead of attention for efficient global spatial mixing.
    
    Structure:
        x -> LayerNorm -> GlobalFilter -> +
        |                                 |
        └-> LayerNorm -> MLP ------------> +
    
    Args:
        dim (int): Feature dimension
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Default: 4.0
        drop (float): Dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation function. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        h (int): Spatial height for GlobalFilter. Default: 14
        w (int): Frequency width for GlobalFilter. Default: 8
    """
    
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        
        # First normalization before GlobalFilter
        self.norm1 = norm_layer(dim)
        
        # Global Filter for spatial mixing
        self.filter = GlobalFilter(dim, h=h, w=w)
        
        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Second normalization before MLP
        self.norm2 = norm_layer(dim)
        
        # MLP for channel mixing
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )

    def forward(self, x):
        """
        Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        # Apply filter block with residual connection
        # Note: The original code has a specific structure that can be expanded as:
        # x = x + drop_path(mlp(norm2(filter(norm1(x)))))
        # This applies norm->filter->norm->mlp with a single residual
        x = x + self.drop_path(
            self.mlp(self.norm2(self.filter(self.norm1(x))))
        )
        
        return x


class PatchEmbed(nn.Module):
    """
    Convert 2D images into patch embeddings.
    
    Splits the input image into non-overlapping patches and projects them
    into an embedding space using a convolutional layer. This is the first
    stage of the GFNet that converts spatial images into a sequence of tokens.
    
    For medical imaging, this allows the model to process different anatomical
    regions as separate tokens while maintaining spatial relationships.
    
    Args:
        img_size (int or tuple): Input image dimensions. Default: 224
        patch_size (int or tuple): Size of each patch. Default: 16
        in_chans (int): Number of input channels. Default: 1 (grayscale MRI)
        embed_dim (int): Embedding dimension. Default: 768
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # Calculate number of patches
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Convolutional layer for patch embedding
        # Kernel size and stride equal to patch_size creates non-overlapping patches
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        """
        Convert image to patch embeddings.
        
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Verify input dimensions match expected size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Apply patch projection: (B, C, H, W) -> (B, embed_dim, H', W')
        # where H' = H/patch_size, W' = W/patch_size
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, N)
        x = x.flatten(2)
        
        # Transpose to sequence format: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class GFNet(nn.Module):
    """
    Global Filter Network for Alzheimer's Disease Classification.
    
    A vision model using Fourier-based global filters instead of attention
    mechanisms. Particularly effective for medical imaging where global context
    is crucial for diagnosis.
    
    Architecture:
    1. PatchEmbed: Convert image to patch tokens
    2. Positional Embedding: Add learnable position information
    3. Stacked Blocks: Multiple GlobalFilter + MLP blocks
    4. Classification Head: Global pooling + Linear layer for AD vs Normal
    
    Args:
        img_size (int or tuple): Input image size. Default: 224
        patch_size (int or tuple): Patch size for embedding. Default: 16
        in_chans (int): Input channels (1 for grayscale MRI). Default: 1
        num_classes (int): Number of output classes (2 for binary). Default: 2
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        mlp_ratio (float): MLP expansion ratio. Default: 4.0
        drop_rate (float): Dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module, optional): Normalization layer
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, 
                 embed_dim=768, depth=12, mlp_ratio=4., drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Calculate spatial dimensions for GlobalFilter
        h = img_size // patch_size
        w = h // 2 + 1  # For real FFT
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stack of GFNet blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                h=h,
                w=w
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for linear and normalization layers.
        
        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Extract features from input image.
        
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Feature vector of shape (B, embed_dim)
        """
        # Convert to patches and add positional embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Global average pooling over patches
        x = self.norm(x).mean(1)
        
        return x

    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


def gfnet_small(num_classes=2, img_size=224, **kwargs):
    """
    GFNet-Small variant for Alzheimer's classification.
    
    Args:
        num_classes (int): Number of classes. Default: 2 (Normal, AD)
        img_size (int): Input image size. Default: 224
        **kwargs: Additional arguments for GFNet
        
    Returns:
        GFNet: Small model instance
    """
    model = GFNet(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def gfnet_base(num_classes=2, img_size=224, **kwargs):
    """
    GFNet-Base variant for Alzheimer's classification.
    
    Args:
        num_classes (int): Number of classes. Default: 2 (Normal, AD)
        img_size (int): Input image size. Default: 224
        **kwargs: Additional arguments for GFNet
        
    Returns:
        GFNet: Base model instance
    """
    model = GFNet(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def gfnet_large(num_classes=2, img_size=224, **kwargs):
    """
    GFNet-Large variant for Alzheimer's classification.
    
    Args:
        num_classes (int): Number of classes. Default: 2 (Normal, AD)
        img_size (int): Input image size. Default: 224
        **kwargs: Additional arguments for GFNet
        
    Returns:
        GFNet: Large model instance
    """
    model = GFNet(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=18,
        mlp_ratio=4,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.2,
        **kwargs
    )
    return model
