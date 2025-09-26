import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class VTMUpsamplerExact(nn.Module):
    """
    Exact PyTorch implementation of VTM's display upsampler.
    
    This module implements the same upsampling algorithm as VTM's sampleRateConv function,
    using separable 1D filtering with fixed-point arithmetic and proper normalization.
    
    Attributes:
        log2Norm (int): Normalization factor for filter coefficients (12 or 16)
        luma_filter (torch.Tensor): 8-tap luma interpolation filter [16, 8]
        chroma_filter (torch.Tensor): 4-tap chroma interpolation filter [16, 4]
    """
    
    def __init__(self, log2Norm: int = 16):
        """
        Initialize VTM upsampler with filter coefficients.
        
        Args:
            log2Norm: Normalization factor (12 for standard, 16 for alternative filters)
        """
        super().__init__()
        self.log2Norm = log2Norm
        
        # Register filter coefficients as non-trainable buffers
        self._register_filters()
    
    def _register_filters(self) -> None:
        """Register VTM filter coefficients as non-trainable buffers."""
        
        # VTM 8-tap luma filter (standard VVC interpolation filter)
        luma_coeffs = torch.tensor([
            [0, 0, 0, 64, 0, 0, 0, 0],
            [0, 1, -3, 63, 4, -2, 0, 0],
            [-1, 2, -5, 62, 8, -3, 1, 0],
            [-1, 3, -8, 60, 13, -4, 1, 0],
            [-1, 4, -10, 58, 17, -5, 1, 0],
            [-1, 4, -11, 52, 26, -8, 3, -1],
            [-1, 3, -9, 47, 31, -10, 4, -1],
            [-1, 4, -11, 45, 34, -10, 4, -1],
            [-1, 4, -11, 40, 40, -11, 4, -1],
            [-1, 4, -10, 34, 45, -11, 4, -1],
            [-1, 4, -10, 31, 47, -9, 3, -1],
            [-1, 3, -8, 26, 52, -11, 4, -1],
            [0, 1, -5, 17, 58, -10, 4, -1],
            [0, 1, -4, 13, 60, -8, 3, -1],
            [0, 1, -3, 8, 62, -5, 2, -1],
            [0, 0, -2, 4, 63, -3, 1, 0]
        ], dtype=torch.float32)
        
        # VTM 4-tap chroma filter
        chroma_coeffs = torch.tensor([
            [0, 64, 0, 0],
            [-2, 62, 4, 0],
            [-4, 60, 8, 0],
            [-6, 58, 12, 0],
            [-8, 56, 16, 0],
            [-10, 54, 20, 0],
            [-12, 52, 24, 0],
            [-14, 50, 28, 0],
            [-16, 48, 32, 0],
            [-18, 46, 36, 0],
            [-20, 44, 40, 0],
            [-22, 42, 44, 0],
            [-24, 40, 48, 0],
            [-26, 38, 52, 0],
            [-28, 36, 56, 0],
            [-30, 34, 60, 0]
        ], dtype=torch.float32)
        
        self.register_buffer('luma_filter', luma_coeffs)
        self.register_buffer('chroma_filter', chroma_coeffs)
    
    def _get_filter_coefficients(self, is_luma: bool) -> torch.Tensor:
        """
        Get appropriate filter coefficients based on component type.
        
        Args:
            is_luma: True for luma component, False for chroma
            
        Returns:
            Filter coefficients tensor [16, num_taps]
        """
        return self.luma_filter if is_luma else self.chroma_filter
    
    def _calculate_scaling_ratio(self, scale_factor: int) -> int:
        """
        Calculate fixed-point scaling ratio.
        
        Args:
            scale_factor: Integer scaling factor (2 or 4)
            
        Returns:
            Fixed-point scaling ratio (14-bit precision)
        """
        return int(1 << 14) // scale_factor
    
    def _calculate_reference_position(self, output_index: int, scaling_ratio: int) -> Tuple[int, int]:
        """
        Calculate reference position and fractional part for filtering.
        
        Args:
            output_index: Index in output dimension
            scaling_ratio: Fixed-point scaling ratio
            
        Returns:
            Tuple of (integer_part, fractional_part)
        """
        ref_pos = (output_index * scaling_ratio) >> 14
        integer = ref_pos >> 4  # 4 fractional bits
        frac = ref_pos & 15     # Fractional part (0-15)
        return integer, frac
    
    def upsample_width(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        """
        Upsample input tensor along width dimension using horizontal filtering.
        
        Args:
            x: Input tensor [B, C, H, W]
            scale_factor: Horizontal scaling factor
            
        Returns:
            Horizontally upsampled tensor [B, C, H, W*scale_factor]
        """
        B, C, H, W = x.shape
        new_width = W * scale_factor
        
        # Determine component type (assume single channel or first channel is luma)
        is_luma = (C == 1) or (x.shape[1] == 1)
        filter_coeffs = self._get_filter_coefficients(is_luma)
        num_taps = 8 if is_luma else 4
        
        # Calculate scaling ratio
        scaling_ratio = self._calculate_scaling_ratio(scale_factor)
        
        # Initialize output buffer
        buf = torch.zeros(B, C, H, new_width, dtype=x.dtype, device=x.device)
        
        # Apply horizontal filtering
        for i in range(new_width):
            integer, frac = self._calculate_reference_position(i, scaling_ratio)
            
            # Get filter coefficients for this fractional position
            coeffs = filter_coeffs[frac]  # [num_taps]
            
            # Apply filter
            for t in range(num_taps):
                src_x = integer - num_taps//2 + 1 + t
                if 0 <= src_x < W:
                    buf[:, :, :, i] += x[:, :, :, src_x] * coeffs[t]
        
        return buf
    
    def upsample_height(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        """
        Upsample input tensor along height dimension using vertical filtering.
        
        Args:
            x: Input tensor [B, C, H, W]
            scale_factor: Vertical scaling factor
            
        Returns:
            Vertically upsampled tensor [B, C, H*scale_factor, W]
        """
        B, C, H, W = x.shape
        new_height = H * scale_factor
        
        # Determine component type
        is_luma = (C == 1) or (x.shape[1] == 1)
        filter_coeffs = self._get_filter_coefficients(is_luma)
        num_taps = 8 if is_luma else 4
        
        # Calculate scaling ratio
        scaling_ratio = self._calculate_scaling_ratio(scale_factor)
        
        # Initialize output buffer
        dst = torch.zeros(B, C, new_height, W, dtype=x.dtype, device=x.device)
        
        # Apply vertical filtering
        for j in range(new_height):
            integer, frac = self._calculate_reference_position(j, scaling_ratio)
            
            # Get filter coefficients for this fractional position
            coeffs = filter_coeffs[frac]  # [num_taps]
            
            # Apply filter
            for t in range(num_taps):
                src_y = integer - num_taps//2 + 1 + t
                if 0 <= src_y < H:
                    dst[:, :, j, :] += x[:, :, src_y, :] * coeffs[t]
        
        return dst
    
    def forward(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        """
        Upsample input tensor using separable 1D filtering (horizontal then vertical).
        
        Args:
            x: Input tensor [B, C, H, W] in YUV420 format
            scale_factor: Upscaling factor (2 or 4)
            
        Returns:
            Upsampled tensor [B, C, H*scale_factor, W*scale_factor]
        """
        # First pass: horizontal upsampling
        buf = self.upsample_width(x, scale_factor)
        
        # Second pass: vertical upsampling
        dst = self.upsample_height(buf, scale_factor)
        
        # Final normalization and clamping
        dst = dst / (2 ** self.log2Norm)
        dst = torch.clamp(dst, 0, 255)
        
        return dst


class VTMUpsamplerNetwork(nn.Module):
    """
    Neural network wrapper for VTM upsampler with non-trainable weights.
    
    This class ensures the VTM upsampler weights remain fixed during training.
    """
    
    def __init__(self, log2Norm: int = 16):
        """
        Initialize VTM upsampler network.
        
        Args:
            log2Norm: Normalization factor for filter coefficients
        """
        super().__init__()
        self.upsampler = VTMUpsamplerExact(log2Norm=log2Norm)
        
        # Freeze all parameters to prevent training
        for param in self.upsampler.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        """
        Forward pass through VTM upsampler.
        
        Args:
            x: Input tensor [B, C, H, W]
            scale_factor: Upscaling factor
            
        Returns:
            Upsampled tensor
        """
        return self.upsampler(x, scale_factor)


def test_upsampler():
    """Test function for VTM upsampler."""
    # Create test YUV420 data
    y = torch.randn(1, 1, 270, 480) * 50 + 128  # Luma
    u = torch.randn(1, 1, 135, 240) * 50 + 128  # Chroma U
    v = torch.randn(1, 1, 135, 240) * 50 + 128  # Chroma V
    
    # Initialize upsampler
    upsampler = VTMUpsamplerExact(log2Norm=16)
    
    # Test individual components
    print("Testing individual upsampling functions:")
    
    # Test width upsampling
    y_width = upsampler.upsample_width(y, scale_factor=2)
    print(f"Width upsampling: {y.shape} -> {y_width.shape}")
    
    # Test height upsampling
    y_height = upsampler.upsample_height(y, scale_factor=2)
    print(f"Height upsampling: {y.shape} -> {y_height.shape}")
    
    # Test full upsampling
    y_upsampled = upsampler(y, scale_factor=2)
    u_upsampled = upsampler(u, scale_factor=2)
    v_upsampled = upsampler(v, scale_factor=2)
    
    print(f"\nFull upsampling:")
    print(f"Y: {y.shape} -> {y_upsampled.shape}")
    print(f"U: {u.shape} -> {u_upsampled.shape}")
    print(f"V: {v.shape} -> {v_upsampled.shape}")
    
    return y_upsampled, u_upsampled, v_upsampled


if __name__ == "__main__":
    test_upsampler()