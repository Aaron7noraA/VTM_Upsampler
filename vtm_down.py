import torch
import torch.nn as nn
import numpy as np

class VTMDownsampler(nn.Module):
    """
    VTM RPR downsampler implementation.
    Uses 12-tap SRC (Sample Rate Conversion) filters for downsampling.
    """
    
    def __init__(self):
        super().__init__()
        
        # VTM's downsampling filter coefficients (12-tap filters)
        # 8 different filter sets (D=1 to D=7) based on scaling ratio
        downsampling_filters = [
            # D = 1 (scaling ratio ~1.0)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, -6, 127, 7, -2, 0, 0, 0, 0],
                [0, 0, 0, 3, -12, 125, 16, -5, 1, 0, 0, 0],
                [0, 0, 0, 4, -16, 120, 26, -7, 1, 0, 0, 0],
                [0, 0, 0, 5, -18, 114, 36, -10, 1, 0, 0, 0],
                [0, 0, 0, 5, -20, 107, 46, -12, 2, 0, 0, 0],
                [0, 0, 0, 5, -21, 99, 57, -15, 3, 0, 0, 0],
                [0, 0, 0, 5, -20, 89, 68, -18, 4, 0, 0, 0],
                [0, 0, 0, 4, -19, 79, 79, -19, 4, 0, 0, 0],
                [0, 0, 0, 4, -18, 68, 89, -20, 5, 0, 0, 0],
                [0, 0, 0, 3, -15, 57, 99, -21, 5, 0, 0, 0],
                [0, 0, 0, 2, -12, 46, 107, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -10, 36, 114, -18, 5, 0, 0, 0],
                [0, 0, 0, 1, -7, 26, 120, -16, 4, 0, 0, 0],
                [0, 0, 0, 1, -5, 16, 125, -12, 3, 0, 0, 0],
                [0, 0, 0, 0, -2, 7, 127, -6, 2, 0, 0, 0]
            ],
            # D = 2 (scaling ratio ~1.1)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -4, 127, 6, -2, 0, 0, 0, 0],
                [0, 0, 0, 2, -8, 125, 12, -4, 1, 0, 0, 0],
                [0, 0, 0, 3, -12, 122, 18, -6, 1, 0, 0, 0],
                [0, 0, 0, 4, -16, 118, 24, -8, 2, 0, 0, 0],
                [0, 0, 0, 4, -18, 113, 31, -10, 2, 0, 0, 0],
                [0, 0, 0, 4, -19, 107, 38, -12, 3, 0, 0, 0],
                [0, 0, 0, 4, -18, 100, 46, -14, 4, 0, 0, 0],
                [0, 0, 0, 3, -17, 92, 54, -16, 4, 0, 0, 0],
                [0, 0, 0, 3, -15, 83, 63, -18, 5, 0, 0, 0],
                [0, 0, 0, 2, -13, 73, 73, -20, 5, 0, 0, 0],
                [0, 0, 0, 2, -11, 63, 83, -18, 4, 0, 0, 0],
                [0, 0, 0, 1, -9, 54, 92, -17, 4, 0, 0, 0],
                [0, 0, 0, 1, -7, 46, 100, -18, 4, 0, 0, 0],
                [0, 0, 0, 1, -5, 38, 107, -19, 4, 0, 0, 0],
                [0, 0, 0, 0, -3, 31, 113, -18, 4, 0, 0, 0]
            ],
            # D = 3 (scaling ratio ~1.2)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -3, 127, 5, -2, 0, 0, 0, 0],
                [0, 0, 0, 2, -6, 125, 10, -4, 1, 0, 0, 0],
                [0, 0, 0, 2, -9, 122, 15, -6, 2, 0, 0, 0],
                [0, 0, 0, 3, -12, 118, 20, -8, 2, 0, 0, 0],
                [0, 0, 0, 3, -14, 113, 26, -10, 3, 0, 0, 0],
                [0, 0, 0, 3, -15, 107, 32, -12, 3, 0, 0, 0],
                [0, 0, 0, 3, -15, 100, 38, -14, 4, 0, 0, 0],
                [0, 0, 0, 2, -14, 92, 45, -16, 4, 0, 0, 0],
                [0, 0, 0, 2, -12, 83, 52, -18, 5, 0, 0, 0],
                [0, 0, 0, 2, -10, 73, 60, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -8, 63, 68, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -6, 54, 76, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -4, 45, 84, -18, 4, 0, 0, 0],
                [0, 0, 0, 0, -2, 37, 92, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, -1, 29, 100, -14, 3, 0, 0, 0]
            ],
            # D = 4 (scaling ratio ~1.3)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, -2, 127, 4, -2, 0, 0, 0, 0],
                [0, 0, 0, 1, -5, 125, 8, -4, 1, 0, 0, 0],
                [0, 0, 0, 2, -7, 122, 12, -6, 2, 0, 0, 0],
                [0, 0, 0, 2, -9, 118, 16, -8, 2, 0, 0, 0],
                [0, 0, 0, 2, -11, 113, 21, -10, 3, 0, 0, 0],
                [0, 0, 0, 2, -12, 107, 26, -12, 3, 0, 0, 0],
                [0, 0, 0, 2, -12, 100, 32, -14, 4, 0, 0, 0],
                [0, 0, 0, 2, -11, 92, 38, -16, 4, 0, 0, 0],
                [0, 0, 0, 1, -10, 83, 45, -18, 5, 0, 0, 0],
                [0, 0, 0, 1, -8, 73, 52, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -6, 63, 60, -20, 5, 0, 0, 0],
                [0, 0, 0, 1, -4, 54, 68, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -2, 45, 76, -18, 4, 0, 0, 0],
                [0, 0, 0, 0, -1, 37, 84, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 29, 92, -14, 3, 0, 0, 0]
            ],
            # D = 5 (scaling ratio ~1.4)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -2, 127, 3, -2, 0, 0, 0, 0],
                [0, 0, 0, 1, -4, 125, 6, -4, 1, 0, 0, 0],
                [0, 0, 0, 1, -6, 122, 9, -6, 2, 0, 0, 0],
                [0, 0, 0, 1, -8, 118, 12, -8, 2, 0, 0, 0],
                [0, 0, 0, 1, -9, 113, 16, -10, 3, 0, 0, 0],
                [0, 0, 0, 1, -10, 107, 20, -12, 3, 0, 0, 0],
                [0, 0, 0, 1, -10, 100, 25, -14, 4, 0, 0, 0],
                [0, 0, 0, 1, -9, 92, 30, -16, 4, 0, 0, 0],
                [0, 0, 0, 1, -8, 83, 36, -18, 5, 0, 0, 0],
                [0, 0, 0, 1, -6, 73, 42, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -4, 63, 48, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -2, 54, 54, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -1, 45, 60, -18, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 37, 66, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 29, 72, -14, 3, 0, 0, 0]
            ],
            # D = 6 (scaling ratio ~1.5)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 127, 2, -2, 0, 0, 0, 0],
                [0, 0, 0, 0, -3, 125, 4, -4, 1, 0, 0, 0],
                [0, 0, 0, 0, -4, 122, 6, -6, 2, 0, 0, 0],
                [0, 0, 0, 0, -5, 118, 8, -8, 2, 0, 0, 0],
                [0, 0, 0, 0, -6, 113, 11, -10, 3, 0, 0, 0],
                [0, 0, 0, 0, -6, 107, 14, -12, 3, 0, 0, 0],
                [0, 0, 0, 0, -6, 100, 18, -14, 4, 0, 0, 0],
                [0, 0, 0, 0, -5, 92, 22, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, -4, 83, 27, -18, 5, 0, 0, 0],
                [0, 0, 0, 0, -3, 73, 32, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -2, 63, 37, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, -1, 54, 42, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 45, 47, -18, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 37, 52, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 29, 57, -14, 3, 0, 0, 0]
            ],
            # D = 7 (scaling ratio ~1.6)
            [
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 127, 1, -2, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 125, 2, -4, 1, 0, 0, 0],
                [0, 0, 0, 0, -2, 122, 3, -6, 2, 0, 0, 0],
                [0, 0, 0, 0, -3, 118, 4, -8, 2, 0, 0, 0],
                [0, 0, 0, 0, -3, 113, 6, -10, 3, 0, 0, 0],
                [0, 0, 0, 0, -3, 107, 8, -12, 3, 0, 0, 0],
                [0, 0, 0, 0, -3, 100, 11, -14, 4, 0, 0, 0],
                [0, 0, 0, 0, -2, 92, 14, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, -1, 83, 18, -18, 5, 0, 0, 0],
                [0, 0, 0, 0, -1, 73, 22, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 63, 26, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 54, 30, -20, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 45, 34, -18, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 37, 38, -16, 4, 0, 0, 0],
                [0, 0, 0, 0, 0, 29, 42, -14, 3, 0, 0, 0]
            ]
        ]
        
        # Register filters as non-trainable buffers
        self.register_buffer('downsampling_filters', torch.tensor(downsampling_filters, dtype=torch.float32))
        
        # VTM constants
        self.SCALING_RATIO_BITS = 14
        self.FILTER_LENGTH = 12
        self.LOG2_NORM = 14  # For downsampling
        
    def _select_filter(self, scaling_ratio):
        """
        Select downsampling filter based on scaling ratio (VTM logic).
        """
        if scaling_ratio > (15 << self.SCALING_RATIO_BITS) / 4:
            return 7
        elif scaling_ratio > (20 << self.SCALING_RATIO_BITS) / 7:
            return 6
        elif scaling_ratio > (5 << self.SCALING_RATIO_BITS) / 2:
            return 5
        elif scaling_ratio > (2 << self.SCALING_RATIO_BITS):
            return 4
        elif scaling_ratio > (5 << self.SCALING_RATIO_BITS) / 3:
            return 3
        elif scaling_ratio > (27 << self.SCALING_RATIO_BITS) / 20:
            return 2
        elif scaling_ratio > (11 << self.SCALING_RATIO_BITS) / 10:
            return 1
        else:
            return 0
    
    @torch.no_grad()
    def forward(self, x, scale_factor):
        """
        Downsample input tensor using VTM's downsampling algorithm.
        
        Args:
            x: Input tensor [B, C, H, W]
            scale_factor: Downsampling factor (e.g., 2.0 for 2x downsampling)
        Returns:
            Downsampled tensor [B, C, H//scale_factor, W//scale_factor]
        """
        B, C, H, W = x.shape
        newH = int(H / scale_factor)
        newW = int(W / scale_factor)
        
        # Calculate scaling ratio (14-bit fixed point)
        scaling_ratio = int(scale_factor * (1 << self.SCALING_RATIO_BITS))
        
        # Select filter based on scaling ratio
        filter_idx = self._select_filter(scaling_ratio)
        filters = self.downsampling_filters[filter_idx]  # [16, 12]
        
        # Horizontal pass: [B, C, H, newW]
        buf = x.new_zeros((B, C, H, newW), dtype=torch.float32)
        
        for i in range(newW):
            # Calculate reference position (14-bit fixed point)
            ref_pos_fixed = int(i * (1 << self.SCALING_RATIO_BITS) / scale_factor)
            ref_pos_phase = ref_pos_fixed >> (self.SCALING_RATIO_BITS - 4)  # 4 fractional bits
            integer = ref_pos_phase >> 4
            frac = ref_pos_phase & 15
            
            # Get filter coefficients
            coeffs = filters[frac]  # [12]
            
            # Apply horizontal filter
            acc = x.new_zeros((B, C, H), dtype=torch.float32)
            for k in range(self.FILTER_LENGTH):
                x_int = max(0, min(W - 1, integer + k - self.FILTER_LENGTH // 2 + 1))
                acc += coeffs[k] * x[:, :, :, x_int]
            buf[:, :, :, i] = acc
        
        # Vertical pass: [B, C, newH, newW]
        out = x.new_zeros((B, C, newH, newW), dtype=torch.float32)
        
        for j in range(newH):
            # Calculate reference position (14-bit fixed point)
            ref_pos_fixed = int(j * (1 << self.SCALING_RATIO_BITS) / scale_factor)
            ref_pos_phase = ref_pos_fixed >> (self.SCALING_RATIO_BITS - 4)  # 4 fractional bits
            integer = ref_pos_phase >> 4
            frac = ref_pos_phase & 15
            
            # Get filter coefficients
            coeffs = filters[frac]  # [12]
            
            # Apply vertical filter
            acc = x.new_zeros((B, C, newW), dtype=torch.float32)
            for k in range(self.FILTER_LENGTH):
                y_int = max(0, min(H - 1, integer + k - self.FILTER_LENGTH // 2 + 1))
                acc += coeffs[k] * buf[:, :, y_int, :]
            
            # Final normalization and clamping (VTM style)
            rounding_offset = 1 << (self.LOG2_NORM - 1)
            result = (acc + rounding_offset) >> self.LOG2_NORM
            out[:, :, j, :] = torch.clamp(result, 0, 255)
        
        return out

# Usage example
if __name__ == "__main__":
    # Test downsampler
    downsampler = VTMDownsampler()
    
    # Create test input
    x = torch.randn(1, 1, 1080, 1920) * 50 + 128  # High-res input
    
    # Downsample by 2x
    y = downsampler(x, scale_factor=2.0)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    
    # Test different scaling ratios
    for scale in [1.5, 2.0, 2.5, 3.0]:
        y = downsampler(x, scale_factor=scale)
        print(f"Scale {scale}: {x.shape} -> {y.shape}")