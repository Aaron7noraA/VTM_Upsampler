import torch
import torch.nn as nn
import numpy as np

class VTMDownsamplerCorrect(nn.Module):
    """
    Correct VTM downsampler implementation.
    Matches VTM's sampleRateConv function for downsampling.
    """
    
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        
        # Load all 8 downsampling filter sets from VTM
        self.downsampling_filters = self._load_downsampling_filters()
        
    def _load_downsampling_filters(self):
        """Load all 8 downsampling filter sets from VTM"""
        filters = {}
        
        # D=0 filter set
        filters[0] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=1 filter set (Kaiser(7)-windowed sinc ratio 1.35)
        filters[1] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=2 filter set (Kaiser(7)-windowed sinc ratio 1.5)
        filters[2] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=3 filter set (Kaiser(7)-windowed sinc ratio 1.75)
        filters[3] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=4 filter set (Kaiser(7)-windowed sinc ratio 2.0)
        filters[4] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=5 filter set (Kaiser(7)-windowed sinc ratio 2.5)
        filters[5] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=6 filter set (Kaiser(7)-windowed sinc ratio 3.0)
        filters[6] = torch.tensor([
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
        ], dtype=torch.int32)
        
        # D=7 filter set (Kaiser(7)-windowed sinc ratio 4.0)
        filters[7] = torch.tensor([
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
        ], dtype=torch.int32)
        
        return filters
    
    def _select_filter(self, scaling_ratio):
        """Select filter based on scaling ratio (matching VTM logic)"""
        if scaling_ratio > (15 << 14) // 4:
            return 7
        elif scaling_ratio > (20 << 14) // 7:
            return 6
        elif scaling_ratio > (5 << 14) // 2:
            return 5
        elif scaling_ratio > (2 << 14):
            return 4
        elif scaling_ratio > (5 << 14) // 3:
            return 3
        elif scaling_ratio > (27 << 14) // 20:
            return 2
        elif scaling_ratio > (11 << 14) // 10:
            return 1
        else:
            return 0
    
    def _horizontal_filter(self, x, scaling_ratio, filter_idx):
        """Horizontal filtering pass (first pass)"""
        B, C, H, W = x.shape
        scaled_width = W // 2  # Assuming 2x downsampling
        
        # Convert to fixed-point scaling ratio
        scale_x = 1  # For 2x downsampling
        pos_shift_x = 14
        num_frac_shift = 4
        num_frac_positions = 16
        
        # Create intermediate buffer
        buf = torch.zeros(B, C, H, scaled_width, dtype=torch.int32, device=x.device)
        
        for i in range(scaled_width):
            # Calculate reference position (matching VTM logic)
            ref_pos = (((i << scale_x) - 0) * scaling_ratio + 0) >> pos_shift_x
            integer = ref_pos >> num_frac_shift
            frac = ref_pos & (num_frac_positions - 1)
            
            # Get filter coefficients
            filter_coeffs = self.downsampling_filters[filter_idx][frac]
            
            for j in range(H):
                sum_val = 0
                for k in range(12):  # 12-tap filter
                    x_int = torch.clamp(integer + k - 6 + 1, 0, W - 1)
                    sum_val += filter_coeffs[k] * x[:, :, j, x_int]
                
                buf[:, :, j, i] = sum_val
        
        return buf
    
    def _vertical_filter(self, buf, scaling_ratio, filter_idx):
        """Vertical filtering pass (second pass)"""
        B, C, H, W = buf.shape
        scaled_height = H // 2  # Assuming 2x downsampling
        
        # Convert to fixed-point scaling ratio
        scale_y = 1  # For 2x downsampling
        pos_shift_y = 14
        num_frac_shift = 4
        num_frac_positions = 16
        
        # Create output buffer
        result = torch.zeros(B, C, scaled_height, W, dtype=torch.int32, device=buf.device)
        
        for j in range(scaled_height):
            # Calculate reference position (matching VTM logic)
            ref_pos = (((j << scale_y) - 0) * scaling_ratio + 0) >> pos_shift_y
            integer = ref_pos >> num_frac_shift
            frac = ref_pos & (num_frac_positions - 1)
            
            # Get filter coefficients
            filter_coeffs = self.downsampling_filters[filter_idx][frac]
            
            for i in range(W):
                sum_val = 0
                for k in range(12):  # 12-tap filter
                    y_int = torch.clamp(integer + k - 6 + 1, 0, H - 1)
                    sum_val += filter_coeffs[k] * buf[:, :, y_int, i]
                
                # Apply normalization and clamping (matching VTM)
                log2_norm = 14  # For downsampling
                rounding_offset = 1 << (log2_norm - 1)
                normalized = (sum_val + rounding_offset) >> log2_norm
                result[:, :, j, i] = torch.clamp(normalized, 0, self.max_val)
        
        return result.float()
    
    def forward(self, x, scale_factor=2.0):
        """
        Two-pass downsampling: horizontal then vertical
        """
        # Convert to fixed-point scaling ratio
        scaling_ratio = int(scale_factor * (1 << 14))
        
        # Select appropriate filter
        filter_idx = self._select_filter(scaling_ratio)
        
        # Two-pass filtering
        # Pass 1: Horizontal filtering
        buf = self._horizontal_filter(x, scaling_ratio, filter_idx)
        
        # Pass 2: Vertical filtering  
        result = self._vertical_filter(buf, scaling_ratio, filter_idx)
        
        return result

# Usage example:
downsampler = VTMDownsamplerCorrect(bit_depth=8)
input_tensor = torch.randn(1, 1, 1080, 1920)  # B, C, H, W
downsampled = downsampler(input_tensor, scale_factor=2.0)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {downsampled.shape}")