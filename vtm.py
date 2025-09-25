import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VTMUpsampler(nn.Module):
    """
    PyTorch implementation of VTM upsampler for neural network integration.
    Supports both standard VVC filters and alternative/ECM filters.
    """
    
    def __init__(self, filter_type='standard', use_luma_filter=True):
        super().__init__()
        self.filter_type = filter_type
        self.use_luma_filter = use_luma_filter
        
        # Define filter coefficients (scaled by 256 for integer arithmetic)
        if filter_type == 'standard':
            # Standard VVC 8-tap luma, 4-tap chroma filters
            self.luma_filter = self._get_standard_luma_filter()
            self.chroma_filter = self._get_standard_chroma_filter()
            self.filter_length = 8 if use_luma_filter else 4
            self.num_frac_positions = 15 if use_luma_filter else 31
            self.log2_norm = 12
        elif filter_type == 'alternative':
            # Alternative 12-tap luma, 6-tap chroma filters
            self.luma_filter = self._get_alternative_luma_filter()
            self.chroma_filter = self._get_alternative_chroma_filter()
            self.filter_length = 12 if use_luma_filter else 6
            self.num_frac_positions = 15 if use_luma_filter else 31
            self.log2_norm = 16
        elif filter_type == 'ecm':
            # ECM 12-tap luma, 6-tap chroma filters
            self.luma_filter = self._get_ecm_luma_filter()
            self.chroma_filter = self._get_ecm_chroma_filter()
            self.filter_length = 12 if use_luma_filter else 6
            self.num_frac_positions = 15 if use_luma_filter else 31
            self.log2_norm = 16
        
        # Convert to PyTorch tensors
        self.register_buffer('luma_filter_tensor', torch.tensor(self.luma_filter, dtype=torch.float32))
        self.register_buffer('chroma_filter_tensor', torch.tensor(self.chroma_filter, dtype=torch.float32))
    
    def _get_standard_luma_filter(self):
        """Standard VVC 8-tap luma filter coefficients from VTM"""
        return np.array([
            [0, 0, 0, 64, 0, 0, 0, 0],
            [0, 1, -3, 63, 4, -2, 1, 0],
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
            [0, 1, -2, 4, 63, -3, 1, 0]
        ], dtype=np.float32) / 256.0  # Normalize by 256
    
    def _get_standard_chroma_filter(self):
        """Standard VVC 4-tap chroma filter coefficients from VTM"""
        return np.array([
            [0, 64, 0, 0],
            [-1, 63, 2, 0],
            [-2, 62, 4, 0],
            [-2, 60, 7, -1],
            [-2, 58, 10, -2],
            [-3, 57, 12, -2],
            [-4, 56, 14, -2],
            [-4, 55, 15, -2],
            [-4, 54, 16, -2],
            [-5, 53, 18, -2],
            [-6, 52, 20, -2],
            [-6, 49, 24, -3],
            [-6, 46, 28, -4],
            [-5, 44, 29, -4],
            [-4, 42, 30, -4],
            [-4, 39, 33, -4],
            [-4, 36, 36, -4],
            [-4, 33, 39, -4],
            [-4, 30, 42, -4],
            [-4, 29, 44, -5],
            [-4, 28, 46, -6],
            [-3, 24, 49, -6],
            [-2, 20, 52, -6],
            [-2, 18, 53, -5],
            [-2, 16, 54, -4],
            [-2, 15, 55, -4],
            [-2, 14, 56, -4],
            [-2, 12, 57, -3],
            [-2, 10, 58, -2],
            [-1, 7, 60, -2],
            [0, 4, 62, -2],
            [0, 2, 63, -1]
        ], dtype=np.float32) / 256.0
    
    def _get_alternative_luma_filter(self):
        """Alternative 12-tap luma filter coefficients from VTM"""
        return np.array([
            [0, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 3, -12, 253, 16, -6, 2, 0, 0, 0],
            [0, 0, -3, 9, -24, 250, 32, -11, 4, -1, 0, 0],
            [0, 0, -4, 12, -32, 241, 52, -18, 8, -4, 2, -1],
            [0, 1, -6, 15, -38, 228, 75, -28, 14, -7, 3, -1],
            [0, 1, -7, 18, -43, 214, 96, -33, 16, -8, 3, -1],
            [1, 0, -6, 17, -44, 196, 119, -40, 20, -10, 4, -1],
            [0, 2, -9, 21, -47, 180, 139, -43, 20, -10, 4, -1],
            [-1, 3, -9, 21, -46, 160, 160, -46, 21, -9, 3, -1],
            [-1, 4, -10, 20, -43, 139, 180, -47, 21, -9, 2, 0],
            [-1, 4, -10, 20, -40, 119, 196, -44, 17, -6, 0, 1],
            [-1, 3, -8, 16, -33, 96, 214, -43, 18, -7, 1, 0],
            [-1, 3, -7, 14, -28, 75, 228, -38, 15, -6, 1, 0],
            [-1, 2, -4, 8, -18, 52, 241, -32, 12, -4, 0, 0],
            [0, 0, -1, 4, -11, 32, 250, -24, 9, -3, 0, 0],
            [0, 0, 0, 2, -6, 16, 253, -12, 3, 0, -1, 1]
        ], dtype=np.float32) / 256.0
    
    def _get_alternative_chroma_filter(self):
        """Alternative 6-tap chroma filter coefficients from VTM"""
        return np.array([
            [0, 0, 256, 0, 0, 0],
            [1, -6, 256, 6, -1, 0],
            [2, -11, 254, 14, -4, 1],
            [4, -18, 252, 23, -6, 1],
            [6, -24, 249, 32, -9, 2],
            [6, -26, 244, 41, -12, 3],
            [7, -30, 239, 53, -18, 5],
            [8, -34, 235, 61, -19, 5],
            [10, -38, 228, 72, -22, 6],
            [10, -39, 220, 84, -26, 7],
            [10, -40, 213, 94, -29, 8],
            [11, -42, 205, 105, -32, 9],
            [11, -42, 196, 116, -35, 10],
            [11, -42, 186, 128, -37, 10],
            [11, -42, 177, 138, -38, 10],
            [11, -41, 167, 148, -40, 11],
            [11, -41, 158, 158, -41, 11],
            [11, -40, 148, 167, -41, 11],
            [10, -38, 138, 177, -42, 11],
            [10, -37, 128, 186, -42, 11],
            [10, -35, 116, 196, -42, 11],
            [9, -32, 105, 205, -42, 11],
            [8, -29, 94, 213, -40, 10],
            [7, -26, 84, 220, -39, 10],
            [6, -22, 72, 228, -38, 10],
            [5, -19, 61, 235, -34, 8],
            [5, -18, 53, 239, -30, 7],
            [3, -12, 41, 244, -26, 6],
            [2, -9, 32, 249, -24, 6],
            [1, -6, 23, 252, -18, 4],
            [1, -4, 14, 254, -11, 2],
            [0, -1, 6, 256, -6, 1]
        ], dtype=np.float32) / 256.0
    
    def _get_ecm_luma_filter(self):
        """ECM 12-tap luma filter coefficients from VTM"""
        # Note: You would need to copy the actual m_lumaFilter12 coefficients from VTM
        # For now, using alternative filter as placeholder
        return self._get_alternative_luma_filter()
    
    def _get_ecm_chroma_filter(self):
        """ECM 6-tap chroma filter coefficients from VTM"""
        # Note: You would need to copy the actual m_chromaFilter6 coefficients from VTM
        # For now, using alternative filter as placeholder
        return self._get_alternative_chroma_filter()
    
    def forward(self, x, scale_factor=2.0):
        """
        Forward pass for upsampling
        
        Args:
            x: Input tensor [B, C, H, W]
            scale_factor: Upsampling factor (e.g., 2.0 for 2x upsampling)
        
        Returns:
            Upsampled tensor [B, C, H*scale_factor, W*scale_factor]
        """
        B, C, H, W = x.shape
        scale_factor = int(scale_factor) if scale_factor == int(scale_factor) else scale_factor
        
        if scale_factor == 1.0:
            return x
        
        # Use the appropriate filter
        if self.use_luma_filter:
            filter_tensor = self.luma_filter_tensor
        else:
            filter_tensor = self.chroma_filter_tensor
        
        # Convert to separable 2D filtering
        # Horizontal pass
        x_h = self._horizontal_filter(x, filter_tensor, scale_factor)
        
        # Vertical pass
        x_v = self._vertical_filter(x_h, filter_tensor, scale_factor)
        
        return x_v
    
    def _horizontal_filter(self, x, filter_tensor, scale_factor):
        """Horizontal filtering pass"""
        B, C, H, W = x.shape
        new_W = int(W * scale_factor)
        
        # Create output tensor
        x_h = torch.zeros(B, C, H, new_W, device=x.device, dtype=x.dtype)
        
        for i in range(new_W):
            # Calculate reference position (fixed-point arithmetic)
            ref_pos = int((i * 16384) / scale_factor)  # 14-bit fixed point
            integer = ref_pos >> 4  # numFracShift = 4 for luma
            frac = ref_pos & 15     # numFracPositions = 15 for luma
            
            # Get filter coefficients
            f = filter_tensor[frac]  # [filter_length]
            
            # Apply horizontal filter
            for k in range(self.filter_length):
                x_int = max(0, min(W-1, integer + k - self.filter_length//2 + 1))
                x_h[:, :, :, i] += f[k] * x[:, :, :, x_int]
        
        return x_h
    
    def _vertical_filter(self, x, filter_tensor, scale_factor):
        """Vertical filtering pass"""
        B, C, H, W = x.shape
        new_H = int(H * scale_factor)
        
        # Create output tensor
        x_v = torch.zeros(B, C, new_H, W, device=x.device, dtype=x.dtype)
        
        for j in range(new_H):
            # Calculate reference position
            ref_pos = int((j * 16384) / scale_factor)
            integer = ref_pos >> 4
            frac = ref_pos & 15
            
            # Get filter coefficients
            f = filter_tensor[frac]
            
            # Apply vertical filter
            for k in range(self.filter_length):
                y_int = max(0, min(H-1, integer + k - self.filter_length//2 + 1))
                x_v[:, :, j, :] += f[k] * x[:, :, y_int, :]
        
        # Apply normalization (equivalent to >> log2Norm)
        x_v = x_v / (2 ** self.log2_norm)
        
        return x_v

# Optimized version using PyTorch operations
class VTMUpsamplerOptimized(nn.Module):
    """
    Optimized PyTorch implementation using built-in operations
    """
    
    def __init__(self, filter_type='alternative', use_luma_filter=True):
        super().__init__()
        self.filter_type = filter_type
        self.use_luma_filter = use_luma_filter
        
        # Get filter coefficients
        if filter_type == 'alternative':
            if use_luma_filter:
                self.filter_coeffs = self._get_alternative_luma_filter()
                self.filter_length = 12
            else:
                self.filter_coeffs = self._get_alternative_chroma_filter()
                self.filter_length = 6
        else:
            if use_luma_filter:
                self.filter_coeffs = self._get_standard_luma_filter()
                self.filter_length = 8
            else:
                self.filter_coeffs = self._get_standard_chroma_filter()
                self.filter_length = 4
        
        self.log2_norm = 16 if filter_type == 'alternative' else 12
        self.num_frac_positions = 15 if use_luma_filter else 31
        
        # Convert to PyTorch tensors
        self.register_buffer('filter_tensor', torch.tensor(self.filter_coeffs, dtype=torch.float32))
    
    def _get_alternative_luma_filter(self):
        """Alternative 12-tap luma filter coefficients"""
        return np.array([
            [0, 0, 0, 0, 0, 256, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 3, -12, 253, 16, -6, 2, 0, 0, 0],
            [0, 0, -3, 9, -24, 250, 32, -11, 4, -1, 0, 0],
            [0, 0, -4, 12, -32, 241, 52, -18, 8, -4, 2, -1],
            [0, 1, -6, 15, -38, 228, 75, -28, 14, -7, 3, -1],
            [0, 1, -7, 18, -43, 214, 96, -33, 16, -8, 3, -1],
            [1, 0, -6, 17, -44, 196, 119, -40, 20, -10, 4, -1],
            [0, 2, -9, 21, -47, 180, 139, -43, 20, -10, 4, -1],
            [-1, 3, -9, 21, -46, 160, 160, -46, 21, -9, 3, -1],
            [-1, 4, -10, 20, -43, 139, 180, -47, 21, -9, 2, 0],
            [-1, 4, -10, 20, -40, 119, 196, -44, 17, -6, 0, 1],
            [-1, 3, -8, 16, -33, 96, 214, -43, 18, -7, 1, 0],
            [-1, 3, -7, 14, -28, 75, 228, -38, 15, -6, 1, 0],
            [-1, 2, -4, 8, -18, 52, 241, -32, 12, -4, 0, 0],
            [0, 0, -1, 4, -11, 32, 250, -24, 9, -3, 0, 0],
            [0, 0, 0, 2, -6, 16, 253, -12, 3, 0, -1, 1]
        ], dtype=np.float32) / 256.0
    
    def _get_standard_luma_filter(self):
        """Standard VVC 8-tap luma filter coefficients"""
        return np.array([
            [0, 0, 0, 64, 0, 0, 0, 0],
            [0, 1, -3, 63, 4, -2, 1, 0],
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
            [0, 1, -2, 4, 63, -3, 1, 0]
        ], dtype=np.float32) / 256.0
    
    def _get_standard_chroma_filter(self):
        """Standard VVC 4-tap chroma filter coefficients"""
        return np.array([
            [0, 64, 0, 0],
            [-1, 63, 2, 0],
            [-2, 62, 4, 0],
            [-2, 60, 7, -1],
            [-2, 58, 10, -2],
            [-3, 57, 12, -2],
            [-4, 56, 14, -2],
            [-4, 55, 15, -2],
            [-4, 54, 16, -2],
            [-5, 53, 18, -2],
            [-6, 52, 20, -2],
            [-6, 49, 24, -3],
            [-6, 46, 28, -4],
            [-5, 44, 29, -4],
            [-4, 42, 30, -4],
            [-4, 39, 33, -4],
            [-4, 36, 36, -4],
            [-4, 33, 39, -4],
            [-4, 30, 42, -4],
            [-4, 29, 44, -5],
            [-4, 28, 46, -6],
            [-3, 24, 49, -6],
            [-2, 20, 52, -6],
            [-2, 18, 53, -5],
            [-2, 16, 54, -4],
            [-2, 15, 55, -4],
            [-2, 14, 56, -4],
            [-2, 12, 57, -3],
            [-2, 10, 58, -2],
            [-1, 7, 60, -2],
            [0, 4, 62, -2],
            [0, 2, 63, -1]
        ], dtype=np.float32) / 256.0
    
    def forward(self, x, scale_factor=2.0):
        """
        Optimized forward pass using PyTorch operations
        """
        B, C, H, W = x.shape
        
        if scale_factor == 1.0:
            return x
        
        # Use F.interpolate with custom filter
        # This is a simplified version - for exact VTM behavior, use the loop-based version
        return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)

# Usage example
def create_vtm_upsampler(filter_type='alternative', use_luma_filter=True):
    """
    Factory function to create VTM upsampler
    
    Args:
        filter_type: 'standard', 'alternative', or 'ecm'
        use_luma_filter: True for luma, False for chroma
    
    Returns:
        VTMUpsampler instance
    """
    return VTMUpsampler(filter_type=filter_type, use_luma_filter=use_luma_filter)

# Example usage in a neural network
class VTMUpsamplingNetwork(nn.Module):
    def __init__(self, input_channels=3, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Create VTM upsamplers for each component
        self.luma_upsampler = VTMUpsampler(filter_type='alternative', use_luma_filter=True)
        self.chroma_upsampler = VTMUpsampler(filter_type='alternative', use_luma_filter=False)
        
        # Optional: Add learnable components
        self.conv = nn.Conv2d(input_channels, input_channels, 3, padding=1)
    
    def forward(self, x):
        # Apply VTM upsampling
        x_upsampled = self.luma_upsampler(x, self.scale_factor)
        
        # Optional: Add learnable processing
        x_processed = self.conv(x_upsampled)
        
        return x_processed

# Test the implementation
if __name__ == "__main__":
    # Create test input
    x = torch.randn(1, 3, 64, 64)
    
    # Create VTM upsampler
    upsampler = VTMUpsampler(filter_type='alternative', use_luma_filter=True)
    
    # Test upsampling
    y = upsampler(x, scale_factor=2.0)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")