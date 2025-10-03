import torch
import torch.nn as nn

class VTMDownsampler(nn.Module):
    """
    Efficient VTM downsampler using the same approach as the fast upsampler.
    Matches VTM's sampleRateConv function for downsampling.
    """
    
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth
        self.max_val = (1 << bit_depth) - 1
        
        # Load all 8 downsampling filter sets from VTM
        self._load_filters()
        
    def _load_filters(self):
        """Load all 8 downsampling filter sets from VTM"""
        # D=0 filter set (scaling ratio ~1.0)
        taps_d0 = [
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
        ]
        
        # D=1 filter set (Kaiser(7)-windowed sinc ratio 1.35)
        taps_d1 = [
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
        ]
        
        # D=2 filter set (Kaiser(7)-windowed sinc ratio 1.5)
        taps_d2 = [
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
        ]
        
        # D=3 filter set (Kaiser(7)-windowed sinc ratio 1.75)
        taps_d3 = [
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
        ]
        
        # D=4 filter set (Kaiser(7)-windowed sinc ratio 2.0)
        taps_d4 = [
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
        ]
        
        # D=5 filter set (Kaiser(7)-windowed sinc ratio 2.5)
        taps_d5 = [
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
        ]
        
        # D=6 filter set (Kaiser(7)-windowed sinc ratio 3.0)
        taps_d6 = [
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
        ]
        
        # D=7 filter set (Kaiser(7)-windowed sinc ratio 4.0)
        taps_d7 = [
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
        ]
        
        # Store all filter sets
        self.filter_sets = {
            0: torch.tensor(taps_d0, dtype=torch.float32),
            1: torch.tensor(taps_d1, dtype=torch.float32),
            2: torch.tensor(taps_d2, dtype=torch.float32),
            3: torch.tensor(taps_d3, dtype=torch.float32),
            4: torch.tensor(taps_d4, dtype=torch.float32),
            5: torch.tensor(taps_d5, dtype=torch.float32),
            6: torch.tensor(taps_d6, dtype=torch.float32),
            7: torch.tensor(taps_d7, dtype=torch.float32)
        }
        
        # Filter parameters
        self.numFracShift = 4  # 16 fractional phases
        self.numFracMask = 15
        self.filterLength = 12
        self.log2Norm = 14  # For downsampling
        
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
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, scale: float):
        """
        Efficient two-pass downsampling using the same approach as the fast upsampler.
        
        Args:
            x: [B,C,H,W] float tensor
            scale: < 1.0 (e.g., 0.5 for 2x downsampling)
        Returns:
            [B,C,round(H*scale), round(W*scale)]
        """
        B, C, H, W = x.shape
        newW = int(round(W * scale))
        newH = int(round(H * scale))
        
        # Convert to fixed-point scaling ratio
        scaling_ratio = int(scale * (1 << 14))
        
        # Select appropriate filter
        filter_idx = self._select_filter(scaling_ratio)
        taps = self.filter_sets[filter_idx]
        
        # Horizontal pass: [B,C,H,newW]
        buf = x.new_zeros((B, C, H, newW), dtype=torch.float32)
        center = self.filterLength // 2 - 1
        
        for i in range(newW):
            # Map output column i -> source pos (14-bit fixed point)
            refPosFixed = int(i * (1 << 14) / scale)
            refPosPhase = refPosFixed >> (14 - self.numFracShift)
            integer = refPosPhase >> self.numFracShift
            frac = refPosPhase & self.numFracMask
            f = taps[frac]  # [filterLength]
            
            acc = x.new_zeros((B, C, H), dtype=torch.float32)
            for k in range(self.filterLength):
                xInt = max(0, min(W - 1, integer + k - center))
                acc += f[k] * x[..., xInt]  # Vectorized across B,C,H
            buf[..., i] = acc
        
        # Vertical pass + final normalization: [B,C,newH,newW]
        out = x.new_zeros((B, C, newH, newW), dtype=torch.float32)
        rnd = float(1 << (self.log2Norm - 1))
        
        for j in range(newH):
            refPosFixed = int(j * (1 << 14) / scale)
            refPosPhase = refPosFixed >> (14 - self.numFracShift)
            integer = refPosPhase >> self.numFracShift
            frac = refPosPhase & self.numFracMask
            f = taps[frac]
            
            acc = x.new_zeros((B, C, newW), dtype=torch.float32)
            for k in range(self.filterLength):
                yInt = max(0, min(H - 1, integer + k - center))
                acc += f[k] * buf[:, :, yInt, :]  # Vectorized across B,C,W
            out[:, :, j, :] = (acc + rnd) / float(1 << self.log2Norm)
        
        return out

# Usage example
if __name__ == "__main__":
    # Test the downsampler
    downsampler = VTMDownsampler(bit_depth=8)
    input_tensor = torch.randn(1, 1, 1080, 1920)  # B, C, H, W
    
    with torch.no_grad():
        downsampled = downsampler(input_tensor, scale=0.5)  # 2x downsampling
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {downsampled.shape}")
        
        # Test different scaling ratios
        downsampled_4x = downsampler(input_tensor, scale=0.25)  # 4x downsampling
        print(f"4x downsampled shape: {downsampled_4x.shape}")