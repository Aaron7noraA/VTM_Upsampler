class VTMUpsamplerOptimized(nn.Module):
    """
    Optimized PyTorch implementation using actual VTM filters
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
        Optimized forward pass using actual VTM filters with PyTorch operations
        """
        B, C, H, W = x.shape
        
        if scale_factor == 1.0:
            return x
        
        # Use actual VTM filtering with PyTorch operations
        return self._vtm_filter_2d(x, scale_factor)
    
    def _vtm_filter_2d(self, x, scale_factor):
        """2D VTM filtering using PyTorch operations"""
        B, C, H, W = x.shape
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)
        
        # Horizontal filtering
        x_h = self._horizontal_filter_pytorch(x, scale_factor)
        
        # Vertical filtering
        x_v = self._vertical_filter_pytorch(x_h, scale_factor)
        
        return x_v
    
    def _horizontal_filter_pytorch(self, x, scale_factor):
        """Horizontal filtering using PyTorch operations"""
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
            f = self.filter_tensor[frac]  # [filter_length]
            
            # Apply horizontal filter using PyTorch operations
            for k in range(self.filter_length):
                x_int = max(0, min(W-1, integer + k - self.filter_length//2 + 1))
                x_h[:, :, :, i] += f[k] * x[:, :, :, x_int]
        
        return x_h
    
    def _vertical_filter_pytorch(self, x, scale_factor):
        """Vertical filtering using PyTorch operations"""
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
            f = self.filter_tensor[frac]
            
            # Apply vertical filter using PyTorch operations
            for k in range(self.filter_length):
                y_int = max(0, min(H-1, integer + k - self.filter_length//2 + 1))
                x_v[:, :, j, :] += f[k] * x[:, :, y_int, :]
        
        # Apply normalization (equivalent to >> log2Norm)
        x_v = x_v / (2 ** self.log2_norm)
        
        return x_v