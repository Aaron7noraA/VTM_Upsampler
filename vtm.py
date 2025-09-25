# vtm_upsampler.py
import torch
import torch.nn as nn

class VTMUpsampler(nn.Module):
    """
    VTM-like separable upsampler with fixed integer taps and single final normalization.
    - filter_set: 'standard' (8/4 taps, log2Norm=12) or 'alt' (12/6 taps, log2Norm=16)
    - use_luma: True -> luma path (16 phases), False -> chroma path (32 phases)
    """
    def __init__(self, filter_set: str = 'standard', use_luma: bool = True):
        super().__init__()
        if filter_set not in ('standard', 'alt'):
            raise ValueError("filter_set must be 'standard' or 'alt'")
        self.filter_set = filter_set
        self.use_luma = use_luma

        if filter_set == 'standard':
            # 8-tap luma (16x8), rows sum ~64; 4-tap chroma (32x4), rows sum ~64
            luma = [
                [0,0,0,64,0,0,0,0],
                [0,1,-3,63,4,-2,1,0],
                [-1,2,-5,62,8,-3,1,0],
                [-1,3,-8,60,13,-4,1,0],
                [-1,4,-10,58,17,-5,1,0],
                [-1,4,-11,52,26,-8,3,-1],
                [-1,3,-9,47,31,-10,4,-1],
                [-1,4,-11,45,34,-10,4,-1],
                [-1,4,-11,40,40,-11,4,-1],
                [-1,4,-10,34,45,-11,4,-1],
                [-1,4,-10,31,47,-9,3,-1],
                [-1,3,-8,26,52,-11,4,-1],
                [0,1,-5,17,58,-10,4,-1],
                [0,1,-4,13,60,-8,3,-1],
                [0,1,-3,8,62,-5,2,-1],
                [0,1,-2,4,63,-3,1,0],
            ]
            chroma = [
                [0,64,0,0],[-1,63,2,0],[-2,62,4,0],[-2,60,7,-1],
                [-2,58,10,-2],[-3,57,12,-2],[-4,56,14,-2],[-4,55,15,-2],
                [-4,54,16,-2],[-5,53,18,-2],[-6,52,20,-2],[-6,49,24,-3],
                [-6,46,28,-4],[-5,44,29,-4],[-4,42,30,-4],[-4,39,33,-4],
                [-4,36,36,-4],[-4,33,39,-4],[-4,30,42,-4],[-4,29,44,-5],
                [-4,28,46,-6],[-3,24,49,-6],[-2,20,52,-6],[-2,18,53,-5],
                [-2,16,54,-4],[-2,15,55,-4],[-2,14,56,-4],[-2,12,57,-3],
                [-2,10,58,-2],[-1,7,60,-2],[0,4,62,-2],[0,2,63,-1],
            ]
            self.log2Norm = 12
        else:
            # Alternative 12-tap luma (16x12), rows sum ~256; 6-tap chroma (32x6), rows sum ~256
            luma = [
                [0,0,0,0,0,256,0,0,0,0,0,0],
                [1,-1,0,3,-12,253,16,-6,2,0,0,0],
                [0,0,-3,9,-24,250,32,-11,4,-1,0,0],
                [0,0,-4,12,-32,241,52,-18,8,-4,2,-1],
                [0,1,-6,15,-38,228,75,-28,14,-7,3,-1],
                [0,1,-7,18,-43,214,96,-33,16,-8,3,-1],
                [1,0,-6,17,-44,196,119,-40,20,-10,4,-1],
                [0,2,-9,21,-47,180,139,-43,20,-10,4,-1],
                [-1,3,-9,21,-46,160,160,-46,21,-9,3,-1],
                [-1,4,-10,20,-43,139,180,-47,21,-9,2,0],
                [-1,4,-10,20,-40,119,196,-44,17,-6,0,1],
                [-1,3,-8,16,-33,96,214,-43,18,-7,1,0],
                [-1,3,-7,14,-28,75,228,-38,15,-6,1,0],
                [-1,2,-4,8,-18,52,241,-32,12,-4,0,0],
                [0,0,-1,4,-11,32,250,-24,9,-3,0,0],
                [0,0,0,2,-6,16,253,-12,3,0,-1,1],
            ]
            chroma = [
                [0,0,256,0,0,0],[1,-6,256,6,-1,0],[2,-11,254,14,-4,1],[4,-18,252,23,-6,1],
                [6,-24,249,32,-9,2],[6,-26,244,41,-12,3],[7,-30,239,53,-18,5],[8,-34,235,61,-19,5],
                [10,-38,228,72,-22,6],[10,-39,220,84,-26,7],[10,-40,213,94,-29,8],[11,-42,205,105,-32,9],
                [11,-42,196,116,-35,10],[11,-42,186,128,-37,10],[11,-42,177,138,-38,10],[11,-41,167,148,-40,11],
                [11,-41,158,158,-41,11],[11,-40,148,167,-41,11],[10,-38,138,177,-42,11],[10,-37,128,186,-42,11],
                [10,-35,116,196,-42,11],[9,-32,105,205,-42,11],[8,-29,94,213,-40,10],[7,-26,84,220,-39,10],
                [6,-22,72,228,-38,10],[5,-19,61,235,-34,8],[5,-18,53,239,-30,7],[3,-12,41,244,-26,6],
                [2,-9,32,249,-24,6],[1,-6,23,252,-18,4],[1,-4,14,254,-11,2],[0,-1,6,256,-6,1],
            ]
            self.log2Norm = 16

        if use_luma:
            taps = luma
            self.numFracShift = 4  # 16 fractional phases
            self.numFracMask = 15
        else:
            taps = chroma
            self.numFracShift = 5  # 32 fractional phases
            self.numFracMask = 31

        self.filterLength = len(taps[0])
        self.register_buffer('taps', torch.tensor(taps, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, scale: float):
        """
        x: [B,C,H,W] float tensor
        scale: > 0 (e.g., 2.0, 4.0)
        returns: [B,C,round(H*scale), round(W*scale)]
        """
        B,C,H,W = x.shape
        newW = int(round(W*scale))
        newH = int(round(H*scale))

        # Horizontal pass: [B,C,H,newW]
        buf = x.new_zeros((B, C, H, newW), dtype=torch.float32)
        center = self.filterLength // 2 - 1
        for i in range(newW):
            # Map output column i -> source pos (14-bit fixed point), then to (integer, fracPhase)
            refPosFixed = int(i * (1 << 14) / scale)
            refPosPhase = refPosFixed >> (14 - self.numFracShift)
            integer = refPosPhase >> self.numFracShift
            frac = refPosPhase & self.numFracMask
            f = self.taps[frac]  # [filterLength]
            acc = x.new_zeros((B,C,H), dtype=torch.float32)
            for k in range(self.filterLength):
                xInt = max(0, min(W - 1, integer + k - center))
                acc += f[k] * x[..., xInt]
            buf[..., i] = acc

        # Vertical pass + final normalization: [B,C,newH,newW]
        out = x.new_zeros((B, C, newH, newW), dtype=torch.float32)
        rnd = float(1 << (self.log2Norm - 1))
        for j in range(newH):
            refPosFixed = int(j * (1 << 14) / scale)
            refPosPhase = refPosFixed >> (14 - self.numFracShift)
            integer = refPosPhase >> self.numFracShift
            frac = refPosPhase & self.numFracMask
            f = self.taps[frac]
            acc = x.new_zeros((B,C,newW), dtype=torch.float32)
            for k in range(self.filterLength):
                yInt = max(0, min(H - 1, integer + k - center))
                acc += f[k] * buf[:, :, yInt, :]
            out[:, :, j, :] = (acc + rnd) / float(1 << self.log2Norm)

        return out

# Example
if __name__ == "__main__":
    x = torch.rand(1, 1, 8, 8)  # toy input
    up = VTMUpsampler(filter_set='alt', use_luma=True)  # 12-tap luma, log2Norm=16
    y = up(x, scale=2.0)
    print(x.shape, "->", y.shape)