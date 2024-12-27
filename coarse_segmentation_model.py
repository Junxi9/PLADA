import torch
import torch.nn as nn

class CoarseSegmentationModel(nn.Module):
    def __init__(self, elu=True):
        super(CoarseSegmentationModel, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=False)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=False)

        # 添加一个输出层，用于生成初步的分割结果
        self.out_tr_coarse = nn.Conv3d(256, 3, kernel_size=1)  # 3表示类别数

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        
        # 生成初步的分割结果
        coarse_output = self.out_tr_coarse(out256)
        
        return coarse_output
