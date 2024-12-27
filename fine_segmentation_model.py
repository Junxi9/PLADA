import torch
import torch.nn as nn

class FineSegmentationModel(nn.Module):
    def __init__(self, elu=True):
        super(FineSegmentationModel, self).__init__()
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll=False)

    def forward(self, x, coarse_output):
        # 将粗分割结果与原始输入拼接在一起
        concatenated_input = torch.cat((x, coarse_output), dim=1)
        
        # 解码器部分
        out = self.up_tr256(concatenated_input, None)  # 这里假设没有skip connection
        out = self.up_tr128(out, None)
        out = self.up_tr64(out, None)
        out = self.up_tr32(out, None)
        fine_output = self.out_tr(out)
        
        return fine_output
