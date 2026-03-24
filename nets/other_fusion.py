import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from nets.enssm import EnSSM


# Vision Mamba块的简化实现
class VisionMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mixer = EnSSM(8,
                                   dim,
                                   dim,
                                   1024,
                                   activation="GELU",
                                   dropout=0.4,
                                   causal=False,
                                   )


    def forward(self, x):
        x = self.mixer(x)
        return x



# 编码器模型 - 
class other_VisionMambaEncoder(nn.Module):
    def __init__(self,embed_dim=32):
        super().__init__()

        self.enmamba = VisionMambaBlock(embed_dim)

    def forward(self, x):
        
        b,c,h,w = x.shape
        sequence = h*w
        x = x.view(b,c,sequence).permute(0,2,1)

        #添加位置
        pos_embedding = nn.Parameter(torch.zeros(1,sequence,c)).to(x.device)
        # 初始化位置编码
        nn.init.trunc_normal_(pos_embedding, std=0.02)
        x = x+pos_embedding

        x = self.enmamba(x)
        x = self.enmamba(x)
        x = self.enmamba(x)

        B,L,C = x.shape
        h = w = int(L**0.5)
        x = x.permute(0,2,1).view(b,c,h,w)
        
        return x