import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from nets.enssm import EnSSM


# making BCM
class BCMBlock(nn.Module):
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



# encoder model -
class other_VisionMambaEncoder(nn.Module):
    def __init__(self,embed_dim=32):
        super().__init__()

        self.enmamba = BCMBlock(embed_dim)

    def forward(self, x):
        
        b,c,h,w = x.shape
        sequence = h*w
        x = x.view(b,c,sequence).permute(0,2,1)

        #adding position
        pos_embedding = nn.Parameter(torch.zeros(1,sequence,c)).to(x.device)
        # position encoder
        nn.init.trunc_normal_(pos_embedding, std=0.02)
        x = x+pos_embedding

        x = self.enmamba(x)
        x = self.enmamba(x)
        x = self.enmamba(x)

        B,L,C = x.shape
        h = w = int(L**0.5)
        x = x.permute(0,2,1).view(b,c,h,w)
        
        return x