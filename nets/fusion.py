import torch
from torch import nn
import torch.nn.functional as F

# MPFF
class MLFusion(nn.Module):
    def __init__(self, norm, act,channel):
        super().__init__()

        self.attn_conv = nn.ModuleList()
        for i in range(3):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(channel[i], channel[i], 1, bias=False),
                norm(channel[i]),
                act(),
            ))
        self.fusion_last = nn.Sequential(
                nn.Conv2d(480, 480, 1, bias=False),
                norm(480),
                act(),
            )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        for i in range(3):
            x = feature_list[3-i]
            attn = self.attn_conv[2-i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            x_h,x_w  = feature_list[2-i].size(2), feature_list[2-i].size(3)
            x = F.interpolate(x, size=(x_h, x_w), mode='bilinear', align_corners=True)
            x = torch.cat((x, feature_list[2-i]), dim=1)
            feature_list[2-i] = x
        x = feature_list[0]
        attn = self.fusion_last(x)
        attn = self.pool(attn)
        attn = self.sigmoid(attn)

        x = attn * x + x
        return x

class Catfusion(nn.Module):
    def __init__(self, norm, act,channel):
        super().__init__()
        self.atten = nn.Sequential(
                nn.Conv2d(channel, channel, 1, bias=False),
                norm(channel),
                act(),
            )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
            x = feature_list
            attn = self.atten(x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            
            return x