import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from nets.backbone import BN_MOMENTUM, hrnet_classification
# main
# from nets.enssm import EnSSM
from nets.fusion import Catfusion, MLFusion
from nets.other_fusion import other_VisionMambaEncoder



class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        #删除相关的网络层
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
 
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list
#adapting the segmenta tion task
class HRnet(nn.Module):
    #pretrained:loading others weight, keep the False
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)

        last_inp_channels   = np.int64(np.sum(self.backbone.model.pre_stage_channels))

        #output the channels about num_classes
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.fusion = MLFusion(nn.BatchNorm2d,nn.ReLU,[448,384,256])

        self.enssm_encoder = other_VisionMambaEncoder(
                                                embed_dim=480,
                                      )
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone(inputs)
        a= self.fusion(x)
        
        x = self.enssm_encoder(a)
        x = x+a
        # for i in range(4):
        #     x[i]  = torch.cat([x[i],w[i]],dim=1)
        #     x[i]  = self.cbam[i](x[i])

        x = self.last_layer(x)
        x = F.interpolate(x,size=(H,W), mode='bilinear',align_corners=True)
        #feature_map_data_list = [x[0, i].detach().cpu().numpy() for i in range(x.shape[1])]
        # plt.figure(figsize=(12, 8))
        #
        # for i, feature_map_data in enumerate(feature_map_data_list):
        #     plt.subplot(1, 4, i + 1)
        #     plt.imshow(feature_map_data, cmap="viridis")
        #     plt.title(f"Feature Map {i + 1}")
        #     plt.axis('off')
        return x
