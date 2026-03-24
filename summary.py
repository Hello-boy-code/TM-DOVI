#--------------------------------------------#
#   该部分代码用于看网络结构,查看浮点运算的速度，和参数量
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.hrnet import HRnet
# computer total params
if __name__ == "__main__":
    input_shape     = [480, 480]
    num_classes     = 4
    backbone        = 'hrnetv2_w32'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = HRnet(num_classes = num_classes, backbone = backbone, pretrained=False).to(device)
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    _, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    params   = clever_format([ params], "%.3f")

    print('Total params: %s' % (params))
