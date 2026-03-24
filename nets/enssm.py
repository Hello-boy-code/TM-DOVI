import torch
from torch import nn
from torch.nn import LayerNorm

from .bimamba import Mamba as BiMamba
class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,

    ):
        super().__init__()

        if activation == "GELU":
            activation = torch.nn.GELU


        bidirectional = False
        if causal or  bidirectional:
            pass
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',

            )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def forward(
        self,
        x, inference_params = None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out
class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)

        # 残差连接
        if self.conv is not None:
            x = self.conv(x)

        out = out + x  # 残差相加
        return out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[480,240,120],
        d_ffn=1024,
        activation='Swish',
        dropout=0.4,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        
        cnn_list.append(CNNEncoderLayer(
                    input_size = input_size ,
                    output_size = output_sizes,
                    dropout=dropout
                ))
        mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        x,
        inference_params = None,
    ):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
            )

        return out