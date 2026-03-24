import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from nets.enssm import EnSSM
from nets.gen import gilbert2d

#希尔伯特扫描方式变化
# Vision Mamba块的简化实现
class VisionMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = EnSSM(8,
                                   dim,
                                   dim,
                                   1024,
                                   activation="GELU",
                                   dropout=0.4,
                                   causal=False,
                                   )


    def forward(self, x):
 
        x = self.norm1(x)
        
        x = self.mixer(x)
        x = x
        return x
class Phillips(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.mamba = VisionMambaBlock(embed_dim)
    def forward(self,image):
        seq, indices, shape = self.image_to_hilbert_sequence(image = image)
        seq = self.mamba(seq)
        output = self.hilbert_sequence_to_image(seq,indices,shape)

        return output


    def image_to_hilbert_sequence(self, image: torch.Tensor):
        """
        输入: image (PyTorch Tensor, 形状: (B, C, W, H))
        输出: hilbert_sequence (Tensor, 形状: (B, H*W, C)), hilbert_indices (Tensor, 形状: (H, W))
        """
        # 确保输入是带有批量维度的 Tensor
        assert len(image.shape) == 4, "输入图像必须包含批量维度 (B, C, W, H)"
        B, C, W, H = image.shape  # 注意维度顺序: (批量, 通道, 宽度, 高度)
        
        # 生成希尔伯特曲线坐标（使用 numpy 生成，后转为 Tensor）
        coords = list(gilbert2d(W, H))  # 传入宽度和高度（注意顺序）
        coords = np.array(coords, dtype=np.int32)
        
        # 创建希尔伯特索引矩阵（numpy 数组，后转为 Tensor）
        hilbert_indices = np.zeros((H, W), dtype=np.int32)
        for idx, (x, y) in enumerate(coords):
            hilbert_indices[y, x] = idx  # y 是行索引，x 是列索引
        
        # 将 numpy 数组转为 Tensor 并移动到输入图像的设备
        coords_tensor = torch.from_numpy(coords).to(image.device)
        hilbert_indices_tensor = torch.from_numpy(hilbert_indices).to(image.device)
        
        # 调整图像维度顺序为 (B, H, W, C) 以匹配坐标索引
        image_reshaped = image.permute(0, 3, 2, 1)  # (B, C, W, H) -> (B, H, W, C)
        
        # 按坐标提取像素（利用 PyTorch 索引，自动处理批量维度）
        # coords_tensor[:, 1] 是行索引 (H 维度), coords_tensor[:, 0] 是列索引 (W 维度)
        hilbert_sequence = image_reshaped[
            torch.arange(B).view(-1, 1),  # 批量索引
            coords_tensor[:, 1],          # 行索引
            coords_tensor[:, 0],          # 列索引
            :                           # 通道索引
        ]
        
        # 确保输出形状为 (B, H*W, C)
        assert hilbert_sequence.shape == (B, H*W, C), f"形状错误，期望 (B, {H*W}, C)，得到 {hilbert_sequence.shape}"
        
        return hilbert_sequence, hilbert_indices_tensor, (H, W, C)


    

    def hilbert_sequence_to_image(self, hilbert_sequence: torch.Tensor, hilbert_indices: torch.Tensor, image_shape: tuple):
        """
        将希尔伯特序列恢复为图像（向量化实现，支持批量）
        
        参数:
        hilbert_sequence: 希尔伯特序列，形状 (B, L, C) （L=H*W）
        hilbert_indices: 希尔伯特索引矩阵，形状 (H, W)
        image_shape: 图像形状元组 (H, W, C)
        
        返回:
        restored_image: 恢复的图像，形状 (B, C, H, W)
        """
        H, W, C = image_shape
        B, L, _ = hilbert_sequence.shape
        assert L == H * W, f"序列长度 {L} 与图像尺寸 {H*W} 不匹配"
        
        # 确保索引矩阵与序列在同一设备，并转为 int64 类型
        hilbert_indices = hilbert_indices.to(hilbert_sequence.device).to(torch.long)
        
        # 展平索引矩阵为一维，值为序列中的索引
        flat_indices = hilbert_indices.flatten()  # 形状 (H*W,)
        
        # 对每个样本进行向量化恢复
        # 步骤1：将序列从 (B, L, C) 转为 (B, C, L)
        seq_transposed = hilbert_sequence.permute(0, 2, 1)  # 形状 (B, C, L)
        
        # 步骤2：根据索引矩阵重新排列像素
        # 使用 gather 操作，沿维度 2 插入像素到正确位置
        restored_image = seq_transposed.new_zeros((B, C, H, W))  # 初始化输出
        restored_image = seq_transposed.gather(dim=2, index=flat_indices[None, None, :].expand(B, C, -1))
        
        # 步骤3：调整维度顺序为 (B, C, H, W)
        # gather 结果形状为 (B, C, H*W)，需重塑为 (B, C, H, W)
        restored_image = restored_image.reshape(B, C, H, W)
        
        return restored_image

# 上采样块 - 使用卷积和插值处理非整除情况
class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x,h,w):
        x = self.last_layer(x)
        x = F.interpolate(x,size=(h,w), mode='bilinear',align_corners=True)
        return x,h*2,w*2


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 计算分块后的特征图尺寸
        self.feat_size = img_size // patch_size  # 例如：224//16 = 14
        
        # 分块数量
        self.num_patches = self.feat_size * self.feat_size
        
        # 分块投影层
        self.proj = nn.Conv2d(
            embed_dim, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 二维位置编码（可学习）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, embed_dim)
        )
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        参数:
            x: 输入图像/特征图，形状 [B, C, H, W]
            
        返回:
            patches: 分块后的特征序列，形状 [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        
        # 确保输入尺寸与模型配置匹配
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # 分块并投影到嵌入空间
        x = self.proj(x)  # [B, embed_dim, feat_size, feat_size]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        return x,self.pos_embed
class FixEmbedding(nn.Module):
    def __init__(self, img_size, patch_size):
        """
        Mamba模型输出的二维特征图复原模块
        
        参数:
            img_size: 原始图像尺寸（如224）
            patch_size: 分块大小（如16）
            embed_dim: Mamba输出的特征维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.feat_size = img_size // patch_size  # 分块后的网格尺寸（如14）
        
        # 计算复原后的特征图尺寸
        self.upsampled_size = img_size  # 可设置为更大尺寸实现超分辨率
        

        
    def forward(self, x, pos_embed=None):
        """
        参数:
            x: Mamba输出，形状 [B, seq_len, embed_dim]
            pos_embed: 位置编码（如有需要可移除）
            
        返回:
            reconstructed: 复原的二维特征图，形状 [B, target_channels, H, W]
        """
        B, seq_len, embed_dim = x.shape

        # 确保序列长度匹配分块数
        assert seq_len == self.feat_size * self.feat_size, \
            f"Sequence length {seq_len} doesn't match patch number {self.feat_size*self.feat_size}."
        
        # 移除位置编码（如果有）
        if pos_embed is not None:
            x = x - pos_embed  # 从序列中减去位置编码
        

        
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.feat_size, w=self.feat_size)
        # 形状：[B, embed_dim, feat_size, feat_size]
        
        # 扩展分块空间维度
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, feat_size, feat_size, 1, 1]
        x = x.expand(-1, -1, -1, -1, self.patch_size, self.patch_size)
        # 形状：[B, embed_dim, feat_size, feat_size, patch_size, patch_size]
        
        # 合并分块
        x = x.reshape(
            B, 
            embed_dim, 
            self.feat_size * self.patch_size, 
            self.feat_size * self.patch_size
        )
  
        return x
# 编码器模型 - 
class VisionMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=480, img_size=120, patch_size=4, embed_dim=480):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim


        # 编码器层 - 调整结构以达到120x120的输出
        self.encoder_layers = nn.ModuleList([
            # 第一层: 不提升分辨率
            PatchEmbedding(img_size,patch_size,embed_dim),
            VisionMambaBlock(embed_dim),
            FixEmbedding(img_size,patch_size),
            Phillips(embed_dim),

            # 第二层: 升高分辨率为原来的2倍 (120→240) 
            UpsampleBlock(embed_dim, embed_dim//2),
            PatchEmbedding(img_size*2,patch_size,embed_dim//2),
            #channel (32->64)
            VisionMambaBlock(embed_dim//2),
            FixEmbedding(img_size*2,patch_size),
            Phillips(embed_dim//2),

        ])

    def forward(self, x):
        B, C, H, W = x.shape

        h = H*2
        w = W*2

        # 编码器前向传播
        
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, UpsampleBlock):
                x,h,w = layer(x,h,w)
            elif isinstance(layer,PatchEmbedding):
                x,pos = layer(x)
            elif isinstance(layer,FixEmbedding):
                x = layer(x,pos)
            elif isinstance(layer,Phillips):
                x = layer(x)
            else:
                x = layer(x)

                

        return x



