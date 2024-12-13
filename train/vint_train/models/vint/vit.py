import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from vit_pytorch import SimpleViT
import pdb


class ViT(nn.Module):
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,  # 观察编码的大小，默认为512
        context_size: int = 5,                    # 上下文大小，默认为5
        image_size: int = 128,                    # 图像大小，默认为128
        patch_size: int = 16,                     # 补丁大小，默认为16
        mha_num_attention_heads: Optional[int] = 4,# 多头注意力机制中的头数，默认为4
        mha_num_attention_layers: Optional[int] = 4,# 注意力层的数量，默认为4
    ) -> None:
        """
        ViT类构造函数，用于初始化视觉Transformer模型。
        """
        super(ViT, self).__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        if type(image_size) == int:
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_width = image_size[0]
            self.image_height = image_size[1]

        # 初始化MaskedGoalViT模块
        self.ViT = MaskedGoalViT(
            context_size=context_size,
            image_size=(self.image_height, self.image_width*(self.context_size + 2)),
            patch_size=self.patch_size,
            dim=obs_encoding_size,  # 使用obs_encoding_size作为维度
            depth=mha_num_attention_layers,
            heads=mha_num_attention_heads,
            mlp_dim=obs_encoding_size
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，将输入图像和目标图像传入模型并返回最终表示。

        参数：
        - obs_img: 观察图像张量
        - goal_img: 目标图像张量
        - input_goal_mask: 输入目标掩码张量

        返回：
        - final_repr: 最终输出的特征表示
        """
        # 将观察图像按通道分割成列表
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        # 将目标图像添加到观察图像列表中
        obsgoal_img_list = obs_img_list + [goal_img]
        # 在最后一个维度上拼接所有图像
        x = torch.cat(obsgoal_img_list, dim=-1)

        # 确保输入形状正确
        assert len(x.shape) == 4, "input image shape is not 4D"
        assert x.shape[1] == 3, "input image channel is not 3"
        assert x.shape[2] == self.image_height, f"input image height is not {self.image_height}"
        assert x.shape[3] == self.image_width*(self.context_size + 2), f"input image width is not {self.image_width}*(context_size + 2)"
       
        # 通过ViT模块获取最终表示
        final_repr = self.ViT(x)
        
        return final_repr

# 辅助函数

def pair(t):
    """
    将输入转换为元组，如果输入已经是元组则直接返回。

    参数：
    - t: 输入值

    返回：
    - (t, t) 如果t不是元组，否则返回t
    """
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    """
    生成二维位置嵌入（sine-cosine）。

    参数：
    - patches: 输入补丁张量
    - temperature: 温度参数用于缩放
    - dtype: 数据类型

    返回：
    - pe: 位置嵌入张量
    """
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 创建网格坐标
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    
    # 确保特征维度是4的倍数
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    # 计算位置嵌入
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# 类定义

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        """
        前馈神经网络模块。

        参数：
        - dim: 输入维度
        - hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入张量

        返回：
        - 输出张量
        """
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        """
        自注意力机制模块。

        参数：
        - dim: 输入维度
        - heads: 注意力头的数量
        - dim_head: 每个头的维度
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask):
        """
        前向传播函数。

        参数：
        - x: 输入张量
        - mask: 掩码张量

        返回：
        - out: 输出张量
        """
        x = self.norm(x)

        # 计算Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力得分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 扩展mask以适应批次
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        
        attn = self.attend(dots + mask) 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        """
        Transformer模块，由多个自注意力层和前馈层组成。

        参数：
        - dim: 输入维度
        - depth: 层数
        - heads: 注意力头的数量
        - dim_head: 每个头的维度
        - mlp_dim: 前馈网络的隐藏层维度
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x, mask):
        """
        前向传播函数。

        参数：
        - x: 输入张量
        - mask: 掩码张量

        返回：
        - x: 输出张量
        """
        for attn, ff in self.layers:
            x = attn(x, mask) + x  # 残差连接
            x = ff(x) + x          # 残差连接
        return x


# 实现带有目标掩码的ViT
class MaskedGoalViT(nn.Module):
    def __init__(self, *, context_size, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        """
        带有目标掩码的视觉Transformer模块。

        参数：
        - context_size: 上下文大小
        - image_size: 图像大小
        - patch_size: 补丁大小
        - dim: 特征维度
        - depth: 层数
        - heads: 注意力头的数量
        - mlp_dim: 前馈网络的隐藏层维度
        - channels: 输入图像的通道数
        - dim_head: 每个头的维度
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸可以被补丁大小整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.h = image_height // patch_height
        self.w = image_width // patch_width
        patch_dim = channels * patch_height * patch_width

        # 补丁嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Transformer层
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        # 初始化目标掩码
        self.goal_mask = torch.ones((self.h, self.w))
        assert self.w % (context_size + 2) == 0, "context_size must be a factor of numbers of patches in width"
        self.goal_mask[:, -self.w//(context_size + 2):] = 0  # 设置目标区域为0
        self.goal_mask = rearrange(self.goal_mask, 'h w -> (h w)')
        self.no_mask = torch.ones(self.h*self.w)
        self.all_masks = torch.stack([self.no_mask, self.goal_mask], dim=0)
        self.no_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        self.goal_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        
        # 构建交叉掩码
        for i in range(self.h*self.w):
            for j in range(self.h*self.w):
                if self.goal_mask[i] + self.goal_mask[j] < 2:
                    self.goal_cross_mask[i, j] = 0
                    
        self.all_cross_masks = torch.stack([self.no_cross_mask, self.goal_cross_mask], dim=0)
        self.mean_mask = self.all_masks / self.all_masks.mean(dim=1, keepdim=True)

        # 将掩码中的0替换为-1e9，以便在softmax中忽略这些位置
        self.all_cross_masks = torch.where(self.all_cross_masks == 0, -1e9, 0.0)
        self.all_masks = torch.where(self.all_masks == 0, -1e9, 0.0)


    def forward(self, img, input_goal_mask=None):
        """
        前向传播函数。

        参数：
        - img: 输入图像张量
        - input_goal_mask: 输入目标掩码（可选）

        返回：
        - x: 最终输出的特征表示
        """
        b, c, h, w, dtype = *img.shape, img.dtype
        device = img.device

        if input_goal_mask is None:
            input_goal_mask = torch.zeros(b, dtype=torch.int64)

        # 根据输入目标掩码选择相应的交叉掩码
        final_mask = torch.index_select(self.all_cross_masks.to(device), 0, input_goal_mask.to(device))

        # 获取补丁嵌入
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)  # 计算位置嵌入
        x = rearrange(x, 'b ... d -> b (...) d') + pe  # 添加位置嵌入

        # 通过Transformer处理
        x = self.transformer(x, mask=final_mask)
        final_mask = torch.index_select(self.mean_mask.to(device), 0, input_goal_mask.to(device)).unsqueeze(-1)
        x = x * final_mask  # 应用最终掩码
        x = x.mean(dim=1)   # 对时间步进行平均

        x = self.to_latent(x)  # 转换为潜在空间
        return x
