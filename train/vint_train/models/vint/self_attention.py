import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # 计算位置编码，创建一个形状为 (max_seq_len, d_model) 的零张量
        pos_enc = torch.zeros(max_seq_len, d_model)
        # 创建一个从0到max_seq_len的序列，并增加一个维度以便后续操作
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算每个维度的缩放因子
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 使用正弦和余弦函数填充偶数和奇数索引的位置编码
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        # 增加一个维度，使其形状变为 (1, max_seq_len, d_model)
        pos_enc = pos_enc.unsqueeze(0)

        # 将位置编码注册为缓冲区，以避免在保存模型时将其视为参数
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # 将位置编码添加到输入x中
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        # 初始化位置编码模块
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        # 定义自注意力层
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        # 堆叠多个自注意力层形成解码器
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        # 定义输出层，将最终的嵌入映射到目标输出尺寸
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        # 如果存在位置编码，则对输入进行位置编码处理
        if self.positional_encoding: x = self.positional_encoding(x)
        # 通过自注意力解码器处理输入
        x = self.sa_decoder(x)
        # 当前x的形状为 [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)  # 展平为 [batch_size, seq_len * embed_dim]
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)  # 通过线性层
            x = F.relu(x)  # 应用ReLU激活函数
        return x
