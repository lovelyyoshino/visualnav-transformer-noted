import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder

class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,  # 上下文中使用的先前观察数量
        len_traj_pred: Optional[int] = 5,  # 未来预测的路径点数量
        learn_angle: Optional[bool] = True,  # 是否学习机器人的偏航角
        obs_encoder: Optional[str] = "efficientnet-b0",  # 用于编码观察的EfficientNet架构名称
        obs_encoding_size: Optional[int] = 512,  # 观察图像编码的大小
        late_fusion: Optional[bool] = False,  # 是否使用后期融合
        mha_num_attention_heads: Optional[int] = 2,  # 多头自注意力机制中的头数
        mha_num_attention_layers: Optional[int] = 2,  # 自注意力层的数量
        mha_ff_dim_factor: Optional[int] = 4,  # 前馈网络维度因子
    ) -> None:
        """
        ViNT类：使用基于Transformer的架构来编码（当前和过去的）视觉观察和目标，
        使用EfficientNet CNN，并以与具体实现无关的方式预测时间距离和归一化动作。
        Args:
            context_size (int): 使用多少个先前观察作为上下文
            len_traj_pred (int): 预测未来多少个路径点
            learn_angle (bool): 是否预测机器人的偏航角
            obs_encoder (str): 用于编码观察的EfficientNet架构名称（例如："efficientnet-b0"）
            obs_encoding_size (int): 观察图像编码的大小
            goal_encoding_size (int): 目标图像编码的大小
        """
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)  # 初始化观察编码器
            self.num_obs_features = self.obs_encoder._fc.in_features  # 获取观察特征数量
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)  # 后期融合时初始化目标编码器
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)  # 否则，使用观察+目标通道初始化目标编码器
            self.num_goal_features = self.goal_encoder._fc.in_features  # 获取目标特征数量
        else:
            raise NotImplementedError
        
        # 如果观察特征数量不等于指定的编码大小，则进行线性压缩
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()  # 保持原样
        
        # 如果目标特征数量不等于指定的编码大小，则进行线性压缩
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()  # 保持原样

        # 初始化多层解码器
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        
        # 距离预测器
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        
        # 动作预测器
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数，用于处理输入的观察图像和目标图像并生成输出。
        
        Args:
            obs_img (torch.tensor): 输入的观察图像张量
            goal_img (torch.tensor): 输入的目标图像张量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 返回距离预测和动作预测
        """

        # 获取融合后的观察和目标编码
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)  # 提取目标编码
        else:
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)  # 将观察和目标合并
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)  # 提取目标编码
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)  # 平均池化
        #这个是从efficientnet_pytorch中的model.py中的EfficientNet类中的
        # __init__方法中的GlobalParams类中的include_top属性，
        # 是否包括最后的分类层
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)  # 展平
            goal_encoding = self.goal_encoder._dropout(goal_encoding)  # 应用dropout
        # 当前goal_encoding的形状为 [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)  # 压缩目标编码
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)  # 增加一个维度
        # 当前goal_encoding的形状为 [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size  # 确保形状正确
        
        # 根据上下文大小将观察分割成多个部分
        # 图像大小为 [batch_size, 3*self.context_size, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # 图像大小为 [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # 获取观察编码
        obs_encoding = self.obs_encoder.extract_features(obs_img)  # 提取观察编码
        # 当前大小为 [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)  # 平均池化
        # 当前大小为 [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)  # 展平
            obs_encoding = self.obs_encoder._dropout(obs_encoding)  # 应用dropout
        # 当前大小为 [batch_size, self.context_size + 2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)  # 压缩观察编码
        # 当前大小为 [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # 重塑obs_encoding为 [context + 1, batch, encoding_size]，注意顺序翻转
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)  # 转置
        # 当前大小为 [batch_size, self.context_size + 1, self.obs_encoding_size]

        # 将目标编码连接到观察编码上
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)  # 拼接
        final_repr = self.decoder(tokens)  # 解码，使用自注意力机制
        # 当前大小为 [batch_size, 32]

        dist_pred = self.dist_predictor(final_repr)  # 距离预测
        action_pred = self.action_predictor(final_repr)  # 动作预测

        # 扩展输出以匹配标签的大小
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # 将位置增量转换为路径点
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # 规范化角度预测
        return dist_pred, action_pred  # 返回距离预测和动作预测
