import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
# LOAD DATA CONFIG
# 加载数据配置文件
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

# POPULATE ACTION STATS
# 填充动作统计信息
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    计算距离和动作预测的损失。

    参数：
        dist_label (torch.Tensor): 距离标签。
        action_label (torch.Tensor): 动作标签。
        dist_pred (torch.Tensor): 预测的距离。
        action_pred (torch.Tensor): 预测的动作。
        alpha (float): 动作损失的权重。
        learn_angle (bool): 是否学习动作的角度。
        action_mask (torch.Tensor, optional): 动作掩码，默认为None。

    返回：
        dict: 包含各类损失的字典，包括总损失。
    """
    # 计算均方误差损失
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        # 在非批次维度上减少以获得每个批次元素的损失
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # 遮蔽无效输入（对于负值或当观察与目标之间的距离较大时）
    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    # 计算动作关键点的余弦相似度
    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        action_pred[:, :, :2], action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(action_pred[:, :, :2], start_dim=1),
        torch.flatten(action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "dist_loss": dist_loss,  # 距离损失
        "action_loss": action_loss,  # 动作损失
        "action_waypts_cos_sim": action_waypts_cos_similairity,  # 单一动作关键点余弦相似度
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim,  # 多个动作关键点余弦相似度
    }

    if learn_angle:
        # 如果需要学习角度，则计算角度的余弦相似度
        action_orien_cos_sim = action_reduce(F.cosine_similarity(
            action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
        ))
        multi_action_orien_cos_sim = action_reduce(F.cosine_similarity(
            torch.flatten(action_pred[:, :, 2:], start_dim=1),
            torch.flatten(action_label[:, :, 2:], start_dim=1),
            dim=-1,
            )
        )
        results["action_orien_cos_sim"] = action_orien_cos_sim  # 单一动作方向余弦相似度
        results["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim  # 多个动作方向余弦相似度

    # 计算总损失
    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss  # 总损失

    return results  # 返回包含所有损失的结果字典

def _log_data(
    i,
    epoch,
    num_batches,
    normalized,
    project_folder,
    num_images_log,
    loggers,
    obs_image,
    goal_image,
    action_pred,
    action_label,
    dist_pred,
    dist_label,
    goal_pos,
    dataset_index,
    use_wandb,
    mode,
    use_latest,
    wandb_log_freq=1,
    print_log_freq=1,
    image_log_freq=1,
    wandb_increment_step=True,
):
    """
    将数据记录到wandb并打印到控制台。

    参数：
        i (int): 当前批次索引。
        epoch (int): 当前训练轮数。
        num_batches (int): 总批次数量。
        normalized (bool): 数据是否经过归一化处理。
        project_folder (str): 项目文件夹路径。
        num_images_log (int): 要记录的图像数量。
        loggers (dict): 日志记录器字典。
        obs_image (torch.Tensor): 观察图像。
        goal_image (torch.Tensor): 目标图像。
        action_pred (torch.Tensor): 预测的动作。
        action_label (torch.Tensor): 实际的动作标签。
        dist_pred (torch.Tensor): 预测的距离。
        dist_label (torch.Tensor): 实际的距离标签。
        goal_pos (torch.Tensor): 目标位置。
        dataset_index (torch.Tensor): 数据集索引。
        use_wandb (bool): 是否使用wandb进行日志记录。
        mode (str): 当前模式（如“train”或“eval”）。
        use_latest (bool): 是否使用最新的数据。
        wandb_log_freq (int): wandb日志记录频率。
        print_log_freq (int): 控制台打印频率。
        image_log_freq (int): 图像日志记录频率。
        wandb_increment_step (bool): 是否在wandb中增加步骤计数。
    """
    data_log = {}
    # 遍历所有日志记录器，收集当前批次的数据
    for key, logger in loggers.items():
        if use_latest:
            # 如果使用最新数据，则获取最新值
            data_log[logger.full_name()] = logger.latest()
            # 每隔一定批次打印一次日志信息
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            # 否则获取平均值
            data_log[logger.full_name()] = logger.average()
            # 每隔一定批次打印一次日志信息
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    # 如果启用wandb且满足记录频率条件，则记录数据
    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    # 如果满足图像记录频率条件，则可视化和记录图像
    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            mode,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image),
            to_numpy(goal_image),
            to_numpy(dataset_index),
            to_numpy(goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            mode,
            normalized,
            project_folder,
            epoch,
            num_images_log,
            use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    训练模型一个周期。

    参数：
        model (nn.Module): 要训练的模型。
        optimizer (Adam): 使用的优化器。
        dataloader (DataLoader): 用于训练的数据加载器。
        transform (transforms): 应用的变换。
        device (torch.device): 使用的设备。
        project_folder (str): 保存图像的文件夹。
        epoch (int): 当前轮数。
        alpha (float): 动作损失的权重。
        learn_angle (bool): 是否学习动作的角度。
        print_log_freq (int): 打印损失的频率。
        image_log_freq (int): 记录图像的频率。
        num_images_log (int): 要记录的图像数量。
        use_wandb (bool): 是否使用wandb。
        use_tqdm (bool): 是否使用tqdm显示进度条。
    """
    model.train()  # 设置模型为训练模式
    # 初始化各类损失的日志记录器
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    
    # 将所有日志记录器放入字典中
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    # 如果需要学习角度，则添加相关日志记录器
    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    num_batches = len(dataloader)  # 获取总批次数
    tqdm_iter = tqdm.tqdm(
        dataloader,
        disable=not use_tqdm,  # 根据设置决定是否显示进度条
        dynamic_ncols=True,
        desc=f"Training epoch {epoch}",
    )
    
    # 遍历每个批次的数据
    for i, data in enumerate(tqdm_iter):
        (
            obs_image,
            goal_image,
            action_label,
            dist_label,
            goal_pos,
            dataset_index,
            action_mask,
        ) = data  # 解包数据
        
        # 将观察图像分割成多个部分，并应用变换
        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)  # 可视化最后一张观察图像
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)  # 合并处理后的观察图像

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)  # 可视化目标图像
        
        goal_image = transform(goal_image).to(device)  # 转移目标图像到指定设备
        model_outputs = model(obs_image, goal_image)  # 模型前向传播

        dist_label = dist_label.to(device)  # 转移标签到指定设备
        action_label = action_label.to(device)
        action_mask = action_mask.to(device)

        optimizer.zero_grad()  # 清空梯度
      
        dist_pred, action_pred = model_outputs  # 获取模型输出的预测结果

        # 计算损失
        losses = _compute_losses(
            dist_label=dist_label,
            action_label=action_label,
            dist_pred=dist_pred,
            action_pred=action_pred,
            alpha=alpha,
            learn_angle=learn_angle,
            action_mask=action_mask,
        )

        losses["total_loss"].backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 记录损失数据
        for key, value in losses.items():
            if key in loggers:
                logger = loggers[key]
                logger.log_data(value.item())

        # 调用_log_data函数记录数据
        _log_data(
            i=i,
            epoch=epoch,
            num_batches=num_batches,
            normalized=normalized,
            project_folder=project_folder,
            num_images_log=num_images_log,
            loggers=loggers,
            obs_image=viz_obs_image,
            goal_image=viz_goal_image,
            action_pred=action_pred,
            action_label=action_label,
            dist_pred=dist_pred,
            dist_label=dist_label,
            goal_pos=goal_pos,
            dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq,
            print_log_freq=print_log_freq,
            image_log_freq=image_log_freq,
            use_wandb=use_wandb,
            mode="train",
            use_latest=True,
        )

def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,

):
    """
    在给定评估数据集上评估模型。

    参数：
        eval_type (string): f"{data_type}_{eval_type}"（例如“recon_train”，“gs_test”等）。
        model (nn.Module): 要评估的模型。
        dataloader (DataLoader): 用于评估的数据加载器。
        transform (transforms): 应用到图像的变换。
        device (torch.device): 用于评估的设备。
        project_folder (string): 项目文件夹路径。
        epoch (int): 当前轮数。
        alpha (float): 动作损失的权重。
        learn_angle (bool): 是否学习动作的角度。
        num_images_log (int): 要记录的图像数量。
        use_wandb (bool): 是否使用wandb进行日志记录。
        eval_fraction (float): 用于评估的数据比例。
        use_tqdm (bool): 是否使用tqdm进行日志记录。
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化不同类型的日志记录器
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    action_waypts_cos_sim_logger = Logger("action_waypts_cos_sim", eval_type)
    multi_action_waypts_cos_sim_logger = Logger("multi_action_waypts_cos_sim", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "action_waypts_cos_sim": action_waypts_cos_sim_logger,
        "multi_action_waypts_cos_sim": multi_action_waypts_cos_sim_logger,
        "total_loss": total_loss_logger,
    }

    # 如果需要学习角度，则初始化相关日志记录器
    if learn_angle:
        action_orien_cos_sim_logger = Logger("action_orien_cos_sim", eval_type)
        multi_action_orien_cos_sim_logger = Logger("multi_action_orien_cos_sim", eval_type)
        loggers["action_orien_cos_sim"] = action_orien_cos_sim_logger
        loggers["multi_action_orien_cos_sim"] = multi_action_orien_cos_sim_logger

    # 获取批次数量并根据评估比例调整
    num_batches = len(dataloader)
    num_batches = max(int(num_batches * eval_fraction), 1)

    viz_obs_image = None
    
    # 禁用梯度计算以节省内存和加快速度
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            disable=not use_tqdm,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
        )
        
        # 遍历每个批次的数据
        for i, data in enumerate(tqdm_iter):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            # 将观察图像分割成多个部分，并调整可视化图像大小
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # 调整目标图像大小并转换为张量
            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
            goal_image = transform(goal_image).to(device)

            # 使用模型进行前向传播，获取预测结果
            model_outputs = model(obs_image, goal_image)

            # 将标签转移到指定设备
            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            # 解包模型输出
            dist_pred, action_pred = model_outputs

            # 计算损失
            losses = _compute_losses(
                dist_label=dist_label,
                action_label=action_label,
                dist_pred=dist_pred,
                action_pred=action_pred,
                alpha=alpha,
                learn_angle=learn_angle,
                action_mask=action_mask,
            )

            # 记录损失值
            for key, value in losses.items():
                if key in loggers:
                    logger = loggers[key]
                    logger.log_data(value.item())

    # 将数据记录到wandb/控制台，并从最后一批选择可视化内容
    _log_data(
        i=i,
        epoch=epoch,
        num_batches=num_batches,
        normalized=normalized,
        project_folder=project_folder,
        num_images_log=num_images_log,
        loggers=loggers,
        obs_image=viz_obs_image,
        goal_image=viz_goal_image,
        action_pred=action_pred,
        action_label=action_label,
        goal_pos=goal_pos,
        dist_pred=dist_pred,
        dist_label=dist_label,
        dataset_index=dataset_index,
        use_wandb=use_wandb,
        mode=eval_type,
        use_latest=False,
        wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()

# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
):
    """
    计算距离和动作预测的损失。

    参数：
        ema_model: 指数移动平均模型，用于生成预测。
        noise_scheduler: 噪声调度器，用于处理输入噪声。
        batch_obs_images: 批次观察图像。
        batch_goal_images: 批次目标图像。
        batch_dist_label (torch.Tensor): 距离标签。
        batch_action_label (torch.Tensor): 动作标签。
        device (torch.device): 计算所用的设备。
        action_mask (torch.Tensor): 动作掩码，用于过滤无效输入。
    """

    pred_horizon = batch_action_label.shape[1]  # 预测时间范围
    action_dim = batch_action_label.shape[2]     # 动作维度

    # 通过模型生成输出
    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
    )
    
    uc_actions = model_output_dict['uc_actions']  # 无条件动作
    gc_actions = model_output_dict['gc_actions']  # 条件动作
    gc_distance = model_output_dict['gc_distance']  # 条件距离

    # 计算条件距离损失
    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # 对非批次维度进行归约，以获得每个批次元素的损失
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # 屏蔽无效输入（对于负样本或当观察与目标之间的距离较大时）
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    # 计算无条件和条件动作损失
    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    # 计算无条件和条件动作的余弦相似度
    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results
def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    训练模型一个epoch。

    参数:
        model: 要训练的模型
        ema_model: 指数移动平均模型
        optimizer: 使用的优化器
        dataloader: 用于训练的数据加载器
        transform: 应用的图像变换
        device: 使用的设备
        noise_scheduler: 用于训练的噪声调度器 
        project_folder: 保存图像的文件夹
        epoch: 当前的epoch
        alpha: 动作损失的权重
        print_log_freq: 打印损失的频率
        image_log_freq: 日志记录图像的频率
        num_images_log: 日志记录的图像数量
        use_wandb: 是否使用wandb进行日志记录
    """
    # 限制goal_mask_prob在[0, 1]之间
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()  # 设置模型为训练模式
    num_batches = len(dataloader)  # 获取批次数

    # 初始化多个Logger以记录不同类型的损失和相似性
    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data
            
            # 将观察图像分割成多个通道，并调整大小
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]  # 批次大小

            # 生成随机目标掩码
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
            
            # 获取距离标签
            distance = distance.float().to(device)

            deltas = get_delta(actions)  # 获取动作增量
            ndeltas = normalize_data(deltas, ACTION_STATS)  # 标准化数据
            naction = from_numpy(ndeltas).to(device)  # 转换为张量并转移到设备上
            assert naction.shape[-1] == 2, "action dim must be 2"  # 确保动作维度为2

            # 预测距离
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)  # 计算均方误差损失
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 +(1 - goal_mask.float()).mean())  # 加权损失

            # 为动作添加噪声
            noise = torch.randn(naction.shape, device=device)

            # 为每个数据点采样扩散迭代
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # 根据每个扩散迭代的噪声幅度将噪声添加到干净的图像中
            noisy_action = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # 预测噪声残差
            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                # 在非批处理维度上减少，以获取每个批处理元素的损失
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

            # L2损失
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # 总损失
            loss = alpha * dist_loss + (1-alpha) * diffusion_loss

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新模型权重的指数移动平均
            ema_model.step(model)

            # 日志记录
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                wandb.log({"total_loss": loss_cpu})
                wandb.log({"dist_loss": dist_loss.item()})
                wandb.log({"diffusion_loss": diffusion_loss.item()})
            elif use_wandb == False and i % print_log_freq == 0 and print_log_freq != 0:
                print(f"Total Loss: {loss_cpu}")
                print(f"Dist Loss: {dist_loss.item()}")
                print(f"Diffusion Loss: {diffusion_loss.item()}")


            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                            ema_model.averaged_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    "train",
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    """
    在给定评估数据集上评估模型。

    参数:
        eval_type (string): f"{data_type}_{eval_type}"（例如 "recon_train", "gs_test"等）
        ema_model (nn.Module): 要评估的模型的指数移动平均版本
        dataloader (DataLoader): 用于评估的数据加载器
        transform (transforms): 应用于图像的变换
        device (torch.device): 用于评估的设备
        noise_scheduler: 用于评估的噪声调度器 
        project_folder (string): 项目文件夹路径
        epoch (int): 当前的epoch
        print_log_freq (int): 打印日志的频率 
        wandb_log_freq (int): 向wandb记录的频率
        image_log_freq (int): 日志记录图像的频率
        alpha (float): 动作损失的权重
        num_images_log (int): 日志记录的图像数量
        eval_fraction (float): 用于评估的数据比例
        use_wandb (bool): 是否使用wandb进行日志记录
    """
    # 限制goal_mask_prob在[0, 1]之间
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model = ema_model.averaged_model  # 使用EMA模型
    ema_model.eval()  # 设置模型为评估模式
    
    num_batches = len(dataloader)  # 获取批次数

    # 初始化多个Logger以记录不同类型的损失和相似性
    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=print_log_freq)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=print_log_freq)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    
    num_batches = max(int(num_batches * eval_fraction), 1)  # 计算实际评估的批次数

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches), 
        total=num_batches, 
        dynamic_ncols=True, 
        desc=f"Evaluating {eval_type} for epoch {epoch}", 
        leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask,
            ) = data
            
            # 将观察图像分割成多个通道，并调整大小
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)

            B = actions.shape[0]  # 批次大小

            # 生成随机目标掩码
            rand_goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            goal_mask = torch.ones_like(rand_goal_mask).long().to(device)
            no_mask = torch.zeros_like(rand_goal_mask).long().to(device)

            # 通过随机掩码条件编码输入
            rand_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=rand_goal_mask)

            # 无掩码条件编码输入
            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
            obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

            # 通过目标掩码条件编码输入
            goal_mask_cond = ema_model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)

            distance = distance.to(device)

            deltas = get_delta(actions)  # 获取动作增量
            ndeltas = normalize_data(deltas, ACTION_STATS)  # 标准化数据
            naction = from_numpy(ndeltas).to(device)  # 转换为张量并转移到设备上
            assert naction.shape[-1] == 2, "action dim must be 2"  # 确保动作维度为2

            # 为动作添加噪声
            noise = torch.randn(naction.shape, device=device)

            # 为每个数据点采样扩散迭代
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # 添加噪声到动作
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            ### 随机掩码错误 ###
            # 预测噪声残差
            rand_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=rand_mask_cond)
            
            # L2损失
            rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)
            
            ### 无掩码错误 ###
            # 预测噪声残差
            no_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=obsgoal_cond)
            
            # L2损失
            no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

            ### 目标掩码错误 ###
            # 预测噪声残差
            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps, global_cond=goal_mask_cond)
            
            # L2损失
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)
            
            # 日志记录
            loss_cpu = rand_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb  and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                wandb.log({"diffusion_eval_loss (random masking)": rand_mask_loss})
                wandb.log({"diffusion_eval_loss (no masking)": no_mask_loss})
                wandb.log({"diffusion_eval_loss (goal masking)": goal_mask_loss})
            elif use_wandb == False and i % print_log_freq == 0 and print_log_freq != 0:
                print(f"diffusion_eval_loss (random masking): {rand_mask_loss}")
                print(f"diffusion_eval_loss (no masking): {no_mask_loss}")
                print(f"diffusion_eval_loss (goal masking): {goal_mask_loss}")


            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_nomad(
                            ema_model,
                            noise_scheduler,
                            batch_obs_images,
                            batch_goal_images,
                            distance.to(device),
                            actions.to(device),
                            device,
                            action_mask.to(device),
                        )
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_diffusion_action_distribution(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    batch_goal_images,
                    batch_viz_obs_images,
                    batch_viz_goal_images,
                    actions,
                    distance,
                    goal_pos,
                    device,
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    30,
                    use_wandb,
                )
# 归一化数据
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])  # 将数据重塑为二维数组，行数不固定，列数为最后一个维度的大小
    stats = {
        'min': np.min(data, axis=0),  # 计算每个特征的最小值
        'max': np.max(data, axis=0)   # 计算每个特征的最大值
    }
    return stats

def normalize_data(data, stats):
    # 将数据归一化到 [0,1] 范围内
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # 将数据归一化到 [-1, 1] 范围内
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2  # 将数据从 [-1, 1] 转换回 [0, 1]
    data = ndata * (stats['max'] - stats['min']) + stats['min']  # 恢复原始数据范围
    return data

def get_delta(actions):
    # 在第一个动作前添加零向量
    ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]  # 计算相邻动作之间的差异
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)，表示扩散模型输出的动作序列
    # 返回: (B, T-1)，即去掉第一帧后的动作序列
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)  # 重塑形状以便处理
    ndeltas = to_numpy(ndeltas)  # 转换为 NumPy 数组
    ndeltas = unnormalize_data(ndeltas, action_stats)  # 反归一化
    actions = np.cumsum(ndeltas, axis=1)  # 计算累积和得到实际动作
    return from_numpy(actions).to(device)  # 转换回张量并返回

def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)  # 创建目标掩码
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)  # 编码观察图像和目标图像
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)  # 扩展条件以适应样本数量

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)  # 创建无掩码
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)  # 编码没有目标掩码的观察和目标图像
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)  # 同样扩展条件

    # 从高斯噪声初始化动作
    noisy_diffusion_output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)  # 随机生成初始噪声
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:  # 遍历所有时间步
        # 预测噪声
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # 逆扩散步骤（去除噪声）
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)  # 获取未条件化的动作

    # 再次从高斯噪声初始化动作
    noisy_diffusion_output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        # 预测噪声
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )

        # 逆扩散步骤（去除噪声）
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  # 展平条件
    gc_actions = get_action(diffusion_output, ACTION_STATS)  # 获取条件化的动作
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)  # 预测距离

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
):
    """绘制来自探索模型的样本。"""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):  # 如果可视化路径不存在，则创建
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]  # 限制批次大小
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []  # 用于存储 WandB 图像列表

    pred_horizon = batch_action_label.shape[1]  # 动作预测的时间跨度
    action_dim = batch_action_label.shape[2]  # 动作的维度

    # 按批次拆分
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))  # 收集未条件化的动作
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))  # 收集条件化的动作
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))  # 收集条件化的距离

    # 合并结果
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # 拆分为每个观察的动作
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]  # 计算平均距离
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]  # 计算标准差

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log  # 确保长度一致

    np_distance_labels = to_numpy(batch_distance_labels)  # 转换距离标签为 NumPy 数组

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)  # 创建子图
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)  # 合并轨迹
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]  # 设置颜色
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]  # 设置透明度

        # 创建机器人位置 (0, 0) 和目标位置的点数组
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # 将通道移动到最后一个维度
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)  # 显示观察图像
        ax[2].imshow(goal_image)  # 显示目标图像

        # 设置标题
        ax[0].set_title(f"扩散动作预测")
        ax[1].set_title(f"观察")
        ax[2].set_title(f"目标: 标签={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # 调整图表大小
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")  # 保存路径
        plt.savefig(save_path)  # 保存图像
        wandb_list.append(wandb.Image(save_path))  # 添加到 WandB 列表
        plt.close(fig)  # 关闭图表
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)  # 记录到 WandB
