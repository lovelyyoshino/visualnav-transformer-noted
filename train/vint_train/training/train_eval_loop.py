import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    训练和评估模型多个周期（vint或gnm模型）

    参数:
        train_model: 是否训练模型
        model: 要训练的模型
        optimizer: 使用的优化器
        scheduler: 使用的学习率调度器
        dataloader: 训练数据集的数据加载器
        test_dataloaders: 测试用的数据加载器字典
        transform: 应用于图像的变换
        epochs: 训练的轮数
        device: 训练所用设备
        project_folder: 保存检查点和日志的文件夹
        normalized: 是否对动作空间进行归一化
        wandb_log_freq: 日志记录到wandb的频率
        print_log_freq: 控制台打印的频率
        image_log_freq: 将图像记录到wandb的频率
        num_images_log: 记录到wandb的图像数量
        current_epoch: 从哪个epoch开始训练
        alpha: 距离损失与动作损失之间的权衡
        learn_angle: 是否学习角度
        use_wandb: 是否记录到wandb
        eval_fraction: 用于评估的训练数据比例
    """
    assert 0 <= alpha <= 1  # 确保alpha在合理范围内
    latest_path = os.path.join(project_folder, f"latest.pth")  # 最新模型保存路径

    for epoch in range(current_epoch, current_epoch + epochs):  # 遍历每个epoch
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            # 调用train函数进行训练
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        avg_total_test_loss = []  # 存储平均测试损失
        for dataset_type in test_dataloaders:  # 对每种测试数据集进行评估
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]  # 获取当前数据集的加载器

            # 调用evaluate函数进行评估
            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)  # 添加总评估损失

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),  # 平均测试损失
            "scheduler": scheduler
        }
        # 记录平均评估损失
        wandb.log({}, commit=False)

        if scheduler is not None:
            # 根据调度器类型调用相应的方法
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))  # 基于评估损失调整学习率
            else:
                scheduler.step()  # 普通步进更新学习率
        wandb.log({
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "lr": optimizer.param_groups[0]["lr"],  # 当前学习率
        }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")  # 每个epoch的模型保存路径
        torch.save(checkpoint, latest_path)  # 保存最新模型
        torch.save(checkpoint, numbered_path)  # 保存每个epoch的模型

    # 刷新最后一组评估日志
    wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    训练和评估模型多个周期（vint或gnm模型）

    参数:
        model: 要训练的模型
        optimizer: 使用的优化器
        lr_scheduler: 使用的学习率调度器
        noise_scheduler: 噪声调度器
        dataloader: 训练数据集的数据加载器
        test_dataloaders: 测试用的数据加载器字典
        transform: 应用于图像的变换
        goal_mask_prob: 在训练期间掩盖目标标记的概率
        epochs: 训练的轮数
        device: 训练所用设备
        project_folder: 保存检查点和日志的文件夹
        wandb_log_freq: 日志记录到wandb的频率
        print_log_freq: 控制台打印的频率
        image_log_freq: 将图像记录到wandb的频率
        num_images_log: 记录到wandb的图像数量
        current_epoch: 从哪个epoch开始训练
        alpha: 距离损失与动作损失之间的权衡
        use_wandb: 是否记录到wandb
        eval_fraction: 用于评估的训练数据比例
        eval_freq: 评估频率
    """
    latest_path = os.path.join(project_folder, f"latest.pth")  # 最新模型保存路径
    ema_model = EMAModel(model=model,power=0.75)  # 创建EMA模型
    
    for epoch in range(current_epoch, current_epoch + epochs):  # 遍历每个epoch
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            # 调用train_nomad函数进行训练
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()  # 更新学习率

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")  # EMA模型保存路径
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)  # 保存EMA模型
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")  # 模型保存路径
        torch.save(model.state_dict(), numbered_path)  # 保存模型
        torch.save(model.state_dict(), latest_path)  # 保存最新模型
        print(f"Saved model to {numbered_path}")

        # 保存优化器状态
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # 保存调度器状态
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0:  # 如果达到评估频率
            for dataset_type in test_dataloaders:  # 对每种测试数据集进行评估
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                # 调用evaluate_nomad进行评估
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],  # 当前学习率
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()  # 更新学习率

        # 记录平均评估损失
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # 刷新最后一组评估日志
    wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """从检查点加载模型。"""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)  # 加载nomad模型状态
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)  # 加载多GPU模型状态
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)  # 加载单GPU模型状态


def load_ema_model(ema_model, state_dict: dict) -> None:
    """从检查点加载EMA模型。"""
    ema_model.load_state_dict(state_dict)  # 加载EMA模型状态


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])  # 创建表格显示模块和参数数量
    total_params = 0  # 总参数计数
    for name, parameter in model.named_parameters():  # 遍历模型中的所有参数
        if not parameter.requires_grad: continue  # 跳过不需要梯度的参数
        params = parameter.numel()  # 获取参数数量
        table.add_row([name, params])  # 添加行到表格
        total_params+=params  # 累加参数数量
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")  # 打印可训练参数总数
    return total_params  # 返回总参数数量
