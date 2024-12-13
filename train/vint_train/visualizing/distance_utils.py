import os
import wandb
import numpy as np
from typing import List, Optional, Tuple
from vint_train.visualizing.visualize_utils import numpy_to_img
import matplotlib.pyplot as plt


def visualize_dist_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    batch_dist_preds: np.ndarray,
    batch_dist_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
    dist_error_threshold: float = 3.0,
):
    """
    可视化观察-目标图像对的距离分类预测和标签。

    参数:
        batch_obs_images (np.ndarray): 观察图像批次 [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): 目标图像批次 [batch_size, height, width, channels]
        batch_dist_preds (np.ndarray): 距离预测批次 [batch_size]
        batch_dist_labels (np.ndarray): 距离标签批次 [batch_size]
        eval_type (string): {data_type}_{eval_type}（例如 recon_train, gs_test 等）
        epoch (int): 当前训练轮数
        num_images_preds (int): 要可视化的图像数量
        use_wandb (bool): 是否使用 wandb 来记录图像
        save_folder (str): 保存图像的文件夹。如果为 None，则不保存图像
        display (bool): 是否显示图像
        rounding (int): 四舍五入到小数点后的位数
        dist_error_threshold (float): 用于将距离预测分类为正确或错误的阈值（仅用于可视化目的）
    """
    # 创建可视化路径
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)  # 如果目录不存在则创建

    # 确保所有输入数组长度相同
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_dist_preds)
        == len(batch_dist_labels)
    )

    batch_size = batch_obs_images.shape[0]  # 获取批次大小
    wandb_list = []  # 初始化 wandb 图像列表

    for i in range(min(batch_size, num_images_preds)):
        # 对预测和标签进行四舍五入
        dist_pred = np.round(batch_dist_preds[i], rounding)
        dist_label = np.round(batch_dist_labels[i], rounding)

        # 将 NumPy 数组转换为图像格式
        obs_image = numpy_to_img(batch_obs_images[i])
        goal_image = numpy_to_img(batch_goal_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")  # 设置保存路径

        text_color = "black"
        # 根据预测与标签之间的差异设置文本颜色
        if abs(dist_pred - dist_label) > dist_error_threshold:
            text_color = "red"

        # 显示距离预测结果
        display_distance_pred(
            [obs_image, goal_image],
            ["Observation", "Goal"],
            dist_pred,
            dist_label,
            text_color,
            save_path,
            display,
        )
        
        # 如果使用 wandb 则记录图像
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    
    # 使用 wandb 记录所有图像
    if use_wandb:
        wandb.log({f"{eval_type}_dist_prediction": wandb_list}, commit=False)


def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
):
    """
    可视化观察-目标图像对的成对距离分类预测和标签。

    参数:
        batch_obs_images (np.ndarray): 观察图像批次 [batch_size, height, width, channels]
        batch_close_images (np.ndarray): 近目标图像批次 [batch_size, height, width, channels]
        batch_far_images (np.ndarray): 远目标图像批次 [batch_size, height, width, channels]
        batch_close_preds (np.ndarray): 近目标预测批次 [batch_size]
        batch_far_preds (np.ndarray): 远目标预测批次 [batch_size]
        batch_close_labels (np.ndarray): 近目标标签批次 [batch_size]
        batch_far_labels (np.ndarray): 远目标标签批次 [batch_size]
        eval_type (string): {data_type}_{eval_type}（例如 recon_train, gs_test 等）
        save_folder (str): 保存图像的文件夹。如果为 None，则不保存图像
        epoch (int): 当前训练轮数
        num_images_preds (int): 要可视化的图像数量
        use_wandb (bool): 是否使用 wandb 来记录图像
        display (bool): 是否显示图像
        rounding (int): 四舍五入到小数点后的位数
    """
    # 创建可视化路径
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)  # 如果目录不存在则创建

    # 确保所有输入数组长度相同
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )

    batch_size = batch_obs_images.shape[0]  # 获取批次大小
    wandb_list = []  # 初始化 wandb 图像列表

    for i in range(min(batch_size, num_images_preds)):
        # 对预测和标签进行四舍五入
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)

        # 将 NumPy 数组转换为图像格式
        obs_image = numpy_to_img(batch_obs_images[i])
        close_image = numpy_to_img(batch_close_images[i])
        far_image = numpy_to_img(batch_far_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")  # 设置保存路径

        # 根据近和远预测的比较设置文本颜色
        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        # 显示成对距离预测结果
        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
        )
        
        # 如果使用 wandb 则记录图像
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    
    # 使用 wandb 记录所有图像
    if use_wandb:
        wandb.log({f"{eval_type}_pairwise_classification": wandb_list}, commit=False)


def display_distance_pred(
    imgs: list,
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    """
    显示距离预测及其对应的图像。

    参数:
        imgs (list): 要显示的图像列表
        titles (list): 每个图像的标题列表
        dist_pred (float): 距离预测值
        dist_label (float): 距离标签值
        text_color (str): 标题文本颜色
        save_path (Optional[str]): 保存图像的路径，如果为 None 则不保存
        display (bool): 是否直接显示图像
    """
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))  # 创建子图

    # 设置总标题，包括预测和标签信息
    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

    # 遍历图像并设置各自的标题
    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)  # 显示图像
        axis.set_title(title)  # 设置标题
        axis.xaxis.set_visible(False)  # 隐藏 x 轴
        axis.yaxis.set_visible(False)  # 隐藏 y 轴

    # 调整图形大小
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    # 如果提供了保存路径，则保存图像
    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)  # 如果不需要显示，则关闭图形
