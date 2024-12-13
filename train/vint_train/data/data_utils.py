import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (160, 120)  # 可视化图像的大小
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # 所有图像在训练中都被中心裁剪为4:3的宽高比


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    """
    获取数据文件路径

    Args:
        data_folder (str): 数据文件夹路径
        f (str): 文件名
        time (int): 时间戳
        data_type (str): 数据类型，默认为'image'

    Returns:
        str: 完整的数据文件路径
    """
    data_ext = {
        "image": ".jpg",
        # 可以在这里添加更多数据类型
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    """
    根据偏航角生成旋转矩阵

    Args:
        yaw (float): 偏航角（弧度）

    Returns:
        np.ndarray: 旋转矩阵
    """
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    将全局坐标转换为局部坐标

    Args:
        positions (np.ndarray): 要转换的位置数组
        curr_pos (np.ndarray): 当前的位置
        curr_yaw (float): 当前的偏航角

    Returns:
        np.ndarray: 局部坐标系中的位置
    """
    rotmat = yaw_rotmat(curr_yaw)  # 计算当前偏航角的旋转矩阵
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]  # 如果是二维坐标，只取前两行和前两列
    elif positions.shape[-1] == 3:
        pass  # 三维坐标保持不变
    else:
        raise ValueError("Invalid position shape")  # 抛出异常以处理无效形状

    return (positions - curr_pos).dot(rotmat)  # 转换到局部坐标系


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    计算路径点之间的增量

    Args:
        waypoints (torch.Tensor): 路径点张量

    Returns:
        torch.Tensor: 增量张量
    """
    num_params = waypoints.shape[1]  # 获取路径点的参数数量
    origin = torch.zeros(1, num_params)  # 创建一个原点
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)  # 前一个路径点
    deltas = waypoints - prev_waypoints  # 计算增量
    if num_params > 2:
        return calculate_sin_cos(deltas)  # 如果参数大于2，计算正弦和余弦
    return deltas  # 返回增量


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    计算路径点的角度的正弦和余弦值

    Args:
        waypoints (torch.Tensor): 路径点张量

    Returns:
        torch.Tensor: 包含正弦和余弦的路径点张量
    """
    assert waypoints.shape[1] == 3  # 确保输入的路径点具有三个参数
    angle_repr = torch.zeros_like(waypoints[:, :2])  # 初始化角度表示
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])  # 计算余弦
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])  # 计算正弦
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)  # 合并结果


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    对图像进行变换和调整大小

    Args:
        img (Image.Image): 输入图像
        transform (transforms): 图像变换操作
        image_resize_size (Tuple[int, int]): 调整后的图像大小
        aspect_ratio (float): 宽高比，默认为4:3

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 可视化图像和经过变换的图像
    """
    w, h = img.size  # 获取图像的宽和高
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # 按照高度裁剪以保持比例
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))  # 按照宽度裁剪以保持比例
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)  # 调整可视化图像大小
    viz_img = TF.to_tensor(viz_img)  # 转换为张量
    img = img.resize(image_resize_size)  # 调整图像大小
    transf_img = transform(img)  # 应用变换
    return viz_img, transf_img  # 返回可视化图像和变换后的图像


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    调整图像大小并按比例裁剪

    Args:
        img (Image.Image): 输入图像
        image_resize_size (Tuple[int, int]): 调整后的图像大小
        aspect_ratio (float): 宽高比，默认为4:3

    Returns:
        torch.Tensor: 调整后图像的张量
    """
    w, h = img.size  # 获取图像的宽和高
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # 按照高度裁剪以保持比例
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))  # 按照宽度裁剪以保持比例
    img = img.resize(image_resize_size)  # 调整图像大小
    resize_img = TF.to_tensor(img)  # 转换为张量
    return resize_img  # 返回调整后的图像张量


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    从路径加载图像并进行变换

    Args:
        path (Union[str, io.BytesIO]): 图像路径或字节流
        image_resize_size (Tuple[int, int]): 调整后的图像大小

    Returns:
        torch.Tensor: 调整后的图像作为张量
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)  # 加载图像并调整大小
