import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List
import wandb
import yaml
import torch
import torch.nn as nn
from vint_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

# 加载数据配置文件 data_config.yaml
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def visualize_traj_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    使用自我中心可视化比较预测路径与真实路径的关键点。这种可视化用于数据集中的最后一个批次。

    参数：
        batch_obs_images (np.ndarray): 观察图像的批次 [batch_size, height, width, channels]
        batch_goal_images (np.ndarray): 目标图像的批次 [batch_size, height, width, channels]
        dataset_names: 对应于数据集名称的索引
        batch_goals (np.ndarray): 目标位置的批次 [batch_size, 2]
        batch_pred_waypoints (np.ndarray): 预测的关键点的批次 [batch_size, horizon, 4] 或 [batch_size, horizon, 2] 或 [batch_size, num_trajs_sampled horizon, {2 or 4}]
        batch_label_waypoints (np.ndarray): 标签关键点的批次 [batch_size, T, 4] 或 [batch_size, horizon, 2]
        eval_type (string): f"{data_type}_{eval_type}" (例如 "recon_train", "gs_test" 等)
        normalized (bool): 关键点是否经过归一化处理
        save_folder (str): 保存图像的文件夹。如果为 None，则不保存图像
        epoch (int): 当前的训练轮数
        num_images_preds (int): 要可视化的图像数量
        use_wandb (bool): 是否使用 wandb 来记录图像
        display (bool): 是否显示图像
    """
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    # 创建保存目录
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    # 确保输入数组长度一致
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
    )

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        obs_img = numpy_to_img(batch_obs_images[i])  # 将观察图像转换为numpy格式
        goal_img = numpy_to_img(batch_goal_images[i])  # 将目标图像转换为numpy格式
        dataset_name = dataset_names[int(dataset_indices[i])]  # 获取数据集名称
        goal_pos = batch_goals[i]  # 获取目标位置
        pred_waypoints = batch_pred_waypoints[i]  # 获取预测的关键点
        label_waypoints = batch_label_waypoints[i]  # 获取标签的关键点

        # 如果关键点是归一化的，进行反归一化处理
        if normalized:
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        # 比较预测的关键点和标签的关键点
        compare_waypoints_pred_to_label(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))  # 记录到wandb中
    if use_wandb:
        wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)  # 提交日志


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    使用自我中心可视化比较预测路径与真实路径的关键点。

    参数：
        obs_img: 观察图像
        goal_img: 目标图像
        dataset_name: 在 data_config.yaml 中找到的数据集名称（例如 "recon"）
        goal_pos: 图像中的目标位置
        pred_waypoints: 图像中的预测关键点
        label_waypoints: 图像中的标签关键点
        save_path: 保存图形的路径
        display: 是否显示图形
    """

    fig, ax = plt.subplots(1, 3)  # 创建子图
    start_pos = np.array([0, 0])  # 起始位置
    if len(pred_waypoints.shape) > 2:
        trajs = [*pred_waypoints, label_waypoints]  # 包含多个轨迹
    else:
        trajs = [pred_waypoints, label_waypoints]  # 单个轨迹

    # 绘制轨迹和点
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    ax[2].imshow(goal_img)  # 显示目标图像

    fig.set_size_inches(18.5, 10.5)  # 设置图形大小
    ax[0].set_title(f"Action Prediction")  # 设置标题
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )  # 保存图形

    if not display:
        plt.close(fig)  # 不显示则关闭图形


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
):
    """
    在图像上绘制轨迹和点。如果没有数据集的相机内参配置，则图像将按原样绘制。
    
    参数：
        ax: matplotlib 轴
        img: 要绘制的图像
        dataset_name: 在 data_config.yaml 中找到的数据集名称（例如 "recon"）
        list_trajs: 轨迹列表，每条轨迹是形状为 (horizon, 2)（如果没有偏航）或 (horizon, 4)（如果有偏航）的numpy数组
        list_points: 点的列表，每个点是形状为 (2,) 的numpy数组
        traj_colors: 轨迹颜色列表
        point_colors: 点颜色列表
    """
    assert len(list_trajs) <= len(traj_colors), "轨迹颜色不足"
    assert len(list_points) <= len(point_colors), "点颜色不足"
    assert (
        dataset_name in data_config
    ), f"未在 data/data_config.yaml 中找到数据集 {dataset_name}"

    ax.imshow(img)  # 显示图像
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]  # 相机高度
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]  # 相机x方向偏移

        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]  # 焦距x
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]  # 焦距y
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]  # 主点x坐标
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]  # 主点y坐标
        camera_matrix = gen_camera_matrix(fx, fy, cx, cy)  # 生成相机矩阵

        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]  # 畸变系数k1
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]  # 畸变系数k2
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]  # 畸变系数p1
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]  # 畸变系数p2
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]  # 畸变系数k3
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])  # 畸变系数数组

        for i, traj in enumerate(list_trajs):
            xy_coords = traj[:, :2]  # (horizon, 2)
            traj_pixels = get_pos_pixels(
                xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
            )  # 获取像素位置
            if len(traj_pixels.shape) == 2:
                ax.plot(
                    traj_pixels[:250, 0],
                    traj_pixels[:250, 1],
                    color=traj_colors[i],
                    lw=2.5,
                )  # 绘制轨迹

        for i, point in enumerate(list_points):
            if len(point.shape) == 1:
                # 为点添加维度
                point = point[None, :2]
            else:
                point = point[:, :2]
            pt_pixels = get_pos_pixels(
                point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
            )  # 获取点的像素位置
            ax.plot(
                pt_pixels[:250, 0],
                pt_pixels[:250, 1],
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )  # 绘制点
        ax.xaxis.set_visible(False)  # 隐藏x轴
        ax.yaxis.set_visible(False)  # 隐藏y轴
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))  # 设置x轴范围
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))  # 设置y轴范围


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    绘制可能具有偏航的轨迹和点。

    参数：
        ax: matplotlib 轴
        list_trajs: 轨迹列表，每条轨迹是形状为 (horizon, 2)（如果没有偏航）或 (horizon, 4)（如果有偏航）的numpy数组
        list_points: 点的列表，每个点是形状为 (2,) 的numpy数组
        traj_colors: 轨迹颜色列表
        point_colors: 点颜色列表
        traj_labels: 轨迹标签列表
        point_labels: 点标签列表
        traj_alphas: 轨迹透明度列表
        point_alphas: 点透明度列表
        quiver_freq: 向量场频率（如果轨迹数据包含机器人的偏航）
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "轨迹颜色不足"
    assert len(list_points) <= len(point_colors), "点颜色不足"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "轨迹标签不足"
    assert point_labels is None or len(list_points) == len(point_labels), "点标签不足"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # 轨迹数据也包括机器人的偏航
            bearings = gen_bearings_from_waypoints(traj)  # 从关键点生成朝向
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    
    # 将图例放置在图下方
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def angle_to_unit_vector(theta):
    """将角度转换为单位向量。"""
    return np.array([np.cos(theta), np.sin(theta)])


def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """从关键点生成朝向，(x, y, sin(theta), cos(theta))。"""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # 标签是sin/cos表示
            v = waypoints[i, 2:]
            # 归一化v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # 标签是弧度表示
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    使用提供的相机参数将3D坐标投影到2D图像平面。

    参数：
        xy: 形状为 (batch_size, horizon, 2) 的数组，表示 (x, y) 坐标
        camera_height: 相机离地面的高度（以米为单位）
        camera_x_offset: 相机相对于汽车中心的偏移（以米为单位）
        camera_matrix: 表示相机内参的3x3矩阵
        dist_coeffs: 畸变系数向量

    返回：
        uv: 形状为 (batch_size, horizon, 2) 的数组，表示2D图像平面上的 (u, v) 坐标
    """
    batch_size, horizon, _ = xy.shape

    # 创建3D坐标，相机位于给定高度
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # 创建虚拟旋转和平移向量
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset  # 应用相机偏移
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)  # 转换坐标系
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)  # 重塑输出形状

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    使用提供的相机参数将3D坐标投影到2D图像平面。
    
    参数：
        points: 形状为 (batch_size, horizon, 2) 的数组，表示 (x, y) 坐标
        camera_height: 相机离地面的高度（以米为单位）
        camera_x_offset: 相机相对于汽车中心的偏移（以米为单位）
        camera_matrix: 表示相机内参的3x3矩阵
        dist_coeffs: 畸变系数向量

    返回：
        pixels: 形状为 (batch_size, horizon, 2) 的数组，表示2D图像平面上的 (u, v) 坐标
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]  # 翻转x坐标
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    生成相机内参矩阵。

    参数：
        fx: x方向的焦距
        fy: y方向的焦距
        cx: 主点x坐标
        cy: 主点y坐标
    
    返回：
        相机矩阵
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
