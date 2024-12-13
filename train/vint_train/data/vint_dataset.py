import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
    ):
        """
        主ViNT数据集类

        参数:
            data_folder (string): 所有图像数据的目录
            data_split_folder (string): 包含filepaths.txt的目录，该文件列出了数据集中所有轨迹名称，每行一个
            dataset_name (string): 数据集名称 [recon, go_stanford, scand, tartandrive等]
            waypoint_spacing (int): 路径点之间的间距
            min_dist_cat (int): 使用的最小距离类别
            max_dist_cat (int): 使用的最大距离类别
            negative_mining (bool): 是否使用ViNG论文中的负采样（Shah et al.）(https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): 如果这是一个动作数据集，则预测路径点的长度
            learn_angle (bool): 如果这是一个动作数据集，是否学习每个预测路径点的机器人偏航角
            context_size (int): 用作上下文的先前观察数量
            context_type (str): 是否使用时间、随机或随机时间上下文
            end_slack (int): 在轨迹末尾忽略的时间步数
            goals_per_obs (int): 每个观察中要采样的目标数量
            normalize (bool): 是否对距离或动作进行归一化
            goal_type (str): 要用于目标的数据类型。目前仅支持“image”。
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type必须是temporal、randomized或randomized_temporal之一"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # 加载数据/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"在data_config.yaml中未找到数据集{self.dataset_name}"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # 使用此索引从data_config.yaml检索数据集名称
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()  # 加载索引
        self._build_caches()  # 构建缓存
        
        if self.learn_angle:
            self.num_action_params = 3  # 学习角度时参数为3
        else:
            self.num_action_params = 2  # 不学习角度时参数为2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None  # 不序列化图像缓存
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()  # 重建缓存

    def _build_caches(self, use_tqdm: bool = True):
        """
        使用LMDB构建图像缓存以加快加载速度
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # 将所有轨迹加载到内存中。这些应该已经被加载，但以防万一。
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        如果缓存文件不存在，通过遍历数据集并将每个图像写入缓存来创建它
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"正在为{self.dataset_name}构建LMDB缓存"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # 以只读模式重新打开缓存文件
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        构建一个包含元组(轨迹名称, 时间, 最大目标距离)的索引
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        从未来的同一路径中抽样一个目标。
        返回: (轨迹名称, 目标时间, 目标是否为负样本)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()  # 抽样负目标
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        从（可能）不同的轨迹中抽样一个目标。
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        为数据集中每个观察生成元组列表(obs_traj_name, goal_traj_name, obs_time, goal_time)
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # 如果index_to_data已存在则加载（节省时间）
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # 如果index_to_data文件不存在，则创建它
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)  # 转换图像数据
        except TypeError:
            print(f"无法加载图像 {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]  # 获取偏航角
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]  # 获取位置
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]  # 获取目标位置

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)  # 压缩维度

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])  # 填充偏航角
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)  # 填充位置

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} 和 {(self.len_traj_pred + 1,)} 应该相等"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} 和 {(self.len_traj_pred + 1, 2)} 应该相等"

        waypoints = to_local_coords(positions, positions[0], yaw[0])  # 转换为局部坐标系
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])  # 转换目标位置为局部坐标

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} 和 {(self.len_traj_pred + 1, 2)} 应该相等"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]  # 计算相对于第一个点的偏航变化
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)  # 合并动作
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing  # 归一化动作
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing  # 归一化目标位置

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} 和 {(self.len_traj_pred, self.num_action_params)} 应该相等"

        return actions, goal_pos  # 返回动作和目标位置
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]  # 从缓存获取轨迹
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)  # 从文件加载轨迹数据
            self.trajectory_cache[trajectory_name] = traj_data  # 缓存轨迹数据
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)  # 返回数据集大小

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        参数:
            i (int): 第i个数据点的索引
        返回:
            包含上下文、观察、目标、转换后的上下文、转换后的观察、转换后的目标、距离标签和动作标签的张量元组
                obs_image (torch.Tensor): 形状为[3, H, W]的张量，包含机器人的观察图像
                goal_image (torch.Tensor): 形状为[3, H, W]的张量，包含子目标图像 
                dist_label (torch.Tensor): 形状为(1,)的张量，包含观察与目标之间的距离标签
                action_label (torch.Tensor): 形状为(5, 2)或(5, 4)（如果训练时考虑角度），包含观察与目标之间的动作标签
                which_dataset (torch.Tensor): 数据集中数据点的索引[用于可视化多个数据集时识别数据集]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)  # 抽样目标

        # 加载图像
        context = []
        if self.context_type == "temporal":
            # 从区间[0, curr_time)中采样最后self.context_size次
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]  # 创建上下文

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context  # 加载上下文图像
        ])

        # 加载目标图像
        goal_image = self._load_image(f_goal, goal_time)

        # 加载其他轨迹数据
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} 和 {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} 和 {goal_traj_len}"

        # 计算动作
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # 计算距离
        if goal_is_negative:
            distance = self.max_dist_cat  # 负样本距离设为最大值
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing  # 正样本距离
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} 和 {curr_time} 应该相隔{self.waypoint_spacing}的整数倍"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)  # 计算正弦余弦表示
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)  # 动作掩码条件
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
