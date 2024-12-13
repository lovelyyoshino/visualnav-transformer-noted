import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"  # 顶图像目录
MODEL_WEIGHTS_PATH = "../model_weights"  # 模型权重路径
ROBOT_CONFIG_PATH ="../config/robot.yaml"  # 机器人配置文件路径
MODEL_CONFIG_PATH = "../config/models.yaml"  # 模型配置文件路径
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)  # 加载机器人配置
MAX_V = robot_config["max_v"]  # 最大线速度
MAX_W = robot_config["max_w"]  # 最大角速度
RATE = robot_config["frame_rate"]  # 帧率

# GLOBALS
context_queue = []  # 上下文队列，用于存储观察到的图像
context_size = None  # 上下文大小
subgoal = []  # 子目标列表

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否使用GPU
print("Using device:", device)

def callback_obs(msg):
    """
    回调函数，处理接收到的图像消息并将其添加到上下文队列中。
    
    参数:
        msg: 接收到的ROS图像消息
    """
    obs_img = msg_to_pil(msg)  # 将ROS图像消息转换为PIL图像
    if context_size is not None:
        if len(context_queue) < context_size + 1:  # 如果上下文队列未满
            context_queue.append(obs_img)  # 添加新图像
        else:
            context_queue.pop(0)  # 移除最旧的图像
            context_queue.append(obs_img)  # 添加新图像


def main(args: argparse.Namespace):
    global context_size

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)  # 加载模型参数

    model_config_path = model_paths[args.model]["config_path"]  # 获取模型配置路径
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)  # 加载模型参数

    context_size = model_params["context_size"]  # 设置上下文大小

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]  # 获取模型权重路径
    if os.path.exists(ckpth_path):  # 检查权重文件是否存在
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")  # 抛出异常
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置模型为评估模式

    # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))  # 按照节点顺序加载顶图
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"  # 顶图目录
    num_nodes = len(os.listdir(topomap_dir))  # 节点数量
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])  # 构建每个顶图的路径
        topomap.append(PILImage.open(image_path))  # 打开并添加顶图

    closest_node = 0  # 最近节点索引
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"  # 确保目标节点有效
    if args.goal_node == -1:
        goal_node = len(topomap) - 1  # 如果目标节点为-1，则设置为最后一个节点
    else:
        goal_node = args.goal_node  # 否则设置为用户提供的目标节点
    reached_goal = False  # 是否达到目标标志

    # ROS
    rospy.init_node("EXPLORATION", anonymous=False)  # 初始化ROS节点
    rate = rospy.Rate(RATE)  # 设置循环频率
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)  # 订阅图像话题
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  # 发布航点话题
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)  # 发布采样动作话题
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)  # 发布目标到达状态

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]  # 获取扩散迭代次数
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )  # 初始化噪声调度器
    
    # navigation loop
    while not rospy.is_shutdown():  # 当ROS未关闭时循环
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)  # 初始化选择的航点
        if len(context_queue) > model_params["context_size"]:  # 如果上下文队列足够长
            if model_params["model_type"] == "nomad":  # 使用nomad模型
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)  # 转换观察图像
                obs_images = torch.split(obs_images, 3, dim=1)  # 分割图像通道
                obs_images = torch.cat(obs_images, dim=1)  # 合并图像
                obs_images = obs_images.to(device)  # 移动到设备
                mask = torch.zeros(1).long().to(device)  # 创建输入掩码

                start = max(closest_node - args.radius, 0)  # 定义搜索起始节点
                end = min(closest_node + args.radius + 1, goal_node)  # 定义搜索结束节点
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]  # 转换目标图像
                goal_image = torch.concat(goal_image, dim=0)  # 合并目标图像

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))  # 编码观察和目标条件
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)  # 预测距离
                dists = to_numpy(dists.flatten())  # 转换为numpy数组
                min_idx = np.argmin(dists)  # 找到最近节点索引
                closest_node = min_idx + start  # 更新最近节点
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)  # 计算子目标索引
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)  # 获取对应的观察条件

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)  # 重复以适应样本数
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)  # 重复以适应样本数
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)  # 从高斯噪声初始化动作
                    naction = noisy_action  # 保存当前动作

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)  # 设置时间步

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:  # 遍历所有时间步
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )  # 预测噪声
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample  # 执行逆扩散步骤
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))  # 获取最终动作
                sampled_actions_msg = Float32MultiArray()  # 创建消息对象
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))  # 拼接数据
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)  # 发布采样动作
                naction = naction[0]  # 选择第一个动作
                chosen_waypoint = naction[args.waypoint]  # 选择航点
            else:  # 如果不是nomad模型
                start = max(closest_node - args.radius, 0)  # 定义搜索起始节点
                end = min(closest_node + args.radius + 1, goal_node)  # 定义搜索结束节点
                distances = []  # 存储距离
                waypoints = []  # 存储航点
                batch_obs_imgs = []  # 批量观察图像
                batch_goal_data = []  # 批量目标数据
                for i, sg_img in enumerate(topomap[start: end + 1]):  # 遍历目标图像
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])  # 转换观察图像
                    goal_data = transform_images(sg_img, model_params["image_size"])  # 转换目标图像
                    batch_obs_imgs.append(transf_obs_img)  # 添加到批量观察图像
                    batch_goal_data.append(goal_data)  # 添加到批量目标数据
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)  # 合并并移动到设备
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)  # 合并并移动到设备

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)  # 预测距离和航点
                distances = to_numpy(distances)  # 转换为numpy数组
                waypoints = to_numpy(waypoints)  # 转换为numpy数组
                # look for closest node
                min_dist_idx = np.argmin(distances)  # 找到最近节点索引
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:  # 如果距离大于阈值
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]  # 选择航点
                    closest_node = start + min_dist_idx  # 更新最近节点
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]  # 选择下一个航点
                    closest_node = min(start + min_dist_idx + 1, goal_node)  # 更新最近节点
        
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  # 正规化航点
        waypoint_msg = Float32MultiArray()  # 创建航点消息
        waypoint_msg.data = chosen_waypoint  # 设置航点数据
        waypoint_pub.publish(waypoint_msg)  # 发布航点
        reached_goal = closest_node == goal_node  # 检查是否到达目标
        goal_pub.publish(reached_goal)  # 发布目标到达状态
        if reached_goal:
            print("Reached goal! Stopping...")  # 输出到达目标信息
        rate.sleep()  # 等待下一次循环


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")  # 命令行参数解析器
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # 选择默认航点
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",  # 顶图像路径
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",  # 从探索模型中采样的动作数量
    )
    args = parser.parse_args()  # 解析命令行参数
    print(f"Using {device}")  # 输出使用的设备
    main(args)  # 调用主函数
