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

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否使用GPU
print("Using device:", device)

def callback_obs(msg):
    """
    回调函数，处理接收到的图像消息并将其添加到上下文队列中。
    
    参数:
        msg: 接收到的ROS图像消息
    """
    obs_img = msg_to_pil(msg)  # 将ROS图像消息转换为PIL格式
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

    model_config_path = model_paths[args.model]["config_path"]  # 获取指定模型的配置路径
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)  # 加载模型具体参数

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
    model = model.to(device)  # 将模型移动到设备上
    model.eval()  # 设置模型为评估模式

    num_diffusion_iters = model_params["num_diffusion_iters"]  # 获取扩散迭代次数
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )  # 初始化扩散调度器

    # ROS
    rospy.init_node("EXPLORATION", anonymous=False)  # 初始化ROS节点
    rate = rospy.Rate(RATE)  # 设置循环频率
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)  # 订阅图像话题
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  # 发布航点话题
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)  # 发布采样动作话题

    print("Registered with master node. Waiting for image observations...")

    while not rospy.is_shutdown():  # 循环直到ROS关闭
        # EXPLORATION MODE
        waypoint_msg = Float32MultiArray()  # 创建航点消息
        if (
                len(context_queue) > model_params["context_size"]
            ):  # 确保上下文队列有足够的图像
            
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)  # 转换图像
            obs_images = obs_images.to(device)  # 移动到设备
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)  # 随机生成目标图像
            mask = torch.ones(1).long().to(device)  # 创建掩码以忽略目标

            # infer action
            with torch.no_grad():
                # encoder vision features
                obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)  # 编码视觉特征
                
                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:  # 检查输出形状
                    obs_cond = obs_cond.repeat(args.num_samples, 1)  # 重复以匹配样本数量
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)  # 重复以匹配样本数量
                
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (args.num_samples, model_params["len_traj_pred"], 2), device=device)  # 从高斯噪声初始化动作
                naction = noisy_action  # 保存当前动作

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)  # 设置调度器时间步

                start_time = time.time()  # 开始计时
                for k in noise_scheduler.timesteps[:]:  # 遍历每个时间步
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
                    ).prev_sample  # 执行逆扩散步骤（去噪）
                print("time elapsed:", time.time() - start_time)  # 打印耗时

            naction = to_numpy(get_action(naction))  # 获取最终动作并转换为numpy数组
            
            sampled_actions_msg = Float32MultiArray()  # 创建采样动作消息
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))  # 合并数据
            sampled_actions_pub.publish(sampled_actions_msg)  # 发布采样动作消息

            naction = naction[0]  # 根据启发式选择动作

            chosen_waypoint = naction[args.waypoint]  # 选择航点

            if model_params["normalize"]:  # 如果需要归一化
                chosen_waypoint *= (MAX_V / RATE)  # 进行归一化
            waypoint_msg.data = chosen_waypoint  # 设置航点消息数据
            waypoint_pub.publish(waypoint_msg)  # 发布航点消息
            print("Published waypoint")
        rate.sleep()  # 睡眠以保持速率


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")  # 命令行参数解析器
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",  # 模型名称
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # 默认航点索引
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",  # 航点索引帮助信息
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",  # 采样动作数量
    )
    args = parser.parse_args()  # 解析命令行参数
    print(f"Using {device}")  # 打印使用的设备
    main(args)  # 调用主函数
