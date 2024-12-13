import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag

# utils
from vint_train.process_data.process_data_utils import *

def main(args: argparse.Namespace):
    # 加载配置文件
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 如果输出目录不存在，则创建该目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 递归遍历输入目录中的所有文件夹，获取以.bag结尾的文件路径
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    
    # 如果指定了处理的轨迹数量，则限制处理的文件数量
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # 处理循环
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)  # 尝试打开rosbag文件
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")  # 打印错误信息并跳过此文件
            continue

        # 生成轨迹名称，由文件路径的最后两部分组成，并去掉扩展名
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # 从rosbag中加载图像和里程计数据
        bag_img_data, bag_traj_data = get_images_and_odom(
            b,
            config[args.dataset_name]["imtopics"],  # 图像主题
            config[args.dataset_name]["odomtopics"],  # 里程计主题
            eval(config[args.dataset_name]["img_process_func"]),  # 图像处理函数
            eval(config[args.dataset_name]["odom_process_func"]),  # 里程计处理函数
            rate=args.sample_rate,  # 采样率
            ang_offset=config[args.dataset_name]["ang_offset"],  # 角度偏移
        )

        # 检查是否成功提取到所需的数据
        if bag_img_data is None or bag_traj_data is None:
            print(f"{bag_path} did not have the topics we were looking for. Skipping...")
            continue
        
        # 移除反向运动的轨迹
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        # 保存每条有效的轨迹数据
        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"  # 为每条轨迹命名
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)  # 创建轨迹文件夹路径
            
            # 如果轨迹文件夹不存在，则创建它
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            
            # 将轨迹数据保存为pkl文件
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            
            # 将图像数据保存到磁盘
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{i}.jpg"))  # 保存图像为JPEG格式


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 获取重建输入目录和输出目录的参数
    # 添加数据集名称
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in vint/process_data/process_bags_config.yaml)",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../dataset/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../dataset/tartan_drive/)",
    )
    # 要处理的轨迹数量
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    # 采样率
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    # 输出数据集名称（大写）
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)  # 调用主函数进行处理
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
