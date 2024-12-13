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

    # 递归遍历输入目录中的所有文件夹，获取以.bag结尾且包含"diff"的文件路径
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag") and "diff" in file:
                bag_files.append(os.path.join(root, file))
    # 如果指定了处理的轨迹数量，则限制处理的文件数量
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # 处理循环
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)  # 尝试加载rosbag文件
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")  # 出现错误时跳过该文件
            continue

        # 生成轨迹名称，由文件路径最后两部分组成，并去掉扩展名
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # 从bag文件中加载图像数据和里程计数据
        bag_img_data, bag_traj_data = get_images_and_odom_2(
            b,
            ['/usb_cam_front/image_raw', '/chosen_subgoal'],
            ['/odom'],
            rate=args.sample_rate,
        )
  
        if bag_img_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )  # 如果没有找到所需主题则跳过
            continue
        
        # 创建用于保存轨迹数据的文件夹
        traj_folder = os.path.join(args.output_dir, traj_name)
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)
        
        obs_images = bag_img_data["/usb_cam_front/image_raw"]  # 获取观察图像
        diff_images = bag_img_data["/chosen_subgoal"]  # 获取差异图像
        for i, img_data in enumerate(zip(obs_images, diff_images)):
            obs_image, diff_image = img_data
            # 将观察图像和差异图像保存到磁盘
            obs_image.save(os.path.join(traj_folder, f"{i}.jpg"))
            diff_image.save(os.path.join(traj_folder, f"diff_{i}.jpg"))

        # 保存轨迹数据到pkl文件
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(bag_traj_data['/odom'], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 获取重建输入目录和输出目录的参数
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
    # 打印开始处理信息
    print(f"STARTING PROCESSING DIFF DATASET")
    main(args)
    # 打印完成处理信息
    print(f"FINISHED PROCESSING DIFF DATASET")
