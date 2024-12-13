import h5py  # 导入h5py库，用于处理HDF5文件格式
import os  # 导入os库，用于操作系统相关的功能，如路径和目录管理
import pickle  # 导入pickle库，用于序列化和反序列化Python对象
from PIL import Image  # 从PIL库导入Image模块，用于图像处理
import io  # 导入io库，用于处理字节流
import argparse  # 导入argparse库，用于解析命令行参数
import tqdm  # 导入tqdm库，用于显示进度条


def main(args: argparse.Namespace):
    recon_dir = os.path.join(args.input_dir, "recon_release")  # 拼接输入目录与'recon_release'子目录
    output_dir = args.output_dir  # 获取输出目录

    # 如果输出目录不存在，则创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取重建数据集中的所有文件夹
    filenames = os.listdir(recon_dir)  # 列出重建目录下的所有文件名
    if args.num_trajs >= 0:
        filenames = filenames[: args.num_trajs]  # 如果指定了轨迹数量，截取相应数量的文件名

    # 处理循环
    for filename in tqdm.tqdm(filenames, desc="Trajectories processed"):
        # 提取不带扩展名的文件名
        traj_name = filename.split(".")[0]
        # 加载hdf5文件
        try:
            h5_f = h5py.File(os.path.join(recon_dir, filename), "r")  # 打开HDF5文件进行读取
        except OSError:
            print(f"Error loading {filename}. Skipping...")  # 如果加载失败，打印错误信息并跳过
            continue
        
        # 提取位置和偏航数据
        position_data = h5_f["jackal"]["position"][:, :2]  # 提取前两列作为位置信息，xy信息
        yaw_data = h5_f["jackal"]["yaw"][()]  # 提取偏航数据
        # 将数据保存到字典中
        traj_data = {"position": position_data, "yaw": yaw_data}
        traj_folder = os.path.join(output_dir, traj_name)  # 创建以轨迹名称命名的文件夹
        os.makedirs(traj_folder, exist_ok=True)  # 如果文件夹已存在则不报错地创建

        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)  # 将轨迹数据序列化并保存为pkl文件
        
        # 保存图像数据到磁盘
        for i in range(h5_f["images"]["rgb_left"].shape[0]):  # 遍历左侧RGB图像
            img = Image.open(io.BytesIO(h5_f["images"]["rgb_left"][i]))  # 从字节流中打开图像
            img.save(os.path.join(traj_folder, f"{i}.jpg"))  # 将图像保存为JPEG格式


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象用于解析命令行参数
    # 获取重建输入目录和输出目录的参数
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the recon_dataset",  # 输入目录的帮助说明
        required=True,  # 此参数为必需项
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dataset/recon/",  # 默认输出目录
        type=str,
        help="path for processed recon dataset (default: dataset/recon/)",  # 输出目录的帮助说明
    )
    # 要处理的轨迹数量
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,  # 默认值为-1，表示处理所有轨迹
        type=int,
        help="number of trajectories to process (default: -1, all)",  # 轨迹数量的帮助说明
    )

    args = parser.parse_args()  # 解析命令行参数
    print("STARTING PROCESSING RECON DATASET")  # 打印开始处理的信息
    main(args)  # 调用主函数进行处理
    print("FINISHED PROCESSING RECON DATASET")  # 打印完成处理的信息
