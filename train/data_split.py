import argparse
import os
import shutil
import random


def remove_files_in_dir(dir_path: str):
    """
    删除指定目录中的所有文件和子目录。

    参数:
        dir_path (str): 需要清空的目录路径。
    """
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            # 如果是文件或链接，删除它
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # 如果是目录，递归删除该目录及其内容
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def main(args: argparse.Namespace):
    """
    主函数，用于处理数据集并将其分为训练集和测试集。

    参数:
        args (argparse.Namespace): 命令行参数对象，包含输入输出目录、数据集名称等信息。
    """
    # 获取数据目录中包含 'traj_data.pkl' 文件的文件夹名称
    folder_names = [
        f
        for f in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(args.data_dir, f))
    ]

    # 随机打乱文件夹名称列表
    random.shuffle(folder_names)

    # 将文件夹名称划分为训练集和测试集
    split_index = int(args.split * len(folder_names))  # 根据给定的比例计算切分索引
    train_folder_names = folder_names[:split_index]  # 训练集文件夹名称
    test_folder_names = folder_names[split_index:]     # 测试集文件夹名称

    # 创建训练集和测试集的目录
    train_dir = os.path.join(args.data_splits_dir, args.dataset_name, "train")
    test_dir = os.path.join(args.data_splits_dir, args.dataset_name, "test")
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")  # 清空已有目录
            remove_files_in_dir(dir_path)  # 调用函数清空目录
        else:
            print(f"Creating {dir_path}")  # 创建新目录
            os.makedirs(dir_path)

    # 将训练集和测试集的文件夹名称写入文本文件
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name + "\n")  # 写入训练集文件夹名称

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")  # 写入测试集文件夹名称


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir", "-i", help="Directory containing the data", required=True
    )
    parser.add_argument(
        "--dataset-name", "-d", help="Name of the dataset", required=True
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.8, help="Train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", default="vint_train/data/data_splits", help="Data splits directory"
    )
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 调用主函数
    print("Done")  # 完成提示
