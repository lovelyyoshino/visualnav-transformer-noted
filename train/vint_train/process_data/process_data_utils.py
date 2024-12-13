import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF

IMAGE_SIZE = (160, 120)  # 定义图像的目标大小
IMAGE_ASPECT_RATIO = 4 / 3  # 定义图像的宽高比


def process_images(im_list: List, img_process_func) -> List:
    """
    处理来自发布ROS图像主题的图像数据，将其转换为PIL图像列表
    :param im_list: 图像消息列表
    :param img_process_func: 用于处理每个图像的函数
    :return: 处理后的PIL图像列表
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)  # 调用传入的图像处理函数
        images.append(img)
    return images


def process_tartan_img(msg) -> Image:
    """
    将来自tartan_drive数据集的sensor_msgs/Image类型的图像消息处理为PIL图像
    :param msg: ROS图像消息
    :return: 处理后的PIL图像
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255  # 转换为numpy数组并缩放到0-255
    img = img.astype(np.uint8)  # 转换为无符号8位整数
    img = np.moveaxis(img, 0, -1)  # 改变轴顺序以获得正确的图像方向
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 从RGB转换为BGR格式
    img = Image.fromarray(img)  # 转换为PIL图像
    return img


def process_locobot_img(msg) -> Image:
    """
    将来自locobot数据集的sensor_msgs/Image类型的图像消息处理为PIL图像
    :param msg: ROS图像消息
    :return: 处理后的PIL图像
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)  # 从字节流中读取图像数据并重塑形状
    pil_image = Image.fromarray(img)  # 转换为PIL图像
    return pil_image


def process_scand_img(msg) -> Image:
    """
    将来自scand数据集的sensor_msgs/CompressedImage类型的图像消息处理为PIL图像
    :param msg: ROS压缩图像消息
    :return: 处理后的PIL图像
    """
    img = Image.open(io.BytesIO(msg.data))  # 将压缩图像数据解码为PIL图像
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # 中心裁剪图像以符合4:3的宽高比
    img = img.resize(IMAGE_SIZE)  # 调整图像大小到指定尺寸
    return img


############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    """
    将来自sacson数据集的图像消息处理为PIL图像
    :param msg: ROS图像消息
    :return: 处理后的PIL图像
    """
    np_arr = np.fromstring(msg.data, np.uint8)  # 从字节流中读取图像数据
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 解码为OpenCV图像
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # 从BGR转换为RGB格式
    pil_image = Image.fromarray(image_np)  # 转换为PIL图像
    return pil_image


#######################################################################


def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    处理来自nav_msgs/Odometry主题的里程计数据，提取位置和偏航角
    :param odom_list: 里程计消息列表
    :param odom_process_func: 用于处理每个里程计消息的函数
    :param ang_offset: 偏航角的附加偏移量
    :return: 包含位置和偏航角的字典
    """
    xys = []  # 存储位置
    yaws = []  # 存储偏航角
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)  # 调用处理函数获取位置和偏航角
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}  # 返回包含位置和偏航角的字典


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    从nav_msgs/Odometry消息中提取位置和偏航角
    :param odom_msg: 里程计消息
    :param ang_offset: 偏航角的附加偏移量
    :return: 位置坐标和偏航角
    """
    position = odom_msg.pose.pose.position  # 获取位置
    orientation = odom_msg.pose.pose.orientation  # 获取朝向（四元数）
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset  # 计算偏航角并添加偏移量
    )
    return [position.x, position.y], yaw  # 返回位置和偏航角


############ Add custom odometry processing functions here ############


#######################################################################


def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    从rosbag文件中获取图像和里程计数据
    :param bag: rosbag文件对象
    :param imtopics: 图像数据的主题名称
    :param odomtopics: 里程计数据的主题名称
    :param img_process_func: 处理图像数据的函数
    :param odom_process_func: 处理里程计数据的函数
    :param rate: 数据采样率
    :param ang_offset: 里程计数据的角度偏移量
    :return: 处理后的图像数据和轨迹数据
    """
    # 检查bag文件是否同时包含两个主题
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag文件不包含两个主题
        return None, None

    synced_imdata = []  # 同步的图像数据
    synced_odomdata = []  # 同步的里程计数据
    currtime = bag.get_start_time()  # 获取bag文件的起始时间

    curr_imdata = None
    curr_odomdata = None

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg  # 当前图像消息
        elif topic == odomtopic:
            curr_odomdata = msg  # 当前里程计消息
        if (t.to_sec() - currtime) >= 1.0 / rate:  # 根据设定的采样率同步数据
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func)  # 处理图像数据
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )  # 处理里程计数据

    return img_data, traj_data  # 返回处理后的图像和轨迹数据


def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    检查给定两点的位置和偏航角是否表示轨迹是向后移动
    :param pos1: 第一个点的位置
    :param yaw1: 第一个点的偏航角
    :param pos2: 第二个点的位置
    :param eps: 容差值
    :return: 如果轨迹向后移动则返回True，否则返回False
    """
    dx, dy = pos2 - pos1  # 计算两个点之间的差异
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps  # 判断是否向后移动


# 剔除轨迹中的非正速度段
def filter_backwards(
    img_list: List[Image.Image],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    剔除轨迹中的非正速度段
    :param img_list: 图像列表
    :param traj_data: 包含位置和偏航角的数据字典
    :param start_slack: 在轨迹开始时忽略的点数
    :param end_slack: 在轨迹结束时忽略的点数
    :return: 剔除后的轨迹和对应的起始时间
    """
    traj_pos = traj_data["position"]  # 提取位置数据
    traj_yaws = traj_data["yaw"]  # 提取偏航角数据
    cut_trajs = []  # 存储剔除后的轨迹
    start = True  # 标记是否在新轨迹的开始

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_traj_data = zip(*traj_pair)  # 拆分成新的图像和轨迹数据
        new_traj_data = np.array(new_traj_data)  # 转换为numpy数组
        new_traj_pos = new_traj_data[:, :2]  # 提取位置
        new_traj_yaws = new_traj_data[:, 2]  # 提取偏航角
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]  # 前一个位置
        yaw1 = traj_yaws[i - 1]  # 前一个偏航角
        pos2 = traj_pos[i]  # 当前的位置
        if not is_backwards(pos1, yaw1, pos2):  # 检查是否向后移动
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]  # 开始新的轨迹对
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))  # 添加最后一段轨迹
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )  # 继续添加轨迹对
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))  # 完成当前轨迹
            start = True
    return cut_trajs  # 返回剔除后的轨迹


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    将四元数转换为偏航角
    :param x: 四元数x分量
    :param y: 四元数y分量
    :param z: 四元数z分量
    :param w: 四元数w分量
    :return: 对应的偏航角（弧度）
    """
    t3 = 2.0 * (w * z + x * y)  # 计算公式的一部分
    t4 = 1.0 - 2.0 * (y * y + z * z)  # 计算公式的一部分
    yaw = np.arctan2(t3, t4)  # 计算偏航角
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    将ROS图像消息转换为numpy数组
    :param msg: ROS图像消息
    :param nchannels: 通道数量
    :param empty_value: 空值填充
    :param output_resolution: 输出分辨率
    :param aggregate: 聚合方式
    :return: 转换后的numpy数组
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)  # 设置输出分辨率

    is_rgb = "8" in msg.encoding  # 检查编码是否为RGB
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()  # 从字节流中读取数据
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()  # 从字节流中读取浮点数据

    data = data.reshape(msg.height, msg.width, nchannels)  # 重塑数据形状

    if empty_value:
        mask = np.isclose(abs(data), empty_value)  # 创建掩膜以识别空值
        fill_value = np.percentile(data[~mask], 99)  # 找到非空值的99百分位数作为填充值
        data[mask] = fill_value  # 填充空值

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )  # 调整图像大小

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])  # 小端聚合
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])  # 大端聚合

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)  # 扩展维度
    else:
        data = np.moveaxis(data, 2, 0)  # 切换到通道优先

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )  # 归一化数据

    return data  # 返回最终的numpy数组
