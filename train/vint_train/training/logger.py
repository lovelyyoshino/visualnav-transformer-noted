import numpy as np


class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        初始化Logger类的实例

        Args:
            name (str): 指标名称
            dataset (str): 数据集名称
            window_size (int, optional): 移动平均窗口大小，默认为10。
            rounding (int, optional): 四舍五入的小数位数，默认为4。
        """
        self.data = []  # 存储日志数据的列表
        self.name = name  # 指标名称
        self.dataset = dataset  # 数据集名称
        self.rounding = rounding  # 小数位数
        self.window_size = window_size  # 移动平均窗口大小

    def display(self) -> str:
        """
        显示最新值、平均值和移动平均值的字符串表示。

        Returns:
            str: 包含最新值、移动平均值和平均值的格式化字符串
        """
        latest = round(self.latest(), self.rounding)  # 获取并四舍五入最新值
        average = round(self.average(), self.rounding)  # 获取并四舍五入平均值
        moving_average = round(self.moving_average(), self.rounding)  # 获取并四舍五入移动平均值
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output  # 返回格式化后的输出字符串

    def log_data(self, data: float):
        """
        记录新的数据点，如果数据不是NaN，则将其添加到数据列表中。

        Args:
            data (float): 要记录的数据点
        """
        if not np.isnan(data):  # 检查数据是否为NaN
            self.data.append(data)  # 将有效数据添加到列表中

    def full_name(self) -> str:
        """
        获取指标的完整名称，包括数据集信息。

        Returns:
            str: 格式化的完整名称字符串
        """
        return f"{self.name} ({self.dataset})"  # 返回格式化的完整名称

    def latest(self) -> float:
        """
        获取最新记录的数据点。

        Returns:
            float: 最新的数据点，如果没有数据则返回NaN
        """
        if len(self.data) > 0:  # 如果有数据
            return self.data[-1]  # 返回最后一个数据点
        return np.nan  # 否则返回NaN

    def average(self) -> float:
        """
        计算所有记录数据的平均值。

        Returns:
            float: 所有数据的平均值，如果没有数据则返回NaN
        """
        if len(self.data) > 0:  # 如果有数据
            return np.mean(self.data)  # 返回数据的平均值
        return np.nan  # 否则返回NaN

    def moving_average(self) -> float:
        """
        计算最近window_size个数据的移动平均值。

        Returns:
            float: 最近window_size个数据的移动平均值，如果数据不足则返回整体平均值
        """
        if len(self.data) > self.window_size:  # 如果数据量大于窗口大小
            return np.mean(self.data[-self.window_size :])  # 返回最近window_size个数据的平均值
        return self.average()  # 否则返回整体平均值
