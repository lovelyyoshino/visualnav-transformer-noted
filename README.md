# 通用导航模型：GNM、ViNT 和 NoMaD

**贡献者**：Dhruv Shah, Ajay Sridhar, Nitish Dashora, Catherine Glossop, Kyle Stachowicz, Arjun Bhorkar, Kevin Black, Noriaki Hirose, Sergey Levine

_加州大学伯克利分校人工智能研究所_

[项目页面](https://general-navigation-models.github.io) | [引用](https://github.com/robodhruv/visualnav-transformer#citing) | [预训练模型](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

通用导航模型是基于多样化、跨实体训练数据训练的通用目标条件视觉导航策略，能够在零样本情况下控制多种不同的机器人。它们还可以高效地微调或适应新的机器人和下游任务。我们的模型家族在以下研究论文中进行了描述（并在不断增加）：
1. [GNM：驱动任何机器人的通用导航模型](https://sites.google.com/view/drive-any-robot)（_2022年10月_，在ICRA 2023上展示）
2. [ViNT：视觉导航的基础模型](https://general-navigation-models.github.io/vint/index.html)（_2023年6月_，在CoRL 2023上展示）
3. [NoMaD：用于导航和探索的目标掩蔽扩散策略](https://general-navigation-models.github.io/nomad/index.html)（_2023年10月_）

## 概述
该代码库包含用于使用您自己的数据训练我们模型家族的代码、预训练模型检查点，以及在TurtleBot2/LoCoBot机器人上部署的示例代码。该代码库遵循[GNM](https://github.com/PrieureDeSion/drive-any-robot)的组织结构。

- `./train/train.py`：用于在您的自定义数据上训练或微调ViNT模型的训练脚本。
- `./train/vint_train/models/`：包含GNM、ViNT及一些基线模型的文件。
- `./train/process_*.py`：处理rosbag或其他格式的机器人轨迹以生成训练数据的脚本。
- `./deployment/src/record_bag.sh`：在机器人目标环境中收集演示轨迹作为ROS bag的脚本。该轨迹经过下采样以生成环境的拓扑图。
- `./deployment/src/create_topomap.sh`：将演示轨迹的ROS bag转换为机器人可以用来导航的拓扑图的脚本。
- `./deployment/src/navigate.sh`：在机器人上部署训练好的GNM/ViNT/NoMaD模型以在生成的拓扑图中导航到所需目标的脚本。请参见下面相关部分以获取配置设置。
- `./deployment/src/explore.sh`：在机器人上部署训练好的NoMaD模型以随机探索其环境的脚本。请参见下面相关部分以获取配置设置。

## 训练

该子文件夹包含处理数据集和从您自己的数据训练模型的代码。

### 先决条件

该代码库假设可以访问运行Ubuntu（在18.04和20.04上测试）、Python 3.7+和具有CUDA 10+的GPU的工作站。它还假设可以访问conda，但您可以修改它以与其他虚拟环境包或本地设置一起使用。

### 设置
在`vint_release/`（最上层）目录中运行以下命令：
1. 设置conda环境：
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. 激活conda环境：
    ```
    conda activate vint_train
    ```
3. 安装vint_train包：
    ```bash
    pip install -e train/
    ```
4. 从此[仓库](https://github.com/real-stanford/diffusion_policy)安装`diffusion_policy`包：
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```

### 数据处理
在[论文](https://general-navigation-models.github.io)中，我们在公开可用和未发布数据集的组合上进行训练。以下是用于训练的公开可用数据集的列表；请联系各自的作者以获取未发布数据的访问权限。
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2（修改版）](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

我们建议您下载这些（以及您可能想要训练的任何其他数据集）并运行以下处理步骤。

#### 数据处理 

我们提供了一些示例脚本来处理这些数据集，无论是直接从rosbag还是从自定义格式（如HDF5）：
1. 使用相关参数运行`process_bags.py`，或使用`process_recon.py`处理RECON HDF5。您还可以通过遵循我们下面的结构手动添加自己的数据集（如果您添加自定义数据集，请查看[自定义数据集](#custom-datasets)部分）。
2. 在您的数据集文件夹上运行`data_split.py`，并使用相关参数。

在数据处理的第一步之后，处理后的数据集应具有以下结构：

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

每个`*.jpg`文件包含来自机器人的前视RGB观察，并且它们是按时间标记的。`traj_data.pkl`文件是轨迹的里程计数据。它是一个带有以下键的pickle字典：
- `"position"`：一个np.ndarray [T, 2]，表示每个图像观察时机器人的xy坐标。
- `"yaw"`：一个np.ndarray [T,]，表示每个图像观察时机器人的偏航角。

在数据处理的第二步之后，处理后的数据拆分应在`vint_release/train/vint_train/data/data_splits/`中具有以下结构：

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### 训练您的通用导航模型
在`vint_release/train`目录中运行：
```bash
python train.py -c <path_of_train_config_file>
```
预制的配置yaml文件位于`train/config`目录中。 

#### 自定义配置文件
您可以使用其中一个预制的yaml文件作为起点，并根据需要更改值。`config/vint.yaml`是一个不错的选择，因为它包含注释参数。`config/defaults.yaml`包含默认配置值（请勿直接使用此配置文件进行训练，因为它未指定任何用于训练的数据集）。

#### 自定义数据集
确保您的数据集和数据拆分目录遵循[数据处理](#data-processing)部分中提供的结构。找到`train/vint_train/data/data_config.yaml`并附加以下内容：

```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

找到您的训练配置文件，并在`datasets`参数下添加以下文本（可以自由更改`end_slack`、`goals_per_obs`和`negative_mining`的值）：
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # 从每个轨迹的末尾切掉多少时间步（以防许多轨迹以碰撞结束）
    goals_per_obs: 1 # 每个观察采样多少个目标
    negative_mining: True # 来自ViNG论文的负采样（Shah等人）
```

#### 从检查点训练您的模型
您还可以从已发布结果中加载现有检查点，而不是从头开始训练。
在`vint_release/train/config/`中的.yaml配置文件中添加`load_run: <project_name>/<log_run_name>`。您要加载的`*.pth`文件应保存在此文件结构中，并重命名为“latest”：`vint_release/train/logs/<project_name>/<log_run_name>/latest.pth`。这使得从先前运行的检查点进行训练变得简单，因为日志默认以这种方式保存。注意：如果您从先前运行加载检查点，请检查`vint_release/train/logs/<project_name>/`中的运行名称，因为代码会在配置yaml文件中为每个运行名称附加日期字符串，以避免重复的运行名称。

如果您想使用我们的检查点，可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载`*.pth`文件。

## 部署
该子文件夹包含加载预训练ViNT并在开源[LoCoBot室内机器人平台](http://www.locobot.org/)上部署的代码，使用[NVIDIA Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=asc_df_B0BZJTQ5YP/?tag=hyprod-20&linkCode=df0&hvadid=652427572954&hvpos=&hvnetw=g&hvrand=12520404772764575478&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1013585&hvtargid=pla-2112361227514&psc=1&gclid=CjwKCAjw4P6oBhBsEiwAKYVkq7dqJEwEPz0K-H33oN7MzjO0hnGcAJDkx2RdT43XZHdSWLWHKDrODhoCmnoQAvD_BwE)。它可以轻松适应在其他机器人上运行，研究人员已经能够独立地在以下机器人上部署它——Clearpath Jackal、DJI Tello、Unitree A1、TurtleBot2、Vizbot——以及在CARLA等模拟环境中。

### LoCoBot设置

该软件在运行Ubuntu 20.04的LoCoBot上进行了测试。

#### 软件安装（按此顺序）
1. ROS: [ros-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
2. ROS包： 
    ```bash
    sudo apt-get install ros-noetic-usb-cam ros-noetic-joy
    ```
3. [kobuki](http://wiki.ros.org/kobuki/Tutorials/Installation)
4. Conda 
    - 安装anaconda/miniconda等以管理环境
    - 使用environment.yml创建conda环境（在`vint_release/`目录中运行）
        ```bash
        conda env create -f deployment/deployment_environment.yaml
        ```
    - 激活环境 
        ```bash
        conda activate nomad_train
        ```
    - （推荐）添加到`~/.bashrc`： 
        ```bash
        echo “conda activate nomad_train” >> ~/.bashrc 
        ```
5. 安装`vint_train`包（在`vint_release/`目录中运行）。命令用于在开发模式下安装 Python 包，其中 -e 代表“editable”：
    ```bash
    pip install -e train/
    ```
6. 从此[仓库](https://github.com/real-stanford/diffusion_policy)安装`diffusion_policy`包：
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
7. （推荐）如果未安装，请安装[tmux](https://github.com/tmux/tmux/wiki/Installing)。
    许多bash脚本依赖于tmux来启动多个屏幕以执行不同的命令。这对于调试非常有用，因为您可以看到每个屏幕的输出。

#### 硬件要求
- LoCoBot: http://locobot.org（仅导航堆栈）
- 广角RGB相机：[示例](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W)。`vint_locobot.launch`文件使用与ELP鱼眼广角相机兼容的相机参数，您可以根据自己的需要进行修改。根据需要在`vint_release/deployment/config/camera.yaml`中调整相机参数（用于可视化）。
- [操纵杆](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[键盘遥控](http://wiki.ros.org/teleop_twist_keyboard)，可与Linux配合使用。将操纵杆的_死区开关_的索引映射添加到`vint_release/deployment/config/joystick.yaml`。您可以在[wiki](https://wiki.ros.org/joy)中找到常见操纵杆的按钮到索引的映射。

### 加载模型权重

将模型权重`*.pth`文件保存在`vint_release/deployment/model_weights`文件夹中。我们的模型权重在[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)中。

### 收集拓扑图

_确保在`vint_release/deployment/src/`目录中运行这些脚本。_

本节讨论了一种简单的方法来为部署创建目标环境的拓扑图。为简单起见，我们将使用机器人在“路径跟随”模式下，即给定环境中的单个轨迹，任务是沿着相同的轨迹到达目标。环境可能有新的/动态障碍物、光照变化等。

#### 记录rosbag： 
```bash
./record_bag.sh <bag_name>
```

运行此命令以使用操纵杆和相机遥控机器人。此命令将打开三个窗口：
1. `roslaunch vint_locobot.launch`：此启动文件打开相机的`usb_cam`节点、操纵杆的joy节点和机器人的移动底盘节点。
2. `python joy_teleop.py`：此python脚本启动一个节点，读取joy主题的输入并将其输出到遥控机器人的底盘的主题上。
3. `rosbag record /usb_cam/image_raw -o <bag_name>`：此命令不会立即运行（您必须按Enter）。它将在vint_release/deployment/topomaps/bags目录中运行，我们建议您在此处存储rosbags。

一旦准备好记录bag，运行`rosbag record`脚本并遥控机器人沿着您希望机器人跟随的地图。当您完成路径记录时，终止`rosbag record`命令，然后终止tmux会话。

#### 制作拓扑图： 
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

此命令将打开3个窗口：
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`：此命令在`/vint_release/deployment/topomaps/images`中创建一个目录，并在每秒播放bag时将图像保存为地图中的节点。
3. `rosbag play -r 1.5 <bag_filename>`：此命令以5倍速度播放rosbag，因此python脚本实际上每1.5秒记录一个节点。`<bag_filename>`应为完整的bag名称，带有.bag扩展名。您可以在`make_topomap.sh`文件中更改此值。该命令在您按Enter之前不会运行，您应仅在python脚本给出等待消息后按Enter。一旦播放bag，请转到运行python脚本的屏幕，以便在rosbag停止播放时终止它。

当bag停止播放时，终止tmux会话。

### 运行模型 
#### 导航
_确保在`vint_release/deployment/src/`目录中运行此脚本。_

```bash
./navigate.sh “--model <model_name> --dir <topomap_dir>”
```

要部署已发布结果中的模型之一，我们发布了可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载的模型检查点。

`<model_name>`是`vint_release/deployment/config/models.yaml`文件中模型的名称。在此文件中，您为每个模型指定这些参数（使用默认值）：
- `config_path`（str）：用于训练模型的`vint_release/train/config/`中的*.yaml文件的路径
- `ckpt_path`（str）：`vint_release/deployment/model_weights/`中*.pth文件的路径

确保这些配置与您用于训练模型的配置匹配。我们提供权重的模型的配置在yaml文件中供您参考。

`<topomap_dir>`是`vint_release/deployment/topomaps/images`中包含与拓扑图节点对应的图像的目录名称。图像按名称从0到N排序。

此命令将打开4个窗口：

1. `roslaunch vint_locobot.launch`：此启动文件打开相机的usb_cam节点、操纵杆的joy节点和机器人的移动底盘的多个节点。
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`：此python脚本启动一个节点，从`/usb_cam/image_raw`主题读取图像观察，将观察和地图输入模型，并将动作发布到`/waypoint`主题。
3. `python joy_teleop.py`：此python脚本启动一个节点，读取joy主题的输入并将其输出到遥控机器人的底盘的主题上。
4. `python pd_controller.py`：此python脚本启动一个节点，从`/waypoint`主题（来自模型的航点）读取消息并输出速度以导航机器人的底盘。

当机器人完成导航时，终止`pd_controller.py`脚本，然后终止tmux会话。如果您希望在机器人导航时控制它，`joy_teleop.py`脚本允许您使用操纵杆进行控制。

#### 探索
_确保在`vint_release/deployment/src/`目录中运行此脚本。_

```bash
./exploration.sh “--model <model_name>”
```

要部署已发布结果中的模型之一，我们发布了可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载的模型检查点。

`<model_name>`是`vint_release/deployment/config/models.yaml`文件中模型的名称（请注意，只有NoMaD适用于探索）。在此文件中，您为每个模型指定这些参数（使用默认值）：
- `config_path`（str）：用于训练模型的`vint_release/train/config/`中的*.yaml文件的路径
- `ckpt_path`（str）：`vint_release/deployment/model_weights/`中*.pth文件的路径

确保这些配置与您用于训练模型的配置匹配。我们提供权重的模型的配置在yaml文件中供您参考。

`<topomap_dir>`是`vint_release/deployment/topomaps/images`中包含与拓扑图节点对应的图像的目录名称。图像按名称从0到N排序。

此命令将打开4个窗口：

1. `roslaunch vint_locobot.launch`：此启动文件打开相机的usb_cam节点、操纵杆的joy节点和机器人的移动底盘的多个节点。
2. `python explore.py --model <model_name>`：此python脚本启动一个节点，从`/usb_cam/image_raw`主题读取图像观察，将观察和地图输入模型，并将探索动作发布到`/waypoint`主题。
3. `python joy_teleop.py`：此python脚本启动一个节点，读取joy主题的输入并将其输出到遥控机器人的底盘的主题上。
4. `python pd_controller.py`：此python脚本启动一个节点，从`/waypoint`主题（来自模型的航点）读取消息并输出速度以导航机器人的底盘。

当机器人完成导航时，终止`pd_controller.py`脚本，然后终止tmux会话。如果您希望在机器人导航时控制它，`joy_teleop.py`脚本允许您使用操纵杆进行控制。


### Isaac-Sim-NoMaD 工作流程

#### 步骤 1：录制 Rosbag
1. 进入 `visualnav-transformer/deployment/src/` 目录：
   ```bash
   cd visualnav-transformer/deployment/src/
   ```

2. 执行脚本开始录制：
   ```bash
   sh record_bag_isaac.sh
   ```

3. 在新终端中，进入 topomaps 目录：
   ```bash
   cd ../topomaps/bags
   ```

4. 将相关主题录制到 Rosbag 中：
   ```bash
   rosbag record /rgb /odom -O warehouse_turtlebot
   ```

5. 完成后，立即按 `Ctrl+C` 停止录制。

#### 步骤 2：创建拓扑图
1. 返回 `visualnav-transformer/deployment/src/` 目录：
   ```bash
   cd visualnav-transformer/deployment/src/
   ```

2. 运行脚本以创建拓扑图：
   ```bash
   sh create_topomap_isaac.sh <topomap_name> <bag_filename> <rosbag_playback_rate>
   ```

   例如：
   ```bash
   sh create_topomap_isaac.sh warehouse_turtlebot warehouse_turtlebot 1.5
   ```

#### 步骤 3：使用预训练模型导航
1. 仍在 `visualnav-transformer/deployment/src/` 目录中，执行导航脚本：
   ```bash
   sh navigate_isaac.sh
   ```
