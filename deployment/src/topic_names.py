# ROS通信的主题名称

# 图像观察主题
FRONT_IMAGE_TOPIC = "/usb_cam_front/image_raw"  # 前方摄像头捕获的原始图像数据
REVERSE_IMAGE_TOPIC = "/usb_cam_reverse/image_raw"  # 后方摄像头捕获的原始图像数据
#IMAGE_TOPIC = "/usb_cam/image_raw"  # 通用摄像头捕获的原始图像数据
IMAGE_TOPIC = "/rgb"   # for issac


# 探索相关主题
SUBGOALS_TOPIC = "/subgoals"  # 子目标列表，用于导航和路径规划
GRAPH_NAME_TOPIC = "/graph_name"  # 当前使用的图形名称，可能用于标识不同的地图或环境
WAYPOINT_TOPIC = "/waypoint"  # 当前的航点信息，机器人需要到达的位置
REVERSE_MODE_TOPIC = "/reverse_mode"  # 指示机器人是否处于倒退模式的主题
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"  # 采样输出结果，可能是算法生成的决策或路径
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"  # 表示机器人已达到目标的通知
SAMPLED_WAYPOINTS_GRAPH_TOPIC = "/sampled_waypoints_graph"  # 采样的航点图，用于可视化或进一步处理
BACKTRACKING_IMAGE_TOPIC = "/backtracking_image"  # 回溯过程中的图像数据，用于调试或分析
FRONTIER_IMAGE_TOPIC = "/frontier_image"  # 边界区域的图像数据，可能用于探索新区域
SUBGOALS_SHAPE_TOPIC = "/subgoal_shape"  # 子目标的形状信息，可能影响机器人的行为
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"  # 采样的动作选择，供后续决策参考
ANNOTATED_IMAGE_TOPIC = "/annotated_image"  # 带注释的图像数据，用于可视化和调试
CURRENT_NODE_IMAGE_TOPIC = "/current_node_image"  # 当前节点的图像数据，帮助理解当前状态
FLIP_DIRECTION_TOPIC = "/flip_direction"  # 翻转方向的指令，可能用于改变运动轨迹
TURNING_TOPIC = "/turning"  # 转向状态的信息，表示机器人正在转弯
SUBGOAL_GEN_RATE_TOPIC = "/subgoal_gen_rate"  # 子目标生成速率的设置，影响探索频率
MARKER_TOPIC = "/visualization_marker_array"  # 可视化标记数组，用于在RViz等工具中显示信息
VIZ_NAV_IMAGE_TOPIC = "/nav_image"  # 导航图像数据，用于实时监控和可视化

# 可视化主题
CHOSEN_SUBGOAL_TOPIC = "/chosen_subgoal"  # 被选定的子目标，用于反馈和展示

# 记录在机器人上的主题
ODOM_TOPIC = "/odom"  # 里程计信息，提供机器人的位置和姿态
BUMPER_TOPIC = "/mobile_base/events/bumper"  # 碰撞传感器事件，检测与障碍物的接触
JOY_BUMPER_TOPIC = "/joy_bumper"  # 手柄碰撞事件，来自游戏手柄的输入

# 移动机器人
