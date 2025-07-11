"""
配置文件 - 包含所有可调整的参数
"""

# 环境参数
ENV_CONFIG = {
    'num_lanes': 5,                    # 车道数量
    'start_position': -1,              # 起点位置
    'end_position': 5,                 # 终点位置
    'traffic_light_cycle': 20,         # 红绿灯周期（步数）- 10步绿灯，10步红灯
    'car_spawn_probability': 0.3,      # 车辆生成概率
}
# 注：机器人需要6步到达终点，绿灯持续10步，给予充足时间

# Q-Learning参数
QL_CONFIG = {
    'alpha': 0.1,                      # 学习率
    'gamma': 0.9,                      # 折扣因子
    'epsilon': 1.0,                    # 探索率初始值
    'epsilon_decay': 0.995,            # 探索率衰减
    'epsilon_min': 0.01,               # 最小探索率
}

# 奖励设置
REWARD_CONFIG = {
    'goal_reward': 50,                 # 到达终点奖励
    'collision_penalty': -100,         # 碰撞惩罚
    'step_penalty': -1,                # 每步惩罚
}

# 训练参数
TRAIN_CONFIG = {
    'num_episodes': 1000,              # 训练回合数
    'max_steps_per_episode': 100,      # 每回合最大步数
    'save_interval': 100,              # 保存模型间隔
}

# v2.0算法配置
ALGORITHM_CONFIG = {
    'algorithm': 'linear_fa',          # 'q_table' or 'linear_fa'
    'environment': 'enhanced',         # 'classic' or 'enhanced'
    'continuous_mode': True,           # 是否使用连续位置模式
}

# 线性函数近似配置
LINEAR_FA_CONFIG = {
    'n_features': 9,                   # 特征数量
    'alpha': 0.01,                     # 学习率（比Q-table更小）
    'gamma': 0.9,                      # 折扣因子
    'epsilon': 1.0,                    # 探索率初始值
    'epsilon_decay': 0.995,            # 探索率衰减
    'epsilon_min': 0.01,               # 最小探索率
    'weight_init': 'zeros',            # 权重初始化 ('zeros', 'random', 'small_random')
}

# 可视化参数
VIS_CONFIG = {
    'window_width': 1200,              # 窗口宽度
    'window_height': 800,              # 窗口高度
    'fps': 60,                         # 帧率（训练时）
    'demo_fps': 2,                     # 帧率（演示时）
    'fast_fps': 0,                     # 快速训练（0=无延时）
    'colors': {
        'background': (240, 240, 240),
        'road': (100, 100, 100),
        'lane_divider': (255, 255, 255),
        'car': (255, 0, 0),
        'robot': (0, 0, 255),
        'goal': (0, 255, 0),
        'text': (0, 0, 0),
        'red_light': (255, 0, 0),
        'green_light': (0, 255, 0),
    }
}