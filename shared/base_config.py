"""
基础配置文件 - 所有版本共享的配置
"""

# 基础环境参数
BASE_ENV_CONFIG = {
    'num_lanes': 5,                    # 车道数量
    'start_position': -1,              # 起点位置
    'end_position': 5,                 # 终点位置
    'traffic_light_cycle': 20,         # 红绿灯周期（步数）- 10步绿灯，10步红灯
    'car_spawn_probability': 0.3,      # 车辆生成概率
}

# 基础奖励设置
BASE_REWARD_CONFIG = {
    'goal_reward': 50,                 # 到达终点奖励
    'collision_penalty': -100,         # 碰撞惩罚
    'step_penalty': -1,                # 每步惩罚
}

# 基础训练参数
BASE_TRAIN_CONFIG = {
    'num_episodes': 1000,              # 训练回合数
    'max_steps_per_episode': 100,      # 每回合最大步数
    'save_interval': 100,              # 保存模型间隔
}

# 基础可视化参数
BASE_VIS_CONFIG = {
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

class BaseConfig:
    """所有版本的基础配置类"""
    
    def __init__(self, version="unknown"):
        self.version = version
        self.env_config = BASE_ENV_CONFIG.copy()
        self.reward_config = BASE_REWARD_CONFIG.copy()
        self.train_config = BASE_TRAIN_CONFIG.copy()
        self.vis_config = BASE_VIS_CONFIG.copy()
    
    def get_model_save_path(self, episode=None):
        """获取模型保存路径"""
        if episode is None:
            return f"saved_models/{self.version}/model_final.pkl"
        else:
            return f"saved_models/{self.version}/model_episode_{episode}.pkl"
    
    def get_log_path(self, timestamp=None):
        """获取日志保存路径"""
        if timestamp is None:
            return f"logs/{self.version}/training.txt"
        else:
            return f"logs/{self.version}/training_{timestamp}.txt"