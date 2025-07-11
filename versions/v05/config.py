"""
v0.5版本配置文件 - 基础Q-Learning配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.base_config import BaseConfig

class V05Config(BaseConfig):
    def __init__(self):
        super().__init__("v05")
        
        # v0.5特有的Q-Learning参数
        self.ql_config = {
            'alpha': 0.1,                      # 学习率
            'gamma': 0.9,                      # 折扣因子
            'epsilon': 1.0,                    # 探索率初始值
            'epsilon_decay': 0.995,            # 探索率衰减
            'epsilon_min': 0.01,               # 最小探索率
        }

# 为了向后兼容，保留原有的配置变量
config = V05Config()

ENV_CONFIG = config.env_config
REWARD_CONFIG = config.reward_config
TRAIN_CONFIG = config.train_config
VIS_CONFIG = config.vis_config
QL_CONFIG = config.ql_config