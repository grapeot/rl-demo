"""
v2.0版本配置文件 - 线性函数近似配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.base_config import BaseConfig

class V20Config(BaseConfig):
    def __init__(self):
        super().__init__("v20")
        
        # v2.0算法配置
        self.algorithm_config = {
            'algorithm': 'linear_fa',          # 'q_table' or 'linear_fa'
            'environment': 'enhanced',         # 'classic' or 'enhanced'
            'continuous_mode': True,           # 是否使用连续位置模式
        }
        
        # v2.0线性函数近似配置
        self.linear_fa_config = {
            'n_features': 9,                   # 特征数量
            'alpha': 0.01,                     # 学习率（比Q-table更小）
            'gamma': 0.9,                      # 折扣因子
            'epsilon': 1.0,                    # 探索率初始值
            'epsilon_decay': 0.995,            # 探索率衰减
            'epsilon_min': 0.01,               # 最小探索率
            'weight_init': 'zeros',            # 权重初始化 ('zeros', 'random', 'small_random')
        }

# 为了向后兼容，保留原有的配置变量
config = V20Config()

ENV_CONFIG = config.env_config
REWARD_CONFIG = config.reward_config
TRAIN_CONFIG = config.train_config
VIS_CONFIG = config.vis_config
ALGORITHM_CONFIG = config.algorithm_config
LINEAR_FA_CONFIG = config.linear_fa_config