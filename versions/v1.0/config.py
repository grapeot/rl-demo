"""
v1.0版本配置文件 - 扩展Q-Learning配置
"""

from shared.base_config import BaseConfig

class V10Config(BaseConfig):
    def __init__(self):
        super().__init__("v1.0")
        
        # v1.0特有的Q-Learning参数
        self.ql_config = {
            'alpha': 0.1,                      # 学习率
            'gamma': 0.9,                      # 折扣因子
            'epsilon': 1.0,                    # 探索率初始值
            'epsilon_decay': 0.995,            # 探索率衰减
            'epsilon_min': 0.01,               # 最小探索率
        }
        
        # v1.0增强的训练参数
        self.train_config.update({
            'num_episodes': 3000,              # 增加训练回合数应对更复杂的状态空间
        })
        
        # v1.0增强的奖励设置
        self.reward_config.update({
            'goal_reward': 100,                # 增加目标奖励
            'step_penalty': -2,                # 增加步骤惩罚，鼓励效率
            'timeout_penalty': -50,            # 新增：绿灯时间用完的惩罚
        })

# 为了向后兼容，保留原有的配置变量
config = V10Config()

ENV_CONFIG = config.env_config
REWARD_CONFIG = config.reward_config
TRAIN_CONFIG = config.train_config
VIS_CONFIG = config.vis_config
QL_CONFIG = config.ql_config