"""
线性Q-Learning算法实现
使用线性函数近似替代Q-table
"""
import numpy as np
import pickle
import random
from typing import Dict, List, Union, Any, Optional
from collections import defaultdict
from .feature_engineering import extract_features, get_feature_names, validate_features

class LinearQLearning:
    """线性函数近似的Q-Learning智能体"""
    
    def __init__(self, n_features: int = 9, n_actions: int = 2, 
                 alpha: float = 0.01, gamma: float = 0.9, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, weight_init: str = 'zeros'):
        """
        初始化线性Q-Learning智能体
        
        Args:
            n_features: 特征数量
            n_actions: 动作数量
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索率初始值
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            weight_init: 权重初始化方式 ('zeros', 'random', 'small_random')
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化权重矩阵：每个动作一个权重向量
        self.weights = self._initialize_weights(weight_init)
        
        # 训练统计
        self.episode = 0
        self.total_reward_history = []
        self.td_errors = []  # TD误差历史
        
        # 配置信息
        self.config = {
            'n_features': n_features,
            'n_actions': n_actions,
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'epsilon_min': epsilon_min,
            'weight_init': weight_init
        }
    
    def _initialize_weights(self, method: str) -> np.ndarray:
        """
        初始化权重矩阵
        
        Args:
            method: 初始化方法
        
        Returns:
            权重矩阵 (n_actions, n_features)
        """
        if method == 'zeros':
            return np.zeros((self.n_actions, self.n_features), dtype=np.float32)
        elif method == 'random':
            return np.random.normal(0, 0.1, (self.n_actions, self.n_features)).astype(np.float32)
        elif method == 'small_random':
            return np.random.uniform(-0.01, 0.01, (self.n_actions, self.n_features)).astype(np.float32)
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")
    
    def get_q_values(self, state: Any) -> np.ndarray:
        """
        计算所有动作的Q值
        
        Args:
            state: 状态对象
        
        Returns:
            Q值数组
        """
        q_values = np.zeros(self.n_actions, dtype=np.float32)
        
        for action in range(self.n_actions):
            features = extract_features(state, action)
            q_values[action] = np.dot(self.weights[action], features)
        
        return q_values
    
    def get_q_value(self, state: Any, action: int) -> float:
        """
        计算特定状态-动作对的Q值
        
        Args:
            state: 状态对象
            action: 动作索引
        
        Returns:
            Q值
        """
        features = extract_features(state, action)
        return np.dot(self.weights[action], features)
    
    def choose_action(self, state: Any, valid_actions: Optional[List[int]] = None) -> int:
        """
        使用ε-greedy策略选择动作
        
        Args:
            state: 状态对象
            valid_actions: 有效动作列表
        
        Returns:
            选择的动作索引
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        
        # ε-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最高的动作
            q_values = self.get_q_values(state)
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def update(self, state: Any, action: int, reward: float, 
               next_state: Any, done: bool) -> float:
        """
        更新权重
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        
        Returns:
            TD误差
        """
        # 计算当前Q值
        features = extract_features(state, action)
        current_q = np.dot(self.weights[action], features)
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)
        
        # 计算TD误差
        td_error = target_q - current_q
        
        # 更新权重
        self.weights[action] += self.alpha * td_error * features
        
        # 记录TD误差
        self.td_errors.append(abs(td_error))
        
        return td_error
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1
    
    def get_policy(self) -> Dict[str, int]:
        """
        获取当前策略（用于调试）
        
        Returns:
            策略字典
        """
        # 注意：对于函数近似，策略是连续的，这里只返回一些示例状态
        from feature_engineering import create_test_state
        
        policy = {}
        for pos in range(-1, 6):
            for light in [0, 1]:
                test_state = create_test_state(pos, light, 5)
                q_values = self.get_q_values(test_state)
                best_action = np.argmax(q_values)
                policy[f"pos_{pos}_light_{light}"] = best_action
        
        return policy
    
    def save(self, filename: str):
        """
        保存模型到文件
        
        Args:
            filename: 文件名
        """
        save_data = {
            'weights': self.weights,
            'epsilon': self.epsilon,
            'episode': self.episode,
            'config': self.config,
            'td_errors': self.td_errors[-1000:],  # 只保存最近1000个TD误差
            'total_reward_history': self.total_reward_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Linear Q-Learning model saved to {filename}")
    
    def load(self, filename: str):
        """
        从文件加载模型
        
        Args:
            filename: 文件名
        """
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        
        # 恢复权重
        self.weights = save_data['weights']
        
        # 恢复其他参数
        self.epsilon = save_data.get('epsilon', self.epsilon_min)
        self.episode = save_data.get('episode', 0)
        self.td_errors = save_data.get('td_errors', [])
        self.total_reward_history = save_data.get('total_reward_history', [])
        
        # 更新配置
        if 'config' in save_data:
            self.config.update(save_data['config'])
        
        print(f"Linear Q-Learning model loaded from {filename}")
    
    def get_weight_stats(self) -> Dict[str, Any]:
        """
        获取权重统计信息
        
        Returns:
            统计信息字典
        """
        all_weights = self.weights.flatten()
        
        return {
            'num_parameters': len(all_weights),
            'weight_mean': np.mean(all_weights),
            'weight_std': np.std(all_weights),
            'weight_max': np.max(all_weights),
            'weight_min': np.min(all_weights),
            'weight_norm': np.linalg.norm(all_weights)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（权重绝对值的平均值）
        
        Returns:
            特征重要性字典
        """
        feature_names = get_feature_names()
        importance = {}
        
        for i, name in enumerate(feature_names):
            # 计算该特征在所有动作中的平均重要性
            importance[name] = np.mean(np.abs(self.weights[:, i]))
        
        return importance
    
    def get_td_error_stats(self) -> Dict[str, float]:
        """
        获取TD误差统计信息
        
        Returns:
            TD误差统计
        """
        if not self.td_errors:
            return {'mean': 0, 'std': 0, 'max': 0, 'recent_mean': 0}
        
        recent_errors = self.td_errors[-100:]  # 最近100个
        
        return {
            'mean': np.mean(self.td_errors),
            'std': np.std(self.td_errors),
            'max': np.max(self.td_errors),
            'recent_mean': np.mean(recent_errors)
        }
    
    def debug_prediction(self, state: Any) -> Dict[str, Any]:
        """
        调试预测结果
        
        Args:
            state: 状态对象
        
        Returns:
            调试信息
        """
        debug_info = {}
        
        # 获取特征
        features_forward = extract_features(state, 0)
        features_backward = extract_features(state, 1)
        
        # 获取Q值
        q_values = self.get_q_values(state)
        
        # 获取特征名称
        feature_names = get_feature_names()
        
        debug_info['features_forward'] = dict(zip(feature_names, features_forward))
        debug_info['features_backward'] = dict(zip(feature_names, features_backward))
        debug_info['q_values'] = {'forward': q_values[0], 'backward': q_values[1]}
        debug_info['best_action'] = np.argmax(q_values)
        debug_info['weight_contribution'] = {
            'forward': dict(zip(feature_names, self.weights[0] * features_forward)),
            'backward': dict(zip(feature_names, self.weights[1] * features_backward))
        }
        
        return debug_info

# 向后兼容的包装器
class LinearQLearningWrapper:
    """
    向后兼容的包装器，提供与原始QLearningAgent相同的接口
    """
    
    def __init__(self, n_actions: int = 2, config: Optional[Dict] = None):
        """
        初始化包装器
        
        Args:
            n_actions: 动作数量
            config: 配置字典
        """
        if config is None:
            config = {}
        
        # 提取线性函数近似的配置
        linear_config = config.get('linear_fa', {})
        
        self.agent = LinearQLearning(
            n_features=linear_config.get('n_features', 9),
            n_actions=n_actions,
            alpha=linear_config.get('alpha', 0.01),
            gamma=config.get('gamma', 0.9),
            epsilon=config.get('epsilon', 1.0),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            epsilon_min=config.get('epsilon_min', 0.01),
            weight_init=linear_config.get('weight_init', 'zeros')
        )
    
    def choose_action(self, state: Any, valid_actions: Optional[List[int]] = None) -> int:
        """选择动作"""
        return self.agent.choose_action(state, valid_actions)
    
    def update(self, state: Any, action: int, reward: float, 
               next_state: Any, done: bool):
        """更新模型"""
        return self.agent.update(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.agent.decay_epsilon()
    
    def get_policy(self) -> Dict[str, int]:
        """获取策略"""
        return self.agent.get_policy()
    
    def save(self, filename: str):
        """保存模型"""
        self.agent.save(filename)
    
    def load(self, filename: str):
        """加载模型"""
        self.agent.load(filename)
    
    def get_q_value_stats(self) -> Dict[str, Any]:
        """获取Q值统计（兼容接口）"""
        return self.agent.get_weight_stats()
    
    # 暴露线性函数近似特有的方法
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.agent.get_feature_importance()
    
    def get_td_error_stats(self) -> Dict[str, float]:
        """获取TD误差统计"""
        return self.agent.get_td_error_stats()
    
    def debug_prediction(self, state: Any) -> Dict[str, Any]:
        """调试预测"""
        return self.agent.debug_prediction(state)

# 测试函数
if __name__ == "__main__":
    from feature_engineering import create_test_state
    
    # 创建智能体
    agent = LinearQLearning(n_features=9, n_actions=2)
    
    # 创建测试状态
    test_state = create_test_state(2.0, 1, 5)
    
    # 测试Q值计算
    q_values = agent.get_q_values(test_state)
    print("Q values:", q_values)
    
    # 测试动作选择
    action = agent.choose_action(test_state)
    print("Chosen action:", action)
    
    # 测试更新
    next_state = create_test_state(2.5, 1, 4)
    td_error = agent.update(test_state, action, 1.0, next_state, False)
    print("TD error:", td_error)
    
    # 测试统计信息
    weight_stats = agent.get_weight_stats()
    print("Weight stats:", weight_stats)
    
    # 测试特征重要性
    importance = agent.get_feature_importance()
    print("Feature importance:", importance)
    
    # 测试调试信息
    debug_info = agent.debug_prediction(test_state)
    print("Debug info keys:", list(debug_info.keys()))