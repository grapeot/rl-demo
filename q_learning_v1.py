"""
Q-Learning算法实现 v1.0 - 支持扩展状态空间
"""
import random
import pickle
from collections import defaultdict
from config import QL_CONFIG

class QLearningAgentV1:
    """Q-Learning智能体 v1.0 - 针对420状态空间优化"""
    
    def __init__(self, n_actions=2, config=None):
        self.config = config or QL_CONFIG.copy()
        self.n_actions = n_actions
        
        # v1.0调整：更大状态空间需要调整探索策略
        if 'epsilon_decay' not in self.config:
            self.config['epsilon_decay'] = 0.998  # 更慢的衰减
        if 'epsilon_min' not in self.config:
            self.config['epsilon_min'] = 0.02    # 保持更高的最小探索率
        
        # Q-Table: {state: [Q_value_for_each_action]}
        self.q_table = defaultdict(lambda: [0.0] * n_actions)
        
        # 超参数
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon']
        self.epsilon_decay = self.config['epsilon_decay']
        self.epsilon_min = self.config['epsilon_min']
        
        # 训练统计
        self.episode = 0
        self.total_reward_history = []
        
        # v1.0特定：状态访问频率统计
        self.state_visit_count = defaultdict(int)
    
    def choose_action(self, state, valid_actions=None):
        """使用ε-greedy策略选择动作，针对v1.0状态空间优化"""
        # 如果没有提供有效动作，假设所有动作都有效
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        
        # 记录状态访问
        self.state_visit_count[state] += 1
        
        # ε-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择一个有效动作
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最高的有效动作
            q_values = self.q_table[state]
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def update(self, state, action, reward, next_state, done):
        """更新Q值"""
        # 获取当前Q值
        current_q = self.q_table[state][action]
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            # 找出下一状态的最大Q值
            max_next_q = max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # 更新Q值
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1
    
    def get_policy(self):
        """获取当前策略（每个状态的最佳动作）"""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = max(range(len(q_values)), key=lambda a: q_values[a])
        return policy
    
    def save(self, filename):
        """保存Q-Table到文件"""
        save_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'config': self.config,
            'state_visit_count': dict(self.state_visit_count)
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"v1.0 Model saved to {filename}")
    
    def load(self, filename):
        """从文件加载Q-Table"""
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        
        # 恢复Q-Table
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)
        for state, q_values in save_data['q_table'].items():
            self.q_table[state] = q_values
        
        # 恢复其他参数
        self.epsilon = save_data.get('epsilon', self.epsilon_min)
        self.episode = save_data.get('episode', 0)
        self.state_visit_count = defaultdict(int)
        if 'state_visit_count' in save_data:
            self.state_visit_count.update(save_data['state_visit_count'])
        
        print(f"v1.0 Model loaded from {filename}")
    
    def get_q_value_stats(self):
        """获取Q-Table的统计信息"""
        if not self.q_table:
            return {
                'num_states': 0,
                'avg_q_value': 0,
                'max_q_value': 0,
                'min_q_value': 0,
                'state_coverage': 0
            }
        
        all_q_values = []
        for q_values in self.q_table.values():
            all_q_values.extend(q_values)
        
        # v1.0状态空间覆盖率 (理论最大420状态)
        theoretical_max_states = 7 * 2 * 10 * 3  # robot_lane * light_status * countdown * car
        coverage = len(self.q_table) / theoretical_max_states * 100
        
        return {
            'num_states': len(self.q_table),
            'avg_q_value': sum(all_q_values) / len(all_q_values),
            'max_q_value': max(all_q_values),
            'min_q_value': min(all_q_values),
            'state_coverage': coverage,
            'theoretical_max_states': theoretical_max_states
        }
    
    def get_state_exploration_stats(self):
        """获取状态探索统计"""
        if not self.state_visit_count:
            return {
                'explored_states': 0,
                'avg_visits_per_state': 0,
                'max_visits': 0,
                'min_visits': 0
            }
        
        visits = list(self.state_visit_count.values())
        return {
            'explored_states': len(self.state_visit_count),
            'avg_visits_per_state': sum(visits) / len(visits),
            'max_visits': max(visits),
            'min_visits': min(visits)
        }