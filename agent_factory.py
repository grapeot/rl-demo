"""
智能体工厂 - 统一创建不同类型的智能体
"""
from typing import Dict, Any, Optional, Union
from config import ALGORITHM_CONFIG, LINEAR_FA_CONFIG, QL_CONFIG

class AgentFactory:
    """智能体工厂类"""
    
    @staticmethod
    def create_agent(algorithm: Optional[str] = None, 
                    config: Optional[Dict[str, Any]] = None) -> Any:
        """
        创建智能体实例
        
        Args:
            algorithm: 算法类型 ('q_table', 'linear_fa')
            config: 配置字典
        
        Returns:
            智能体实例
        """
        if algorithm is None:
            algorithm = ALGORITHM_CONFIG.get('algorithm', 'q_table')
        
        if config is None:
            config = {}
        
        if algorithm == 'q_table':
            return AgentFactory._create_q_table_agent(config)
        elif algorithm == 'linear_fa':
            return AgentFactory._create_linear_fa_agent(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @staticmethod
    def _create_q_table_agent(config: Dict[str, Any]) -> Any:
        """创建Q-table智能体"""
        from q_learning import QLearningAgent
        
        # 合并配置
        merged_config = QL_CONFIG.copy()
        merged_config.update(config)
        
        return QLearningAgent(n_actions=2, config=merged_config)
    
    @staticmethod
    def _create_linear_fa_agent(config: Dict[str, Any]) -> Any:
        """创建线性函数近似智能体"""
        from linear_q_learning import LinearQLearning
        
        # 合并配置
        merged_config = LINEAR_FA_CONFIG.copy()
        merged_config.update(config)
        
        return LinearQLearning(
            n_features=merged_config['n_features'],
            n_actions=2,
            alpha=merged_config['alpha'],
            gamma=merged_config['gamma'],
            epsilon=merged_config['epsilon'],
            epsilon_decay=merged_config['epsilon_decay'],
            epsilon_min=merged_config['epsilon_min'],
            weight_init=merged_config['weight_init']
        )
    
    @staticmethod
    def create_environment(environment: Optional[str] = None,
                          continuous: Optional[bool] = None) -> Any:
        """
        创建环境实例
        
        Args:
            environment: 环境类型 ('classic', 'enhanced')
            continuous: 是否使用连续模式
        
        Returns:
            环境实例
        """
        if environment is None:
            environment = ALGORITHM_CONFIG.get('environment', 'classic')
        
        if continuous is None:
            continuous = ALGORITHM_CONFIG.get('continuous_mode', False)
        
        if environment == 'classic':
            from environment import RoadEnvironment
            return RoadEnvironment()
        elif environment == 'enhanced':
            from enhanced_environment import EnhancedRoadEnvironment
            return EnhancedRoadEnvironment(continuous_mode=continuous)
        else:
            raise ValueError(f"Unknown environment: {environment}")
    
    @staticmethod
    def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
        """
        获取算法信息
        
        Args:
            algorithm: 算法类型
        
        Returns:
            算法信息字典
        """
        if algorithm == 'q_table':
            return {
                'name': 'Q-Table Learning',
                'description': '经典的表格型Q-Learning算法',
                'state_space': 'discrete',
                'memory_usage': 'high',
                'convergence_speed': 'fast',
                'scalability': 'poor'
            }
        elif algorithm == 'linear_fa':
            return {
                'name': 'Linear Function Approximation',
                'description': '线性函数近似的Q-Learning算法',
                'state_space': 'continuous',
                'memory_usage': 'low',
                'convergence_speed': 'medium',
                'scalability': 'good'
            }
        else:
            return {'error': f'Unknown algorithm: {algorithm}'}
    
    @staticmethod
    def compare_algorithms() -> Dict[str, Dict[str, Any]]:
        """
        比较不同算法
        
        Returns:
            比较结果字典
        """
        return {
            'q_table': AgentFactory.get_algorithm_info('q_table'),
            'linear_fa': AgentFactory.get_algorithm_info('linear_fa')
        }

# 统一的训练接口
class UnifiedTrainer:
    """统一训练接口"""
    
    def __init__(self, algorithm: str = 'q_table', 
                 environment: str = 'classic',
                 continuous: bool = False):
        """
        初始化统一训练器
        
        Args:
            algorithm: 算法类型
            environment: 环境类型
            continuous: 是否使用连续模式
        """
        self.algorithm = algorithm
        self.environment_type = environment
        self.continuous = continuous
        
        # 创建智能体和环境
        self.agent = AgentFactory.create_agent(algorithm)
        self.environment = AgentFactory.create_environment(environment, continuous)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
    
    def train(self, num_episodes: int, max_steps: int = 100, 
              save_interval: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        训练智能体
        
        Args:
            num_episodes: 训练回合数
            max_steps: 每回合最大步数
            save_interval: 保存间隔
            verbose: 是否显示训练过程
        
        Returns:
            训练结果
        """
        successful_episodes = 0
        
        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # 选择动作
                valid_actions = self.environment.get_valid_actions(state)
                action = self.agent.choose_action(state, valid_actions)
                
                # 执行动作
                next_state, reward, done = self.environment.step(action)
                
                # 更新智能体
                self.agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    # 检查是否成功
                    if hasattr(self.environment, 'robot_position'):
                        if self.environment.robot_position >= self.environment.end_position:
                            successful_episodes += 1
                    break
            
            # 记录统计信息
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # 衰减探索率
            self.agent.decay_epsilon()
            
            # 计算成功率
            if episode >= 49:  # 最近50个回合
                recent_successes = sum(1 for i in range(episode-49, episode+1) 
                                     if self.episode_rewards[i] > 0)
                self.success_rate.append(recent_successes / 50.0)
            
            # 显示进度
            if verbose and (episode + 1) % save_interval == 0:
                avg_reward = sum(self.episode_rewards[-save_interval:]) / save_interval
                recent_success_rate = self.success_rate[-1] if self.success_rate else 0
                print(f"Episode {episode + 1}: "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Success Rate: {recent_success_rate:.2%}, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rate': self.success_rate,
            'final_success_rate': successful_episodes / num_episodes,
            'algorithm': self.algorithm,
            'environment': self.environment_type,
            'continuous': self.continuous
        }
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save(filename)
    
    def load_model(self, filename: str):
        """加载模型"""
        self.agent.load(filename)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if hasattr(self.agent, 'get_weight_stats'):
            # 线性函数近似的统计
            return {
                'agent_stats': self.agent.get_weight_stats(),
                'feature_importance': self.agent.get_feature_importance(),
                'td_error_stats': self.agent.get_td_error_stats()
            }
        else:
            # Q-table的统计
            return {
                'agent_stats': self.agent.get_q_value_stats()
            }

# 测试函数
if __name__ == "__main__":
    # 测试智能体工厂
    print("Testing Agent Factory...")
    
    # 创建Q-table智能体
    q_agent = AgentFactory.create_agent('q_table')
    print(f"Q-table agent created: {type(q_agent)}")
    
    # 创建线性函数近似智能体
    linear_agent = AgentFactory.create_agent('linear_fa')
    print(f"Linear FA agent created: {type(linear_agent)}")
    
    # 创建环境
    classic_env = AgentFactory.create_environment('classic')
    enhanced_env = AgentFactory.create_environment('enhanced', continuous=True)
    print(f"Classic environment: {type(classic_env)}")
    print(f"Enhanced environment: {type(enhanced_env)}")
    
    # 比较算法
    comparison = AgentFactory.compare_algorithms()
    print("Algorithm comparison:")
    for alg, info in comparison.items():
        print(f"  {alg}: {info}")
    
    # 测试统一训练器
    print("\nTesting Unified Trainer...")
    trainer = UnifiedTrainer('linear_fa', 'enhanced', continuous=True)
    
    # 快速训练测试
    results = trainer.train(num_episodes=10, verbose=True)
    print(f"Training completed. Final success rate: {results['final_success_rate']:.2%}")