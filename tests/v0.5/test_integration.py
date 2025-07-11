"""
集成测试 - 测试主要组件的集成
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import RoadEnvironment
from q_learning import QLearningAgent

def test_training_loop():
    """测试基本的训练循环"""
    print("Testing basic training loop...")
    
    # 创建环境和智能体
    env = RoadEnvironment()
    agent = QLearningAgent(n_actions=len(env.actions))
    
    # 运行几个episode
    num_episodes = 10
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            # 选择动作
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 更新Q值
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
        
        if episode % 5 == 0:
            print(f"Episode {episode}: Steps={steps}, Total Reward={total_reward}, Epsilon={agent.epsilon:.3f}")
    
    # 检查是否学到了一些状态
    stats = agent.get_q_value_stats()
    print(f"\nFinal statistics:")
    print(f"States explored: {stats['num_states']}")
    print(f"Average Q-value: {stats['avg_q_value']:.2f}")
    
    assert stats['num_states'] > 0, "No states were explored!"
    print("\nTraining loop test passed!")

def test_deterministic_behavior():
    """测试训练后的确定性行为"""
    print("\nTesting deterministic behavior after training...")
    
    # 创建环境和智能体
    env = RoadEnvironment()
    agent = QLearningAgent(n_actions=len(env.actions))
    
    # 快速训练
    for _ in range(50):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
    
    # 设置为纯利用模式
    agent.epsilon = 0
    
    # 测试确定性行为
    state = env.reset()
    actions_taken = []
    
    for _ in range(5):
        valid_actions = env.get_valid_actions(state)
        action = agent.choose_action(state, valid_actions)
        actions_taken.append(action)
        
        # 多次选择应该得到相同的动作
        for _ in range(3):
            same_action = agent.choose_action(state, valid_actions)
            assert same_action == action, "Deterministic behavior failed!"
        
        next_state, _, done = env.step(action)
        if done:
            break
        state = next_state
    
    print(f"Deterministic actions taken: {actions_taken}")
    print("Deterministic behavior test passed!")

if __name__ == "__main__":
    test_training_loop()
    test_deterministic_behavior()
    print("\nAll tests passed!")