"""
主程序入口 v2.0 - 支持线性函数近似的机器人过马路训练
"""
import pygame
import sys
import os
import argparse
from datetime import datetime

from .agent_factory import AgentFactory, UnifiedTrainer
from .config import TRAIN_CONFIG, ALGORITHM_CONFIG
from .visualizer import Visualizer

def train_with_factory(algorithm='linear_fa', environment='enhanced', 
                      continuous=True, num_episodes=None, visualizer=None):
    """使用工厂模式的训练函数"""
    num_episodes = num_episodes or TRAIN_CONFIG['num_episodes']
    max_steps = TRAIN_CONFIG['max_steps_per_episode']
    save_interval = TRAIN_CONFIG['save_interval']
    
    # 创建保存目录
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 创建统一训练器
    trainer = UnifiedTrainer(algorithm, environment, continuous)
    
    # 如果有可视化器，更新其环境和智能体引用
    if visualizer:
        visualizer.env = trainer.environment
        visualizer.agent = trainer.agent
    
    # 训练日志
    log_filename = f"logs/training_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"Starting training with {algorithm} algorithm...")
    print(f"Environment: {environment}, Continuous: {continuous}")
    print(f"Training for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"Saving model every {save_interval} episodes")
    
    try:
        for episode in range(num_episodes):
            # 重置环境
            state = trainer.environment.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # 处理Pygame事件
                if visualizer:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining interrupted by user")
                            return
                
                # 选择动作
                valid_actions = trainer.environment.get_valid_actions(state)
                action = trainer.agent.choose_action(state, valid_actions)
                
                # 执行动作
                next_state, reward, done = trainer.environment.step(action)
                total_reward += reward
                
                # 更新智能体
                trainer.agent.update(state, action, reward, next_state, done)
                
                # 可视化更新
                if visualizer:
                    visualizer.update(state, action, reward, done)
                    visualizer.clock.tick(visualizer.fps)
                
                state = next_state
                steps += 1
            
            # 更新探索率
            trainer.agent.decay_epsilon()
            trainer.episode_rewards.append(total_reward)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = sum(trainer.episode_rewards[-10:]) / min(10, len(trainer.episode_rewards))
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Steps: {steps} | "
                      f"Total Reward: {total_reward:.1f} | "
                      f"Avg Reward (10): {avg_reward:.1f} | "
                      f"ε: {trainer.agent.epsilon:.3f}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                model_filename = f"saved_models/{algorithm}_episode_{episode + 1}.pkl"
                trainer.agent.save(model_filename)
            
            # 写入日志
            with open(log_filename, 'a') as f:
                f.write(f"{episode + 1},{steps},{total_reward},{trainer.agent.epsilon}\n")
        
        # 保存最终模型
        final_model_filename = f"saved_models/{algorithm}_final.pkl"
        trainer.agent.save(final_model_filename)
        print(f"\nTraining completed! Final model saved to {final_model_filename}")
        
        # 打印统计信息
        stats = trainer.get_stats()
        print(f"\nAlgorithm Statistics:")
        
        if algorithm == 'linear_fa':
            agent_stats = stats['agent_stats']
            print(f"  Number of parameters: {agent_stats['num_parameters']}")
            print(f"  Weight mean: {agent_stats['weight_mean']:.4f}")
            print(f"  Weight std: {agent_stats['weight_std']:.4f}")
            print(f"  Weight norm: {agent_stats['weight_norm']:.4f}")
            
            # 特征重要性
            importance = stats['feature_importance']
            print(f"\nFeature Importance:")
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {imp:.4f}")
            
            # TD误差统计
            td_stats = stats['td_error_stats']
            print(f"\nTD Error Statistics:")
            print(f"  Mean: {td_stats['mean']:.4f}")
            print(f"  Std: {td_stats['std']:.4f}")
            print(f"  Max: {td_stats['max']:.4f}")
            print(f"  Recent Mean: {td_stats['recent_mean']:.4f}")
        else:
            agent_stats = stats['agent_stats']
            print(f"  Number of states explored: {agent_stats['num_states']}")
            print(f"  Average Q-value: {agent_stats['avg_q_value']:.2f}")
            print(f"  Max Q-value: {agent_stats['max_q_value']:.2f}")
            print(f"  Min Q-value: {agent_stats['min_q_value']:.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # 保存当前进度
        interrupt_model_filename = f"saved_models/{algorithm}_interrupted_episode_{episode + 1}.pkl"
        trainer.agent.save(interrupt_model_filename)
        print(f"Current progress saved to {interrupt_model_filename}")

def demonstrate_with_factory(algorithm='linear_fa', environment='enhanced', 
                           continuous=True, model_path=None, visualizer=None):
    """使用工厂模式的演示函数"""
    print("\nStarting demonstration mode...")
    print("Press ESC to exit")
    
    # 创建智能体和环境
    agent = AgentFactory.create_agent(algorithm)
    env = AgentFactory.create_environment(environment, continuous)
    
    # 加载模型
    if model_path:
        agent.load(model_path)
    else:
        # 尝试加载最终模型
        final_model = f"saved_models/{algorithm}_final.pkl"
        if os.path.exists(final_model):
            agent.load(final_model)
        else:
            print("No trained model found. Please train first or specify model with --load")
            return
    
    # 创建或更新可视化器
    if not visualizer:
        visualizer = Visualizer(env, agent)
    else:
        # 更新可视化器的环境和智能体引用
        visualizer.env = env
        visualizer.agent = agent
    
    # 设置演示模式的帧率（更慢）
    visualizer.set_fps(visualizer.config['demo_fps'])
    
    # 设置agent为纯利用模式
    agent.epsilon = 0
    
    running = True
    episode_count = 0
    
    while running:
        # 运行一个episode
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_count += 1
        
        print(f"\nStarting episode {episode_count}...")
        
        # 显示初始状态
        visualizer.update(state, None, 0, False, show_initial=True)
        visualizer.clock.tick(visualizer.fps)
        
        while not done:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
            
            if not running:
                break
            
            # 选择最优动作
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 打印决策信息
            action_name = env.actions[action]
            if algorithm == 'linear_fa':
                # 线性函数近似的调试信息
                debug_info = agent.debug_prediction(state)
                q_values = debug_info['q_values']
                print(f"Step {steps}: State pos={getattr(state, 'robot_position', 'N/A'):.1f} "
                      f"light={getattr(state, 'light_status', 'N/A')} -> Action: {action_name} "
                      f"(Q-values: Forward={q_values['forward']:.2f}, Backward={q_values['backward']:.2f})")
            else:
                # Q-table的调试信息
                q_values = agent.q_table[state]
                print(f"Step {steps}: State {state} -> Action: {action_name} "
                      f"(Q-values: Forward={q_values[0]:.2f}, Backward={q_values[1]:.2f})")
            
            # 可视化
            visualizer.update(state, action, reward, done)
            visualizer.clock.tick(visualizer.fps)
            
            state = next_state
            steps += 1
        
        if running:
            print(f"Episode {episode_count} finished! Steps: {steps}, Total Reward: {total_reward}")
            # 等待一下再开始下一个episode
            import time
            time.sleep(2)
    
    print("\nDemonstration ended")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Robot Road Crossing v2.0 - Linear Function Approximation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo', 'both'],
                       help='Mode: train, demo, or both')
    parser.add_argument('--algorithm', type=str, default='linear_fa', choices=['q_table', 'linear_fa'],
                       help='Algorithm: q_table or linear_fa')
    parser.add_argument('--environment', type=str, default='enhanced', choices=['classic', 'enhanced'],
                       help='Environment: classic or enhanced')
    parser.add_argument('--continuous', action='store_true', default=True,
                       help='Use continuous position mode')
    parser.add_argument('--discrete', action='store_true',
                       help='Use discrete position mode (overrides --continuous)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization during training')
    parser.add_argument('--load', type=str, default=None,
                       help='Load pre-trained model from file')
    parser.add_argument('--fast', action='store_true',
                       help='Fast training mode (no frame rate limit)')
    
    args = parser.parse_args()
    
    # 处理连续/离散模式
    continuous = args.continuous and not args.discrete
    
    # 创建可视化器（如果需要）
    visualizer = None
    if not args.no_vis or args.mode in ['demo', 'both']:
        # 创建临时环境和智能体用于可视化器初始化
        temp_env = AgentFactory.create_environment(args.environment, continuous)
        temp_agent = AgentFactory.create_agent(args.algorithm)
        visualizer = Visualizer(temp_env, temp_agent)
        
        # 设置快速训练模式
        if args.fast and args.mode in ['train', 'both']:
            visualizer.set_fps(visualizer.config['fast_fps'])
    
    try:
        if args.mode == 'train':
            train_with_factory(args.algorithm, args.environment, continuous, 
                             args.episodes, visualizer)
        elif args.mode == 'demo':
            demonstrate_with_factory(args.algorithm, args.environment, continuous, 
                                   args.load, visualizer)
        elif args.mode == 'both':
            train_with_factory(args.algorithm, args.environment, continuous, 
                             args.episodes, visualizer)
            demonstrate_with_factory(args.algorithm, args.environment, continuous, 
                                   None, visualizer)
    
    finally:
        if visualizer:
            visualizer.close()

def compare_algorithms():
    """比较不同算法的性能"""
    print("Comparing Q-table vs Linear Function Approximation...")
    
    algorithms = ['q_table', 'linear_fa']
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm}...")
        trainer = UnifiedTrainer(algorithm, 'enhanced', continuous=True)
        
        # 短时间训练测试
        result = trainer.train(num_episodes=100, verbose=False)
        results[algorithm] = result
        
        print(f"  Final success rate: {result['final_success_rate']:.2%}")
        print(f"  Average episode length: {sum(result['episode_lengths'])/len(result['episode_lengths']):.1f}")
        print(f"  Average reward: {sum(result['episode_rewards'])/len(result['episode_rewards']):.2f}")
    
    # 生成比较报告
    print("\n" + "="*50)
    print("ALGORITHM COMPARISON REPORT")
    print("="*50)
    
    for algorithm in algorithms:
        result = results[algorithm]
        print(f"\n{algorithm.upper()}:")
        print(f"  Success Rate: {result['final_success_rate']:.2%}")
        print(f"  Avg Episode Length: {sum(result['episode_lengths'])/len(result['episode_lengths']):.1f}")
        print(f"  Avg Reward: {sum(result['episode_rewards'])/len(result['episode_rewards']):.2f}")
        
        if result['success_rate']:
            final_success_rate = result['success_rate'][-1]
            print(f"  Final 50-episode Success Rate: {final_success_rate:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_algorithms()
    else:
        main()