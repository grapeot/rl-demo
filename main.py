"""
主程序入口 - 机器人过马路Q-Learning训练
"""
import pygame
import sys
import os
import argparse
from datetime import datetime

from environment import RoadEnvironment
from q_learning import QLearningAgent
from visualizer import Visualizer
from config import TRAIN_CONFIG

def train(env, agent, visualizer=None, num_episodes=None):
    """训练主循环"""
    num_episodes = num_episodes or TRAIN_CONFIG['num_episodes']
    max_steps = TRAIN_CONFIG['max_steps_per_episode']
    save_interval = TRAIN_CONFIG['save_interval']
    
    # 创建保存目录
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 训练日志
    log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"Saving model every {save_interval} episodes")
    
    try:
        for episode in range(num_episodes):
            # 重置环境
            state = env.reset()
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
                valid_actions = env.get_valid_actions(state)
                action = agent.choose_action(state, valid_actions)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                # 更新Q值
                agent.update(state, action, reward, next_state, done)
                
                # 可视化更新
                if visualizer:
                    visualizer.update(state, action, reward, done)
                
                state = next_state
                steps += 1
            
            # 更新探索率
            agent.decay_epsilon()
            agent.total_reward_history.append(total_reward)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = sum(agent.total_reward_history[-10:]) / min(10, len(agent.total_reward_history))
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Steps: {steps} | "
                      f"Total Reward: {total_reward:.1f} | "
                      f"Avg Reward (10): {avg_reward:.1f} | "
                      f"ε: {agent.epsilon:.3f}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                model_filename = f"saved_models/q_table_episode_{episode + 1}.pkl"
                agent.save(model_filename)
            
            # 写入日志
            with open(log_filename, 'a') as f:
                f.write(f"{episode + 1},{steps},{total_reward},{agent.epsilon}\n")
        
        # 保存最终模型
        final_model_filename = "saved_models/q_table_final.pkl"
        agent.save(final_model_filename)
        print(f"\nTraining completed! Final model saved to {final_model_filename}")
        
        # 打印统计信息
        stats = agent.get_q_value_stats()
        print(f"\nQ-Table Statistics:")
        print(f"  Number of states explored: {stats['num_states']}")
        print(f"  Average Q-value: {stats['avg_q_value']:.2f}")
        print(f"  Max Q-value: {stats['max_q_value']:.2f}")
        print(f"  Min Q-value: {stats['min_q_value']:.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # 保存当前进度
        interrupt_model_filename = f"saved_models/q_table_interrupted_episode_{episode + 1}.pkl"
        agent.save(interrupt_model_filename)
        print(f"Current progress saved to {interrupt_model_filename}")

def demonstrate(env, agent, visualizer):
    """演示训练好的策略"""
    print("\nStarting demonstration mode...")
    print("Press SPACE to start next episode, ESC to exit")
    
    # 设置演示模式的帧率（更慢）
    visualizer.set_fps(visualizer.config['demo_fps'])
    
    # 设置agent为纯利用模式
    agent.epsilon = 0
    
    running = True
    waiting = True
    
    while running:
        # 等待用户按键
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        waiting = False
            
            # 绘制等待界面
            visualizer.draw()
            wait_text = visualizer.font.render("Press SPACE to start episode", True, (0, 0, 0))
            text_rect = wait_text.get_rect(center=(visualizer.screen.get_width() // 2, 
                                                  visualizer.screen.get_height() // 2))
            visualizer.screen.blit(wait_text, text_rect)
            pygame.display.flip()
        
        if not running:
            break
        
        # 运行一个episode
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nStarting new episode...")
        
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
            q_values = agent.q_table[state]
            print(f"Step {steps}: State {state} -> Action: {action_name} "
                  f"(Q-values: Forward={q_values[0]:.2f}, Backward={q_values[1]:.2f})")
            
            # 可视化
            visualizer.update(state, action, reward, done)
            
            state = next_state
            steps += 1
        
        if running:
            print(f"Episode finished! Steps: {steps}, Total Reward: {total_reward}")
            waiting = True
    
    print("\nDemonstration ended")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Robot Road Crossing Q-Learning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'demo', 'both'],
                       help='Mode: train, demo, or both')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization during training')
    parser.add_argument('--load', type=str, default=None,
                       help='Load pre-trained model from file')
    parser.add_argument('--fast', action='store_true',
                       help='Fast training mode (no frame rate limit)')
    
    args = parser.parse_args()
    
    # 创建环境和智能体
    env = RoadEnvironment()
    agent = QLearningAgent(n_actions=len(env.actions))
    
    # 加载已有模型
    if args.load:
        agent.load(args.load)
        print(f"Loaded model from {args.load}")
    
    # 创建可视化器（如果需要）
    visualizer = None
    if not args.no_vis or args.mode in ['demo', 'both']:
        visualizer = Visualizer(env, agent)
        # 设置快速训练模式
        if args.fast and args.mode in ['train', 'both']:
            visualizer.set_fps(visualizer.config['fast_fps'])
    
    try:
        if args.mode == 'train':
            train(env, agent, visualizer, args.episodes)
        elif args.mode == 'demo':
            if not args.load:
                # 尝试加载最终模型
                final_model = "saved_models/q_table_final.pkl"
                if os.path.exists(final_model):
                    agent.load(final_model)
                else:
                    print("No trained model found. Please train first or specify model with --load")
                    return
            if not visualizer:
                visualizer = Visualizer(env, agent)
            demonstrate(env, agent, visualizer)
        elif args.mode == 'both':
            train(env, agent, visualizer, args.episodes)
            if not visualizer:
                visualizer = Visualizer(env, agent)
            demonstrate(env, agent, visualizer)
    
    finally:
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()