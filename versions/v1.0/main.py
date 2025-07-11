"""
主程序入口 v1.0 - 机器人过马路Q-Learning训练（纯v1.0版本）
"""
import pygame
import sys
import os
import argparse
from datetime import datetime

from .environment import RoadEnvironmentV1
from .q_learning import QLearningAgentV1
from .visualizer import VisualizerV1
from .config import TRAIN_CONFIG

def train(env, agent, visualizer=None, num_episodes=None):
    """训练主循环"""
    import pygame
    
    num_episodes = num_episodes or TRAIN_CONFIG['num_episodes']
    max_steps = TRAIN_CONFIG['max_steps_per_episode']
    save_interval = TRAIN_CONFIG['save_interval']
    
    # 创建保存目录
    os.makedirs('saved_models/v1.0', exist_ok=True)
    os.makedirs('logs/v1.0', exist_ok=True)
    
    # 训练日志
    log_filename = f"logs/v1.0/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"Starting v1.0 training for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"Saving model every {save_interval} episodes")
    
    # 写入CSV表头
    with open(log_filename, 'w') as f:
        f.write("episode,steps,total_reward,epsilon\\n")
    
    clock = pygame.time.Clock()
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新Q值
            agent.update_q_value(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_reward += reward
            step += 1
            
            # 可视化
            if visualizer:
                visualizer.update(env, agent, state, action, reward, done, 
                                episode, step, total_reward, agent.epsilon)
                
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Training interrupted by user")
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # 暂停/继续
                            paused = True
                            while paused:
                                for pause_event in pygame.event.get():
                                    if pause_event.type == pygame.QUIT:
                                        return
                                    elif pause_event.type == pygame.KEYDOWN:
                                        if pause_event.key == pygame.K_SPACE:
                                            paused = False
                                        elif pause_event.key == pygame.K_ESCAPE:
                                            return
                                clock.tick(10)  # 降低暂停时的CPU使用率
                        elif event.key == pygame.K_ESCAPE:
                            return
                
                clock.tick(visualizer.fps)
        
        # 记录训练数据
        with open(log_filename, 'a') as f:
            f.write(f"{episode},{step},{total_reward},{agent.epsilon:.6f}\\n")
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model_filename = f"saved_models/v1.0/q_table_v1.0_episode_{episode + 1}.pkl"
            agent.save_model(model_filename)
            print(f"Episode {episode + 1}: Steps={step}, Reward={total_reward:.2f}, Epsilon={agent.epsilon:.4f}")
    
    # 保存最终模型
    final_model_filename = f"saved_models/v1.0/q_table_v1.0_final.pkl"
    agent.save_model(final_model_filename)
    print(f"Training completed. Final model saved to {final_model_filename}")

def demo(env, agent, visualizer=None, model_path=None):
    """演示训练好的模型"""
    import pygame
    
    if model_path:
        agent.load_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print("No model loaded. Using current agent state.")
    
    # 演示模式设置
    agent.epsilon = 0.0  # 关闭探索
    
    print("Demo mode - Press SPACE to pause/resume, ESC to exit")
    
    clock = pygame.time.Clock()
    episode = 0
    
    while True:
        episode += 1
        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        print(f"\\n=== Demo Episode {episode} ===")
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if visualizer:
                visualizer.update(env, agent, state, action, reward, done, 
                                episode, step, total_reward, agent.epsilon)
                
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # 暂停/继续
                            paused = True
                            while paused:
                                for pause_event in pygame.event.get():
                                    if pause_event.type == pygame.QUIT:
                                        return
                                    elif pause_event.type == pygame.KEYDOWN:
                                        if pause_event.key == pygame.K_SPACE:
                                            paused = False
                                        elif pause_event.key == pygame.K_ESCAPE:
                                            return
                                clock.tick(10)
                        elif event.key == pygame.K_ESCAPE:
                            return
                
                clock.tick(visualizer.demo_fps)
        
        print(f"Episode {episode} completed: Steps={step}, Reward={total_reward:.2f}")
        
        # 等待一下再开始下一个演示
        pygame.time.wait(2000)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器人过马路 Q-Learning v1.0')
    parser.add_argument('--mode', choices=['train', 'demo', 'both'], default='train',
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练回合数')
    parser.add_argument('--no-vis', action='store_true',
                       help='不显示可视化界面')
    parser.add_argument('--fast', action='store_true',
                       help='快速训练模式（无延时）')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（用于演示）')
    
    args = parser.parse_args()
    
    # 创建环境和智能体
    env = RoadEnvironmentV1()
    agent = QLearningAgentV1(n_actions=2)
    
    # 创建可视化器
    visualizer = None
    if not args.no_vis:
        visualizer = VisualizerV1(env, agent)
        if args.fast:
            visualizer.fps = 0  # 无延时
    
    try:
        if args.mode == 'train':
            train(env, agent, visualizer, args.episodes)
        elif args.mode == 'demo':
            demo(env, agent, visualizer, args.model)
        elif args.mode == 'both':
            # 先训练
            train(env, agent, visualizer, args.episodes)
            # 再演示
            if visualizer:
                print("\\n切换到演示模式...")
                pygame.time.wait(2000)
                demo(env, agent, visualizer)
    
    except KeyboardInterrupt:
        print("\\n程序被用户中断")
    finally:
        if visualizer:
            pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()