"""
主程序入口 v1.0 - 支持v0.5和v1.0环境的机器人过马路Q-Learning训练
"""
import pygame
import sys
import os
import argparse
from datetime import datetime

# v0.5 imports
from environment import RoadEnvironment
from q_learning import QLearningAgent
from visualizer import Visualizer

# v1.0 imports
from environment_v1 import RoadEnvironmentV1
from q_learning_v1 import QLearningAgentV1
from visualizer_v1 import VisualizerV1

from config import TRAIN_CONFIG

def train(env, agent, visualizer=None, num_episodes=None, version="v1.0"):
    """训练主循环 - 支持v0.5和v1.0"""
    import pygame
    
    num_episodes = num_episodes or TRAIN_CONFIG['num_episodes']
    
    # v1.0需要更多episode训练
    if version == "v1.0" and num_episodes == TRAIN_CONFIG['num_episodes']:
        num_episodes = 1500  # 缩减到1500个episode
        print(f"v1.0 mode detected: increasing episodes to {num_episodes}")
    
    max_steps = TRAIN_CONFIG['max_steps_per_episode']
    save_interval = TRAIN_CONFIG['save_interval']
    
    # 创建保存目录
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 训练日志
    log_filename = f"logs/training_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"Starting {version} training for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps}")
    print(f"Saving model every {save_interval} episodes")
    
    # 写入CSV表头
    with open(log_filename, 'w') as f:
        f.write("episode,steps,total_reward,epsilon\\n")
    
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
                            print("\\nTraining interrupted by user")
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
            if (episode + 1) % 50 == 0:  # v1.0减少打印频率
                avg_reward = sum(agent.total_reward_history[-50:]) / min(50, len(agent.total_reward_history))
                
                progress_info = (
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Steps: {steps} | "
                    f"Total Reward: {total_reward:.1f} | "
                    f"Avg Reward (50): {avg_reward:.1f} | "
                    f"ε: {agent.epsilon:.3f}"
                )
                
                # v1.0显示状态空间信息
                if version == "v1.0" and hasattr(agent, 'get_q_value_stats'):
                    stats = agent.get_q_value_stats()
                    progress_info += f" | States: {stats['num_states']} | Coverage: {stats.get('state_coverage', 0):.1f}%"
                
                print(progress_info)
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                model_filename = f"saved_models/q_table_{version}_episode_{episode + 1}.pkl"
                agent.save(model_filename)
            
            # 写入日志
            with open(log_filename, 'a') as f:
                f.write(f"{episode + 1},{steps},{total_reward},{agent.epsilon}\\n")
        
        # 保存最终模型
        final_model_filename = f"saved_models/q_table_{version}_final.pkl"
        agent.save(final_model_filename)
        print(f"\\nTraining completed! Final model saved to {final_model_filename}")
        
        # 打印统计信息
        stats = agent.get_q_value_stats()
        print(f"\\nQ-Table Statistics:")
        print(f"  Number of states explored: {stats['num_states']}")
        print(f"  Average Q-value: {stats['avg_q_value']:.2f}")
        print(f"  Max Q-value: {stats['max_q_value']:.2f}")
        print(f"  Min Q-value: {stats['min_q_value']:.2f}")
        
        # v1.0特定统计
        if version == "v1.0" and hasattr(agent, 'get_state_exploration_stats'):
            exp_stats = agent.get_state_exploration_stats()
            print(f"\\nState Exploration Statistics:")
            print(f"  State coverage: {stats.get('state_coverage', 0):.1f}% ({stats['num_states']}/{stats.get('theoretical_max_states', 420)})")
            print(f"  Avg visits per state: {exp_stats['avg_visits_per_state']:.1f}")
            print(f"  Max visits to a state: {exp_stats['max_visits']}")
        
        # 截图等待机制：训练完成后等待用户截图
        if visualizer:
            print(f"\\n📸 Training completed! Updating final visualization for screenshot...")
            
            # 更新最终可视化显示
            final_state = env._get_state()
            visualizer.update(final_state, None, 0, True, show_initial=True)
            
            print("🖼️  Ready for screenshot!")
            print("   - Success rate and death rate charts are displayed")
            print("   - Press any key in the visualization window to continue")
            print("   - Or close the window manually when done")
            
            # 等待用户操作
            import pygame
            waiting_for_screenshot = True
            while waiting_for_screenshot:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_screenshot = False
                    elif event.type == pygame.KEYDOWN:
                        print("   User input detected - continuing...")
                        waiting_for_screenshot = False
                
                # 保持窗口更新
                visualizer.draw()
                visualizer.clock.tick(10)  # 低帧率等待
        
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user")
        # 保存当前进度
        interrupt_model_filename = f"saved_models/q_table_{version}_interrupted_episode_{episode + 1}.pkl"
        agent.save(interrupt_model_filename)
        print(f"Current progress saved to {interrupt_model_filename}")
        
        # 中断后也提供截图机会
        if visualizer:
            print(f"\\n📸 Training interrupted! Updating visualization for screenshot...")
            
            # 更新可视化显示
            try:
                current_state = env._get_state()
                visualizer.update(current_state, None, 0, True, show_initial=True)
                
                print("🖼️  Ready for screenshot!")
                print("   - Current progress charts are displayed")
                print("   - Press any key in the visualization window to continue")
                print("   - Or close the window manually when done")
                
                # 等待用户操作
                import pygame
                waiting_for_screenshot = True
                while waiting_for_screenshot:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting_for_screenshot = False
                        elif event.type == pygame.KEYDOWN:
                            print("   User input detected - continuing...")
                            waiting_for_screenshot = False
                    
                    # 保持窗口更新
                    visualizer.draw()
                    visualizer.clock.tick(10)  # 低帧率等待
            except:
                print("   Screenshot mode failed, continuing...")
                pass

def demonstrate(env, agent, visualizer, version="v1.0"):
    """演示训练好的策略"""
    print(f"\\nStarting {version} demonstration mode...")
    print("Press ESC to exit, SPACE to pause/resume")
    
    # 设置演示模式的帧率（更慢）
    visualizer.set_fps(visualizer.config['demo_fps'])
    
    # 设置agent为纯利用模式
    agent.epsilon = 0
    
    running = True
    episode_count = 0
    paused = False
    
    while running:
        # 运行一个episode
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_count += 1
        
        print(f"\\nStarting episode {episode_count}...")
        
        # 显示初始状态
        visualizer.update(state, None, 0, False, show_initial=True)
        
        while not done and running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
            
            if not running or paused:
                if paused:
                    # 在暂停时仍然渲染当前帧
                    visualizer.draw()
                continue
            
            # 选择最优动作
            valid_actions = env.get_valid_actions(state)
            action = agent.choose_action(state, valid_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # 打印决策信息
            action_name = env.actions[action]
            q_values = agent.q_table[state]
            
            # v1.0显示更详细的状态信息
            if version == "v1.0":
                print(f"Step {steps}: State(robot={state.robot_lane}, light={state.light_status}({state.light_countdown}), car={state.next_lane_car}) -> "
                      f"Action: {action_name} (Q-values: Forward={q_values[0]:.2f}, Backward={q_values[1]:.2f})")
            else:
                print(f"Step {steps}: State {state} -> Action: {action_name} "
                      f"(Q-values: Forward={q_values[0]:.2f}, Backward={q_values[1]:.2f})")
            
            # 可视化
            visualizer.update(state, action, reward, done)
            
            state = next_state
            steps += 1
        
        if running:
            print(f"Episode {episode_count} finished! Steps: {steps}, Total Reward: {total_reward}")
            # 等待一下再开始下一个episode
            import time
            time.sleep(2)
    
    print("\\nDemonstration ended")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Robot Road Crossing Q-Learning v1.0')
    parser.add_argument('--version', type=str, default='v1.0', choices=['v0.5', 'v1.0'],
                       help='Environment version: v0.5 or v1.0')
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
    
    print(f"Starting Robot Road Crossing {args.version}")
    print(f"Mode: {args.mode}")
    
    # 创建环境、智能体和可视化器
    if args.version == "v0.5":
        env = RoadEnvironment()
        agent = QLearningAgent(n_actions=len(env.actions))
        visualizer_class = Visualizer
    else:  # v1.0
        env = RoadEnvironmentV1()
        agent = QLearningAgentV1(n_actions=len(env.actions))
        visualizer_class = VisualizerV1
    
    # 加载已有模型
    if args.load:
        if os.path.exists(args.load):
            agent.load(args.load)
            print(f"Loaded model from {args.load}")
        else:
            print(f"Model file {args.load} not found!")
            return
    
    # 创建可视化器（如果需要）
    visualizer = None
    if not args.no_vis or args.mode in ['demo', 'both']:
        visualizer = visualizer_class(env, agent)
        # 设置快速训练模式
        if args.fast and args.mode in ['train', 'both']:
            visualizer.set_fps(visualizer.config['fast_fps'])
    
    try:
        if args.mode == 'train':
            train(env, agent, visualizer, args.episodes, args.version)
        elif args.mode == 'demo':
            if not args.load:
                # 尝试加载对应版本的最终模型
                final_model = f"saved_models/q_table_{args.version}_final.pkl"
                if os.path.exists(final_model):
                    agent.load(final_model)
                    print(f"Loaded {args.version} model from {final_model}")
                else:
                    print(f"No trained {args.version} model found. Please train first or specify model with --load")
                    print(f"Expected file: {final_model}")
                    return
            if not visualizer:
                visualizer = visualizer_class(env, agent)
            demonstrate(env, agent, visualizer, args.version)
        elif args.mode == 'both':
            train(env, agent, visualizer, args.episodes, args.version)
            if not visualizer:
                visualizer = visualizer_class(env, agent)
            demonstrate(env, agent, visualizer, args.version)
    
    finally:
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()