#!/usr/bin/env python3
"""
反龟缩机制测试脚本
用于验证v1.0环境是否成功解决"站着不动"问题
"""

from environment_v1 import RoadEnvironmentV1
from q_learning_v1 import QLearningAgentV1

def test_anti_stuck_mechanism():
    """测试反龟缩机制效果"""
    print("🧪 测试反龟缩机制")
    print("=" * 50)
    
    env = RoadEnvironmentV1()
    agent = QLearningAgentV1(n_actions=len(env.actions))
    
    # 1. 测试奖励机制
    print("\n📋 1. 奖励机制测试:")
    
    # 绿灯前进
    env.time_step = 0  # 绿灯开始
    state = env.reset()
    next_state, reward, done = env.step(0)  # Forward
    print(f"   绿灯前进: {reward:+3.0f} (期望: 0)")
    
    # 起点等待
    env.reset()
    env.robot_position = -1
    next_state, reward, done = env.step(1)  # Backward
    print(f"   起点等待: {reward:+3.0f} (期望: -3)")
    
    # 2. 训练测试
    print(f"\n🏃 2. 训练效果测试 (300 episodes):")
    
    total_episodes = 300
    stuck_episodes = 0
    start_forward = 0
    start_backward = 0
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_stuck = True
        
        for step in range(40):
            action = agent.choose_action(state)
            
            # 统计起点行为
            if state.robot_lane == -1:
                if action == 0:
                    start_forward += 1
                else:
                    start_backward += 1
            
            # 检查是否离开起点
            if state.robot_lane > -1:
                episode_stuck = False
            
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
        
        if episode_stuck:
            stuck_episodes += 1
        
        # 进度显示
        if (episode + 1) % 100 == 0:
            current_stuck_rate = stuck_episodes / (episode + 1) * 100
            print(f"   Episode {episode+1:3d}: 卡住率 {current_stuck_rate:4.1f}%")
    
    # 3. 结果分析
    print(f"\n📊 3. 最终结果:")
    stuck_rate = stuck_episodes / total_episodes * 100
    
    if start_forward + start_backward > 0:
        forward_rate = start_forward / (start_forward + start_backward) * 100
    else:
        forward_rate = 0
    
    print(f"   卡住率: {stuck_rate:4.1f}% (目标: <25%)")
    print(f"   起点前进率: {forward_rate:4.1f}% (目标: >50%)")
    
    # 4. Q值样例
    print(f"\n🧠 4. 学习到的Q值 (起点状态):")
    start_states = [s for s in agent.q_table.keys() if s.robot_lane == -1]
    
    for i, state in enumerate(start_states[:3]):
        q_values = agent.q_table[state]
        preference = "Forward ✅" if q_values[0] > q_values[1] else "Backward ⚠️"
        print(f"   状态 {i+1}: Forward({q_values[0]:+5.2f}) vs Backward({q_values[1]:+5.2f}) → {preference}")
    
    # 5. 总结
    print(f"\n🎯 5. 反龟缩机制评估:")
    
    success_criteria = [
        ("卡住率 < 25%", stuck_rate < 25),
        ("起点前进率 > 50%", forward_rate > 50),
        ("Q值偏好前进", len(start_states) > 0 and 
         sum(1 for s in start_states if agent.q_table[s][0] > agent.q_table[s][1]) > len(start_states)/2)
    ]
    
    passed = sum(1 for _, condition in success_criteria if condition)
    
    for criterion, condition in success_criteria:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"   {criterion}: {status}")
    
    overall_status = "🎉 SUCCESS" if passed >= 2 else "⚠️  NEEDS IMPROVEMENT"
    print(f"\n{overall_status}: {passed}/3 criteria passed")
    
    return passed >= 2

if __name__ == "__main__":
    success = test_anti_stuck_mechanism()
    exit(0 if success else 1)