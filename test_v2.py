"""
v2.0功能测试脚本
"""
import sys
import traceback
from feature_engineering import extract_features, create_test_state, validate_features
from linear_q_learning import LinearQLearning
from enhanced_environment import EnhancedRoadEnvironment
from agent_factory import AgentFactory, UnifiedTrainer

def test_feature_engineering():
    """测试特征工程"""
    print("Testing Feature Engineering...")
    
    try:
        # 创建测试状态
        state = create_test_state(robot_pos=2.0, light_status=1, light_countdown=5)
        
        # 提取特征
        features_forward = extract_features(state, 0)
        features_backward = extract_features(state, 1)
        
        # 验证特征
        assert len(features_forward) == 9, f"Expected 9 features, got {len(features_forward)}"
        assert len(features_backward) == 9, f"Expected 9 features, got {len(features_backward)}"
        assert validate_features(features_forward), "Forward features validation failed"
        assert validate_features(features_backward), "Backward features validation failed"
        
        print("  ✓ Feature extraction working correctly")
        print(f"  ✓ Features shape: {features_forward.shape}")
        print(f"  ✓ Feature range: [{features_forward.min():.3f}, {features_forward.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Feature engineering test failed: {e}")
        traceback.print_exc()
        return False

def test_linear_q_learning():
    """测试线性Q-Learning"""
    print("\nTesting Linear Q-Learning...")
    
    try:
        # 创建智能体
        agent = LinearQLearning(n_features=9, n_actions=2, alpha=0.1)
        
        # 创建测试状态
        state = create_test_state()
        
        # 测试Q值计算
        q_values = agent.get_q_values(state)
        assert len(q_values) == 2, f"Expected 2 Q-values, got {len(q_values)}"
        
        # 测试动作选择
        action = agent.choose_action(state)
        assert action in [0, 1], f"Invalid action: {action}"
        
        # 测试权重更新
        next_state = create_test_state(robot_pos=2.5)
        initial_weights = agent.weights.copy()
        td_error = agent.update(state, action, 1.0, next_state, False)
        
        # 权重应该发生变化
        assert not (initial_weights == agent.weights).all(), "Weights not updated"
        
        print("  ✓ Linear Q-Learning working correctly")
        print(f"  ✓ Q-values: {q_values}")
        print(f"  ✓ TD error: {td_error:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Linear Q-Learning test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_environment():
    """测试增强环境"""
    print("\nTesting Enhanced Environment...")
    
    try:
        # 创建环境
        env = EnhancedRoadEnvironment(continuous_mode=True)
        
        # 重置环境
        initial_state = env.reset()
        
        # 检查状态类型
        assert hasattr(initial_state, 'robot_position'), "Missing robot_position attribute"
        assert hasattr(initial_state, 'light_status'), "Missing light_status attribute"
        assert hasattr(initial_state, 'light_countdown'), "Missing light_countdown attribute"
        
        # 测试动作执行
        action = 0  # Forward
        next_state, reward, done = env.step(action)
        
        # 检查状态变化
        assert next_state.robot_position > initial_state.robot_position, "Robot position should increase"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        
        print("  ✓ Enhanced Environment working correctly")
        print(f"  ✓ Initial position: {initial_state.robot_position}")
        print(f"  ✓ After forward: {next_state.robot_position}")
        print(f"  ✓ Reward: {reward}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Enhanced Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_factory():
    """测试智能体工厂"""
    print("\nTesting Agent Factory...")
    
    try:
        # 测试创建不同类型的智能体
        q_agent = AgentFactory.create_agent('q_table')
        linear_agent = AgentFactory.create_agent('linear_fa')
        
        # 测试创建不同类型的环境
        classic_env = AgentFactory.create_environment('classic')
        enhanced_env = AgentFactory.create_environment('enhanced', continuous=True)
        
        print("  ✓ Agent Factory working correctly")
        print(f"  ✓ Q-table agent: {type(q_agent).__name__}")
        print(f"  ✓ Linear FA agent: {type(linear_agent).__name__}")
        print(f"  ✓ Classic environment: {type(classic_env).__name__}")
        print(f"  ✓ Enhanced environment: {type(enhanced_env).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Agent Factory test failed: {e}")
        traceback.print_exc()
        return False

def test_unified_trainer():
    """测试统一训练器"""
    print("\nTesting Unified Trainer...")
    
    try:
        # 创建训练器
        trainer = UnifiedTrainer('linear_fa', 'enhanced', continuous=True)
        
        # 短时间训练
        results = trainer.train(num_episodes=5, verbose=False)
        
        # 检查结果
        assert 'episode_rewards' in results, "Missing episode_rewards"
        assert 'episode_lengths' in results, "Missing episode_lengths"
        assert len(results['episode_rewards']) == 5, "Wrong number of episodes"
        
        print("  ✓ Unified Trainer working correctly")
        print(f"  ✓ Episodes completed: {len(results['episode_rewards'])}")
        print(f"  ✓ Average reward: {sum(results['episode_rewards'])/len(results['episode_rewards']):.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Unified Trainer test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性"""
    print("\nTesting Backward Compatibility...")
    
    try:
        # 测试原始Q-Learning仍然工作
        from q_learning import QLearningAgent
        from environment import RoadEnvironment
        
        env = RoadEnvironment()
        agent = QLearningAgent(n_actions=2)
        
        # 简单的训练循环
        state = env.reset()
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        print("  ✓ Backward compatibility maintained")
        print("  ✓ Original Q-Learning still works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("Running v2.0 Comprehensive Tests")
    print("="*60)
    
    tests = [
        test_feature_engineering,
        test_linear_q_learning,
        test_enhanced_environment,
        test_agent_factory,
        test_unified_trainer,
        test_backward_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! v2.0 implementation is ready.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)