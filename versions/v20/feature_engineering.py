"""
特征工程模块 - 用于线性函数近似的特征提取
"""
import numpy as np
from typing import Dict, List, Union, Any
from collections import namedtuple

# 连续状态定义
ContinuousState = namedtuple('ContinuousState', [
    'robot_position',    # 机器人连续位置
    'light_status',      # 红绿灯状态
    'light_countdown',   # 倒计时
    'car_positions',     # 车辆位置列表
    'car_speeds'         # 车辆速度列表
])

def extract_features(state: Union[ContinuousState, Any], action: int) -> np.ndarray:
    """
    从状态和动作中提取特征向量
    
    Args:
        state: 状态对象（支持连续状态和离散状态）
        action: 动作索引 (0=Forward, 1=Backward)
    
    Returns:
        特征向量 (numpy array)
    """
    features = []
    
    # 1. 偏置项
    features.append(1.0)
    
    # 2. 位置特征
    if hasattr(state, 'robot_position'):
        # 连续位置
        robot_pos = float(state.robot_position)
    else:
        # 离散位置（向后兼容）
        robot_pos = float(state.robot_lane)
    
    features.append(robot_pos / 5.0)  # 归一化位置 [0, 1]
    features.append((5.0 - robot_pos) / 5.0)  # 距离终点 [0, 1]
    
    # 3. 红绿灯特征
    features.append(float(state.light_status))  # 红绿灯状态 [0, 1]
    
    # 4. 倒计时特征
    if hasattr(state, 'light_countdown'):
        features.append(state.light_countdown / 10.0)  # 倒计时比例 [0, 1]
    else:
        # 向后兼容：基于时间步计算倒计时
        features.append(0.5)  # 默认值
    
    # 5. 车辆危险评估
    danger_score = calculate_danger_score(state)
    features.append(danger_score)
    
    # 6. 时间紧迫性
    urgency = calculate_urgency(state)
    features.append(urgency)
    
    # 7. 动作特征
    features.append(1.0 if action == 0 else 0.0)  # Forward action
    features.append(1.0 if action == 1 else 0.0)  # Backward action
    
    return np.array(features, dtype=np.float32)

def calculate_danger_score(state: Union[ContinuousState, Any]) -> float:
    """
    计算当前位置的危险评分
    
    Args:
        state: 状态对象
    
    Returns:
        危险评分 [0, 1]
    """
    if hasattr(state, 'robot_position'):
        robot_pos = state.robot_position
    else:
        robot_pos = float(state.robot_lane)
    
    # 如果不在车道上，无危险
    if robot_pos < 0 or robot_pos >= 5:
        return 0.0
    
    # 如果有车辆位置信息
    if hasattr(state, 'car_positions'):
        current_lane = int(robot_pos)
        if 0 <= current_lane < len(state.car_positions):
            car_pos = state.car_positions[current_lane]
            if car_pos > 0:  # 有车
                # 根据车辆位置计算危险度
                # 车辆在中心位置(1.5)时最危险
                distance = abs(car_pos - 1.5)
                danger = max(0, 1.0 - distance / 1.5)
                
                # 考虑车辆速度
                if hasattr(state, 'car_speeds'):
                    speed = state.car_speeds[current_lane]
                    danger *= (1.0 + speed)
                
                return min(1.0, danger)
    
    # 向后兼容：简单的危险评估
    # 基于红绿灯状态的简单评估
    if state.light_status == 0:  # 红灯时有车
        return 0.7
    else:  # 绿灯时无车
        return 0.1

def calculate_urgency(state: Union[ContinuousState, Any]) -> float:
    """
    计算时间紧迫性
    
    Args:
        state: 状态对象
    
    Returns:
        紧迫性评分 [0, 1]
    """
    if hasattr(state, 'light_countdown'):
        countdown = state.light_countdown
    else:
        # 向后兼容：假设中等紧迫性
        return 0.5
    
    if state.light_status == 1:  # 绿灯
        # 绿灯时间越少，越紧迫
        return 1.0 - (countdown / 10.0)
    else:  # 红灯
        # 红灯时不紧迫
        return 0.0

def calculate_progress_reward(state: Union[ContinuousState, Any], action: int) -> float:
    """
    计算进度奖励特征
    
    Args:
        state: 状态对象
        action: 动作索引
    
    Returns:
        进度奖励 [-1, 1]
    """
    if hasattr(state, 'robot_position'):
        robot_pos = state.robot_position
    else:
        robot_pos = float(state.robot_lane)
    
    # 前进时，距离终点越近奖励越高
    if action == 0:  # Forward
        return robot_pos / 5.0
    else:  # Backward
        return -(robot_pos / 5.0)

def get_feature_names() -> List[str]:
    """
    获取特征名称列表（用于调试和可视化）
    
    Returns:
        特征名称列表
    """
    return [
        'bias',
        'robot_position_norm',
        'distance_to_goal',
        'light_status',
        'light_countdown_norm',
        'danger_score',
        'urgency',
        'action_forward',
        'action_backward'
    ]

def validate_features(features: np.ndarray) -> bool:
    """
    验证特征向量的有效性
    
    Args:
        features: 特征向量
    
    Returns:
        是否有效
    """
    # 检查维度
    if len(features) != 9:
        return False
    
    # 检查偏置项
    if features[0] != 1.0:
        return False
    
    # 检查归一化特征是否在合理范围内
    for i in [1, 2, 3, 4, 5, 6]:
        if not (0.0 <= features[i] <= 1.0):
            return False
    
    # 检查动作特征
    if features[7] + features[8] != 1.0:
        return False
    
    return True

def create_test_state(robot_pos: float = 2.0, light_status: int = 1, 
                     light_countdown: int = 5) -> ContinuousState:
    """
    创建测试用的状态对象
    
    Args:
        robot_pos: 机器人位置
        light_status: 红绿灯状态
        light_countdown: 倒计时
    
    Returns:
        测试状态对象
    """
    return ContinuousState(
        robot_position=robot_pos,
        light_status=light_status,
        light_countdown=light_countdown,
        car_positions=[0.0, 1.5, 0.0, 2.0, 0.0],  # 示例车辆位置
        car_speeds=[0.0, 0.5, 0.0, 0.3, 0.0]      # 示例车辆速度
    )

# 测试函数
if __name__ == "__main__":
    # 创建测试状态
    test_state = create_test_state()
    
    # 提取特征
    features_forward = extract_features(test_state, 0)
    features_backward = extract_features(test_state, 1)
    
    # 验证特征
    print("Forward features:", features_forward)
    print("Backward features:", features_backward)
    print("Feature names:", get_feature_names())
    print("Forward features valid:", validate_features(features_forward))
    print("Backward features valid:", validate_features(features_backward))
    
    # 显示特征含义
    names = get_feature_names()
    print("\nFeature breakdown (Forward):")
    for i, (name, value) in enumerate(zip(names, features_forward)):
        print(f"  {i}: {name} = {value:.3f}")