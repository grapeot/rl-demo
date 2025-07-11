"""
增强版环境模拟器 - 支持连续位置和三格子车道系统
"""
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import namedtuple
from config import ENV_CONFIG, REWARD_CONFIG
from feature_engineering import ContinuousState

class EnhancedRoadEnvironment:
    """增强版机器人过马路环境"""
    
    def __init__(self, config=None, continuous_mode=True):
        """
        初始化增强版环境
        
        Args:
            config: 环境配置
            continuous_mode: 是否使用连续位置模式
        """
        self.config = config or ENV_CONFIG
        self.reward_config = REWARD_CONFIG
        self.continuous_mode = continuous_mode
        
        # 环境参数
        self.num_lanes = self.config['num_lanes']
        self.start_position = self.config['start_position']
        self.end_position = self.config['end_position']
        self.traffic_light_cycle = self.config['traffic_light_cycle']
        self.car_spawn_probability = self.config['car_spawn_probability']
        
        # 三格子车道参数
        self.lane_segments = 3  # 每条车道3个段：右侧(0.5)→中心(1.5)→左侧(2.5)
        self.segment_positions = [0.5, 1.5, 2.5]  # 车辆可能的位置
        
        # 动作定义
        self.actions = ['Forward', 'Backward']
        self.action_to_index = {action: i for i, action in enumerate(self.actions)}
        
        # 车辆系统
        self.lane_cars = []  # 每条车道的车辆信息
        self.car_speeds = []  # 每条车道的车辆速度
        
        # 初始化环境状态
        self.time_step = 0
        self.robot_position = float(self.start_position)
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        if self.continuous_mode:
            self.robot_position = float(self.start_position)
        else:
            self.robot_position = self.start_position
        
        # 初始化车辆系统
        self.lane_cars = [0.0] * self.num_lanes  # 0.0表示无车
        self.car_speeds = [0.0] * self.num_lanes
        
        self.done = False
        
        # 生成初始车辆
        self._spawn_cars()
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 红绿灯状态（基于时间步）
        light_status = 1 if (self.time_step % self.traffic_light_cycle) < (self.traffic_light_cycle // 2) else 0
        
        # 红绿灯倒计时
        cycle_position = self.time_step % self.traffic_light_cycle
        half_cycle = self.traffic_light_cycle // 2
        
        if light_status == 1:  # 绿灯
            light_countdown = half_cycle - cycle_position
        else:  # 红灯
            light_countdown = self.traffic_light_cycle - cycle_position
        
        if self.continuous_mode:
            # 返回连续状态
            return ContinuousState(
                robot_position=self.robot_position,
                light_status=light_status,
                light_countdown=light_countdown,
                car_positions=self.lane_cars.copy(),
                car_speeds=self.car_speeds.copy()
            )
        else:
            # 向后兼容：返回离散状态
            from environment import State
            return State(
                robot_lane=int(self.robot_position) if self.robot_position >= 0 else -1,
                light_status=light_status
            )
    
    def _spawn_cars(self):
        """在车道上生成和更新车辆"""
        # 获取当前红绿灯状态
        light_status = 1 if (self.time_step % self.traffic_light_cycle) < (self.traffic_light_cycle // 2) else 0
        
        for i in range(self.num_lanes):
            # 更新现有车辆
            if self.lane_cars[i] > 0:
                # 车辆向前移动
                self.lane_cars[i] += self.car_speeds[i]
                
                # 检查车辆是否离开车道
                if self.lane_cars[i] >= 3.0:
                    self.lane_cars[i] = 0.0
                    self.car_speeds[i] = 0.0
            else:
                # 可能生成新车辆
                if light_status == 0:  # 红灯时，车辆可以通行
                    if random.random() < self.car_spawn_probability:
                        self.lane_cars[i] = 0.5  # 在右侧生成
                        self.car_speeds[i] = random.uniform(0.3, 0.8)  # 随机速度
                else:  # 绿灯时，车辆停止（不生成新车辆）
                    self.lane_cars[i] = 0.0
                    self.car_speeds[i] = 0.0
    
    def step(self, action):
        """执行动作并返回新状态、奖励和完成标志"""
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")
        
        # 将动作转换为索引（如果需要）
        if isinstance(action, str):
            action_idx = self.action_to_index.get(action)
            if action_idx is None:
                raise ValueError(f"Invalid action: {action}")
        else:
            action_idx = action
            action = self.actions[action_idx]
        
        # 执行动作
        old_position = self.robot_position
        reward = self.reward_config['step_penalty']  # 默认步骤惩罚
        
        if action == 'Forward':
            if self.continuous_mode:
                self.robot_position += 0.5  # 半格子移动
            else:
                self.robot_position += 1
            
            # 检查是否到达终点
            if self.robot_position >= self.end_position:
                reward = self.reward_config['goal_reward']
                self.done = True
                
        elif action == 'Backward':
            if self.continuous_mode:
                self.robot_position = max(float(self.start_position), self.robot_position - 0.5)
            else:
                self.robot_position = max(self.start_position, self.robot_position - 1)
        
        # 更新时间步和车辆
        self.time_step += 1
        self._spawn_cars()
        
        # 检查碰撞
        if not self.done:
            collision_reward = self._check_collision()
            if collision_reward != 0:
                reward = collision_reward
                self.done = True
        
        # 获取新状态
        new_state = self._get_state()
        
        return new_state, reward, self.done
    
    def _check_collision(self) -> float:
        """
        检查碰撞
        
        Returns:
            碰撞奖励（0表示无碰撞，负数表示碰撞）
        """
        # 如果机器人不在车道上，无碰撞
        if self.robot_position < 0 or self.robot_position >= self.num_lanes:
            return 0.0
        
        current_lane = int(self.robot_position)
        
        if 0 <= current_lane < len(self.lane_cars):
            car_pos = self.lane_cars[current_lane]
            
            if car_pos > 0:  # 有车
                # 检查是否在危险区域
                if self.continuous_mode:
                    # 连续模式：检查机器人和车辆的精确位置
                    robot_in_lane = self.robot_position - current_lane  # 机器人在车道内的位置
                    car_segment = int(car_pos)  # 车辆所在段
                    
                    # 如果机器人和车辆在同一段，发生碰撞
                    if abs(robot_in_lane - (car_pos - current_lane)) < 0.3:
                        return self.reward_config['collision_penalty']
                else:
                    # 离散模式：简单检查
                    if car_pos >= 1.0:  # 车辆在中心或以后
                        return self.reward_config['collision_penalty']
        
        return 0.0
    
    def get_valid_actions(self, state=None):
        """获取当前状态下的有效动作"""
        if state is None:
            state = self._get_state()
        
        valid_actions = []
        
        # 前进：如果还没到终点
        if self.robot_position < self.end_position:
            valid_actions.append(0)  # Forward
        
        # 后退：总是可用
        valid_actions.append(1)  # Backward
        
        return valid_actions
    
    def get_danger_level(self, lane: int) -> float:
        """
        获取指定车道的危险等级
        
        Args:
            lane: 车道索引
        
        Returns:
            危险等级 [0, 1]
        """
        if lane < 0 or lane >= len(self.lane_cars):
            return 0.0
        
        car_pos = self.lane_cars[lane]
        if car_pos <= 0:
            return 0.0
        
        # 根据车辆位置计算危险等级
        if car_pos < 1.0:  # 右侧预警
            return 0.3
        elif car_pos < 2.0:  # 中心危险
            return 1.0
        else:  # 左侧离开
            return 0.1
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息（用于调试和可视化）"""
        state = self._get_state()
        
        return {
            'robot_position': self.robot_position,
            'light_status': state.light_status if hasattr(state, 'light_status') else 0,
            'light_countdown': state.light_countdown if hasattr(state, 'light_countdown') else 0,
            'car_positions': self.lane_cars.copy(),
            'car_speeds': self.car_speeds.copy(),
            'danger_levels': [self.get_danger_level(i) for i in range(self.num_lanes)],
            'time_step': self.time_step,
            'done': self.done
        }
    
    def render_text(self):
        """文本方式渲染环境"""
        # 构建显示字符串
        display = []
        
        # 红绿灯
        state = self._get_state()
        light_status = "🟢" if state.light_status == 1 else "🔴"
        countdown = state.light_countdown if hasattr(state, 'light_countdown') else 0
        display.append(f"Traffic Light: {light_status} ({countdown}s)")
        display.append("")
        
        # 起点
        if self.robot_position == self.start_position:
            display.append("Start: 🤖")
        else:
            display.append("Start: [ ]")
        
        # 车道
        for i in range(self.num_lanes):
            lane_str = f"Lane {i}: "
            
            # 机器人位置
            if self.continuous_mode:
                if int(self.robot_position) == i and self.robot_position >= 0:
                    lane_str += "🤖"
                else:
                    lane_str += "[ ]"
            else:
                if self.robot_position == i:
                    lane_str += "🤖"
                else:
                    lane_str += "[ ]"
            
            # 车辆位置
            car_pos = self.lane_cars[i]
            if car_pos > 0:
                if car_pos < 1.0:
                    lane_str += " 🚗R"  # 右侧
                elif car_pos < 2.0:
                    lane_str += " 🚗C"  # 中心
                else:
                    lane_str += " 🚗L"  # 左侧
            else:
                lane_str += " [ ]"
            
            display.append(lane_str)
        
        # 终点
        if self.robot_position >= self.end_position:
            display.append("Goal: 🤖 🏁")
        else:
            display.append("Goal: [ ] 🏁")
        
        return "\n".join(display)

# 向后兼容的工厂函数
def create_environment(enhanced=False, continuous=True):
    """
    创建环境实例
    
    Args:
        enhanced: 是否使用增强版环境
        continuous: 是否使用连续模式
    
    Returns:
        环境实例
    """
    if enhanced:
        return EnhancedRoadEnvironment(continuous_mode=continuous)
    else:
        from environment import RoadEnvironment
        return RoadEnvironment()

# 测试函数
if __name__ == "__main__":
    # 测试增强版环境
    env = EnhancedRoadEnvironment(continuous_mode=True)
    
    print("Initial state:")
    print(env.render_text())
    print()
    
    # 运行几步
    for step in range(5):
        action = random.choice([0, 1])  # 随机动作
        action_name = env.actions[action]
        
        state, reward, done = env.step(action)
        
        print(f"Step {step + 1}: Action={action_name}, Reward={reward:.2f}")
        print(env.render_text())
        print(f"State: {state}")
        print()
        
        if done:
            print("Episode finished!")
            break
    
    # 测试环境信息
    info = env.get_environment_info()
    print("Environment info:")
    for key, value in info.items():
        print(f"  {key}: {value}")