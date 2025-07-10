"""
环境模拟器 - 机器人过马路环境
"""
import random
from collections import namedtuple
from config import ENV_CONFIG, REWARD_CONFIG

# 状态定义
State = namedtuple('State', ['robot_lane', 'light_status', 'car_imminent'])

class RoadEnvironment:
    """机器人过马路环境"""
    
    def __init__(self, config=None):
        self.config = config or ENV_CONFIG
        self.reward_config = REWARD_CONFIG
        
        # 环境参数
        self.num_lanes = self.config['num_lanes']
        self.start_position = self.config['start_position']
        self.end_position = self.config['end_position']
        self.traffic_light_cycle = self.config['traffic_light_cycle']
        self.car_spawn_probability = self.config['car_spawn_probability']
        
        # 动作定义
        self.actions = ['Forward', 'Backward']
        self.action_to_index = {action: i for i, action in enumerate(self.actions)}
        
        # 初始化环境状态
        self.time_step = 0
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.robot_position = self.start_position
        # 保持time_step连续，不重置为0，这样红绿灯会持续变化
        self.cars_in_lanes = [False] * self.num_lanes
        self.done = False
        
        # 生成初始车辆
        self._spawn_cars()
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 红绿灯状态（基于时间步）
        light_status = 1 if (self.time_step % self.traffic_light_cycle) < (self.traffic_light_cycle // 2) else 0
        
        # 下一车道是否有车
        car_imminent = False
        if 0 <= self.robot_position + 1 < self.num_lanes:
            car_imminent = self.cars_in_lanes[self.robot_position + 1]
        
        return State(self.robot_position, light_status, car_imminent)
    
    def _spawn_cars(self):
        """在车道上生成车辆"""
        # 获取当前红绿灯状态
        light_status = 1 if (self.time_step % self.traffic_light_cycle) < (self.traffic_light_cycle // 2) else 0
        
        for i in range(self.num_lanes):
            if light_status == 0:  # 红灯时，车辆可以通行
                if random.random() < self.car_spawn_probability:
                    self.cars_in_lanes[i] = True
                else:
                    self.cars_in_lanes[i] = False
            else:  # 绿灯时，车辆停止（不生成新车辆）
                self.cars_in_lanes[i] = False
    
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
            self.robot_position += 1
            
            # 检查是否到达终点
            if self.robot_position >= self.end_position:
                reward = self.reward_config['goal_reward']
                self.done = True
            # 检查是否碰撞
            elif 0 <= self.robot_position < self.num_lanes and self.cars_in_lanes[self.robot_position]:
                reward = self.reward_config['collision_penalty']
                self.done = True
                
        elif action == 'Backward':
            # 允许在起点后退（相当于原地等待）
            self.robot_position = max(self.start_position, self.robot_position - 1)
        
        # 更新时间步和车辆
        self.time_step += 1
        self._spawn_cars()
        
        # 获取新状态
        new_state = self._get_state()
        
        return new_state, reward, self.done
    
    def get_valid_actions(self, state=None):
        """获取当前状态下的有效动作"""
        if state is None:
            state = self._get_state()
        
        valid_actions = []
        
        # 前进：如果还没到终点
        if state.robot_lane < self.end_position:
            valid_actions.append(0)  # Forward
        
        # 后退：总是可用（在起点时相当于原地等待）
        valid_actions.append(1)  # Backward
            
        return valid_actions
    
    def render_text(self):
        """文本方式渲染环境（用于调试）"""
        # 构建显示字符串
        display = []
        
        # 红绿灯
        light_status = "🟢" if self._get_state().light_status == 1 else "🔴"
        display.append(f"Traffic Light: {light_status}")
        display.append("")
        
        # 起点
        if self.robot_position == self.start_position:
            display.append("Start: 🤖")
        else:
            display.append("Start: [ ]")
        
        # 车道
        for i in range(self.num_lanes):
            lane_str = f"Lane {i}: "
            if self.robot_position == i:
                lane_str += "🤖"
            elif self.cars_in_lanes[i]:
                lane_str += "🚗"
            else:
                lane_str += "[ ]"
            display.append(lane_str)
        
        # 终点
        if self.robot_position >= self.end_position:
            display.append("Goal: 🤖 🏁")
        else:
            display.append("Goal: [ ] 🏁")
        
        return "\n".join(display)