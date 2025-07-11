"""
环境模拟器 v1.0 - 机器人过马路环境（单格子预警系统 + 倒计时）
"""
import random
from collections import namedtuple
from .config import ENV_CONFIG, REWARD_CONFIG

# v1.0状态定义
State = namedtuple('State', ['robot_lane', 'light_status', 'light_countdown', 'next_lane_car'])

class RoadEnvironmentV1:
    """机器人过马路环境 v1.0"""
    
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
        
        # 车辆状态：每个车道的车辆位置 (0=无车, 1=右侧, 2=中心)
        self.cars_in_lanes = [0] * self.num_lanes
        
        # v1.0 反龟缩机制：记录起点停留时间
        self.start_staying_steps = 0
        self.previous_light_status = None
        
        # 初始化环境状态
        self.time_step = 0
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.robot_position = self.start_position
        # 保持time_step连续，不重置为0，这样红绿灯会持续变化
        self.cars_in_lanes = [0] * self.num_lanes
        self.done = False
        
        # 重置反龟缩机制
        self.start_staying_steps = 0
        self.previous_light_status = None
        
        # 生成初始车辆
        self._spawn_cars()
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 红绿灯状态（基于时间步）
        cycle_position = self.time_step % self.traffic_light_cycle
        half_cycle = self.traffic_light_cycle // 2
        light_status = 1 if cycle_position < half_cycle else 0
        
        # 红绿灯倒计时
        if light_status == 1:  # 绿灯
            light_countdown = half_cycle - 1 - cycle_position
        else:  # 红灯
            light_countdown = self.traffic_light_cycle - 1 - cycle_position
        
        # 下一车道的车辆位置
        next_lane_car = 0  # 默认无车
        if 0 <= self.robot_position + 1 < self.num_lanes:
            next_lane_car = self.cars_in_lanes[self.robot_position + 1]
        
        return State(self.robot_position, light_status, light_countdown, next_lane_car)
    
    def _spawn_cars(self):
        """在车道上生成和移动车辆"""
        # 获取当前红绿灯状态
        cycle_position = self.time_step % self.traffic_light_cycle
        half_cycle = self.traffic_light_cycle // 2
        light_status = 1 if cycle_position < half_cycle else 0
        
        # 先移动现有车辆（右侧→中心→消失）
        for i in range(self.num_lanes):
            if self.cars_in_lanes[i] == 1:  # 右侧 -> 中心
                self.cars_in_lanes[i] = 2
            elif self.cars_in_lanes[i] == 2:  # 中心 -> 消失
                self.cars_in_lanes[i] = 0
        
        # 仅在红灯时生成新车辆（在右侧位置）
        if light_status == 0:  # 红灯
            for i in range(self.num_lanes):
                if self.cars_in_lanes[i] == 0 and random.random() < self.car_spawn_probability:
                    self.cars_in_lanes[i] = 1  # 在右侧生成
    
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
        
        # 获取当前状态（用于反龟缩机制）
        current_state = self._get_state()
        
        # 执行动作
        old_position = self.robot_position
        reward = self.reward_config['step_penalty']  # 默认步骤惩罚
        
        # v1.0 反龟缩机制：分析当前情况
        is_at_start = (self.robot_position == self.start_position)
        current_light = current_state.light_status
        light_just_turned_green = (self.previous_light_status == 0 and current_light == 1)
        
        if action == 'Forward':
            if is_at_start:
                # 从起点前进
                if current_light == 1:  # 绿灯时前进，给小奖励
                    reward += 1  # 鼓励在合适时机行动
                self.start_staying_steps = 0  # 重置等待计数
            
            self.robot_position += 1
            
            # 检查是否到达终点
            if self.robot_position >= self.end_position:
                reward = self.reward_config['goal_reward']
                self.done = True
        
        elif action == 'Backward':
            if is_at_start:
                # 在起点选择等待
                self.start_staying_steps += 1
                
                # 等待惩罚：基础惩罚 + 渐进式惩罚
                wait_penalty = -2  # 基础等待惩罚
                
                # 如果绿灯刚开始却选择等待，额外惩罚
                if light_just_turned_green:
                    wait_penalty -= 3  # 错失绿灯机会
                
                # 长时间等待的渐进式惩罚
                if self.start_staying_steps > 5:
                    wait_penalty -= min(self.start_staying_steps - 5, 5)  # 最多额外-5
                
                reward += wait_penalty
            
            # 允许在起点后退（相当于原地等待）
            self.robot_position = max(self.start_position, self.robot_position - 1)
        
        # 更新时间步和车辆
        self.time_step += 1
        self._spawn_cars()
        
        # 检查碰撞（机器人在车道中心，车辆也在中心）
        if not self.done and 0 <= self.robot_position < self.num_lanes:
            if self.cars_in_lanes[self.robot_position] == 2:  # 车辆在中心
                reward = self.reward_config['collision_penalty']
                self.done = True
        
        # 检查超时（绿灯结束但机器人未完成）
        current_state = self._get_state()
        if not self.done and current_state.light_status == 0 and current_state.light_countdown == 9:
            # 绿灯刚结束，检查是否应该给超时惩罚
            if self.robot_position > self.start_position and self.robot_position < self.end_position:
                reward += -20  # 轻微的超时惩罚
        
        # 获取新状态
        new_state = self._get_state()
        
        # 更新红绿灯状态追踪（用于下次判断）
        self.previous_light_status = new_state.light_status
        
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
        
        # 红绿灯信息
        state = self._get_state()
        light_str = "🟢" if state.light_status == 1 else "🔴"
        display.append(f"Traffic Light: {light_str} (countdown: {state.light_countdown})")
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
            if self.robot_position == i:
                lane_str += "🤖"
            else:
                lane_str += "[ ]"
            
            # 车辆位置
            if self.cars_in_lanes[i] == 1:
                lane_str += " | 🚗→"  # 右侧
            elif self.cars_in_lanes[i] == 2:
                lane_str += " | →🚗"  # 中心
            else:
                lane_str += " |   "  # 无车
            
            display.append(lane_str)
        
        # 终点
        if self.robot_position >= self.end_position:
            display.append("Goal: 🤖 🏁")
        else:
            display.append("Goal: [ ] 🏁")
        
        return "\n".join(display)