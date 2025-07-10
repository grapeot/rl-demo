"""
可视化系统 - 使用Pygame展示训练过程和结果
"""
import pygame
import math
from datetime import datetime
from collections import deque
from config import VIS_CONFIG

class Visualizer:
    """可视化系统"""
    
    def __init__(self, env, agent, config=None):
        self.env = env
        self.agent = agent
        self.config = config or VIS_CONFIG
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config['window_width'], self.config['window_height'])
        )
        pygame.display.set_caption("Robot Road Crossing - Q-Learning")
        
        # 字体设置
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # 颜色配置
        self.colors = self.config['colors']
        
        # 区域划分
        width = self.config['window_width']
        height = self.config['window_height']
        
        self.env_area = pygame.Rect(10, 50, int(width * 0.55), int(height * 0.65))
        self.qtable_area = pygame.Rect(int(width * 0.58), 50, int(width * 0.4), int(height * 0.65))
        self.log_area = pygame.Rect(10, int(height * 0.75), width - 20, int(height * 0.22))
        
        # 日志系统
        self.log_messages = deque(maxlen=10)
        
        # 统计信息
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # 时钟控制
        self.clock = pygame.time.Clock()
        self.fps = self.config['fps']
        
    def update(self, state, action, reward, done=False):
        """更新可视化"""
        # 更新统计
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        
        # 添加日志
        action_name = self.env.actions[action] if isinstance(action, int) else action
        self._add_log(f"State: {state}, Action: {action_name}, Reward: {reward}")
        
        # 绘制
        self.draw()
        
        # 控制帧率（0表示无延时）
        if self.fps > 0:
            self.clock.tick(self.fps)
    
    def draw(self):
        """绘制整个界面"""
        # 清空屏幕
        self.screen.fill(self.colors['background'])
        
        # 绘制标题
        self._draw_title()
        
        # 绘制三个主要区域
        self._draw_environment()
        self._draw_qtable()
        self._draw_logs()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_title(self):
        """绘制标题栏"""
        title_text = f"Episode: {self.agent.episode} | ε: {self.agent.epsilon:.3f}"
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            title_text += f" | Avg Reward (last 10): {avg_reward:.1f}"
        
        title_surface = self.font.render(title_text, True, self.colors['text'])
        self.screen.blit(title_surface, (10, 10))
    
    def _draw_environment(self):
        """绘制环境区域"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.env_area, 2)
        
        # 计算布局
        lane_height = self.env_area.height // (self.env.num_lanes + 3)  # +3 for start, end, spacing
        lane_width = self.env_area.width - 40
        x_offset = self.env_area.x + 20
        
        # 获取当前状态
        current_state = self.env._get_state()
        
        # 绘制红绿灯
        light_y = self.env_area.y + 10
        light_color = self.colors['green_light'] if current_state.light_status == 1 else self.colors['red_light']
        pygame.draw.circle(self.screen, light_color, (self.env_area.centerx, light_y + 15), 15)
        light_text = "GREEN" if current_state.light_status == 1 else "RED"
        text_surface = self.small_font.render(f"Light: {light_text}", True, self.colors['text'])
        self.screen.blit(text_surface, (self.env_area.centerx + 25, light_y + 5))
        
        # 绘制起点
        start_y = self.env_area.y + lane_height
        pygame.draw.rect(self.screen, self.colors['goal'], 
                         (x_offset, start_y, lane_width, lane_height - 5), 2)
        text_surface = self.small_font.render("START", True, self.colors['text'])
        self.screen.blit(text_surface, (x_offset + 5, start_y + 5))
        
        # 如果机器人在起点，绘制机器人
        if self.env.robot_position == self.env.start_position:
            self._draw_robot(self.env_area.centerx, start_y + lane_height // 2)
        
        # 绘制车道
        for i in range(self.env.num_lanes):
            y = start_y + (i + 1) * lane_height
            
            # 绘制车道
            pygame.draw.rect(self.screen, self.colors['road'], 
                           (x_offset, y, lane_width, lane_height - 5))
            
            # 绘制车道分隔线
            if i < self.env.num_lanes - 1:
                pygame.draw.line(self.screen, self.colors['lane_divider'],
                               (x_offset, y + lane_height - 5),
                               (x_offset + lane_width, y + lane_height - 5), 2)
            
            # 绘制车道号
            text_surface = self.small_font.render(f"Lane {i}", True, self.colors['lane_divider'])
            self.screen.blit(text_surface, (x_offset + 5, y + 5))
            
            # 绘制车辆
            if self.env.cars_in_lanes[i]:
                self._draw_car(self.env_area.centerx + 50, y + lane_height // 2)
            
            # 绘制机器人
            if self.env.robot_position == i:
                self._draw_robot(self.env_area.centerx, y + lane_height // 2)
        
        # 绘制终点
        goal_y = start_y + (self.env.num_lanes + 1) * lane_height
        pygame.draw.rect(self.screen, self.colors['goal'], 
                        (x_offset, goal_y, lane_width, lane_height - 5), 2)
        text_surface = self.small_font.render("GOAL 🏁", True, self.colors['text'])
        self.screen.blit(text_surface, (x_offset + 5, goal_y + 5))
        
        # 如果机器人到达终点，绘制机器人
        if self.env.robot_position >= self.env.end_position:
            self._draw_robot(self.env_area.centerx, goal_y + lane_height // 2)
    
    def _draw_robot(self, x, y):
        """绘制机器人"""
        pygame.draw.circle(self.screen, self.colors['robot'], (x, y), 20)
        # 绘制简单的机器人特征
        pygame.draw.circle(self.screen, self.colors['background'], (x - 7, y - 7), 3)
        pygame.draw.circle(self.screen, self.colors['background'], (x + 7, y - 7), 3)
    
    def _draw_car(self, x, y):
        """绘制汽车"""
        car_width = 40
        car_height = 20
        pygame.draw.rect(self.screen, self.colors['car'],
                        (x - car_width // 2, y - car_height // 2, car_width, car_height))
        # 绘制车窗
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x - car_width // 2 + 5, y - car_height // 2 + 3, 10, 14))
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x + car_width // 2 - 15, y - car_height // 2 + 3, 10, 14))
    
    def _draw_qtable(self):
        """绘制Q-Table"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.qtable_area, 2)
        
        # 标题
        title_surface = self.font.render("Q-Table", True, self.colors['text'])
        self.screen.blit(title_surface, (self.qtable_area.x + 10, self.qtable_area.y + 5))
        
        # 表头
        header_y = self.qtable_area.y + 35
        headers = ["State", "Forward", "Backward"]
        header_x_positions = [
            self.qtable_area.x + 10,
            self.qtable_area.x + 150,
            self.qtable_area.x + 250
        ]
        
        for header, x in zip(headers, header_x_positions):
            text_surface = self.small_font.render(header, True, self.colors['text'])
            self.screen.blit(text_surface, (x, header_y))
        
        # 绘制分隔线
        pygame.draw.line(self.screen, self.colors['text'],
                        (self.qtable_area.x + 5, header_y + 25),
                        (self.qtable_area.right - 5, header_y + 25), 1)
        
        # 获取当前状态
        current_state = self.env._get_state()
        
        # 绘制Q值
        y_offset = header_y + 35
        max_rows = 15  # 最多显示15行
        row_height = 25
        
        # 对Q-Table按状态排序
        sorted_states = sorted(self.agent.q_table.keys(), 
                             key=lambda s: (s.robot_lane, s.light_status, s.car_imminent))
        
        for i, state in enumerate(sorted_states[:max_rows]):
            q_values = self.agent.q_table[state]
            
            # 高亮当前状态
            if state == current_state:
                highlight_rect = pygame.Rect(
                    self.qtable_area.x + 5,
                    y_offset + i * row_height - 2,
                    self.qtable_area.width - 10,
                    row_height - 2
                )
                pygame.draw.rect(self.screen, (255, 255, 200), highlight_rect)
            
            # 状态文本
            state_text = f"({state.robot_lane}, {state.light_status}, {state.car_imminent})"
            text_surface = self.small_font.render(state_text, True, self.colors['text'])
            self.screen.blit(text_surface, (header_x_positions[0], y_offset + i * row_height))
            
            # Q值
            for j, q_value in enumerate(q_values):
                # 使用颜色编码Q值
                color = self._get_q_value_color(q_value)
                q_text = f"{q_value:.2f}"
                text_surface = self.small_font.render(q_text, True, color)
                self.screen.blit(text_surface, 
                               (header_x_positions[j + 1], y_offset + i * row_height))
        
        # 如果Q-Table条目太多，显示提示
        if len(sorted_states) > max_rows:
            more_text = f"... and {len(sorted_states) - max_rows} more states"
            text_surface = self.small_font.render(more_text, True, self.colors['text'])
            self.screen.blit(text_surface, 
                           (self.qtable_area.x + 10, y_offset + max_rows * row_height))
    
    def _get_q_value_color(self, q_value):
        """根据Q值返回颜色"""
        if q_value > 10:
            return (0, 200, 0)  # 绿色（好）
        elif q_value > 0:
            return (0, 100, 0)  # 深绿色（较好）
        elif q_value == 0:
            return self.colors['text']  # 默认颜色
        elif q_value > -10:
            return (200, 100, 0)  # 橙色（较差）
        else:
            return (200, 0, 0)  # 红色（差）
    
    def _draw_logs(self):
        """绘制日志区域"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.log_area, 2)
        
        # 标题
        title_surface = self.font.render("Logs", True, self.colors['text'])
        self.screen.blit(title_surface, (self.log_area.x + 10, self.log_area.y + 5))
        
        # 绘制日志消息
        y_offset = self.log_area.y + 30
        for i, message in enumerate(self.log_messages):
            text_surface = self.small_font.render(message, True, self.colors['text'])
            self.screen.blit(text_surface, (self.log_area.x + 10, y_offset + i * 20))
    
    def _add_log(self, message, level='INFO'):
        """添加日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_messages.append(f"[{timestamp}] [{level}] {message}")
    
    def set_fps(self, fps):
        """设置帧率"""
        self.fps = fps
    
    def close(self):
        """关闭可视化窗口"""
        pygame.quit()