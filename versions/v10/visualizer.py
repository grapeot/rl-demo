"""
可视化系统 v1.0 - 支持扩展状态空间和倒计时显示
"""
import pygame
import math
from datetime import datetime
from collections import deque
from .config import VIS_CONFIG

class VisualizerV1:
    """可视化系统 v1.0 - 支持420状态空间"""
    
    def __init__(self, env, agent, config=None):
        self.env = env
        self.agent = agent
        self.config = config or VIS_CONFIG
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config['window_width'], self.config['window_height'])
        )
        pygame.display.set_caption("Robot Road Crossing v1.0 - Q-Learning")
        
        # 字体设置
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.tiny_font = pygame.font.Font(None, 16)
        
        # 颜色配置
        self.colors = self.config['colors']
        # v1.0新增颜色
        self.colors.update({
            'car_warning': (255, 150, 0),  # 橙色预警
            'car_danger': (255, 50, 50),   # 红色危险
            'countdown_bg': (50, 50, 50),  # 倒计时背景
            'state_coverage': (100, 150, 255)  # 状态覆盖率颜色
        })
        
        # 区域划分 - 为v1.0优化布局
        width = self.config['window_width']
        height = self.config['window_height']
        
        self.env_area = pygame.Rect(10, 50, int(width * 0.45), int(height * 0.45))
        self.qtable_area = pygame.Rect(int(width * 0.47), 50, int(width * 0.25), int(height * 0.45))
        self.chart_area = pygame.Rect(int(width * 0.74), 50, int(width * 0.24), int(height * 0.45))
        self.log_area = pygame.Rect(10, int(height * 0.52), width - 20, int(height * 0.45))
        
        # 日志系统
        self.log_messages = deque(maxlen=12)
        
        # 统计信息
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # 死亡率统计
        self.death_history = []
        self.death_rate_smoothed = []
        self.smoothing_window = 50
        
        # v1.0特定统计
        self.success_history = []  # 成功完成的记录
        self.success_rate_smoothed = []
        
        # 时钟控制
        self.clock = pygame.time.Clock()
        self.fps = self.config['fps']
        
    def update(self, state, action, reward, done=False, show_initial=False):
        """更新可视化"""
        # 更新统计
        if not show_initial:
            self.current_episode_reward += reward
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                # 记录死亡和成功
                is_death = reward == -100
                is_success = reward == 100  # 假设目标奖励是100
                
                self.death_history.append(is_death)
                self.success_history.append(is_success)
                
                self._update_rates()
                self.current_episode_reward = 0
        
        # 添加日志
        if show_initial:
            self._add_log(f"Initial State: robot={state.robot_lane}, light={state.light_status}, countdown={state.light_countdown}, car={state.next_lane_car}")
        else:
            action_name = self.env.actions[action] if isinstance(action, int) else action
            self._add_log(f"Robot={state.robot_lane}, Light={state.light_status}({state.light_countdown}), Car={state.next_lane_car}, Action={action_name}, Reward={reward}")
        
        # 绘制
        self.draw()
        
        # 控制帧率
        if self.fps > 0:
            self.clock.tick(self.fps)
    
    def draw(self):
        """绘制整个界面"""
        # 清空屏幕
        self.screen.fill(self.colors['background'])
        
        # 绘制标题
        self._draw_title()
        
        # 绘制四个主要区域
        self._draw_environment()
        self._draw_qtable()
        self._draw_charts()
        self._draw_logs()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_title(self):
        """绘制标题栏"""
        title_text = f"v1.0 | Episode: {self.agent.episode} | ε: {self.agent.epsilon:.3f}"
        if self.episode_rewards:
            avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
            title_text += f" | Avg Reward: {avg_reward:.1f}"
        
        # 显示状态空间统计
        stats = self.agent.get_q_value_stats()
        title_text += f" | States: {stats['num_states']}/{stats.get('theoretical_max_states', 420)}"
        title_text += f" | Coverage: {stats.get('state_coverage', 0):.1f}%"
        
        title_surface = self.font.render(title_text, True, self.colors['text'])
        self.screen.blit(title_surface, (10, 10))
    
    def _draw_environment(self):
        """绘制环境区域 - v1.0增强版"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.env_area, 2)
        
        # 计算布局
        lane_height = self.env_area.height // (self.env.num_lanes + 4)  # +4 for spacing and countdown
        lane_width = self.env_area.width - 80  # 增加右侧空间显示车辆位置
        x_offset = self.env_area.x + 40
        
        # 获取当前状态
        current_state = self.env._get_state()
        
        # 绘制红绿灯和倒计时
        light_y = self.env_area.y + 15
        light_color = self.colors['green_light'] if current_state.light_status == 1 else self.colors['red_light']
        pygame.draw.circle(self.screen, light_color, (self.env_area.x + 50, light_y), 20)
        
        # 绘制倒计时
        countdown_text = f"{current_state.light_countdown}"
        countdown_surface = self.font.render(countdown_text, True, self.colors['background'])
        text_rect = countdown_surface.get_rect(center=(self.env_area.x + 50, light_y))
        self.screen.blit(countdown_surface, text_rect)
        
        # 红绿灯状态文本
        light_text = f"{'GREEN' if current_state.light_status == 1 else 'RED'} ({current_state.light_countdown})"
        text_surface = self.small_font.render(light_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.env_area.x + 80, light_y - 10))
        
        # 绘制起点
        start_y = self.env_area.y + lane_height + 10
        pygame.draw.rect(self.screen, self.colors['goal'], 
                         (x_offset, start_y, lane_width, lane_height - 5), 2)
        text_surface = self.small_font.render("START", True, self.colors['text'])
        self.screen.blit(text_surface, (x_offset + 5, start_y + 5))
        
        if self.env.robot_position == self.env.start_position:
            # 保持与车道中机器人位置一致 - 左侧1/3的中心
            section_width = lane_width // 3
            robot_x = x_offset + section_width // 2
            self._draw_robot(robot_x, start_y + lane_height // 2)
        
        # 绘制车道
        for i in range(self.env.num_lanes):
            y = start_y + (i + 1) * lane_height
            
            # 绘制车道背景
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
            
            # v1.0: 绘制车辆位置系统 - 将车道分成三等份
            # 计算三个位置：机器人位置(左)、中心格子(中)、右侧格子(右)
            section_width = lane_width // 3
            robot_x = x_offset + section_width // 2  # 机器人位置（左侧1/3的中心）
            center_x = x_offset + section_width + section_width // 2  # 中心格子（中间1/3的中心）
            right_x = x_offset + 2 * section_width + section_width // 2  # 右侧格子（右侧1/3的中心）
            
            car_status = self.env.cars_in_lanes[i]
            if car_status == 1:  # 右侧预警
                self._draw_car_warning(right_x, y + lane_height // 2)
                warning_text = "⚠️"
                text_surface = self.tiny_font.render(warning_text, True, self.colors['car_warning'])
                self.screen.blit(text_surface, (x_offset + lane_width - 30, y + 5))
            elif car_status == 2:  # 中心危险
                self._draw_car_danger(center_x, y + lane_height // 2)
                danger_text = "🚨"
                text_surface = self.tiny_font.render(danger_text, True, self.colors['car_danger'])
                self.screen.blit(text_surface, (x_offset + lane_width - 30, y + 5))
            
            # 绘制机器人
            if self.env.robot_position == i:
                self._draw_robot(robot_x, y + lane_height // 2)
                
                # 如果机器人和车辆在同一位置且都在中心，高亮显示碰撞
                if car_status == 2:
                    collision_rect = pygame.Rect(x_offset, y, lane_width, lane_height - 5)
                    pygame.draw.rect(self.screen, (255, 0, 0), collision_rect, 3)
        
        # 绘制终点
        goal_y = start_y + (self.env.num_lanes + 1) * lane_height
        pygame.draw.rect(self.screen, self.colors['goal'], 
                        (x_offset, goal_y, lane_width, lane_height - 5), 2)
        text_surface = self.small_font.render("GOAL 🏁", True, self.colors['text'])
        self.screen.blit(text_surface, (x_offset + 5, goal_y + 5))
        
        if self.env.robot_position >= self.env.end_position:
            # 保持与车道中机器人位置一致 - 左侧1/3的中心
            section_width = lane_width // 3
            robot_x = x_offset + section_width // 2
            self._draw_robot(robot_x, goal_y + lane_height // 2)
        
        # v1.0: 绘制下一车道预警信息
        if current_state.robot_lane >= 0 and current_state.robot_lane < self.env.num_lanes:
            next_lane = current_state.robot_lane + 1
            if next_lane < self.env.num_lanes:
                preview_text = f"Next Lane {next_lane}: "
                if current_state.next_lane_car == 0:
                    preview_text += "Clear ✅"
                    color = (0, 150, 0)
                elif current_state.next_lane_car == 1:
                    preview_text += "Warning ⚠️"
                    color = self.colors['car_warning']
                else:
                    preview_text += "Danger 🚨"
                    color = self.colors['car_danger']
                
                text_surface = self.small_font.render(preview_text, True, color)
                self.screen.blit(text_surface, (self.env_area.x + 10, self.env_area.bottom - 30))
    
    def _draw_robot(self, x, y):
        """绘制机器人"""
        pygame.draw.circle(self.screen, self.colors['robot'], (x, y), 15)
        # 绘制简单的机器人特征
        pygame.draw.circle(self.screen, self.colors['background'], (x - 5, y - 5), 2)
        pygame.draw.circle(self.screen, self.colors['background'], (x + 5, y - 5), 2)
    
    def _draw_car_warning(self, x, y):
        """绘制预警位置的汽车（右侧）"""
        car_width = 25
        car_height = 15
        pygame.draw.rect(self.screen, self.colors['car_warning'],
                        (x - car_width // 2, y - car_height // 2, car_width, car_height))
        # 绘制简单的车窗
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x - 8, y - 5, 6, 10))
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x + 2, y - 5, 6, 10))
    
    def _draw_car_danger(self, x, y):
        """绘制危险位置的汽车（中心）"""
        car_width = 35
        car_height = 18
        pygame.draw.rect(self.screen, self.colors['car_danger'],
                        (x - car_width // 2, y - car_height // 2, car_width, car_height))
        # 绘制车窗
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x - 12, y - 6, 8, 12))
        pygame.draw.rect(self.screen, self.colors['background'],
                        (x + 4, y - 6, 8, 12))
    
    def _draw_qtable(self):
        """绘制Q-Table - v1.0压缩显示"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.qtable_area, 2)
        
        # 标题
        title_surface = self.font.render("Q-Table (Top States)", True, self.colors['text'])
        self.screen.blit(title_surface, (self.qtable_area.x + 5, self.qtable_area.y + 5))
        
        # 获取Q-Table统计
        stats = self.agent.get_q_value_stats()
        stats_text = f"States: {stats['num_states']}, Avg Q: {stats['avg_q_value']:.2f}"
        stats_surface = self.tiny_font.render(stats_text, True, self.colors['text'])
        self.screen.blit(stats_surface, (self.qtable_area.x + 5, self.qtable_area.y + 25))
        
        # 表头
        header_y = self.qtable_area.y + 45
        headers = ["State", "Fwd", "Back"]
        header_x_positions = [
            self.qtable_area.x + 5,
            self.qtable_area.x + 120,
            self.qtable_area.x + 170
        ]
        
        for header, x in zip(headers, header_x_positions):
            text_surface = self.tiny_font.render(header, True, self.colors['text'])
            self.screen.blit(text_surface, (x, header_y))
        
        # 绘制分隔线
        pygame.draw.line(self.screen, self.colors['text'],
                        (self.qtable_area.x + 5, header_y + 18),
                        (self.qtable_area.right - 5, header_y + 18), 1)
        
        # 获取当前状态
        current_state = self.env._get_state()
        
        # 绘制Q值 - 只显示访问过的重要状态
        y_offset = header_y + 25
        available_height = self.qtable_area.height - 85
        row_height = 16  # 更小的行高
        max_rows = min(15, available_height // row_height)
        
        # 按访问频率和Q值排序状态
        if hasattr(self.agent, 'state_visit_count') and self.agent.state_visit_count:
            # 优先显示访问频率高的状态
            sorted_states = sorted(self.agent.q_table.keys(), 
                                 key=lambda s: (self.agent.state_visit_count.get(s, 0), 
                                              max(self.agent.q_table[s])), reverse=True)
        else:
            # 按Q值排序
            sorted_states = sorted(self.agent.q_table.keys(), 
                                 key=lambda s: max(self.agent.q_table[s]), reverse=True)
        
        for i, state in enumerate(sorted_states[:max_rows]):
            q_values = self.agent.q_table[state]
            
            # 高亮当前状态
            if state == current_state:
                highlight_rect = pygame.Rect(
                    self.qtable_area.x + 2,
                    y_offset + i * row_height - 1,
                    self.qtable_area.width - 4,
                    row_height
                )
                pygame.draw.rect(self.screen, (255, 255, 200), highlight_rect)
            
            # 状态文本 - 压缩格式
            state_text = f"({state.robot_lane},{state.light_status},{state.light_countdown},{state.next_lane_car})"
            text_surface = self.tiny_font.render(state_text, True, self.colors['text'])
            self.screen.blit(text_surface, (header_x_positions[0], y_offset + i * row_height))
            
            # Q值
            for j, q_value in enumerate(q_values):
                color = self._get_q_value_color(q_value)
                q_text = f"{q_value:.1f}"
                text_surface = self.tiny_font.render(q_text, True, color)
                self.screen.blit(text_surface, 
                               (header_x_positions[j + 1], y_offset + i * row_height))
        
        # 显示更多状态提示
        if len(sorted_states) > max_rows:
            more_text = f"... +{len(sorted_states) - max_rows} more"
            text_surface = self.tiny_font.render(more_text, True, self.colors['text'])
            self.screen.blit(text_surface, 
                           (self.qtable_area.x + 5, y_offset + max_rows * row_height))
    
    def _draw_charts(self):
        """绘制图表区域 - 显示死亡率和成功率"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.chart_area, 2)
        
        # 标题
        title_surface = self.font.render("Success & Death", True, self.colors['text'])
        self.screen.blit(title_surface, (self.chart_area.x + 5, self.chart_area.y + 5))
        
        # 统计信息
        stats_y = self.chart_area.y + 30
        if self.death_history:
            total_episodes = len(self.death_history)
            total_deaths = sum(self.death_history)
            total_success = sum(self.success_history) if self.success_history else 0
            
            current_death_rate = self.death_rate_smoothed[-1] if self.death_rate_smoothed else 0
            current_success_rate = self.success_rate_smoothed[-1] if self.success_rate_smoothed else 0
            
            stats_text = [
                f"Episodes: {total_episodes}",
                f"Deaths: {total_deaths}",
                f"Success: {total_success}",
                f"Death Rate: {current_death_rate:.1%}",
                f"Success Rate: {current_success_rate:.1%}"
            ]
            
            for i, text in enumerate(stats_text):
                text_surface = self.tiny_font.render(text, True, self.colors['text'])
                self.screen.blit(text_surface, (self.chart_area.x + 5, stats_y + i * 16))
        
        # 绘制图表
        if len(self.death_rate_smoothed) > 1:
            chart_start_y = stats_y + 90
            chart_height = self.chart_area.height - 130
            chart_width = self.chart_area.width - 15
            
            # 图表背景
            chart_rect = pygame.Rect(self.chart_area.x + 5, chart_start_y, 
                                   chart_width, chart_height)
            pygame.draw.rect(self.screen, (250, 250, 250), chart_rect)
            pygame.draw.rect(self.screen, self.colors['text'], chart_rect, 1)
            
            # 绘制数据 - 死亡率和成功率
            max_points = min(50, len(self.death_rate_smoothed))
            if max_points > 1:
                step = len(self.death_rate_smoothed) // max_points if len(self.death_rate_smoothed) > max_points else 1
                death_data = self.death_rate_smoothed[::step][-max_points:]
                success_data = self.success_rate_smoothed[::step][-max_points:] if self.success_rate_smoothed else [0] * len(death_data)
                
                # 计算坐标
                x_step = chart_width / (len(death_data) - 1) if len(death_data) > 1 else 0
                
                # 绘制死亡率曲线
                for i in range(len(death_data) - 1):
                    x1 = self.chart_area.x + 5 + i * x_step
                    y1 = chart_start_y + chart_height - (death_data[i] * chart_height)
                    x2 = self.chart_area.x + 5 + (i + 1) * x_step
                    y2 = chart_start_y + chart_height - (death_data[i + 1] * chart_height)
                    
                    pygame.draw.line(self.screen, (255, 0, 0), (x1, y1), (x2, y2), 2)
                
                # 绘制成功率曲线
                if len(success_data) == len(death_data):
                    for i in range(len(success_data) - 1):
                        x1 = self.chart_area.x + 5 + i * x_step
                        y1 = chart_start_y + chart_height - (success_data[i] * chart_height)
                        x2 = self.chart_area.x + 5 + (i + 1) * x_step
                        y2 = chart_start_y + chart_height - (success_data[i + 1] * chart_height)
                        
                        pygame.draw.line(self.screen, (0, 255, 0), (x1, y1), (x2, y2), 2)
    
    def _get_q_value_color(self, q_value):
        """根据Q值返回颜色"""
        if q_value > 20:
            return (0, 200, 0)  # 绿色
        elif q_value > 0:
            return (0, 100, 0)  # 深绿色
        elif q_value == 0:
            return self.colors['text']
        elif q_value > -20:
            return (200, 100, 0)  # 橙色
        else:
            return (200, 0, 0)  # 红色
    
    def _draw_logs(self):
        """绘制日志区域"""
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['text'], self.log_area, 2)
        
        # 标题
        title_surface = self.font.render("Logs (v1.0)", True, self.colors['text'])
        self.screen.blit(title_surface, (self.log_area.x + 10, self.log_area.y + 5))
        
        # 绘制日志消息
        y_offset = self.log_area.y + 30
        for i, message in enumerate(self.log_messages):
            text_surface = self.tiny_font.render(message, True, self.colors['text'])
            self.screen.blit(text_surface, (self.log_area.x + 10, y_offset + i * 18))
    
    def _update_rates(self):
        """更新平滑的死亡率和成功率"""
        if len(self.death_history) < self.smoothing_window:
            window_data_death = self.death_history
            window_data_success = self.success_history
        else:
            window_data_death = self.death_history[-self.smoothing_window:]
            window_data_success = self.success_history[-self.smoothing_window:]
        
        # 计算死亡率和成功率
        death_rate = sum(window_data_death) / len(window_data_death) if window_data_death else 0
        success_rate = sum(window_data_success) / len(window_data_success) if window_data_success else 0
        
        self.death_rate_smoothed.append(death_rate)
        self.success_rate_smoothed.append(success_rate)
    
    def _add_log(self, message, level='INFO'):
        """添加日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        # 压缩日志格式
        compressed_message = f"[{timestamp[-5:]}] {message}"
        self.log_messages.append(compressed_message)
    
    def set_fps(self, fps):
        """设置帧率"""
        self.fps = fps
    
    def close(self):
        """关闭可视化窗口"""
        pygame.quit()