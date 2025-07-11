# 强化学习"机器人安全过马路"项目设计文档 - v0.5

版本：0.5（已完成）  
日期：2025-07-10  
状态：✅ 已实现并标记为v0.5

> **注意**：本文档描述已完成的v0.5版本。v1.0设计请参考 [plan-v1.md](plan-v1.md)

## 第一章：Executive Summary

本项目通过构建一个"机器人安全过马路"的模拟环境，探索强化学习(RL)算法的实现与应用。项目采用渐进式开发策略：

- **v0.5版本**：实现核心Q-Learning算法与基础可视化系统
- **v1.0版本**：扩展环境复杂度，引入二维空间与时间维度
- **v2.0+版本**：升级至深度强化学习算法(DQN, A2C/A3C)

项目价值在于：
1. 提供强化学习算法的直观理解与实践经验
2. 构建可扩展的实验平台，支持算法迭代与环境演进
3. 通过可视化系统实时展示算法学习过程

## 第二章：Execution Plan for v0.5

### 2.1 系统架构总览

v0.5版本将包含四个核心模块：

```
┌─────────────────────────────────────────────────┐
│              主控制器 (Main Controller)           │
│  - 训练循环管理                                   │
│  - 模块间协调                                     │
└─────────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┬────────────────┐
    ▼               ▼               ▼                ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  环境   │    │Q-Learning│    │可视化系统│    │ 日志系统 │
│ 模拟器  │◄───│  引擎    │───►│         │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 2.2 环境模拟器详细设计

#### 2.2.1 环境参数
```python
class Environment:
    def __init__(self):
        self.num_lanes = 5           # 车道数量
        self.start_position = -1     # 起点位置
        self.end_position = 5        # 终点位置
        self.traffic_light_cycle = 10 # 红绿灯周期(步数)
        self.car_spawn_probability = 0.3  # 车辆生成概率
```

#### 2.2.2 状态表示
```python
State = namedtuple('State', ['robot_lane', 'light_status'])
# robot_lane: -1(起点), 0-4(车道), 5(终点)
# light_status: 0(红灯), 1(绿灯)
```

**重要设计决策**：移除了原有的`car_imminent`字段，因为它与车辆生成的随机性存在同步问题，实际上是无效信号。机器人现在纯粹依靠红绿灯状态和自身位置来做决策，这更符合现实的交通规则。

#### 2.2.3 环境动态
- 红绿灯：固定周期切换（10步绿灯，10步红灯）
- 车辆生成：**仅在红灯时**按概率在随机车道生成，绿灯时无车辆
- 碰撞检测：机器人进入有车车道时触发
- 交通规则：绿灯=机器人通行，红灯=车辆通行

**v0.5学习机制说明**：
- 机器人仅基于红绿灯状态和自身位置做决策，无车辆预警信息
- 机器人需要通过试错学习：绿灯时通行相对安全，红灯时前进风险较高
- 这种设计让机器人真正学会遵守交通规则，而不是依赖"透视"能力

**实际训练发现**：
机器人在v0.5环境中学到了一个有趣但保守的策略：**"站着不动"**
- 原因：移动具有风险（绿灯时可能连续前进完成，红灯时直接被撞死-100分），而不动是安全的（只有-1的步骤惩罚）
- 这说明当前奖励结构下，避免-100惩罚比追求+50奖励更重要
- 该发现促使我们重新思考环境设计，引出v1.0的改进方向

### 2.3 Q-Learning算法实现细节

#### 2.3.1 核心数据结构
```python
class QLearningAgent:
    def __init__(self):
        # Q-Table: {state: [Q_forward, Q_backward]}
        self.q_table = defaultdict(lambda: [0.0, 0.0])
        
        # 超参数
        self.alpha = 0.1          # 学习率
        self.gamma = 0.9          # 折扣因子
        self.epsilon = 1.0        # 探索率初始值
        self.epsilon_decay = 0.995 # 探索率衰减
        self.epsilon_min = 0.01   # 最小探索率
```

#### 2.3.2 训练流程
```python
def training_loop():
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 1. 选择动作(ε-greedy策略)
            action = agent.choose_action(state)
            
            # 2. 执行动作，获取反馈
            next_state, reward, done = env.step(action)
            
            # 3. 更新Q值
            agent.update(state, action, reward, next_state, done)
            
            # 4. 可视化更新
            visualizer.update(state, action, reward, agent.q_table)
            
            state = next_state
        
        # 衰减探索率
        agent.decay_epsilon()
```

#### 2.3.3 Q值更新公式实现
```python
def update(self, state, action, reward, next_state, done):
    # 获取当前Q值
    current_q = self.q_table[state][action]
    
    # 计算目标Q值
    if done:
        target_q = reward
    else:
        max_next_q = max(self.q_table[next_state])
        target_q = reward + self.gamma * max_next_q
    
    # 更新Q值
    self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
```

### 2.4 可视化系统设计

#### 2.4.1 布局设计

```
┌─────────────────────────────────────────────────────────────┐
│                        窗口标题栏                              │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│      环境可视化区域           │        Q-Table展示区域        │
│   (60% 窗口宽度)             │      (40% 窗口宽度)          │
│                              │                              │
│   ┌─────────────────┐        │   状态      Forward Backward │
│   │  🚦 红/绿灯      │        │   (-1,0,F)   0.00   -1.00  │
│   ├─────────────────┤        │   (0,0,T)   -100.0   0.00  │
│   │ Lane 0 [🚗]     │        │   (0,1,F)    2.45   -1.00  │
│   │ Lane 1 [  ]     │        │   ...                       │
│   │ Lane 2 [🤖]     │        │                              │
│   │ Lane 3 [🚗]     │        │   当前状态: (2,1,True)      │
│   │ Lane 4 [  ]     │        │   选择动作: Backward        │
│   ├─────────────────┤        │                              │
│   │     终点 🏁      │        │                              │
│   └─────────────────┘        │                              │
│                              │                              │
├──────────────────────────────┴──────────────────────────────┤
│                         日志区域                              │
│  Episode 42: Step 15, Reward: -1, Total: 23                 │
│  [INFO] Robot moved backward to lane 1                      │
│  [WARN] Car approaching in lane 2!                          │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.2 可视化组件实现

```python
class Visualizer:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font = pygame.font.Font(None, 24)
        
        # 区域划分
        self.env_area = pygame.Rect(0, 50, width*0.6, height*0.7)
        self.qtable_area = pygame.Rect(width*0.6, 50, width*0.4, height*0.7)
        self.log_area = pygame.Rect(0, height*0.8, width, height*0.2)
        
        # 颜色定义
        self.colors = {
            'background': (240, 240, 240),
            'road': (100, 100, 100),
            'car': (255, 0, 0),
            'robot': (0, 0, 255),
            'goal': (0, 255, 0),
            'text': (0, 0, 0)
        }
```

#### 2.4.3 实时更新机制

1. **训练模式可视化**
   - 每个训练步骤实时更新
   - 显示当前状态、动作选择、奖励值
   - Q-Table热力图展示（高Q值用暖色，低Q值用冷色）
   - 滚动日志显示最近20条事件

2. **演示模式可视化**
   - 训练完成后的策略展示
   - 慢速播放机器人决策过程
   - 突出显示当前状态在Q-Table中的对应行
   - 显示动作选择理由（Q值比较）

#### 2.4.4 关键可视化功能

```python
def draw_environment(self):
    """绘制环境状态"""
    # 绘制道路
    for i in range(5):
        y = self.env_area.y + i * lane_height
        pygame.draw.rect(self.screen, self.colors['road'], 
                        (self.env_area.x, y, self.env_area.width, lane_height))
    
    # 绘制机器人、车辆、红绿灯等

def draw_qtable(self, q_table, current_state):
    """绘制Q-Table，突出显示当前状态"""
    # 遍历Q-Table，绘制每个状态的Q值
    # 使用颜色编码表示Q值大小
    # 当前状态行高亮显示

def update_log(self, message, level='INFO'):
    """更新日志显示"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    self.log_messages.append(f"[{timestamp}] [{level}] {message}")
    # 保持最新20条消息
```

### 2.5 技术实现里程碑

#### Phase 1: 基础框架搭建 (2天)
- [ ] 环境类基本结构
- [ ] Q-Learning Agent框架
- [ ] Pygame窗口初始化
- [ ] 基础绘图功能

#### Phase 2: 核心功能实现 (3天)
- [ ] 完整的环境动态逻辑
- [ ] Q-Learning算法实现
- [ ] 状态转换与奖励计算
- [ ] 基础可视化渲染

#### Phase 3: 可视化系统完善 (3天)
- [ ] Q-Table实时展示
- [ ] 动画效果优化
- [ ] 日志系统集成
- [ ] 训练进度显示

#### Phase 4: 调试与优化 (2天)
- [ ] 超参数调优
- [ ] 性能优化
- [ ] Bug修复
- [ ] 文档完善

### 2.6 项目文件结构

```
robot-rl-cross-road/
├── main.py              # 主程序入口
├── environment.py       # 环境模拟器
├── q_learning.py        # Q-Learning算法实现
├── visualizer.py        # 可视化系统
├── config.py           # 配置参数
├── utils.py            # 工具函数
├── logs/               # 训练日志
├── saved_models/       # 保存的Q-Table
└── README.md           # 项目说明
```

## 第三章：未来展望

### 3.1 v1.0 - 环境复杂度提升

#### 3.1.1 二维网格世界
- 从1D车道扩展到5x5网格
- 机器人四向移动能力
- 多车辆同时存在
- 车辆运动轨迹预测
- **车辆预警系统**：机器人能看到2-3格范围内的车辆，提前规划躲避路线

#### 3.1.2 时间维度引入
- 红绿灯倒计时显示
- 车辆速度差异化
- 行人穿越模拟
- 天气影响因素

### 3.2 v2.0+ - 算法升级路线

#### 3.2.1 Deep Q-Network (DQN)
- 神经网络替代Q-Table
- 经验回放机制
- 目标网络稳定训练
- 连续状态空间处理

#### 3.2.2 Actor-Critic方法
- A2C算法实现
- 并行环境训练(A3C)
- 策略梯度优化
- 连续动作空间支持

### 3.3 技术栈演进

```
v0.5: Python + Pygame + NumPy
  ↓
v1.0: + Matplotlib (高级可视化)
  ↓  
v2.0: + PyTorch/TensorFlow + Stable-Baselines3
  ↓
v3.0: + Unity ML-Agents (3D仿真)
```

## 第四章：结论

本项目通过循序渐进的方式，从最简单的Q-Learning算法开始，逐步构建一个完整的强化学习实验平台。v0.5版本专注于核心概念的实现和可视化，为后续的算法升级和环境扩展奠定坚实基础。

关键成功因素：
1. 简化初始复杂度，快速验证核心概念
2. 强调可视化，提供直观的学习反馈
3. 模块化设计，支持渐进式功能扩展
4. 详细的技术文档，便于知识传承

通过这个项目，我们不仅能深入理解强化学习的原理，还能获得宝贵的工程实践经验，为探索更高级的AI技术打下基础。