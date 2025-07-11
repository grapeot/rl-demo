# 机器人过马路 - 强化学习演示项目

这是一个使用Q-Learning算法训练机器人学习如何安全通过红绿灯路口的强化学习演示项目。

## 项目特点

- 🤖 简洁的一维道路环境模拟
- 🧠 经典Q-Learning算法实现
- 📊 实时可视化训练过程
- 📈 死亡率统计与进度追踪
- 🎮 Pygame图形界面展示

## 快速开始

### 环境设置

```bash
# 使用uv创建虚拟环境
uv venv venv
source venv/bin/activate

# 安装依赖
uv pip install pygame numpy
```

### 运行项目

```bash
# 训练模型（带可视化）
python main.py --mode train

# 快速训练（无可视化）
python main.py --mode train --no-vis --episodes 500

# 演示训练好的模型
python main.py --mode demo

# 训练并演示
python main.py --mode both
```

## 项目结构

```
robot-rl-cross-road/
├── main.py              # 主程序入口
├── environment.py       # 环境模拟器
├── q_learning.py        # Q-Learning算法
├── visualizer.py        # 可视化系统
├── config.py           # 配置参数
├── tests/              # 测试脚本
├── docs/               # 文档目录
│   └── designs/        # 设计文档
└── CLAUDE.md           # 开发笔记
```

## 环境说明

### 状态空间
- **机器人位置**: 0-5的整数，表示在道路上的位置
- **红绿灯状态**: 0（红灯）或1（绿灯）

### 动作空间
- **前进（Forward）**: 向前移动一格
- **后退（Backward）**: 向后移动一格（在起点时保持原地）

### 游戏规则
- 🚦 绿灯时：机器人可以安全通行
- 🚗 红灯时：有车辆通行，机器人需要等待
- ⏱️ 红绿灯周期：20步（10步绿灯 + 10步红灯）

### 奖励设计
- ✅ 成功到达终点：+50分
- ❌ 与车辆碰撞：-100分
- ⏳ 每一步：-1分（鼓励快速通过）

## 可视化界面

训练过程中的可视化界面分为四个区域：

### 1. 环境显示区（左侧40%）
- 实时展示道路、机器人、车辆位置
- 动态显示红绿灯状态
- 碰撞时的视觉反馈

### 2. Q-Table展示区（中间25%）
- 实时显示Q值表
- 高亮当前状态
- 展示学习进度

### 3. 训练进度图表（右侧29%）
- 📉 死亡率曲线（红色）
- 📊 50回合滑动平均
- 📈 实时性能追踪

### 4. 决策日志区（底部45%）
- 详细记录每一步的决策过程
- 显示选择的动作和奖励
- 追踪碰撞和成功事件

## 测试

```bash
python tests/test_integration.py
```

## 版本规划

### v0.5（当前版本）✅
- 基础Q-Learning实现
- 一维道路环境
- 实时可视化系统

### v1.0（开发中）🚧
- 多格子车道系统
- 车辆倒计时预警
- 改进的状态空间设计

### v2.0+（计划中）📋
- 深度强化学习算法（DQN, A2C/A3C）
- 更复杂的交通场景
- 多智能体协同

详见 [docs/designs/](docs/designs/) 了解各版本的设计文档。

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License