# 机器人安全过马路 - 强化学习项目

使用Q-Learning算法训练机器人学习如何安全过马路的强化学习演示项目。

## 项目特点

- 简化的一维道路环境
- 经典Q-Learning算法实现
- 实时Pygame可视化系统
- 模块化设计，易于扩展

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
├── plan.md             # 详细设计文档
└── CLAUDE.md           # 开发笔记
```

## 环境说明

- **状态空间**: (机器人位置, 红绿灯状态, 下一车道是否有车)
- **动作空间**: Forward（前进）, Backward（后退）
- **奖励机制**:
  - 到达终点: +50
  - 碰撞: -100
  - 每步: -1

## 可视化界面

- 左侧：环境状态展示（道路、机器人、车辆、红绿灯）
- 右侧：Q-Table实时更新，高亮当前状态
- 底部：日志信息

## 测试

```bash
python tests/test_integration.py
```

## 未来计划

- v1.0: 二维网格世界，车辆预警系统
- v2.0+: 深度强化学习算法（DQN, A2C/A3C）

详见 [plan.md](plan.md) 了解完整的项目规划。