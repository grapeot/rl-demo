# Claude Development Notes

## 项目环境设置

本项目使用 `uv` 作为包管理工具。

### 创建和激活虚拟环境
```bash
# 创建虚拟环境
uv venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 安装依赖
```bash
# 使用 uv pip 安装包
uv pip install pygame numpy
```

### 运行项目
```bash
# 训练模型
python main.py --mode train

# 演示训练好的模型
python main.py --mode demo

# 训练并演示
python main.py --mode both

# 无可视化训练（最快）
python main.py --mode train --no-vis

# 快速训练（有可视化但无延时）
python main.py --mode train --fast

# 指定训练回合数
python main.py --mode train --episodes 500 --fast
```

## 项目结构
- `environment.py`: 环境模拟器
- `q_learning.py`: Q-Learning算法实现  
- `visualizer.py`: Pygame可视化系统
- `config.py`: 配置参数
- `main.py`: 主程序入口
- `plan.md`: 项目设计文档

## 测试命令
```bash
# 运行集成测试
python tests/test_integration.py

# 快速验证安装
python -c "from environment import RoadEnvironment; print('✓ 环境模块正常')"
python -c "from q_learning import QLearningAgent; print('✓ Q-Learning模块正常')"
```

## 重要修复记录
- ✅ 交通灯逻辑：绿灯时无车辆，红灯时有车辆
- ✅ 红绿灯周期：20步（10绿+10红），给机器人充足时间
- ✅ 等待机制：机器人可在起点通过后退动作等待
- ✅ 连续时间：红绿灯在多个episode间持续变化
- ✅ 训练效率：快速模式（--fast）和无可视化模式（--no-vis）
- ✅ Demo模式：自动播放（2 FPS），修复初始状态显示
- ✅ 死亡率统计：实时图表显示训练进度和改进情况
- ✅ 状态空间简化：移除car_imminent，消除同步问题和无效信号

## 可视化功能
- **环境区域**：道路、机器人、车辆、红绿灯状态
- **Q-Table区域**：实时Q值更新，当前状态高亮
- **统计图表**：死亡率趋势，50回合滑动平均
- **日志区域**：详细的决策过程记录