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
# 测试环境模块
python tmp_test_environment.py

# 测试Q-Learning模块
python tmp_test_qlearning.py
```