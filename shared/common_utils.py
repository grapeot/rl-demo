"""
通用工具函数 - 所有版本共享的工具函数
"""

import pickle
import os
from datetime import datetime

def save_model(model, path):
    """保存模型到文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    """从文件加载模型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {path}")
    return model

def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def log_message(message, log_path=None):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')
    
    print(formatted_message)