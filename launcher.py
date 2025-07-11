#!/usr/bin/env python3
"""
统一启动器 - 选择版本并启动对应的main.py
"""
import sys
import argparse
import importlib.util
import os

def get_available_versions():
    """获取可用版本列表"""
    versions_dir = "versions"
    if not os.path.exists(versions_dir):
        return []
    
    versions = []
    for item in os.listdir(versions_dir):
        version_path = os.path.join(versions_dir, item)
        if os.path.isdir(version_path):
            main_file = os.path.join(version_path, "main.py")
            if os.path.exists(main_file):
                versions.append(item)
    
    return sorted(versions)

def load_version_module(version):
    """动态加载版本模块"""
    try:
        # 使用importlib动态导入
        module_name = f"versions.{version}.main"
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f"Error loading version {version}: {e}")
        return None

def main():
    """主函数"""
    available_versions = get_available_versions()
    
    if not available_versions:
        print("No versions found in versions/ directory")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='RL-Demo 统一启动器')
    parser.add_argument('--version', choices=available_versions, 
                       required=True, help='选择版本')
    parser.add_argument('--mode', choices=['train', 'demo', 'both'], 
                       default='train', help='运行模式')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练回合数')
    parser.add_argument('--no-vis', action='store_true',
                       help='不显示可视化界面')
    parser.add_argument('--fast', action='store_true',
                       help='快速训练模式（无延时）')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（用于演示）')
    
    # 解析已知参数，其余参数传给版本模块
    args, unknown_args = parser.parse_known_args()
    
    print(f"Launching version {args.version}...")
    print(f"Available versions: {', '.join(available_versions)}")
    
    # 动态导入对应版本的主模块
    version_module = load_version_module(args.version)
    if version_module is None:
        print(f"Failed to load version {args.version}")
        sys.exit(1)
    
    # 重构参数列表传给版本模块
    version_args = [
        '--mode', args.mode,
    ]
    
    if args.episodes is not None:
        version_args.extend(['--episodes', str(args.episodes)])
    
    if args.no_vis:
        version_args.append('--no-vis')
    
    if args.fast:
        version_args.append('--fast')
    
    if args.model is not None:
        version_args.extend(['--model', args.model])
    
    # 添加未知参数
    version_args.extend(unknown_args)
    
    # 临时替换sys.argv来传递参数
    original_argv = sys.argv
    try:
        sys.argv = [f"versions/{args.version}/main.py"] + version_args
        
        # 调用版本的main函数
        if hasattr(version_module, 'main'):
            version_module.main()
        else:
            print(f"Version {args.version} does not have a main() function")
            sys.exit(1)
    
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()