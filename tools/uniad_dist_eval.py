#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='config file path')
    parser.add_argument('ckpt', help='checkpoint file path')
    # 允许接受额外的参数，为了兼容原来的调用方式
    parser.add_argument('extra_args', nargs='*', help='additional arguments')
    args = parser.parse_args()
    
    CFG = args.cfg
    CKPT = args.ckpt
    
    # 设置工作目录 (与bash脚本保持一致的逻辑)
    # WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
    WORK_DIR = CFG.replace('configs', 'work_dirs')
    if WORK_DIR.endswith('.py'):
        WORK_DIR = WORK_DIR[:-3] + '/'
    else:
        WORK_DIR = WORK_DIR + '/'
    
    # 创建日志目录
    log_dir = os.path.join(WORK_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取当前时间戳 (与bash脚本保持一致的格式)
    # T=`date +%m%d%H%M`
    timestamp = datetime.now().strftime('%m%d%H%M')
    
    # 构建并执行命令
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'test.py'),
        CFG,
        CKPT,
        '--launcher', 'none',
        '--eval', 'bbox',
        '--show-dir', WORK_DIR
    ]
    
    # 设置环境变量 (与bash脚本保持一致)
    # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{os.path.dirname(__file__)}/..:{env.get('PYTHONPATH', '')}"
    
    # 执行命令并将输出保存到日志文件 (与bash脚本保持一致的行为)
    log_file = os.path.join(log_dir, f'eval.{timestamp}')
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, universal_newlines=True)
        
        # 实时输出并写入日志，同时过滤掉包含 "ModulatedDeformConvPack" 的行
        for line in process.stdout:
            # 过滤掉包含 "ModulatedDeformConvPack" 的输出行，模拟 grep -v 的行为
            if "ModulatedDeformConvPack" not in line:
                print(line, end='')
                f.write(line)
        
        process.wait()

if __name__ == '__main__':
    main()