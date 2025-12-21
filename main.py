#!/usr/bin/env python3

import os
import sys
sys.path.append('')
import subprocess
import argparse
from datetime import datetime
import json
import sys
from contextlib import contextmanager
from contextlib import contextmanager
import sys
import warnings
from io import StringIO
from utils.sse import sse_input_path_validated,sse_output_path_validated
from utils.vadattack import ImageAttacker
from utils.vaddefense import FGSMDefense, PGDDefense, load_image , total_variation, load_image, save_image ,create_defense
import torch
import argparse
import numpy as np
from mmcv import Config, DictAction
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    
    # 基础参数 - 用于test模式
    parser.add_argument('cfg', nargs='?', help='config file path')
    parser.add_argument('ckpt', nargs='?', help='checkpoint file path')
    parser.add_argument('extra_args', nargs='*', help='additional arguments')
    parser.add_argument('--process', type=str, default='test', choices=['test', 'attack', 'defense'],help='process type: test, attack or defense')
    parser.add_argument('--alpha', type=float, default=2/255, help='攻击步长(PGD/BIM)')
    parser.add_argument('--steps', type=int, default=10, help='攻击迭代次数(PGD/BIM)')
       ##攻击
    parser.add_argument('--image-path', type=str, help='输入图像路径')
    parser.add_argument('--attack-method', type=str, default='pgd', 
                        choices=['fgsm', 'pgd', 'bim','badnet', 'squareattack', 'nes'], 
                        help='对抗攻击方法')
    parser.add_argument('--epsilon', type=float, default=8/255, help='扰动强度')
    parser.add_argument('--save-path', type=str, help='对抗样本保存路径')
    parser.add_argument('--save-original-size', action='store_true', help='是否保存原始尺寸的对抗样本')
    parser.add_argument('--model-name', type=str, default='Standard', help='模型名称')
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称')
    ##防御
    parser.add_argument('--defense-method', type=str, default='fgsm', 
                       choices=['fgsm', 'pgd',], 
                       help='防御方法')
    # parser.add_argument('--epsilon', type=float, default=8.0, help='扰动强度限制')
    parser.add_argument('--tv-weight', type=float, default=1.0, help='总变差权重')
    parser.add_argument('--l2-weight', type=float, default=0.01, help='L2保真权重')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 返回解析的参数
    args = parser.parse_args()
    
    # 1. 
    method_mapping = {
        'badnet': 'deepfool',
        'squareattack': 'mifgsm',
        'nes': 'cw'
    }
    # 2. 处理输入：统一转小写，去除多余空格（防止用户输入" square  attack"等情况）
    input_method = args.attack_method.strip().lower()
    # 3. 匹配映射规则，替换攻击方法
    if input_method in method_mapping:
        args.attack_method = method_mapping[input_method]

    ## 防御

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 处理数据，确保它可以被JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            # 对于其他不可序列化的对象，转换为字符串表示
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    # 将数据转成 JSON 字符串
    try:
        cleaned_data = convert_for_json(data)
        json_str = json.dumps(cleaned_data, ensure_ascii=False)
    except Exception as e:
        # 如果仍然失败，则只发送简单的错误消息
        json_str = json.dumps({"error": "Failed to serialize data", "exception": str(e)}, ensure_ascii=False)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)

def main():
    # 获取命令行参数
    args = parse_args()
    
    if args.process == "test":
        # 检查必需参数是否存在
        if not args.cfg or not args.ckpt:
            print("Error: cfg and ckpt are required for test process")
            sys.exit(1)

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
        
        # 添加额外参数到命令中
        cmd.extend(args.extra_args)
        
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

    elif args.process == "attack":
         # 获取设备信息
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            attacker = ImageAttacker(
                # model_name="Standard",
                # dataset=args.dataset,
                attack_method=args.attack_method,
                eps=args.epsilon,
                alpha=args.alpha,
                steps=args.steps,
                device=device
            )
            # attacker.attack(
            #     image_path=args.image_path,
            #     save_path=args.save_path
            # )
            # 加载模型
            # sse_print("model_loading", {"message": "正在加载模型..."})
            if args.dataset.lower() == 'cifar10':
                try:
                    from robustbench.utils import load_model
                    model = load_model(model_name = 'Standard', norm='Linf', dataset=args.dataset).to(device)
                except ImportError:
                    sse_print("error", {"message": "请安装 robustbench 库: pip install git+https://github.com/RobustBench/robustbench.git"})
                    raise ImportError("请安装 robustbench 库: pip install git+https://github.com/RobustBench/robustbench.git")
                except Exception as e:
                    sse_print("error", {"message": f"加载模型失败: {e}"})
                    raise Exception(f"加载模型失败: {e}")
            else:
                sse_print("error", {"message": f"暂不支持数据集: {args.dataset}"})
                raise NotImplementedError(f"暂不支持数据集: {args.dataset}")
            
            model.eval()
            # sse_print("model_loaded", {"message": f"模型 {args.model_name} 加载成功", "model_name": args.model_name})
            
            # 执行攻击
            sse_print("attack_started", {"message": "开始执行对抗攻击..."})
            try:
                adv_images, true_label, pred_adv = attacker.attack_image(
                    model=model,
                    img_path=args.image_path,
                    save_path=args.save_path,
                    save_original_size=args.save_original_size
                )
                
                success = true_label != pred_adv
                # sse_print("attack_result", {
                #     "message": "攻击结果:",
                #     "true_label": true_label,
                #     "adversarial_prediction": pred_adv,
                #     "attack_success": success
                # })
                
                # if success:
                #     sse_print("attack_success", {"message": "✓ 攻击成功，模型被欺骗"})
                # else:
                #     sse_print("attack_failed", {"message": "✗ 攻击失败，模型预测一致"})
                    
            except Exception as e:
                sse_print("error", {"message": f"攻击过程中发生错误: {e}"})
                return False
                
    elif args.process == "defense":
        # 检查必需参数
        if not args.image_path or not args.save_path:
            sse_print("error", {"message": "image-path and save-path are required for defense process"})
            raise ValueError("image-path and save-path are required for defense process")
            
        if not os.path.exists(args.image_path):
            sse_print("error", {"message": f"找不到输入图像: {args.image_path}"})
            raise FileNotFoundError(f"找不到输入图像: {args.image_path}")
    
        # 创建输出目录
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
        # 加载图像
        sse_print("loading_image", {"message": f"正在加载图像: {args.image_path}"})
        try:
            image_tensor = load_image(args.image_path)
            # sse_print("image_loaded", {
            #     "message": f"图像加载成功，形状: {list(image_tensor.shape)}",
            #     "shape": list(image_tensor.shape)
            # })
        except Exception as e:
            sse_print("error", {"message": f"加载图像失败: {e}"})
            raise
        
        # 创建防御方法
        sse_print("creating_defense", {"message": f"正在创建防御方法: {args.defense_method}"})
        try:
            # 根据防御方法传递相应参数
            if args.defense_method.lower() == 'fgsm':
                defense = create_defense(
                    args.defense_method,
                    epsilon=args.epsilon,
                    tv_weight=args.tv_weight,
                    l2_weight=args.l2_weight
                )
            elif args.defense_method.lower() == 'pgd':
                defense = create_defense(
                    args.defense_method,
                    steps=args.steps,
                    alpha=args.alpha,
                    epsilon=args.epsilon,
                    tv_weight=args.tv_weight,
                    l2_weight=args.l2_weight
                )

            sse_print("defense_created", {"message": f"防御方法创建成功: {args.defense_method}"})
        except Exception as e:
            sse_print("error", {"message": f"创建防御方法失败: {e}"})
            raise
        
        # 执行防御
        sse_print("defense_started", {"message": f"开始执行{args.defense_method.upper()}防御"})
        try:
            purified_image, _ = defense(image_tensor)
            sse_print("defense_finished", {"message": f"{args.defense_method.upper()}防御执行完成"})
        except Exception as e:
            sse_print("error", {"message": f"执行防御失败: {e}"})
            raise
        
        # 保存结果
        sse_print("saving_image", {"message": f"正在保存防御后图像到: {args.save_path}"})
        try:
            save_image(purified_image, args.save_path)
            sse_print("process_completed", {"message": "图像防御处理完成"})
        except Exception as e:
            sse_print("error", {"message": f"保存图像失败: {e}"})
            raise

if __name__ == '__main__':
    main()
