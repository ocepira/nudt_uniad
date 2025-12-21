import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings
import sys
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import logging
import json
from io import StringIO
def sse_print(event: str, data: dict) -> str:
    """
    SSE 打印
    :param event: 事件名称
    :param data: 事件数据（字典或能被 json 序列化的对象）
    :return: SSE 格式字符串
    """
    # 将数据转成 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # 按 SSE 协议格式拼接
    message = f"event: {event}\n" \
              f"data: {json_str}\n"
    print(message, flush=True)


from contextlib import contextmanager
@contextmanager    
def suppress_stdout():
    """临时屏蔽标准输出的上下文管理器"""
    original_stdout = sys.stdout  # 保存原始stdout
    sys.stdout = open(os.devnull, 'w')  # 重定向到空设备
    try:
        yield
    finally:
        sys.stdout.close()  # 关闭空设备文件
        sys.stdout = original_stdout  # 恢复原始stdout
# 创建过滤器屏蔽特定消息
@contextmanager    
def suppress_stdout_stderr():
    """同时抑制 stdout、stderr 和文件描述符"""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    
    # 保存原始文件描述符
    save_stdout_fd = os.dup(1)
    save_stderr_fd = os.dup(2)
    
    try:
        # 重定向到 /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        
        # Python 层面也要重定向
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        yield
    finally:
        # 恢复文件描述符
        os.dup2(save_stdout_fd, 1)
        os.dup2(save_stderr_fd, 2)
        os.close(save_stdout_fd)
        os.close(save_stderr_fd)
        
        # 恢复 sys.stdout 和 sys.stderr
        sys.stdout = save_stdout
        sys.stderr = save_stderr

class UpgradeInfoFilter(logging.Filter):
    def filter(self, record):
        return "ModulatedDeformConvPack" not in record.getMessage()

# 应用到 root logger
logging.root.addFilter(UpgradeInfoFilter())
# 重定向stdout/stderr来过滤特定输出
class FilteredStdout:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
    
    def write(self, text):
        if "ModulatedDeformConvPack" not in text:
            self.original_stdout.write(text)
    
    def flush(self):
        self.original_stdout.flush()

# 应用过滤器
sys.stdout = FilteredStdout(sys.stdout)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

# ===================== 配置 logging，屏蔽 INFO 日志 =====================
import logging

# 全局配置 logging，设置 root 日志级别为 WARNING（屏蔽 INFO/DEBUG）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 清理 root 日志器的默认 handler，避免重复输出
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# 重新添加 handler，强制级别为 WARNING
handler = logging.StreamHandler()
handler.setLevel(logging.WARNING)
root_logger.addHandler(handler)

# 针对 mmcv/mmdet/mmdet3d 等相关库的日志器也强制设为 WARNING
for logger_name in ['root', 'mmcv', 'mmdet', 'mmdet3d', 'mmengine']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False

# ===================== 屏蔽 warnings 警告 =====================
import warnings
warnings.filterwarnings("ignore")

# ===================== 导入其他模块 =====================
from mmdet3d.apis import single_gpu_test
from mmdet3d.apis import single_gpu_test
warnings.filterwarnings("ignore")
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmcv').setLevel(logging.WARNING)
logging.getLogger('mmcv.runner').setLevel(logging.WARNING)
# from utils.attack import attacks



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
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



def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    with suppress_stdout():
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu',strict=False)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        with suppress_stdout_stderr():
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        # assert False
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        with suppress_stdout_stderr():
            outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                            args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            sse_print('writting_results', {'results_path': args.out})
            # print(f'\nwriting results to {args.out}')
            #assert False
            mmcv.dump(outputs, args.out)
            #outputs = mmcv.load(args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            
            try:
                print(dataset.evaluate(outputs, **eval_kwargs))
            except AssertionError as e:
                if "unachieved_thresholds + duplicate_thresholds" in str(e):
                    print("Warning: Tracking evaluation failed due to threshold calculation issue.")
                    print("This is a known issue with the nuScenes tracking evaluation code, ")
                    print("often occurring with small datasets like v1.0-mini.")
                    print("Skipping tracking metrics...")
                else:
                    raise
            except Exception as e:
                print(f"Evaluation failed with error: {e}")
                raise

if __name__ == '__main__':
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method('spawn')
    main()