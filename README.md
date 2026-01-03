# UniAD: Unified Autonomous Driving Framework

本项目基于uniad库实现自动驾驶场景下的模型测试、对抗攻击和防御功能。支持攻击和防御策略，帮助评估和提升自动驾驶系统的鲁棒性。支持nuscenes_tiny数据集

## 功能特性

- **模型测试**: 对自动驾驶模型进行标准测试和评估
- **对抗攻击**: 支持多种对抗攻击方法生成对抗样本
- **防御机制**: 提供多种防御策略增强模型鲁棒性

## 环境变量

| 变量名 | 是否必填 | 描述 |
|--------|---------|------|
| input_path | 选填 | 指定输入路径，在此路径下有权重文件和数据集文件 |
| output_path | 选填 | 指定输出路径，在此路径下保存生成的对抗样本和防御训练的权重 |
| process | 选填 | 指定进程名称，支持枚举值（第一个为默认值）: `test`, `attack`, `defense` ，'adv',|
| image-path | 选填 | 输入图像路径，当process为`attack`或`defense`时默认input下的nuscenes-tiny |
| attack-method | 选填 | 指定攻击方法，若process为`attack`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd`, `bim`,`badnet`, `squareattack`, `nes` |
| defense-method | 选填 | 指定防御方法，若process为`defense`则必填，支持枚举值（第一个为默认值）: `fgsm`, `pgd` |
| save-path | 选填 | 对抗样本保存路径 |
| save-original-size | 选填 | 是否保存原始尺寸的对抗样本 |
| config | 选填 | test config file path，当process为`test`时不填v为默认|
| checkpoint | 选填 | checkpoint file，当process为`test`不填为默认|
| steps | 选填，默认为10 | 攻击迭代次数(PGD/BIM) |
| alpha | 选填，默认为2/255 | 攻击步长(PGD/BIM) |
| epsilon | 选填，默认为8/255 | 扰动强度 |
| device | 选填，默认为0 | 使用哪个gpu |
| workers | 选填，默认为0 | 加载数据集时workers的数量 |

## 下载nuscene_tiny数据集如/data所示
## 下载对应的 /model

## 快速开始
python main.py --process xxx

### Docker 镜像
docker build -t uniad:latest .
测试预训练的自动驾驶模型
