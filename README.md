# 人工智能高阶课程大作业：JDnCNN

![Jittor](https://img.shields.io/badge/Jittor-1.3%2B-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Task](https://img.shields.io/badge/Task-Image%20Denoising-orange)

本项目复现并改进 DnCNN 图像去噪实验，将图像去噪模型迁移到 Jittor，并在盲去噪设置下引入噪声水平图和轻量 U-Net 结构。

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [系统配置](#系统配置)
- [模型结构](#模型结构)
- [训练配置](#训练配置)
- [输出文件](#输出文件)
- [训练](#训练)
- [测试](#测试)
- [实验结果](#实验结果)
- [参考文献](#参考文献)

## 项目结构

```text
.
├── configs/
│   └── default.yaml       # 默认训练、测试和路径配置
├── data/                  # 本地数据集，不纳入 Git 跟踪
├── models/                # 默认模型权重输出目录
├── outputs/
│   └── logs/              # 默认训练日志输出目录
├── scripts/
│   ├── train.py           # 训练入口
│   └── test.py            # 测试入口
├── src/
│   ├── config.py          # YAML 配置读取与命令行覆盖
│   ├── dataset.py         # HDF5 数据预处理与 Jittor Dataset
│   ├── models.py          # U-Net 去噪网络
│   └── utils.py           # 初始化、指标、数据增强工具
├── tests/                 # 配置与入口行为测试
├── test.py                # 兼容入口
├── requirements.txt
└── README.md
```

## 环境配置

建议使用 Python 3.9+。如果本机有 CUDA，Jittor 会根据环境编译并调用 GPU 后端。

```bash
conda create -n myDnCNN python=3.9 -y
conda activate myDnCNN
pip install -r requirements.txt
```

## 系统配置

项目主要依赖如下：

- `jittor`：模型训练和推理框架。
- `numpy`、`opencv-python`、`h5py`：图像读取、数值处理和 HDF5 数据缓存。
- `tensorboardX`：记录训练曲线。
- `scikit-image`：计算 PSNR 和 SSIM。
- `PyYAML`：读取 `configs/default.yaml`。

默认配置文件是 `configs/default.yaml`。脚本启动时会先读取 YAML，再用命令行参数覆盖对应字段。因此，修改默认实验配置只需要改 YAML；临时实验可以直接传命令行参数。

## 模型结构

当前模型定义在 `src/models.py`，主体是轻量 U-Net 去噪网络。

- 输入为 2 个通道：带噪灰度图像和噪声水平图。
- 编码器包含 3 层下采样模块，用于提取多尺度特征。
- 解码器包含 3 层上采样模块，并使用 skip connection 恢复空间细节。
- 网络输出预测噪声残差。
- 复原图像通过 `restored = noisy - predicted_noise` 得到，并裁剪到 `[0, 1]`。

`num_of_layers` 参数仅为兼容早期 DnCNN 命令保留，当前实际结构由 `UNet` 类决定。

## 训练配置

训练默认参数集中在 `configs/default.yaml`：

```yaml
train:
  preprocess: false
  batch_size: 128
  epochs: 10
  milestone: 6
  lr: 0.001
  mode: B
  noiseL: 25
  val_noiseL: 25
  num_workers: 4
  use_cuda: true
```

关键参数说明：

- `preprocess`：是否先把 `data/train` 和 `data/Set12` 预处理成 HDF5。
- `mode`：`S` 表示固定噪声水平训练，`B` 表示盲去噪训练。
- `noiseL`：`mode=S` 时使用的训练噪声水平。
- `val_noiseL`：验证阶段使用的噪声水平。
- `milestone`：达到该 epoch 后学习率变为原来的 1/10。
- `use_cuda`：是否启用 CUDA，最终是否可用由 Jittor 后端和本机环境决定。

命令行参数优先级高于 YAML。例如：

```bash
python -m scripts.train --epochs 20 --batch-size 64 --lr 0.0005
```

## 输出文件

默认路径同样由 `configs/default.yaml` 控制：

```yaml
paths:
  data_dir: data
  train_h5: train.h5
  val_h5: val.h5
  model_dir: models
  checkpoint_name: net.pkl
  log_dir: outputs/logs/DnCNN-B
  log_file: outputs/logs/DnCNN-B.log
```

训练会生成以下文件：

- `train.h5`、`val.h5`：预处理后的训练和验证缓存。
- `models/net.pkl`：默认模型权重文件。
- `outputs/logs/DnCNN-B/`：TensorBoard 事件文件。
- `outputs/logs/DnCNN-B.log`：控制台训练日志。

`data/`、`outputs/`、`models/*.pkl`、`train.h5`、`val.h5` 已写入 `.gitignore`，避免把数据集、模型权重和训练产物提交到仓库。

## 训练

首次训练前，如果还没有生成 HDF5 缓存，需要开启预处理：

```bash
python -m scripts.train --preprocess true
```

已经生成 `train.h5` 和 `val.h5` 后，可以直接训练：

```bash
python -m scripts.train
```

常用覆盖示例：

```bash
python -m scripts.train --epochs 20 --batch-size 64 --mode B --val_noiseL 25
```

如果希望长期修改默认值，编辑 `configs/default.yaml`；如果只想临时覆盖某次运行，使用命令行参数。

## 测试

测试默认读取 `models/net.pkl`。默认测试集和噪声水平由 `configs/default.yaml` 的 `eval` 字段决定：

```yaml
eval:
  test_data: Set12
  test_noiseL: 25
```

运行默认测试：

```bash
python -m scripts.test
```

指定测试集和噪声水平：

```bash
python -m scripts.test --test_data Set68 --test_noiseL 50
```

指定模型路径：

```bash
python -m scripts.test --model-dir models --checkpoint-name net.pkl
```

## 实验结果

**Set68 / BSD68（中等噪声）**

| 噪声水平 sigma | 基线 DnCNN-B PSNR | 本方法 PSNR | 基线 DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 15 | 31.60 | 31.62 | 0.891 | 0.891 | +0.02 dB |
| 25 | 29.15 | 29.16 | 0.827 | 0.829 | +0.01 dB |
| 50 | 26.20 | 26.20 | 0.714 | 0.715 | 基本持平 |

**Set68 / BSD68（高噪声盲去噪）**

| 噪声水平 sigma | DnCNN-B PSNR | 本方法 PSNR | DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 26.20 | 26.11 | 0.715 | 0.708 | -0.09 dB |
| 75 | 17.89 | **24.47** | 0.294 | **0.621** | **+6.58 dB** |
| 100 | 13.65 | **22.95** | 0.160 | **0.508** | **+9.30 dB** |

## 参考文献

- Zhang et al., [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189/)
- Original MATLAB implementation: [cszn/DnCNN](https://github.com/cszn/DnCNN)
