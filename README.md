# Stereo Odometry on Orin NX

SuperPoint + LightGlue 双目视觉里程计，运行在 **NVIDIA Jetson Orin NX 16GB** (JetPack 7.2)。

## 环境信息

| 项目 | 值 |
|------|-----|
| 设备 | NVIDIA Jetson Orin NX 16GB (CC 8.7) |
| JetPack | **7.2** (L4T R39.2.0) |
| CUDA | 13.2 |
| cuDNN | 9.20.0 |
| TensorRT | 10.16.2 |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu132 |
| SSH | nvidia@192.168.1.23 |
| 项目路径 | ~/Documents/stereo_odom |

## 环境搭建

### 1. Python 虚拟环境

```bash
cd ~/Documents/stereo_odom
python3 -m venv stereo_odom
source stereo_odom/bin/activate
```

### 2. pip 国内镜像源

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### 3. 安装 PyTorch

> **⚠️ 坑点：** Orin NX 的 GPU 计算能力为 **8.7 (sm_87)**，而 PyTorch 官方 aarch64 wheel 不包含 sm_87 支持。
> 启动时会出现警告 `Found GPU0 Orin which is of compute capability (CC) 8.7... except {8.7}`。
> **但实测基础 tensor 运算和模型推理都能正常通过**，该警告不影响功能。

```bash
# 直接使用预下载的 wheel（推荐）
pip install torch-2.12.1+cu132-cp312-cp312-manylinux_2_28_aarch64.whl
```

如果从 PyTorch 官方源安装：

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
```

### 4. 安装基础依赖

> **⚠️ 坑点：** `numpy` 必须先于其他依赖安装，否则 PyTorch 导入时会报 `Failed to initialize NumPy` 警告。

```bash
pip install numpy
pip install opencv-python onnx onnxruntime h5py tqdm matplotlib scipy pillow
```

### 5. 下载模型文件

| 文件 | 大小 | 来源 |
|------|------|------|
| `superpoint_v1.pth` | 5.0 MB | [LightGlue Releases](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv) |
| `superpoint_lightglue.pth` | 45.3 MB | 同上 |

```bash
cd ~/Documents/stereo_odom
wget https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth
wget https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth
```

## 模型验证

### 架构说明

```
  左图                   右图
   │                      │
   ▼                      ▼
SuperPoint             SuperPoint
(superpoint_v1.pth)   (superpoint_v1.pth)
   │                      │
   ├─ 关键点检测           ├─ 关键点检测
   └─ 256-dim 描述子       └─ 256-dim 描述子
   │                      │
   └──────────┬───────────┘
              ▼
         LightGlue
    (superpoint_lightglue.pth)
              │
              ├─ 9层 Self-Attention
              ├─ 9层 Cross-Attention
              └─ Sinkhorn 最优传输匹配
              │
              ▼
         匹配点对
```

### 运行验证

```bash
cd ~/Documents/stereo_odom
source stereo_odom/bin/activate
python3 validate_model.py
```

预期输出：

```
SuperPoint: ✅ VALID  (24/24 weights, scores [1,65,30,40])
LightGlue:  ✅ VALID  (260/260 weights, 100/100 points matched)
```

> **注意：** 验证脚本完全从零定义模型架构并加载预训练权重，**不需要安装 `superpoint` 或 `lightglue` 等第三方库**，避免依赖冲突。

### 模型参数

| 模型 | 参数量 | 说明 |
|------|--------|------|
| SuperPoint | **1.30M** | VGG-style CNN: 4层编码器 + 关键点头 + 描述子头 |
| LightGlue | **11.85M** | 9层 Transformer: Self-Attn + Cross-Attn + Sinkhorn |

## 坑点记录

### 🔴 严重

| 坑 | 说明 | 解决 |
|---|------|------|
| **PyTorch CC 8.7 不兼容** | 官方 aarch64 wheel 不支持 Orin NX 的 sm_87 | 不影响推理，可忽略警告 |
| **NVIDIA Jetson wheel 404** | JP 7.1/7.2 的 PyTorch wheel 尚未上传 | 使用通用 wheel 替代 |
| **numpy 缺失** | PyTorch 导入时报 `Failed to initialize NumPy` | 先 `pip install numpy` |

### 🟡 注意

| 注意 | 说明 |
|------|------|
| **国内网络** | NVIDIA 下载服务器访问慢，配置清华镜像源加速 |
| **模型文件 .gitignore** | `.pth` 文件体积大，已排除在版本控制外 |
| **虚拟环境独立** | 所有依赖安装在 venv 内，不影响系统 Python |

## 文件结构

```
~/Documents/stereo_odom/
├── stereo_odom/              # Python 虚拟环境 (gitignored)
├── superpoint_v1.pth         # SuperPoint 权重 (gitignored)
├── superpoint_lightglue.pth  # LightGlue 权重 (gitignored)
├── torch-2.12.1*.whl         # PyTorch wheel (gitignored)
├── validate_model.py         # 模型验证脚本
├── inspect_sp.py             # 模型检查工具
├── create_repo.py            # GitHub 仓库创建工具
├── .gitignore
└── README.md
```

## 参考链接

- [LightGlue: Local Feature Matching at Light Speed](https://github.com/cvg/LightGlue)
- [SuperPoint: Self-Supervised Interest Point Detection](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [JetPack 7.2 Release Notes](https://docs.nvidia.com/jetson/archives/)
