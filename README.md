# Super Mario Bros DQN 强化学习项目

一个使用深度Q网络(DQN)训练AI智能体玩超级马里奥兄弟游戏的强化学习项目。

![Mario DQN Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.26+-green.svg)

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [核心模块详解](#核心模块详解)
- [训练配置](#训练配置)
- [使用说明](#使用说明)
- [结果分析](#结果分析)
- [参考](#参考)

## 🎮 项目简介

本项目实现了一个基于***Double DQN***的强化学习智能体，能够自主学习并掌握超级马里奥兄弟游戏。通过观察游戏画面，智能体学会了如何移动、跳跃以及避开障碍物，最终能够成功通关游戏关卡。

https://github.com/user-attachments/assets/7f7bc03a-1a9d-4baa-905a-8bd4cc2a9e2c


## ✨ 核心特性

- **DQN**: 使用卷积神经网络处理游戏画面
- **Double DQN**: 采用在线网络和目标网络减少过估计
- **经验回放**: 使用replay buffer提高学习效率
- **ε-贪心策略**: 平衡探索与利用
- **环境预处理**: 多层wrapper优化输入状态
- **实时监控**: 完整的训练指标记录和可视化
- **模型保存**: 自动保存训练检查点
- **演示模式**: 加载训练好的模型进行游戏演示

## 📁 项目结构

```
homework_3/
├── main.py                 # 主训练入口
├── demo.py                 # 模型演示
├── auto_demo.py           # 自动演示（无UI）
├── draw_log.py            # 重绘训练图表
├── utils/
│   ├── agent.py           # Mario DQN智能体
│   ├── net.py             # MarioNet神经网络
│   ├── wrappers.py        # 环境包装器
│   └── logger.py          # 训练指标记录器
├── checkpoints/           # 训练检查点目录
│   └── YYYY-MM-DDTHH-MM-SS/
│       ├── log            # 训练日志
│       ├── mario_net_*.chkpt  # 模型检查点
│       └── *.jpg          # 训练图表
└── README.md
```

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenAI Gym 0.26+
- nes-py
- gym-super-mario-bros


## 📦 安装指南

### 1. 克隆项目
```bash
git clone https://github.com/Crohkles/Mario_Agent_using_DDQN
```

### 2. 创建虚拟环境（使用conda）
```bash
conda create -n mario python=3.10
```

### 3. 激活虚拟环境并安装依赖
```bash
conda activate mario
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 快速开始

### 训练新模型
```bash
python main.py
```

训练时会提示选择：
- 运行模式（训练/演示）
- 是否显示游戏画面
- 是否继续之前的训练

### 演示训练好的模型
```bash
python demo.py
```
或者自动选取最近的检查点演示训练好的模型（无画面演示）
```bash 
python auto_demo.py
```


### 重新绘制训练图表
```bash
python draw_log.py
```


## 🔧 核心模块详解

### 1. Mario智能体 (`agent.py`)

实现了完整的DQN算法：

- **网络结构**: 在线网络 + 目标网络
- **经验回放**: 容量10万的replay buffer
- **学习策略**: ε-贪心探索策略
- **参数更新**: 定期同步目标网络

```python
# 核心参数
exploration_rate = 1.0          # 初始探索率
exploration_rate_decay = 0.99999975  # 探索率衰减
exploration_rate_min = 0.1     # 最小探索率
gamma = 0.9                     # 折扣因子
learning_rate = 0.00025         # 学习率
batch_size = 32                 # 批处理大小
```

### 2. 神经网络 (`net.py`)

卷积神经网络结构设计：

```
输入: 4×84×84 (4帧灰度图像)
├── Conv2d(4→32, 8×8, stride=4)  → 20×20×32
├── Conv2d(32→64, 4×4, stride=2) → 9×9×64  
├── Conv2d(64→64, 3×3, stride=1) → 7×7×64
├── Flatten                       → 3136
├── Linear(3136→512)
└── Linear(512→action_dim)       → 2 (向右/向右跳)
```

### 3. 环境预处理 (`wrappers.py`)

多层环境包装器优化输入：

- **SkipFrame**: 跳帧处理（每4帧执行一次动作）
- **GrayScaleObservation**: RGB转灰度图
- **ResizeObservation**: 图像缩放到84×84
- **FrameStack**: 堆叠4帧作为状态输入

### 4. 训练监控 (`logger.py`)

全面的训练指标记录：

- **实时指标**: 奖励、回合长度、损失、Q值
- **移动平均**: 100回合的平均性能
- **可视化**: 自动生成训练曲线图
- **断点续训**: 支持加载历史训练数据

## ⚙️ 训练配置

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `burnin` | 10,000 | 预热期步数 |
| `learn_every` | 3 | 学习频率（每3步学习一次） |
| `sync_every` | 10,000 | 目标网络同步频率 |
| `save_every` | 500,000 | 模型保存频率 |
| `memory_size` | 100,000 | 经验回放缓冲区大小 |

### 训练阶段

1. **预热阶段** (0-10k步): 随机探索填充经验池
2. **学习阶段** (10k+步): 开始DQN学习
3. **收敛阶段**: 探索率降至最小值，策略趋于稳定

## 📖 使用说明

### 训练模式

```bash
python main.py
```

**交互选项**:
- 选择训练模式 (输入1)
- 选择是否显示游戏画面 (y/n)
- 选择是否继续之前的训练 (y/n)

### 演示模式

```bash
python demo.py
```

**功能**:
- 加载训练好的模型
- 实时显示游戏过程
- 统计游戏表现

### 图表重绘

```bash
python draw_log.py
```

**生成文件**:
- `reward_plot.jpg` - 平均奖励曲线
- `length_plot.jpg` - 平均回合长度
- `loss_plot.jpg` - 训练损失
- `q_plot.jpg` - Q值变化

## 📊 结果分析

### 训练指标

- **Mean Reward**: 反映智能体获得奖励的能力
- **Mean Length**: 回合持续时间，越长表示存活越久
- **Mean Loss**: 神经网络训练损失
- **Mean Q Value**: Q函数的平均估值

### 性能表现

根据训练日志显示：
- 训练初期平均奖励约600分
- 经过训练后平均奖励提升至1600+分
- 回合长度从160步提升至300+步
- 探索率逐渐从1.0降至0.1

## 🙏 参考

- [PyTorch官方教程](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

---

**Happy Gaming with AI! 🎮🤖**
