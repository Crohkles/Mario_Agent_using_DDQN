import torch 
import warnings

# 抑制gym_super_mario_bros的overflow警告
warnings.filterwarnings("ignore", message="overflow encountered in scalar subtract")

from pathlib import Path
import datetime

import gym

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from utils.logger import MetricLogger
from utils.agent import Mario

# 选择运行模式
print("请选择运行模式:")
print("1. 训练模式 (默认)")
print("2. 演示模式")
mode_choice = input("请输入选择 (1/2): ").strip()

if mode_choice == '2':
    print("演示模式请运行 demo.py 文件")
    exit()

# 选择是否显示游戏画面
show_game = input("是否显示游戏画面？(y/n, 显示会明显降低训练速度): ").lower().strip() == 'y'
render_mode = 'human' if show_game else 'rgb'

if gym.__version__ <'0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0",render_mode=render_mode,
                                    apply_api_compatibility=True)

# 将动作空间限制为
# 1.right（向右移动）
# 2.right A (向右跳)
env = JoypadSpace(env, [["right"], ["right","A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

# 检查是否使用现有的检查点目录
checkpoints_base = Path("checkpoints")
use_existing = input("是否继续之前的训练？(y/n): ").lower().strip() == 'y'

if use_existing and checkpoints_base.exists():
    # 寻找最新的检查点目录
    existing_dirs = [d for d in checkpoints_base.iterdir() if d.is_dir()]
    if existing_dirs:
        save_dir = max(existing_dirs, key=lambda x: x.stat().st_mtime)
        print(f"继续使用目录: {save_dir}")
    else:
        save_dir = checkpoints_base / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        print(f"没有找到现有目录，创建新目录: {save_dir}")
else:
    save_dir = checkpoints_base / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    print(f"创建新的训练目录: {save_dir}")

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

# 尝试加载已有模型
if use_existing:
    # 寻找最新的检查点文件
    checkpoint_files = []
    for checkpoint_dir in checkpoints_base.glob("*"):
        if checkpoint_dir.is_dir():
            checkpoint_files.extend(checkpoint_dir.glob("mario_net_*.chkpt"))
    
    if checkpoint_files:
        # 找到最新的模型
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"加载最新模型: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=mario.device)
            mario.net.load_state_dict(checkpoint['model'])
            mario.exploration_rate = checkpoint.get('exploration_rate', mario.exploration_rate)
            print(f"成功加载模型，探索率: {mario.exploration_rate:.6f}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将开始新的训练")
    else:
        print("未找到现有模型，将开始新的训练")

logger = MetricLogger(save_dir)

episodes = 40000
print(f"开始训练，最大回合数: {episodes}")

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    step_count = 0

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)
        
        total_reward += reward
        step_count += 1

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state        
        
        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
    

print(f"\n训练完成！共进行了 {e + 1} 个回合")
print("可以运行 demo.py 来查看训练效果")