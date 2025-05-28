"""
Super Mario Bros DQN 演示模式
使用已训练的模型进行游戏演示，不进行学习
"""
import torch 
import warnings
import glob
import time
from pathlib import Path

# 抑制gym_super_mario_bros的overflow警告
warnings.filterwarnings("ignore", message="overflow encountered in scalar subtract")

import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from utils.agent import Mario

class DemoMario:
    """演示模式的Mario智能体，不进行学习"""
    
    def __init__(self, mario_agent):
        self.mario = mario_agent
        # 设置为演示模式：禁用探索，始终选择最优动作
        self.mario.exploration_rate = 0.0
        self.mario.net.eval()  # 设置为评估模式
        
    def act(self, state):
        """只进行利用(exploit)，不进行探索"""
        return self.mario.act(state)

def setup_environment(render_mode='human'):
    """设置游戏环境"""
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", 
                                       render_mode=render_mode,
                                       apply_api_compatibility=True)
    
    # 限制动作空间
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    
    # 应用环境包装器
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    
    return env

def find_best_model(checkpoints_dir):
    """查找最新/最好的模型文件"""
    checkpoint_files = []
    
    # 搜索所有子目录中的检查点文件
    for checkpoint_dir in Path(checkpoints_dir).glob("*"):
        if checkpoint_dir.is_dir():
            checkpoint_files.extend(checkpoint_dir.glob("mario_net_*.chkpt"))
    
    if not checkpoint_files:
        print("未找到任何模型文件！")
        return None
    
    # 按文件修改时间排序，返回最新的
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    print(f"找到模型文件: {latest_checkpoint}")
    return latest_checkpoint

def run_demo(model_path, num_episodes=5, delay=0.05):
    """运行演示"""
    print("=" * 50)
    print("Super Mario Bros DQN 演示模式")
    print("=" * 50)
    
    # 设置环境（显示游戏画面）
    env = setup_environment(render_mode='human')
    
    # 创建Mario智能体
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=Path("demo"))
    
    # 加载模型
    if model_path:
        mario.load(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print("警告: 未加载模型，使用随机策略！")
    
    # 创建演示智能体
    demo_mario = DemoMario(mario)
    
    # 统计信息
    total_rewards = []
    completion_count = 0
    
    print(f"\n开始演示 {num_episodes} 个回合...")
    print("按 Ctrl+C 可以提前退出")
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- 回合 {episode + 1}/{num_episodes} ---")
            
            state = env.reset()
            total_reward = 0
            step_count = 0
            
            while True:
                # 智能体选择动作
                action = demo_mario.act(state)
                
                # 执行动作
                next_state, reward, done, trunc, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # 更新状态
                state = next_state
                
                # 添加延迟以便观看
                if delay > 0:
                    time.sleep(delay)
                
                # 检查游戏结束条件
                if done or info.get("flag_get", False):
                    if info.get("flag_get", False):
                        print(f"🎉 成功完成关卡！奖励: {total_reward:.0f}, 步数: {step_count}")
                        completion_count += 1
                    else:
                        print(f"💀 游戏结束。奖励: {total_reward:.0f}, 步数: {step_count}")
                    break
                
                # 防止无限循环
                if step_count > 10000:
                    print(f"⏰ 达到最大步数限制。奖励: {total_reward:.0f}")
                    break
            
            total_rewards.append(total_reward)
            
            # 显示实时统计
            avg_reward = sum(total_rewards) / len(total_rewards)
            completion_rate = completion_count / (episode + 1) * 100
            print(f"完成率: {completion_rate:.1f}% ({completion_count}/{episode + 1})")
            print(f"平均奖励: {avg_reward:.0f}")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    
    finally:
        env.close()
        
        # 最终统计
        print("\n" + "=" * 50)
        print("演示完成！")
        print("=" * 50)
        if total_rewards:
            print(f"总回合数: {len(total_rewards)}")
            print(f"完成关卡次数: {completion_count}")
            print(f"完成率: {completion_count/len(total_rewards)*100:.1f}%")
            print(f"平均奖励: {sum(total_rewards)/len(total_rewards):.0f}")
            print(f"最高奖励: {max(total_rewards):.0f}")
            print(f"最低奖励: {min(total_rewards):.0f}")

def main():
    """主函数"""
    print("欢迎使用 Super Mario Bros DQN 演示程序！")
    
    # 查找模型文件
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print("未找到checkpoints目录！请先训练模型。")
        return
    
    model_path = find_best_model(checkpoints_dir)
    if not model_path:
        print("未找到模型文件！请先训练模型。")
        return
    
    # 获取用户输入
    try:
        num_episodes = int(input(f"请输入演示回合数 (默认: 5): ") or "5")
        delay_input = input("请输入动作延迟时间(秒) (默认: 0.05): ") or "0.05"
        delay = float(delay_input)
    except ValueError:
        print("输入无效，使用默认值")
        num_episodes = 5
        delay = 0.05
    
    # 运行演示
    run_demo(model_path, num_episodes, delay)

if __name__ == "__main__":
    main()
