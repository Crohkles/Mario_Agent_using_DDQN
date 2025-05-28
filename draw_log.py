"""
重新绘制训练日志图表
使用 MetricLogger 类读取已有的日志文件并重新生成所有训练图表
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils.logger import MetricLogger

def redraw_plots(checkpoint_dir):
    """
    重新绘制指定检查点目录中的所有训练图表
    
    Args:
        checkpoint_dir (str or Path): 检查点目录路径
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"错误: 检查点目录不存在: {checkpoint_path}")
        return
    
    log_file = checkpoint_path / "log"
    if not log_file.exists():
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    print(f"正在读取日志文件: {log_file}")
    
    # 创建 MetricLogger 实例来加载数据
    logger = MetricLogger(checkpoint_path)
    
    # 检查是否有数据
    if not logger.moving_avg_ep_rewards:
        print("警告: 没有找到移动平均数据")
        return
    
    print(f"找到 {len(logger.moving_avg_ep_rewards)} 条记录")
    
    # 创建 x 轴数据 (记录点索引)
    x_data = range(len(logger.moving_avg_ep_rewards))
    
    # 设置matplotlib中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制奖励图表
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x_data, logger.moving_avg_ep_rewards, 'b-', linewidth=1.5, label='平均奖励')
    plt.title('训练进度: 平均奖励', fontsize=14, fontweight='bold')
    plt.xlabel('记录点', fontsize=12)
    plt.ylabel('平均奖励', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制长度图表
    plt.subplot(2, 2, 2)
    plt.plot(x_data, logger.moving_avg_ep_lengths, 'g-', linewidth=1.5, label='平均长度')
    plt.title('训练进度: 平均回合长度', fontsize=14, fontweight='bold')
    plt.xlabel('记录点', fontsize=12)
    plt.ylabel('平均长度', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制损失图表
    plt.subplot(2, 2, 3)
    plt.plot(x_data, logger.moving_avg_ep_avg_losses, 'r-', linewidth=1.5, label='平均损失')
    plt.title('训练进度: 平均损失', fontsize=14, fontweight='bold')
    plt.xlabel('记录点', fontsize=12)
    plt.ylabel('平均损失', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制Q值图表
    plt.subplot(2, 2, 4)
    plt.plot(x_data, logger.moving_avg_ep_avg_qs, 'm-', linewidth=1.5, label='平均Q值')
    plt.title('训练进度: 平均Q值', fontsize=14, fontweight='bold')
    plt.xlabel('记录点', fontsize=12)
    plt.ylabel('平均Q值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    combined_plot_path = checkpoint_path / "combined_training_plots.jpg"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"已保存综合图表: {combined_plot_path}")
    plt.close()
    
    # 分别绘制单独的图表
    metrics = [
        ("ep_rewards", logger.moving_avg_ep_rewards, "平均奖励", "blue"),
        ("ep_lengths", logger.moving_avg_ep_lengths, "平均回合长度", "green"), 
        ("ep_avg_losses", logger.moving_avg_ep_avg_losses, "平均损失", "red"),
        ("ep_avg_qs", logger.moving_avg_ep_avg_qs, "平均Q值", "magenta")
    ]
    
    for metric_name, data, title, color in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, data, color=color, linewidth=2, alpha=0.8)
        plt.title(f'训练进度: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('记录点', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        if data:
            plt.axhline(y=np.mean(data), color='orange', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(data):.2f}')
            plt.axhline(y=np.max(data), color='red', linestyle=':', alpha=0.7, label=f'最大值: {np.max(data):.2f}')
            plt.legend()
        
        # 保存图表
        plot_path = checkpoint_path / f"{metric_name}_plot.jpg"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"已保存 {title} 图表: {plot_path}")
        plt.close()

def main():
    """主函数"""
    print("=" * 60)
    print("训练日志图表重绘工具")
    print("=" * 60)
    
    # 获取检查点目录
    checkpoints_base = Path("checkpoints")
    
    if not checkpoints_base.exists():
        print(f"错误: 检查点基础目录不存在: {checkpoints_base}")
        return
    
    # 查找所有检查点目录
    checkpoint_dirs = [d for d in checkpoints_base.iterdir() if d.is_dir()]
    
    if not checkpoint_dirs:
        print("错误: 没有找到任何检查点目录")
        return
    
    print("找到以下检查点目录:")
    for i, dir_path in enumerate(checkpoint_dirs, 1):
        log_file = dir_path / "log"
        status = "✓" if log_file.exists() else "✗"
        print(f"{i}. {dir_path.name} {status}")
    
    # 让用户选择目录
    while True:
        try:
            choice = input(f"\n请选择要重绘的目录 (1-{len(checkpoint_dirs)}) 或输入 'all' 处理所有目录: ").strip()
            
            if choice.lower() == 'all':
                print("\n开始处理所有目录...")
                for dir_path in checkpoint_dirs:
                    print(f"\n正在处理: {dir_path.name}")
                    redraw_plots(dir_path)
                break
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(checkpoint_dirs):
                    selected_dir = checkpoint_dirs[choice_idx]
                    print(f"\n正在处理: {selected_dir.name}")
                    redraw_plots(selected_dir)
                    break
                else:
                    print("无效选择，请重新输入")
        except ValueError:
            print("无效输入，请输入数字或 'all'")
        except KeyboardInterrupt:
            print("\n操作已取消")
            return
    
    print("\n图表重绘完成！")

if __name__ == "__main__":
    main()

