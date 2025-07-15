#!/usr/bin/env python3
"""
多功能Agent系统入口
支持多种专业化Agent，如验证码处理、购物助手、社交媒体等
"""

import argparse
import os
import time
import uuid
from typing import Optional

from scripts.agent_manager import AgentManager, TaskQueue
from scripts.and_controller import list_all_devices, AndroidController
from scripts.utils import print_with_color
from scripts.base_agent import AgentTask


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多功能AppAgent系统")
    parser.add_argument("--app", help="目标应用名称")
    parser.add_argument("--task", help="任务描述")
    parser.add_argument("--agent", help="指定使用的Agent类型")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    print_with_color("欢迎使用多功能AppAgent系统！", "cyan")
    print_with_color("="*50, "yellow")
    
    # 初始化Agent管理器
    print_with_color("初始化Agent管理器...", "yellow")
    agent_manager = AgentManager(args.config)
    
    # 设置Android设备
    controller = setup_android_device()
    if not controller:
        print_with_color("无法连接Android设备，程序退出", "red")
        return
    
    agent_manager.set_controller(controller)
    
    # 显示可用的Agent
    show_available_agents(agent_manager)
    
    # 根据模式执行
    if args.mode == "single":
        run_single_task(agent_manager, args)
    elif args.mode == "batch":
        run_batch_tasks(agent_manager, args)
    else:
        run_interactive_mode(agent_manager)


def setup_android_device() -> Optional[AndroidController]:
    """设置Android设备"""
    print_with_color("查找Android设备...", "yellow")
    
    device_list = list_all_devices()
    if not device_list:
        print_with_color("未找到Android设备！", "red")
        print_with_color("请确保：", "yellow")
        print_with_color("1. 已安装ADB", "white")
        print_with_color("2. 已启用USB调试", "white")
        print_with_color("3. 设备已正确连接", "white")
        return None
    
    print_with_color(f"发现设备: {device_list}", "green")
    
    # 选择设备
    if len(device_list) == 1:
        device = device_list[0]
        print_with_color(f"自动选择设备: {device}", "green")
    else:
        print_with_color("请选择设备:", "blue")
        for i, dev in enumerate(device_list):
            print_with_color(f"{i+1}. {dev}", "white")
        
        while True:
            try:
                choice = int(input("输入设备编号: ")) - 1
                if 0 <= choice < len(device_list):
                    device = device_list[choice]
                    break
                else:
                    print_with_color("无效选择，请重新输入", "red")
            except ValueError:
                print_with_color("请输入有效数字", "red")
    
    # 创建控制器
    controller = AndroidController(device)
    width, height = controller.get_device_size()
    
    if not width or not height:
        print_with_color("无法获取设备屏幕尺寸", "red")
        return None
    
    print_with_color(f"设备屏幕尺寸: {width}x{height}", "green")
    return controller


def show_available_agents(agent_manager: AgentManager):
    """显示可用的Agent"""
    print_with_color("\n可用的专业化Agent:", "cyan")
    print_with_color("-" * 30, "yellow")
    
    agents_info = agent_manager.get_available_agents()
    if not agents_info:
        print_with_color("没有可用的Agent", "red")
        return
    
    for agent_name, capabilities in agents_info.items():
        print_with_color(f"• {agent_name}", "green")
        for capability in capabilities:
            print_with_color(f"  - {capability}", "white")
    
    print_with_color("-" * 30, "yellow")


def run_single_task(agent_manager: AgentManager, args):
    """运行单个任务"""
    app_name = args.app or input("请输入应用名称: ")
    task_desc = args.task or input("请输入任务描述: ")
    
    # 创建任务
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        description=task_desc,
        app_name=app_name
    )
    
    print_with_color(f"\n执行任务: {task_desc}", "cyan")
    print_with_color(f"目标应用: {app_name}", "cyan")
    
    # 执行任务
    result = agent_manager.execute_task(task)
    
    # 显示结果
    if result.success:
        print_with_color(f"任务完成: {result.message}", "green")
        if result.data:
            print_with_color(f"结果数据: {result.data}", "cyan")
        if result.actions_taken:
            print_with_color(f"执行的操作: {result.actions_taken}", "yellow")
    else:
        print_with_color(f"任务失败: {result.message}", "red")


def run_batch_tasks(agent_manager: AgentManager, args):
    """批量运行任务"""
    print_with_color("批量任务模式", "cyan")
    
    task_queue = TaskQueue()
    
    print_with_color("请输入任务列表（输入'done'结束）:", "yellow")
    while True:
        task_desc = input("任务描述: ").strip()
        if task_desc.lower() == 'done':
            break
        
        if not task_desc:
            continue
        
        app_name = input("应用名称: ").strip()
        priority = input("优先级(1-10, 默认5): ").strip()
        priority = int(priority) if priority.isdigit() else 5
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            description=task_desc,
            app_name=app_name,
            priority=priority
        )
        
        task_queue.add_task(task)
        print_with_color(f"已添加任务: {task_desc}", "green")
    
    # 执行所有任务
    print_with_color(f"\n开始执行 {len(task_queue.tasks)} 个任务...", "cyan")
    
    while True:
        task = task_queue.get_next_task()
        if not task:
            break
        
        print_with_color(f"\n执行任务: {task.description}", "yellow")
        result = agent_manager.execute_task(task)
        
        if result.success:
            task_queue.mark_completed(task)
            print_with_color("任务完成", "green")
        else:
            task_queue.mark_failed(task)
            print_with_color(f"任务失败: {result.message}", "red")
        
        time.sleep(2)  # 任务间隔
    
    # 显示总结
    status = task_queue.get_status()
    print_with_color(f"\n任务执行总结:", "cyan")
    print_with_color(f"完成: {status['completed']}", "green")
    print_with_color(f"失败: {status['failed']}", "red")


def run_interactive_mode(agent_manager: AgentManager):
    """交互式模式"""
    print_with_color("\n进入交互式模式", "cyan")
    print_with_color("输入 'help' 查看可用命令", "yellow")
    
    while True:
        try:
            command = input("\nAppAgent> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print_with_color("再见！", "cyan")
                break
            elif command == 'help':
                show_help()
            elif command == 'agents':
                show_available_agents(agent_manager)
            elif command.startswith('task'):
                handle_task_command(agent_manager, command)
            elif command.startswith('train'):
                handle_train_command(agent_manager, command)
            elif command == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
            else:
                print_with_color("未知命令，输入 'help' 查看帮助", "yellow")
        
        except KeyboardInterrupt:
            print_with_color("\n用户中断，退出程序", "yellow")
            break
        except Exception as e:
            print_with_color(f"错误: {e}", "red")


def show_help():
    """显示帮助信息"""
    help_text = """
可用命令:
• help          - 显示此帮助信息
• agents        - 显示所有可用的Agent
• task          - 执行单个任务
• train <agent> - 训练指定的Agent
• clear         - 清屏
• quit/exit     - 退出程序

使用示例:
• task          - 交互式创建并执行任务
• train captcha_agent - 训练验证码Agent
"""
    print_with_color(help_text, "white")


def handle_task_command(agent_manager: AgentManager, command: str):
    """处理任务命令"""
    app_name = input("应用名称: ").strip()
    task_desc = input("任务描述: ").strip()
    
    if not app_name or not task_desc:
        print_with_color("应用名称和任务描述不能为空", "red")
        return
    
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        description=task_desc,
        app_name=app_name
    )
    
    result = agent_manager.execute_task(task)
    
    if result.success:
        print_with_color(f"任务完成: {result.message}", "green")
    else:
        print_with_color(f"任务失败: {result.message}", "red")


def handle_train_command(agent_manager: AgentManager, command: str):
    """处理训练命令"""
    parts = command.split()
    if len(parts) < 2:
        print_with_color("请指定要训练的Agent名称", "red")
        return
    
    agent_name = parts[1]
    app_name = input("训练应用名称: ").strip()
    task_desc = input("训练任务描述: ").strip()
    
    if not app_name or not task_desc:
        print_with_color("应用名称和任务描述不能为空", "red")
        return
    
    success = agent_manager.train_agent(agent_name, app_name, task_desc)
    if success:
        print_with_color(f"Agent {agent_name} 训练完成", "green")
    else:
        print_with_color(f"Agent {agent_name} 训练失败", "red")


if __name__ == "__main__":
    main() 