#!/usr/bin/env python3
"""
验证码Agent测试脚本
用于测试验证码Agent的各种功能
"""

import os
import sys
import argparse
import time
import uuid
from typing import Optional

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.agent_manager import AgentManager
from scripts.and_controller import list_all_devices, AndroidController
from scripts.utils import print_with_color
from scripts.base_agent import AgentTask


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证码Agent测试工具")
    parser.add_argument("--app", help="目标应用名称，例如：支付宝、微信等")
    parser.add_argument("--type", choices=["text", "slider", "click", "math", "general"], 
                       default="general", help="验证码类型")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--test_all", action="store_true", help="测试所有验证码类型")
    
    args = parser.parse_args()
    
    print_with_color("验证码Agent测试工具", "cyan")
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
    
    # 获取应用名称
    app_name = args.app or input("请输入应用名称: ")
    
    if args.test_all:
        test_all_captcha_types(agent_manager, app_name)
    else:
        captcha_type = args.type
        test_captcha_type(agent_manager, app_name, captcha_type)


def setup_android_device() -> Optional[AndroidController]:
    """设置Android设备"""
    print_with_color("查找Android设备...", "yellow")
    
    device_list = list_all_devices()
    if not device_list:
        print_with_color("未找到Android设备！", "red")
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


def test_captcha_type(agent_manager, app_name, captcha_type):
    """测试特定类型的验证码"""
    print_with_color(f"\n测试 {captcha_type} 类型验证码", "cyan")
    
    # 构建任务描述
    task_desc = f"识别并处理{app_name}中的"
    if captcha_type == "text":
        task_desc += "文本验证码"
    elif captcha_type == "slider":
        task_desc += "滑块验证码"
    elif captcha_type == "click":
        task_desc += "点击验证码"
    elif captcha_type == "math":
        task_desc += "数学验证码"
    else:
        task_desc += "验证码"
    
    # 创建任务
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        description=task_desc,
        app_name=app_name
    )
    
    print_with_color(f"任务: {task_desc}", "yellow")
    input("请将手机屏幕导航到验证码界面，然后按Enter继续...")
    
    # 执行任务
    start_time = time.time()
    result = agent_manager.execute_task(task)
    end_time = time.time()
    
    # 显示结果
    print_with_color("\n测试结果:", "cyan")
    print_with_color(f"耗时: {end_time - start_time:.2f}秒", "yellow")
    
    if result.success:
        print_with_color(f"成功: {result.message}", "green")
        if result.data:
            print_with_color(f"数据: {result.data}", "cyan")
        if result.actions_taken:
            print_with_color(f"执行操作: {result.actions_taken}", "cyan")
    else:
        print_with_color(f"失败: {result.message}", "red")
    
    # 请求用户评估
    print_with_color("\n请评估验证码处理结果:", "yellow")
    print_with_color("1. 完全正确", "green")
    print_with_color("2. 部分正确", "yellow")
    print_with_color("3. 完全错误", "red")
    
    while True:
        try:
            rating = int(input("请输入评分 (1-3): "))
            if 1 <= rating <= 3:
                break
            else:
                print_with_color("请输入1-3之间的数字", "red")
        except ValueError:
            print_with_color("请输入有效数字", "red")
    
    # 获取用户反馈
    feedback = input("请输入详细反馈 (可选): ").strip()
    
    # 保存测试结果
    save_test_result(app_name, captcha_type, result, rating, feedback)
    
    return rating


def test_all_captcha_types(agent_manager, app_name):
    """测试所有类型的验证码"""
    captcha_types = ["text", "slider", "click", "math", "general"]
    results = {}
    
    for captcha_type in captcha_types:
        print_with_color(f"\n测试 {captcha_type} 类型验证码", "cyan")
        input(f"请将手机屏幕导航到 {captcha_type} 验证码界面，然后按Enter继续...")
        
        rating = test_captcha_type(agent_manager, app_name, captcha_type)
        results[captcha_type] = rating
        
        if captcha_type != captcha_types[-1]:
            input("按Enter继续下一个测试...")
    
    # 显示总结
    print_with_color("\n测试总结:", "cyan")
    for captcha_type, rating in results.items():
        status = "成功" if rating == 1 else "部分成功" if rating == 2 else "失败"
        print_with_color(f"{captcha_type}: {status}", "yellow")


def save_test_result(app_name, captcha_type, result, rating, feedback):
    """保存测试结果"""
    test_dir = "./test_results"
    os.makedirs(test_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{test_dir}/captcha_test_{app_name}_{captcha_type}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"应用: {app_name}\n")
        f.write(f"验证码类型: {captcha_type}\n")
        f.write(f"成功: {result.success}\n")
        f.write(f"消息: {result.message}\n")
        if result.data:
            f.write(f"数据: {result.data}\n")
        if result.actions_taken:
            f.write(f"执行操作: {', '.join(result.actions_taken)}\n")
        f.write(f"评分: {rating}\n")
        if feedback:
            f.write(f"反馈: {feedback}\n")
    
    print_with_color(f"测试结果已保存到: {filename}", "green")


def optimize_prompts(test_results_dir="./test_results"):
    """根据测试结果优化提示词"""
    print_with_color("\n开始优化提示词...", "cyan")
    
    # 这里可以实现基于测试结果的提示词优化逻辑
    # 例如：分析成功率低的验证码类型，调整相应的提示词
    
    # 示例实现
    print_with_color("基于测试结果的提示词优化建议:", "yellow")
    print_with_color("1. 文本验证码: 增加对干扰线和扭曲文字的识别能力", "white")
    print_with_color("2. 滑块验证码: 改进目标位置的精确定位", "white")
    print_with_color("3. 点击验证码: 增强对小目标的识别能力", "white")
    print_with_color("4. 数学验证码: 优化表达式解析逻辑", "white")


if __name__ == "__main__":
    main() 