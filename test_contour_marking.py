#!/usr/bin/env python3
"""
测试滑块验证码轮廓标记功能
使用大模型识别并标记滑块和缺口的轮廓边界
"""

import os
import sys
import traceback

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.captcha_agent import CaptchaAgent
from scripts.and_controller import AndroidController, list_all_devices
from scripts.utils import print_with_color
from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel


def main():
    """主函数"""
    print_with_color("滑块验证码轮廓标记测试", "cyan")
    print_with_color("="*50, "yellow")
    
    try:
        # 加载配置
        config = load_config()
        print_with_color(f"使用模型: {config['MODEL']}", "blue")
        
        # 检查API密钥
        if config["MODEL"] == "OpenAI" and (not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "sk-"):
            print_with_color("请在config.yaml中设置OPENAI_API_KEY", "red")
            return
        elif config["MODEL"] == "Qwen" and not config["DASHSCOPE_API_KEY"]:
            print_with_color("请在config.yaml中设置DASHSCOPE_API_KEY", "red")
            return
        
        # 获取设备列表
        device_list = list_all_devices()
        if not device_list:
            print_with_color("未找到Android设备！请连接设备后重试", "red")
            return
        
        print_with_color(f"发现设备: {device_list}", "green")
        
        # 过滤掉离线设备
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("没有在线设备可用！", "red")
            return
        
        # 选择设备
        device = online_devices[0]
        print_with_color(f"使用设备: {device}", "green")
        
        # 创建控制器
        controller = AndroidController(device)
        width, height = controller.get_device_size()
        print_with_color(f"设备屏幕尺寸: {width}x{height}", "green")
        
        # 初始化验证码Agent
        print_with_color("初始化验证码Agent...", "yellow")
        agent = CaptchaAgent(config_path="config.yaml")
        agent.controller = controller
        
        # 初始化模型
        print_with_color("初始化LLM模型...", "yellow")
        if config["MODEL"] == "OpenAI":
            model = OpenAIModel(
                base_url=config["OPENAI_API_BASE"],
                api_key=config["OPENAI_API_KEY"],
                model=config["OPENAI_API_MODEL"],
                temperature=float(config["TEMPERATURE"]),
                max_tokens=int(config["MAX_TOKENS"])
            )
        else:  # Qwen
            model = QwenModel(
                api_key=config["DASHSCOPE_API_KEY"],
                model=config["QWEN_MODEL"]
            )
        
        agent.model = model
        print_with_color("模型初始化完成", "green")
        
        # 确保临时目录存在
        os.makedirs("./temp", exist_ok=True)
        
        # 获取当前屏幕截图
        print_with_color("正在截取当前屏幕...", "yellow")
        screenshot_path = controller.get_screenshot("captcha_contour_test", "./temp")
        if screenshot_path == "ERROR":
            print_with_color("无法获取屏幕截图", "red")
            return
        
        print_with_color(f"屏幕截图已保存: {screenshot_path}", "green")
        
        # 使用新的轮廓标记功能
        print_with_color("开始轮廓标记分析...", "yellow")
        marked_image_path = agent.mark_slider_contours(screenshot_path)
        
        if marked_image_path:
            print_with_color("="*60, "green")
            print_with_color("轮廓标记成功完成！", "green")
            print_with_color(f"标记后的图像保存在: {marked_image_path}", "green")
            print_with_color("="*60, "green")
            print_with_color("标记说明:", "cyan")
            print_with_color("  🟢 绿色双边框: 滑块轮廓边界", "green")
            print_with_color("  🔴 红色双边框: 缺口轮廓边界", "red")
            print_with_color("  🟢 绿色圆点: 滑块中心位置", "green")
            print_with_color("  🔴 红色圆点: 缺口中心位置", "red")
            print_with_color("  🔵 蓝色箭头: 滑动路径和距离", "blue")
            print_with_color("  📝 底部信息: 详细坐标数据", "white")
        else:
            print_with_color("轮廓标记失败", "red")
            print_with_color("请检查:", "yellow")
            print_with_color("  1. 当前页面是否包含滑块验证码", "yellow")
            print_with_color("  2. LLM模型是否正常工作", "yellow")
            print_with_color("  3. 网络连接是否正常", "yellow")
        
        # 询问是否要继续操作
        print_with_color("\n是否要基于标记结果执行滑块操作？(y/n): ", "cyan")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            print_with_color("开始执行滑块操作...", "yellow")
            # 这里可以调用原有的滑块操作功能
            result = agent._solve_slider_captcha(screenshot_path)
            if result.success:
                print_with_color("滑块操作执行成功！", "green")
            else:
                print_with_color(f"滑块操作失败: {result.message}", "red")
        else:
            print_with_color("测试完成，未执行滑块操作", "blue")
    
    except Exception as e:
        print_with_color(f"程序运行出错: {str(e)}", "red")
        traceback.print_exc()


if __name__ == "__main__":
    main() 