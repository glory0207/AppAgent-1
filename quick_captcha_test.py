#!/usr/bin/env python3
"""
快速验证码测试脚本
直接处理当前页面的验证码
"""

import os
import sys
import time
import traceback
from typing import Optional

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.captcha_agent import CaptchaAgent
from scripts.and_controller import AndroidController, list_all_devices
from scripts.base_agent import AgentTask
from scripts.utils import print_with_color
from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel


def main():
    """主函数"""
    print_with_color("验证码快速测试工具", "cyan")
    print_with_color("="*50, "yellow")
    
    try:
        # 加载配置
        config = load_config()
        print_with_color(f"加载配置文件，使用模型: {config['MODEL']}", "blue")
        
        # 检查API密钥
        if config["MODEL"] == "OpenAI" and (not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "sk-"):
            print_with_color("OpenAI API密钥未设置，请在config.yaml中设置OPENAI_API_KEY", "red")
            return
        elif config["MODEL"] == "Qwen" and not config["DASHSCOPE_API_KEY"]:
            print_with_color("Qwen API密钥未设置，请在config.yaml中设置DASHSCOPE_API_KEY", "red")
            return
        
        # 获取设备列表
        device_list = list_all_devices()
        if not device_list:
            print_with_color("未找到Android设备！", "red")
            return
        
        print_with_color(f"发现设备: {device_list}", "green")
        
        # 过滤掉离线设备
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("没有在线设备可用！", "red")
            return
        
        # 自动选择第一个在线设备
        device = online_devices[0]
        print_with_color(f"自动选择设备: {device}", "green")
        
        # 创建控制器和Agent
        controller = AndroidController(device)
        
        # 显示设备屏幕尺寸
        width, height = controller.get_device_size()
        print_with_color(f"设备屏幕尺寸: {width}x{height}", "green")
        
        # 初始化验证码Agent
        print_with_color("初始化验证码Agent...", "yellow")
        try:
            agent = CaptchaAgent(config_path="config.yaml")
            agent.controller = controller
            print_with_color("验证码Agent初始化成功", "green")
            print_with_color(f"支持的能力: {', '.join(agent.capabilities)}", "cyan")
        except Exception as e:
            print_with_color(f"验证码Agent初始化失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 初始化模型
        print_with_color("初始化LLM模型...", "yellow")
        try:
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
            
            # 设置模型
            agent.model = model
            print_with_color("模型初始化完成", "green")
        except Exception as e:
            print_with_color(f"模型初始化失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 获取屏幕截图
        print_with_color("正在截取当前屏幕...", "yellow")
        try:
            # 确保临时目录存在
            os.makedirs("./temp", exist_ok=True)
            
            screenshot_path = controller.get_screenshot("current_screen", "./temp")
            if screenshot_path == "ERROR":
                print_with_color("无法获取屏幕截图", "red")
                return
            print_with_color(f"屏幕截图已保存至: {screenshot_path}", "green")
        except Exception as e:
            print_with_color(f"截图失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 分析验证码类型
        print_with_color("正在分析验证码类型...", "yellow")
        try:
            captcha_type = agent._detect_captcha_type(screenshot_path)
            print_with_color(f"检测到验证码类型: {captcha_type}", "green")
        except Exception as e:
            print_with_color(f"验证码类型检测失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 处理验证码
        print_with_color("开始处理验证码...", "cyan")
        try:
            # 根据类型处理验证码
            if captcha_type == "text":
                result = agent._solve_text_captcha(screenshot_path)
            elif captcha_type == "slider":
                result = agent._solve_slider_captcha(screenshot_path)
            elif captcha_type == "click":
                result = agent._solve_click_captcha(screenshot_path)
            elif captcha_type == "math":
                result = agent._solve_math_captcha(screenshot_path)
            else:
                result = agent._solve_general_captcha(screenshot_path, "处理当前页面验证码")
            
            # 显示结果
            if result.success:
                print_with_color(f"成功: {result.message}", "green")
                if result.data:
                    print_with_color(f"数据: {result.data}", "cyan")
                if result.actions_taken:
                    print_with_color(f"执行操作: {result.actions_taken}", "cyan")
            else:
                print_with_color(f"失败: {result.message}", "red")
        except Exception as e:
            print_with_color(f"验证码处理失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 等待验证结果
        print_with_color("等待验证结果...", "yellow")
        time.sleep(3)
        
        # 确认验证结果
        try:
            confirmation_result = agent._confirm_captcha_result(screenshot_path)
            if confirmation_result:
                print_with_color("验证通过！", "green")
            else:
                print_with_color("验证未通过或无法确认", "yellow")
        except Exception as e:
            print_with_color(f"验证结果确认失败: {str(e)}", "red")
            traceback.print_exc()
    
    except Exception as e:
        print_with_color(f"程序运行出错: {str(e)}", "red")
        traceback.print_exc()


if __name__ == "__main__":
    main() 