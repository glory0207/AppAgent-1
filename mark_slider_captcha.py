#!/usr/bin/env python3
"""
滑块验证码标注工具
用于标注滑块和缺口位置并生成可视化图像
"""

import os
import sys
import cv2
import numpy as np
import time
import traceback
from typing import Dict, Tuple, Optional

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.captcha_agent import CaptchaAgent
from scripts.and_controller import AndroidController, list_all_devices
from scripts.utils import print_with_color
from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel


def main():
    """主函数"""
    print_with_color("🎯 滑块验证码标注工具", "cyan")
    print_with_color("="*50, "yellow")
    
    try:
        # 加载配置
        config = load_config()
        print_with_color(f"📝 加载配置文件，使用模型: {config['MODEL']}", "blue")
        
        # 检查API密钥
        if config["MODEL"] == "OpenAI" and (not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "sk-"):
            print_with_color("❌ OpenAI API密钥未设置，请在config.yaml中设置OPENAI_API_KEY", "red")
            return
        elif config["MODEL"] == "Qwen" and not config["DASHSCOPE_API_KEY"]:
            print_with_color("❌ Qwen API密钥未设置，请在config.yaml中设置DASHSCOPE_API_KEY", "red")
            return
        
        # 获取设备列表
        device_list = list_all_devices()
        if not device_list:
            print_with_color("❌ 未找到Android设备！", "red")
            return
        
        print_with_color(f"📱 发现设备: {device_list}", "green")
        
        # 过滤掉离线设备
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("❌ 没有在线设备可用！", "red")
            return
        
        # 自动选择第一个在线设备
        device = online_devices[0]
        print_with_color(f"✅ 自动选择设备: {device}", "green")
        
        # 创建控制器和Agent
        controller = AndroidController(device)
        
        # 显示设备屏幕尺寸
        width, height = controller.get_device_size()
        print_with_color(f"📱 设备屏幕尺寸: {width}x{height}", "green")
        
        # 初始化验证码Agent
        print_with_color("🤖 初始化验证码Agent...", "yellow")
        try:
            agent = CaptchaAgent(config_path="config.yaml")
            agent.controller = controller
            print_with_color("✅ 验证码Agent初始化成功", "green")
        except Exception as e:
            print_with_color(f"❌ 验证码Agent初始化失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 初始化模型
        print_with_color("🧠 初始化LLM模型...", "yellow")
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
            print_with_color("✅ 模型初始化完成", "green")
        except Exception as e:
            print_with_color(f"❌ 模型初始化失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 获取屏幕截图
        print_with_color("📸 正在截取当前屏幕...", "yellow")
        try:
            # 确保临时目录存在
            os.makedirs("./temp", exist_ok=True)
            
            screenshot_path = controller.get_screenshot("slider_captcha", "./temp")
            if screenshot_path == "ERROR":
                print_with_color("❌ 无法获取屏幕截图", "red")
                return
            print_with_color(f"✅ 屏幕截图已保存至: {screenshot_path}", "green")
        except Exception as e:
            print_with_color(f"❌ 截图失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 分析验证码类型
        print_with_color("🔍 正在分析验证码类型...", "yellow")
        try:
            captcha_type = agent._detect_captcha_type(screenshot_path)
            print_with_color(f"检测到验证码类型: {captcha_type}", "green")
            
            if captcha_type != "slider":
                print_with_color("⚠️ 当前页面不是滑块验证码，请导航到滑块验证码页面", "yellow")
                return
        except Exception as e:
            print_with_color(f"❌ 验证码类型检测失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 分析滑块和缺口位置
        print_with_color("🔍 分析滑块和缺口位置...", "yellow")
        slider_info = analyze_slider_positions(agent, screenshot_path)
        
        if not slider_info:
            print_with_color("❌ 无法分析滑块和缺口位置", "red")
            return
        
        # 标注图像
        print_with_color("🖌️ 正在标注图像...", "yellow")
        marked_image_path = mark_slider_image(screenshot_path, slider_info)
        
        if marked_image_path:
            print_with_color(f"✅ 标注图像已保存至: {marked_image_path}", "green")
        else:
            print_with_color("❌ 图像标注失败", "red")
    
    except Exception as e:
        print_with_color(f"❌ 程序运行出错: {str(e)}", "red")
        traceback.print_exc()


def analyze_slider_positions(agent: CaptchaAgent, screenshot_path: str) -> Optional[Dict]:
    """分析滑块和缺口位置"""
    try:
        # 使用优化后的滑块验证码提示词
        enhanced_prompt = """
请详细分析这个滑块验证码图像，识别以下元素：

1. 滑块的当前位置（精确坐标）
2. 目标缺口位置（精确坐标）
3. 滑块和缺口的尺寸（宽度和高度）

请提供尽可能精确的坐标值，格式如下：
滑块位置：(x1, y1)
滑块尺寸：宽度 x 高度
缺口位置：(x2, y2)
缺口尺寸：宽度 x 高度
滑动距离：xx像素
"""
        
        print_with_color("发送请求到LLM模型...", "blue")
        success, response = agent.model.get_model_response(enhanced_prompt, [screenshot_path])
        
        if not success:
            print_with_color(f"❌ LLM调用失败: {response}", "red")
            return None
        
        print_with_color("✅ 获取到LLM响应", "green")
        print_with_color("解析滑块和缺口位置...", "blue")
        
        # 解析LLM响应
        slider_info = parse_slider_response(response)
        
        if slider_info:
            print_with_color("滑块位置信息:", "cyan")
            print_with_color(f"滑块位置: ({slider_info['slider_x']}, {slider_info['slider_y']})", "cyan")
            print_with_color(f"滑块尺寸: {slider_info['slider_width']}x{slider_info['slider_height']}", "cyan")
            print_with_color(f"缺口位置: ({slider_info['target_x']}, {slider_info['target_y']})", "cyan")
            print_with_color(f"缺口尺寸: {slider_info['target_width']}x{slider_info['target_height']}", "cyan")
            print_with_color(f"滑动距离: {slider_info['distance']}像素", "cyan")
            
            return slider_info
        else:
            print_with_color("❌ 无法解析滑块和缺口位置", "red")
            return None
        
    except Exception as e:
        print_with_color(f"❌ 分析滑块位置失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def parse_slider_response(response: str) -> Optional[Dict]:
    """解析LLM响应中的滑块和缺口位置信息"""
    try:
        import re
        
        slider_info = {}
        
        # 提取滑块位置
        slider_pos_match = re.search(r"滑块位置[：:]\s*\((\d+),?\s*(\d+)\)", response)
        if slider_pos_match:
            slider_info['slider_x'] = int(slider_pos_match.group(1))
            slider_info['slider_y'] = int(slider_pos_match.group(2))
        else:
            # 尝试其他可能的格式
            slider_pos_match = re.search(r"滑块.*?坐标[：:]\s*\((\d+),?\s*(\d+)\)", response)
            if slider_pos_match:
                slider_info['slider_x'] = int(slider_pos_match.group(1))
                slider_info['slider_y'] = int(slider_pos_match.group(2))
            else:
                # 设置默认值
                slider_info['slider_x'] = 100
                slider_info['slider_y'] = 600
        
        # 提取滑块尺寸
        slider_size_match = re.search(r"滑块尺寸[：:]\s*(\d+)\s*[xX×]\s*(\d+)", response)
        if slider_size_match:
            slider_info['slider_width'] = int(slider_size_match.group(1))
            slider_info['slider_height'] = int(slider_size_match.group(2))
        else:
            # 设置默认值
            slider_info['slider_width'] = 50
            slider_info['slider_height'] = 50
        
        # 提取缺口位置
        target_pos_match = re.search(r"缺口位置[：:]\s*\((\d+),?\s*(\d+)\)", response)
        if target_pos_match:
            slider_info['target_x'] = int(target_pos_match.group(1))
            slider_info['target_y'] = int(target_pos_match.group(2))
        else:
            # 尝试其他可能的格式
            target_pos_match = re.search(r"目标位置[：:]\s*\((\d+),?\s*(\d+)\)", response)
            if target_pos_match:
                slider_info['target_x'] = int(target_pos_match.group(1))
                slider_info['target_y'] = int(target_pos_match.group(2))
            else:
                # 设置默认值
                slider_info['target_x'] = 500
                slider_info['target_y'] = 600
        
        # 提取缺口尺寸
        target_size_match = re.search(r"缺口尺寸[：:]\s*(\d+)\s*[xX×]\s*(\d+)", response)
        if target_size_match:
            slider_info['target_width'] = int(target_size_match.group(1))
            slider_info['target_height'] = int(target_size_match.group(2))
        else:
            # 设置默认值
            slider_info['target_width'] = 50
            slider_info['target_height'] = 50
        
        # 计算滑动距离
        distance_match = re.search(r"滑动距离[：:]\s*(\d+)", response)
        if distance_match:
            slider_info['distance'] = int(distance_match.group(1))
        else:
            # 计算距离
            slider_info['distance'] = slider_info['target_x'] - slider_info['slider_x']
        
        return slider_info
        
    except Exception as e:
        print_with_color(f"❌ 解析响应失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def mark_slider_image(image_path: str, slider_info: Dict) -> Optional[str]:
    """标注滑块和缺口位置"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print_with_color(f"❌ 无法读取图像: {image_path}", "red")
            return None
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 创建标注图像（复制原图）
        marked_image = image.copy()
        
        # 标注滑块位置（绿色矩形）
        slider_x = slider_info['slider_x']
        slider_y = slider_info['slider_y']
        slider_width = slider_info['slider_width']
        slider_height = slider_info['slider_height']
        
        cv2.rectangle(
            marked_image, 
            (slider_x, slider_y), 
            (slider_x + slider_width, slider_y + slider_height), 
            (0, 255, 0),  # 绿色
            2
        )
        
        # 添加滑块标签
        cv2.putText(
            marked_image, 
            "Slider", 
            (slider_x, slider_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0),  # 绿色
            2
        )
        
        # 标注缺口位置（红色矩形）
        target_x = slider_info['target_x']
        target_y = slider_info['target_y']
        target_width = slider_info['target_width']
        target_height = slider_info['target_height']
        
        cv2.rectangle(
            marked_image, 
            (target_x, target_y), 
            (target_x + target_width, target_y + target_height), 
            (0, 0, 255),  # 红色
            2
        )
        
        # 添加缺口标签
        cv2.putText(
            marked_image, 
            "Target", 
            (target_x, target_y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255),  # 红色
            2
        )
        
        # 绘制滑动路径（蓝色箭头）
        cv2.arrowedLine(
            marked_image,
            (slider_x + slider_width // 2, slider_y + slider_height // 2),
            (target_x + target_width // 2, target_y + target_height // 2),
            (255, 0, 0),  # 蓝色
            2
        )
        
        # 添加距离标签
        distance = slider_info['distance']
        mid_x = (slider_x + target_x) // 2
        mid_y = slider_y - 30
        
        cv2.putText(
            marked_image, 
            f"Distance: {distance}px", 
            (mid_x, mid_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0),  # 蓝色
            2
        )
        
        # 保存标注图像
        output_path = os.path.join("./temp", "marked_slider_captcha.png")
        cv2.imwrite(output_path, marked_image)
        
        return output_path
        
    except Exception as e:
        print_with_color(f"❌ 标注图像失败: {str(e)}", "red")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 