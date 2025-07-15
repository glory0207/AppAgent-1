#!/usr/bin/env python3
"""
滑块验证码轮廓标注工具
用于标注滑块和缺口位置并生成可视化图像
利用大模型精确识别轮廓并标记
"""

import os
import sys
import cv2
import numpy as np
import time
import traceback
from typing import Dict, Tuple, Optional, List

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.captcha_agent import CaptchaAgent
from scripts.and_controller import AndroidController, list_all_devices
from scripts.utils import print_with_color
from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel


def main():
    """主函数"""
    print_with_color("滑块验证码轮廓标注工具", "cyan")
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
            
            screenshot_path = controller.get_screenshot("slider_captcha", "./temp")
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
            
            if captcha_type != "slider":
                print_with_color("当前页面不是滑块验证码，请导航到滑块验证码页面", "yellow")
                # 仍然继续执行，可能大模型能识别到滑块元素
        except Exception as e:
            print_with_color(f"验证码类型检测失败: {str(e)}", "red")
            traceback.print_exc()
            return
        
        # 分析滑块和缺口的详细轮廓信息
        print_with_color("使用大模型分析滑块和缺口的详细轮廓信息...", "yellow")
        contour_info = analyze_slider_contours(agent, screenshot_path)
        
        if not contour_info:
            print_with_color("无法分析滑块和缺口轮廓信息", "red")
            return
        
        # 标注图像轮廓
        print_with_color("正在标注图像轮廓...", "yellow")
        marked_image_path = mark_slider_contours(screenshot_path, contour_info)
        
        if marked_image_path:
            print_with_color(f"轮廓标注图像已保存至: {marked_image_path}", "green")
            print_with_color("标注说明:", "cyan")
            print_with_color("  - 绿色框: 滑块轮廓", "green")
            print_with_color("  - 红色框: 缺口轮廓", "red")
            print_with_color("  - 蓝色箭头: 滑动路径", "blue")
        else:
            print_with_color("图像轮廓标注失败", "red")
    
    except Exception as e:
        print_with_color(f"程序运行出错: {str(e)}", "red")
        traceback.print_exc()


def analyze_slider_contours(agent: CaptchaAgent, screenshot_path: str) -> Optional[Dict]:
    """使用大模型分析滑块和缺口的详细轮廓信息"""
    try:
        # 使用增强的提示词，让大模型提供更精确的轮廓信息
        enhanced_contour_prompt = """
请非常仔细地分析这个滑块验证码图像，我需要你精确识别滑块和缺口的轮廓边界信息。

请按照以下格式详细描述：

**滑块信息：**
- 滑块类型：[圆形/方形/不规则形状/拖动按钮等]
- 滑块边界坐标：左上角(x1,y1), 右下角(x2,y2)
- 滑块中心坐标：(cx,cy)
- 滑块宽度：xx像素
- 滑块高度：xx像素
- 滑块位置描述：[详细描述滑块在图像中的位置]

**缺口信息：**
- 缺口形状：[圆形/方形/不规则/拼图形状等]
- 缺口边界坐标：左上角(x1,y1), 右下角(x2,y2)
- 缺口中心坐标：(cx,cy)
- 缺口宽度：xx像素
- 缺口高度：xx像素
- 缺口位置描述：[详细描述缺口在背景图中的位置]

**附加信息：**
- 背景图区域：左上角(x1,y1), 右下角(x2,y2)
- 滑动距离：xx像素
- 滑动方向：[水平向右/水平向左/垂直等]

注意事项：
1. 请提供像素级别的精确坐标
2. 所有坐标都是相对于整个截图的绝对坐标
3. 边界坐标要能够完整包围目标元素的轮廓
4. 如果看到多个可能的滑块元素，请选择最明显的那个
5. 缺口通常在背景拼图中，形状与滑块匹配
"""
        
        print_with_color("发送详细轮廓分析请求到LLM模型...", "blue")
        success, response = agent.model.get_model_response(enhanced_contour_prompt, [screenshot_path])
        
        if not success:
            print_with_color(f"LLM调用失败: {response}", "red")
            return None
        
        print_with_color("收到LLM的详细轮廓分析结果", "green")
        print_with_color("="*60, "yellow")
        print_with_color("大模型轮廓分析结果:", "cyan")
        print_with_color("="*60, "yellow")
        print_with_color(response, "white")
        print_with_color("="*60, "yellow")
        
        # 解析LLM响应，提取轮廓信息
        contour_info = parse_contour_response(response)
        
        if contour_info:
            print_with_color("\n解析后的轮廓信息:", "cyan")
            print_with_color("滑块轮廓:", "green")
            slider_info = contour_info['slider']
            print_with_color(f"  - 边界: ({slider_info['x1']},{slider_info['y1']}) -> ({slider_info['x2']},{slider_info['y2']})", "green")
            print_with_color(f"  - 中心: ({slider_info['center_x']},{slider_info['center_y']})", "green")
            print_with_color(f"  - 尺寸: {slider_info['width']}x{slider_info['height']}", "green")
            
            print_with_color("缺口轮廓:", "red")
            gap_info = contour_info['gap']
            print_with_color(f"  - 边界: ({gap_info['x1']},{gap_info['y1']}) -> ({gap_info['x2']},{gap_info['y2']})", "red")
            print_with_color(f"  - 中心: ({gap_info['center_x']},{gap_info['center_y']})", "red")
            print_with_color(f"  - 尺寸: {gap_info['width']}x{gap_info['height']}", "red")
            
            print_with_color(f"滑动距离: {contour_info['distance']}像素", "blue")
            
            return contour_info
        else:
            print_with_color("无法解析轮廓信息", "red")
            return None
        
    except Exception as e:
        print_with_color(f"轮廓分析失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def parse_contour_response(response: str) -> Optional[Dict]:
    """解析LLM响应中的轮廓信息"""
    try:
        import re
        
        contour_info = {
            'slider': {},
            'gap': {},
            'distance': 0
        }
        
        # 解析滑块边界坐标
        slider_boundary_pattern = r"滑块边界坐标[：:]\s*左上角\((\d+),\s*(\d+)\)[，,]\s*右下角\((\d+),\s*(\d+)\)"
        slider_boundary_match = re.search(slider_boundary_pattern, response)
        if slider_boundary_match:
            x1, y1, x2, y2 = map(int, slider_boundary_match.groups())
            contour_info['slider'].update({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1, 'height': y2 - y1
            })
        
        # 解析滑块中心坐标
        slider_center_pattern = r"滑块中心坐标[：:]\s*\((\d+),\s*(\d+)\)"
        slider_center_match = re.search(slider_center_pattern, response)
        if slider_center_match:
            cx, cy = map(int, slider_center_match.groups())
            contour_info['slider'].update({'center_x': cx, 'center_y': cy})
        
        # 解析滑块尺寸（备用方法）
        slider_width_pattern = r"滑块宽度[：:]\s*(\d+)"
        slider_height_pattern = r"滑块高度[：:]\s*(\d+)"
        
        slider_width_match = re.search(slider_width_pattern, response)
        slider_height_match = re.search(slider_height_pattern, response)
        
        if slider_width_match and slider_height_match:
            width = int(slider_width_match.group(1))
            height = int(slider_height_match.group(1))
            if 'width' not in contour_info['slider']:
                contour_info['slider']['width'] = width
            if 'height' not in contour_info['slider']:
                contour_info['slider']['height'] = height
        
        # 解析缺口边界坐标
        gap_boundary_pattern = r"缺口边界坐标[：:]\s*左上角\((\d+),\s*(\d+)\)[，,]\s*右下角\((\d+),\s*(\d+)\)"
        gap_boundary_match = re.search(gap_boundary_pattern, response)
        if gap_boundary_match:
            x1, y1, x2, y2 = map(int, gap_boundary_match.groups())
            contour_info['gap'].update({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1, 'height': y2 - y1
            })
        
        # 解析缺口中心坐标
        gap_center_pattern = r"缺口中心坐标[：:]\s*\((\d+),\s*(\d+)\)"
        gap_center_match = re.search(gap_center_pattern, response)
        if gap_center_match:
            cx, cy = map(int, gap_center_match.groups())
            contour_info['gap'].update({'center_x': cx, 'center_y': cy})
        
        # 解析缺口尺寸（备用方法）
        gap_width_pattern = r"缺口宽度[：:]\s*(\d+)"
        gap_height_pattern = r"缺口高度[：:]\s*(\d+)"
        
        gap_width_match = re.search(gap_width_pattern, response)
        gap_height_match = re.search(gap_height_pattern, response)
        
        if gap_width_match and gap_height_match:
            width = int(gap_width_match.group(1))
            height = int(gap_height_match.group(1))
            if 'width' not in contour_info['gap']:
                contour_info['gap']['width'] = width
            if 'height' not in contour_info['gap']:
                contour_info['gap']['height'] = height
        
        # 解析滑动距离
        distance_pattern = r"滑动距离[：:]\s*(\d+)"
        distance_match = re.search(distance_pattern, response)
        if distance_match:
            contour_info['distance'] = int(distance_match.group(1))
        
        # 设置默认值或计算缺失值
        if not contour_info['slider']:
            print_with_color("使用智能解析方法提取滑块信息...", "yellow")
            contour_info['slider'] = extract_slider_info_smart(response)
        
        if not contour_info['gap']:
            print_with_color("使用智能解析方法提取缺口信息...", "yellow")
            contour_info['gap'] = extract_gap_info_smart(response)
        
        # 计算中心坐标（如果缺失）
        if 'center_x' not in contour_info['slider'] and 'x1' in contour_info['slider']:
            contour_info['slider']['center_x'] = (contour_info['slider']['x1'] + contour_info['slider']['x2']) // 2
            contour_info['slider']['center_y'] = (contour_info['slider']['y1'] + contour_info['slider']['y2']) // 2
        
        if 'center_x' not in contour_info['gap'] and 'x1' in contour_info['gap']:
            contour_info['gap']['center_x'] = (contour_info['gap']['x1'] + contour_info['gap']['x2']) // 2
            contour_info['gap']['center_y'] = (contour_info['gap']['y1'] + contour_info['gap']['y2']) // 2
        
        # 计算滑动距离（如果缺失）
        if contour_info['distance'] == 0 and 'center_x' in contour_info['slider'] and 'center_x' in contour_info['gap']:
            dx = contour_info['gap']['center_x'] - contour_info['slider']['center_x']
            dy = contour_info['gap']['center_y'] - contour_info['slider']['center_y']
            contour_info['distance'] = int((dx**2 + dy**2)**0.5)
        
        # 验证数据完整性
        if validate_contour_info(contour_info):
            return contour_info
        else:
            print_with_color("轮廓信息不完整，使用备用解析方法", "yellow")
            return create_fallback_contour_info(response)
        
    except Exception as e:
        print_with_color(f"解析轮廓响应失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def extract_slider_info_smart(response: str) -> Dict:
    """智能提取滑块信息"""
    import re
    
    slider_info = {}
    
    # 尝试多种坐标格式
    coordinate_patterns = [
        r"\((\d+),\s*(\d+)\)",  # 基本坐标格式
        r"(\d+),\s*(\d+)",      # 简单数字格式
        r"x[：:](\d+).*?y[：:](\d+)",  # x:123 y:456格式
    ]
    
    coordinates = []
    for pattern in coordinate_patterns:
        matches = re.findall(pattern, response)
        coordinates.extend([(int(x), int(y)) for x, y in matches])
    
    if len(coordinates) >= 2:
        # 假设前两个坐标是滑块相关的
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        
        # 确保x1,y1是左上角，x2,y2是右下角
        slider_info = {
            'x1': min(x1, x2), 'y1': min(y1, y2),
            'x2': max(x1, x2), 'y2': max(y1, y2),
            'width': abs(x2 - x1), 'height': abs(y2 - y1),
            'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2
        }
    else:
        # 使用默认估算
        slider_info = {
            'x1': 50, 'y1': 500, 'x2': 100, 'y2': 550,
            'width': 50, 'height': 50,
            'center_x': 75, 'center_y': 525
        }
    
    return slider_info


def extract_gap_info_smart(response: str) -> Dict:
    """智能提取缺口信息"""
    import re
    
    gap_info = {}
    
    # 尝试多种坐标格式
    coordinate_patterns = [
        r"\((\d+),\s*(\d+)\)",  # 基本坐标格式
        r"(\d+),\s*(\d+)",      # 简单数字格式
    ]
    
    coordinates = []
    for pattern in coordinate_patterns:
        matches = re.findall(pattern, response)
        coordinates.extend([(int(x), int(y)) for x, y in matches])
    
    if len(coordinates) >= 4:
        # 假设后面的坐标是缺口相关的
        x1, y1 = coordinates[-2]
        x2, y2 = coordinates[-1]
        
        gap_info = {
            'x1': min(x1, x2), 'y1': min(y1, y2),
            'x2': max(x1, x2), 'y2': max(y1, y2),
            'width': abs(x2 - x1), 'height': abs(y2 - y1),
            'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2
        }
    else:
        # 使用默认估算（假设在右侧）
        gap_info = {
            'x1': 400, 'y1': 300, 'x2': 450, 'y2': 350,
            'width': 50, 'height': 50,
            'center_x': 425, 'center_y': 325
        }
    
    return gap_info


def validate_contour_info(contour_info: Dict) -> bool:
    """验证轮廓信息的完整性"""
    try:
        required_slider_keys = ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
        required_gap_keys = ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
        
        slider_valid = all(key in contour_info['slider'] for key in required_slider_keys)
        gap_valid = all(key in contour_info['gap'] for key in required_gap_keys)
        
        return slider_valid and gap_valid
    except:
        return False


def create_fallback_contour_info(response: str) -> Dict:
    """创建备用轮廓信息"""
    print_with_color("创建备用轮廓信息...", "yellow")
    
    # 基于经验值创建合理的默认轮廓信息
    return {
        'slider': {
            'x1': 80, 'y1': 580, 'x2': 130, 'y2': 630,
            'width': 50, 'height': 50,
            'center_x': 105, 'center_y': 605
        },
        'gap': {
            'x1': 420, 'y1': 320, 'x2': 470, 'y2': 370,
            'width': 50, 'height': 50,
            'center_x': 445, 'center_y': 345
        },
        'distance': 340
    }


def mark_slider_contours(image_path: str, contour_info: Dict) -> Optional[str]:
    """在图像上标注滑块和缺口的轮廓"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print_with_color(f"无法读取图像: {image_path}", "red")
            return None
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        print_with_color(f"图像尺寸: {width}x{height}", "blue")
        
        # 创建标注图像（复制原图）
        marked_image = image.copy()
        
        # 提取滑块和缺口信息
        slider_info = contour_info['slider']
        gap_info = contour_info['gap']
        
        # 标注滑块轮廓（绿色粗边框）
        cv2.rectangle(
            marked_image,
            (slider_info['x1'], slider_info['y1']),
            (slider_info['x2'], slider_info['y2']),
            (0, 255, 0),  # 绿色
            3  # 较粗的线条
        )
        
        # 在滑块轮廓内部画一个稍小的框，形成双边框效果
        cv2.rectangle(
            marked_image,
            (slider_info['x1'] + 2, slider_info['y1'] + 2),
            (slider_info['x2'] - 2, slider_info['y2'] - 2),
            (0, 200, 0),  # 深绿色
            1
        )
        
        # 添加滑块标签和信息
        label_y = slider_info['y1'] - 15
        if label_y < 20:
            label_y = slider_info['y2'] + 25
        
        cv2.putText(
            marked_image,
            f"Slider ({slider_info['width']}x{slider_info['height']})",
            (slider_info['x1'], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),  # 绿色
            2
        )
        
        # 标注滑块中心点
        cv2.circle(
            marked_image,
            (slider_info['center_x'], slider_info['center_y']),
            5,
            (0, 255, 0),  # 绿色
            -1  # 实心圆
        )
        
        # 标注缺口轮廓（红色粗边框）
        cv2.rectangle(
            marked_image,
            (gap_info['x1'], gap_info['y1']),
            (gap_info['x2'], gap_info['y2']),
            (0, 0, 255),  # 红色
            3  # 较粗的线条
        )
        
        # 在缺口轮廓内部画一个稍小的框，形成双边框效果
        cv2.rectangle(
            marked_image,
            (gap_info['x1'] + 2, gap_info['y1'] + 2),
            (gap_info['x2'] - 2, gap_info['y2'] - 2),
            (0, 0, 200),  # 深红色
            1
        )
        
        # 添加缺口标签和信息
        label_y = gap_info['y1'] - 15
        if label_y < 20:
            label_y = gap_info['y2'] + 25
        
        cv2.putText(
            marked_image,
            f"Gap ({gap_info['width']}x{gap_info['height']})",
            (gap_info['x1'], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),  # 红色
            2
        )
        
        # 标注缺口中心点
        cv2.circle(
            marked_image,
            (gap_info['center_x'], gap_info['center_y']),
            5,
            (0, 0, 255),  # 红色
            -1  # 实心圆
        )
        
        # 绘制滑动路径（蓝色粗箭头）
        cv2.arrowedLine(
            marked_image,
            (slider_info['center_x'], slider_info['center_y']),
            (gap_info['center_x'], gap_info['center_y']),
            (255, 0, 0),  # 蓝色
            4,  # 较粗的箭头
            tipLength=0.3
        )
        
        # 添加距离信息
        distance = contour_info['distance']
        mid_x = (slider_info['center_x'] + gap_info['center_x']) // 2
        mid_y = min(slider_info['y1'], gap_info['y1']) - 40
        
        if mid_y < 30:
            mid_y = max(slider_info['y2'], gap_info['y2']) + 30
        
        cv2.putText(
            marked_image,
            f"Distance: {distance}px",
            (mid_x - 80, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),  # 蓝色
            2
        )
        
        # 添加坐标信息到图像角落
        info_lines = [
            f"Slider: ({slider_info['x1']},{slider_info['y1']}) -> ({slider_info['x2']},{slider_info['y2']})",
            f"Gap: ({gap_info['x1']},{gap_info['y1']}) -> ({gap_info['x2']},{gap_info['y2']})",
            f"Centers: ({slider_info['center_x']},{slider_info['center_y']}) -> ({gap_info['center_x']},{gap_info['center_y']})"
        ]
        
        y_offset = height - 80
        for i, line in enumerate(info_lines):
            cv2.putText(
                marked_image,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # 白色
                2
            )
            # 添加黑色描边以提高可读性
            cv2.putText(
                marked_image,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # 黑色
                1
            )
        
        # 保存标注图像
        output_path = os.path.join("./temp", "marked_slider_contours.png")
        success = cv2.imwrite(output_path, marked_image)
        
        if success:
            print_with_color(f"轮廓标注完成，图像大小: {marked_image.shape}", "green")
            return output_path
        else:
            print_with_color("保存标注图像失败", "red")
            return None
        
    except Exception as e:
        print_with_color(f"标注轮廓失败: {str(e)}", "red")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 