#!/usr/bin/env python3
"""
弹窗按钮标注工具
基于analyze_popup.py的分析结果，标注弹窗中需要操作的按钮位置
参考mark_captcha.py的标注方法，生成可视化的按钮标注图像
"""

import os
import sys
import cv2
import numpy as np
import traceback
from typing import Dict, Tuple, Optional, List
import re

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel
from scripts.utils import print_with_color


def main():
    """主函数"""
    print_with_color("弹窗按钮标注工具", "cyan")
    print_with_color("="*60, "yellow")
    
    # 默认使用的测试图片
    default_image = "page/_20250715_163040.png"
    
    print_with_color(f"默认分析图片: {default_image}", "yellow")
    
    # 询问是否使用其他图片
    custom_path = input("使用其他图片路径？(直接按Enter使用默认图片): ").strip()
    
    if custom_path:
        image_path = custom_path
    else:
        image_path = default_image
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print_with_color(f"图片文件不存在: {image_path}", "red")
        return
    
    print_with_color(f"分析图片: {image_path}", "blue")
    
    try:
        # 加载配置
        config = load_config()
        print_with_color(f"使用模型: {config['MODEL']}", "blue")
        
        # 检查API密钥
        if config["MODEL"] == "OpenAI" and (not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "sk-"):
            print_with_color("OpenAI API密钥未设置，请在config.yaml中设置OPENAI_API_KEY", "red")
            return
        elif config["MODEL"] == "Qwen" and not config["DASHSCOPE_API_KEY"]:
            print_with_color("Qwen API密钥未设置，请在config.yaml中设置DASHSCOPE_API_KEY", "red")
            return
        
        # 初始化模型
        if config["MODEL"] == "OpenAI":
            model = OpenAIModel(
                base_url=config["OPENAI_BASE_URL"],
                api_key=config["OPENAI_API_KEY"],
                model=config["OPENAI_MODEL"],
                temperature=float(config.get("TEMPERATURE", 0.0)),
                max_tokens=int(config.get("MAX_TOKENS", 1500))
            )
        else:  # Qwen
            model = QwenModel(
                api_key=config["DASHSCOPE_API_KEY"],
                model=config["QWEN_MODEL"]
            )
        
        # 第一步：分析弹窗内容
        print_with_color("第一步：分析弹窗内容...", "cyan")
        popup_analysis = analyze_popup_content(model, image_path)
        
        if not popup_analysis:
            print_with_color("弹窗内容分析失败", "red")
            return
        
        # 第二步：基于分析结果获取按钮位置信息
        print_with_color("第二步：分析按钮位置...", "cyan")
        button_info = analyze_button_positions(model, image_path, popup_analysis)
        
        if not button_info:
            print_with_color("按钮位置分析失败", "red")
            return
        
        # 第三步：标注按钮
        print_with_color("第三步：标注按钮位置...", "cyan")
        marked_image_path = mark_popup_buttons(image_path, button_info, popup_analysis)
        
        if marked_image_path:
            print_with_color(f"按钮标注图像已保存至: {marked_image_path}", "green")
            print_with_color("标注说明:", "cyan")
            print_with_color("  - 绿色框: 推荐操作的按钮", "green")
            print_with_color("  - 红色框: 需要避免的按钮", "red")
            print_with_color("  - 蓝色箭头: 建议点击位置", "blue")
        else:
            print_with_color("按钮标注失败", "red")
    
    except Exception as e:
        print_with_color(f"程序运行出错: {str(e)}", "red")
        traceback.print_exc()


def analyze_popup_content(model, image_path: str) -> Optional[Dict]:
    """分析弹窗内容，判断弹窗类型和应该进行的操作"""
    
    analysis_prompt = """
请仔细分析这张截图中的弹窗内容，这是一个移动应用的界面。请按照以下要求进行分析：

## 第一步：内容理解
请完整读取并理解弹窗中的所有文本内容，包括：
- 标题文字
- 正文内容
- 按钮文字
- 任何其他可见的文本信息

## 第二步：类型判断
根据文本内容判断这是什么类型的弹窗：
1. **协议弹窗**：包含用户协议、隐私政策、服务条款等需要用户同意的内容
2. **广告弹窗**：包含推广信息、营销内容、应用推荐等可以关闭的内容  
3. **权限弹窗**：请求访问权限的系统弹窗
4. **通知弹窗**：一般信息提示
5. **其他类型**

## 第三步：操作建议
基于弹窗类型和内容，判断用户应该进行什么操作：
- 如果是协议类弹窗，通常需要点击"同意"、"确认"等按钮
- 如果是广告类弹窗，通常需要点击"关闭"、"×"等按钮

## 回答格式
请按以下格式详细回答：

**弹窗文本内容**：
[请完整提取弹窗中的所有文本，包括标题、正文、按钮文字等]

**弹窗类型**：
[协议弹窗/广告弹窗/权限弹窗/通知弹窗/其他]

**内容理解**：
[用自己的话总结这个弹窗想要表达什么，用户看到这个弹窗的目的是什么]

**关键信息**：
[提取与判断操作类型相关的关键词和信息]

**操作建议**：
[明确建议用户应该点击哪个按钮，并详细说明理由]

请特别注意文本的语义含义，准确判断这是需要同意的协议还是需要关闭的广告。
"""
    
    try:
        print_with_color("正在调用大模型分析弹窗内容...", "yellow")
        
        success, response = model.get_model_response(analysis_prompt, [image_path])
        
        if not success:
            print_with_color("模型调用失败", "red")
            print_with_color(f"错误信息: {response}", "red")
            return None
        
        # 显示分析结果
        print_with_color("\n" + "="*80, "yellow")
        print_with_color("弹窗内容分析结果", "cyan")
        print_with_color("="*80, "yellow")
        print_with_color(response, "white")
        print_with_color("="*80, "yellow")
        
        # 解析分析结果
        analysis_result = parse_popup_analysis(response)
        
        if analysis_result:
            print_with_color("\n解析后的弹窗信息:", "cyan")
            print_with_color(f"弹窗类型: {analysis_result['popup_type']}", "green")
            print_with_color(f"操作建议: {analysis_result['action_suggestion']}", "green")
            return analysis_result
        else:
            print_with_color("无法解析弹窗分析结果", "red")
            return None
            
    except Exception as e:
        print_with_color(f"弹窗内容分析失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def parse_popup_analysis(response: str) -> Optional[Dict]:
    """解析弹窗分析结果"""
    try:
        analysis_result = {
            'popup_type': '',
            'content': '',
            'action_suggestion': '',
            'key_info': '',
            'raw_response': response
        }
        
        # 提取弹窗类型
        type_patterns = [
            r"弹窗类型[：:]\s*([^\n]+)",
            r"类型[：:]\s*([^\n]+)"
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, response)
            if match:
                popup_type = match.group(1).strip()
                analysis_result['popup_type'] = popup_type
                break
        
        # 提取弹窗内容
        content_patterns = [
            r"弹窗文本内容[：:]\s*([^*]+?)(?=\*\*|$)",
            r"文本内容[：:]\s*([^*]+?)(?=\*\*|$)"
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                analysis_result['content'] = content
                break
        
        # 提取操作建议
        action_patterns = [
            r"操作建议[：:]\s*([^*]+?)(?=\*\*|$)",
            r"建议操作[：:]\s*([^*]+?)(?=\*\*|$)"
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                action = match.group(1).strip()
                analysis_result['action_suggestion'] = action
                break
        
        # 提取关键信息
        key_patterns = [
            r"关键信息[：:]\s*([^*]+?)(?=\*\*|$)",
            r"关键词[：:]\s*([^*]+?)(?=\*\*|$)"
        ]
        
        for pattern in key_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                key_info = match.group(1).strip()
                analysis_result['key_info'] = key_info
                break
        
        return analysis_result
        
    except Exception as e:
        print_with_color(f"解析弹窗分析结果失败: {str(e)}", "red")
        return None


def analyze_button_positions(model, image_path: str, popup_analysis: Dict) -> Optional[Dict]:
    """基于弹窗分析结果，获取按钮的精确位置信息"""
    
    # 根据弹窗类型和操作建议，构建针对性的按钮定位提示词
    popup_type = popup_analysis.get('popup_type', '')
    action_suggestion = popup_analysis.get('action_suggestion', '')
    
    if "协议" in popup_type or "同意" in action_suggestion:
        target_buttons = ["同意", "确认", "我知道了", "接受", "好的", "确定"]
        avoid_buttons = ["拒绝", "取消", "不同意"]
        primary_color = "green"
        avoid_color = "red"
    elif "广告" in popup_type or "关闭" in action_suggestion:
        target_buttons = ["关闭", "×", "跳过", "取消", "不了", "稍后再说"]
        avoid_buttons = ["立即下载", "马上体验", "了解更多", "去看看"]
        primary_color = "green"
        avoid_color = "red"
    else:
        # 通用按钮检测
        target_buttons = ["确定", "确认", "好的", "关闭", "×"]
        avoid_buttons = ["取消"]
        primary_color = "green"
        avoid_color = "red"
    
    button_prompt = f"""
请仔细分析这个弹窗截图中的所有按钮，我需要你精确识别按钮的位置信息。

基于之前的分析，这个弹窗的类型是：{popup_type}
建议的操作是：{action_suggestion}

请找到以下类型的按钮：

**主要操作按钮**（推荐点击）：{', '.join(target_buttons)}
**避免操作按钮**（不推荐点击）：{', '.join(avoid_buttons)}

对于每个找到的按钮，请提供以下信息：

**按钮1信息：**
- 按钮文字：[具体的按钮文字]
- 按钮类型：[主要操作/避免操作]
- 按钮边界坐标：左上角(x1,y1), 右下角(x2,y2)
- 按钮中心坐标：(cx,cy)
- 按钮宽度：xx像素
- 按钮高度：xx像素
- 按钮位置描述：[详细描述按钮在弹窗中的位置]

**按钮2信息：**
[如果有第二个按钮，按同样格式提供信息]

注意事项：
1. 请提供像素级别的精确坐标
2. 所有坐标都是相对于整个截图的绝对坐标
3. 边界坐标要能够完整包围按钮的轮廓
4. 如果按钮文字不完全匹配目标列表，但语义相似，也要标注
5. 优先识别最明显、最容易点击的按钮
6. 如果有多个相似按钮，请全部标注

请仔细观察按钮的边框、颜色、文字，确保坐标准确。
"""
    
    try:
        print_with_color("正在分析按钮位置信息...", "yellow")
        
        success, response = model.get_model_response(button_prompt, [image_path])
        
        if not success:
            print_with_color("按钮位置分析失败", "red")
            print_with_color(f"错误信息: {response}", "red")
            return None
        
        # 显示按钮分析结果
        print_with_color("\n" + "="*80, "yellow")
        print_with_color("按钮位置分析结果", "cyan")
        print_with_color("="*80, "yellow")
        print_with_color(response, "white")
        print_with_color("="*80, "yellow")
        
        # 解析按钮信息
        button_info = parse_button_response(response, target_buttons, avoid_buttons)
        
        if button_info and button_info['buttons']:
            print_with_color("\n解析后的按钮信息:", "cyan")
            for i, btn in enumerate(button_info['buttons']):
                print_with_color(f"按钮{i+1}: {btn['text']} ({btn['type']})", "green" if btn['type'] == "主要操作" else "red")
                print_with_color(f"  - 边界: ({btn['x1']},{btn['y1']}) -> ({btn['x2']},{btn['y2']})", "white")
                print_with_color(f"  - 中心: ({btn['center_x']},{btn['center_y']})", "white")
                print_with_color(f"  - 尺寸: {btn['width']}x{btn['height']}", "white")
            
            return button_info
        else:
            print_with_color("无法解析按钮位置信息，使用备用方法", "yellow")
            return create_fallback_button_info(response, popup_analysis)
            
    except Exception as e:
        print_with_color(f"按钮位置分析失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def parse_button_response(response: str, target_buttons: List[str], avoid_buttons: List[str]) -> Optional[Dict]:
    """解析按钮位置信息"""
    try:
        button_info = {
            'buttons': [],
            'target_buttons': target_buttons,
            'avoid_buttons': avoid_buttons
        }
        
        # 查找所有按钮信息块
        button_blocks = re.findall(r'\*\*按钮\d+信息[：:]\*\*(.*?)(?=\*\*按钮\d+信息[：:]\*\*|$)', response, re.DOTALL)
        
        for block in button_blocks:
            button = {}
            
            # 解析按钮文字
            text_match = re.search(r'按钮文字[：:]\s*([^\n]+)', block)
            if text_match:
                button['text'] = text_match.group(1).strip().replace('[', '').replace(']', '')
            else:
                continue  # 没有文字信息，跳过这个按钮
            
            # 判断按钮类型
            button_text = button['text']
            if any(target in button_text for target in target_buttons):
                button['type'] = "主要操作"
            elif any(avoid in button_text for avoid in avoid_buttons):
                button['type'] = "避免操作"
            else:
                # 根据语义判断
                if "同意" in button_text or "确认" in button_text or "好" in button_text or "接受" in button_text:
                    button['type'] = "主要操作"
                elif "取消" in button_text or "拒绝" in button_text or "不" in button_text:
                    button['type'] = "避免操作"
                else:
                    button['type'] = "主要操作"  # 默认为主要操作
            
            # 解析边界坐标
            boundary_match = re.search(r'边界坐标[：:]\s*左上角\((\d+),\s*(\d+)\)[，,]\s*右下角\((\d+),\s*(\d+)\)', block)
            if boundary_match:
                x1, y1, x2, y2 = map(int, boundary_match.groups())
                button.update({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1
                })
            
            # 解析中心坐标
            center_match = re.search(r'中心坐标[：:]\s*\((\d+),\s*(\d+)\)', block)
            if center_match:
                cx, cy = map(int, center_match.groups())
                button.update({'center_x': cx, 'center_y': cy})
            elif 'x1' in button:
                # 计算中心坐标
                button['center_x'] = (button['x1'] + button['x2']) // 2
                button['center_y'] = (button['y1'] + button['y2']) // 2
            
            # 解析尺寸（备用方法）
            width_match = re.search(r'宽度[：:]\s*(\d+)', block)
            height_match = re.search(r'高度[：:]\s*(\d+)', block)
            
            if width_match and height_match and 'width' not in button:
                button['width'] = int(width_match.group(1))
                button['height'] = int(height_match.group(1))
            
            # 如果有基本信息，添加到列表
            if 'text' in button and ('x1' in button or 'center_x' in button):
                button_info['buttons'].append(button)
        
        # 如果没有找到标准格式，尝试其他解析方法
        if not button_info['buttons']:
            button_info['buttons'] = extract_buttons_smart(response, target_buttons, avoid_buttons)
        
        return button_info if button_info['buttons'] else None
        
    except Exception as e:
        print_with_color(f"解析按钮响应失败: {str(e)}", "red")
        return None


def extract_buttons_smart(response: str, target_buttons: List[str], avoid_buttons: List[str]) -> List[Dict]:
    """智能提取按钮信息"""
    buttons = []
    
    try:
        # 查找所有坐标对
        coordinate_patterns = [
            r'\((\d+),\s*(\d+)\)',  # 基本坐标格式
            r'(\d+),\s*(\d+)',      # 简单数字格式
        ]
        
        coordinates = []
        for pattern in coordinate_patterns:
            matches = re.findall(pattern, response)
            coordinates.extend([(int(x), int(y)) for x, y in matches])
        
        # 查找按钮文字
        button_texts = []
        for target in target_buttons + avoid_buttons:
            if target in response:
                button_texts.append(target)
        
        # 如果找到了坐标和文字，尝试匹配
        if coordinates and button_texts:
            for i, text in enumerate(button_texts):
                if i * 2 + 1 < len(coordinates):
                    x1, y1 = coordinates[i * 2]
                    x2, y2 = coordinates[i * 2 + 1]
                    
                    # 确保x1,y1是左上角，x2,y2是右下角
                    button = {
                        'text': text,
                        'type': "主要操作" if text in target_buttons else "避免操作",
                        'x1': min(x1, x2), 'y1': min(y1, y2),
                        'x2': max(x1, x2), 'y2': max(y1, y2),
                        'width': abs(x2 - x1), 'height': abs(y2 - y1),
                        'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2
                    }
                    buttons.append(button)
        
    except Exception as e:
        print_with_color(f"智能提取按钮信息失败: {str(e)}", "red")
    
    return buttons


def create_fallback_button_info(response: str, popup_analysis: Dict) -> Dict:
    """创建备用按钮信息"""
    print_with_color("创建备用按钮信息...", "yellow")
    
    popup_type = popup_analysis.get('popup_type', '')
    
    # 根据弹窗类型创建合理的默认按钮信息
    if "协议" in popup_type:
        buttons = [
            {
                'text': '同意',
                'type': '主要操作',
                'x1': 300, 'y1': 600, 'x2': 400, 'y2': 650,
                'width': 100, 'height': 50,
                'center_x': 350, 'center_y': 625
            }
        ]
    elif "广告" in popup_type:
        buttons = [
            {
                'text': '关闭',
                'type': '主要操作',
                'x1': 450, 'y1': 150, 'x2': 480, 'y2': 180,
                'width': 30, 'height': 30,
                'center_x': 465, 'center_y': 165
            }
        ]
    else:
        buttons = [
            {
                'text': '确定',
                'type': '主要操作',
                'x1': 300, 'y1': 600, 'x2': 400, 'y2': 650,
                'width': 100, 'height': 50,
                'center_x': 350, 'center_y': 625
            }
        ]
    
    return {
        'buttons': buttons,
        'target_buttons': [],
        'avoid_buttons': []
    }


def mark_popup_buttons(image_path: str, button_info: Dict, popup_analysis: Dict) -> Optional[str]:
    """在图像上标注弹窗按钮"""
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
        
        # 标注每个按钮
        for i, button in enumerate(button_info['buttons']):
            # 确定颜色
            if button['type'] == "主要操作":
                color = (0, 255, 0)  # 绿色
                label_color = (0, 255, 0)
                thick = 3
            else:
                color = (0, 0, 255)  # 红色
                label_color = (0, 0, 255)
                thick = 2
            
            # 标注按钮边框
            if 'x1' in button:
                cv2.rectangle(
                    marked_image,
                    (button['x1'], button['y1']),
                    (button['x2'], button['y2']),
                    color,
                    thick
                )
                
                # 添加内边框
                cv2.rectangle(
                    marked_image,
                    (button['x1'] + 2, button['y1'] + 2),
                    (button['x2'] - 2, button['y2'] - 2),
                    (color[0] // 2, color[1] // 2, color[2] // 2),
                    1
                )
            
            # 标注按钮中心点
            if 'center_x' in button:
                cv2.circle(
                    marked_image,
                    (button['center_x'], button['center_y']),
                    8,
                    color,
                    -1  # 实心圆
                )
                
                # 添加点击建议箭头（对于主要操作按钮）
                if button['type'] == "主要操作":
                    # 画一个指向按钮的箭头
                    start_point = (button['center_x'] - 60, button['center_y'] - 60)
                    end_point = (button['center_x'] - 15, button['center_y'] - 15)
                    cv2.arrowedLine(
                        marked_image,
                        start_point,
                        end_point,
                        (255, 0, 0),  # 蓝色箭头
                        3,
                        tipLength=0.3
                    )
                    
                    # 添加"点击这里"标签
                    cv2.putText(
                        marked_image,
                        "Click Here",
                        (start_point[0] - 30, start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),  # 蓝色
                        2
                    )
            
            # 添加按钮标签
            label_text = f"{button['text']} ({button['type']})"
            if 'x1' in button:
                label_y = button['y1'] - 15
                if label_y < 20:
                    label_y = button['y2'] + 25
                
                cv2.putText(
                    marked_image,
                    label_text,
                    (button['x1'], label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    label_color,
                    2
                )
            
            # 添加坐标信息
            if 'x1' in button and 'center_x' in button:
                coord_text = f"({button['center_x']},{button['center_y']})"
                cv2.putText(
                    marked_image,
                    coord_text,
                    (button['x1'], button['y2'] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # 白色
                    1
                )
        
        # 添加弹窗分析信息到图像
        popup_type = popup_analysis.get('popup_type', '未知')
        action_suggestion = popup_analysis.get('action_suggestion', '无建议')
        
        info_lines = [
            f"Popup Type: {popup_type}",
            f"Suggestion: {action_suggestion[:50]}...",
            f"Buttons Found: {len(button_info['buttons'])}"
        ]
        
        # 在图像顶部添加信息
        y_offset = 30
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
        
        # 确保临时目录存在
        os.makedirs("./temp", exist_ok=True)
        
        # 保存标注图像
        output_path = os.path.join("./temp", "marked_popup_buttons.png")
        success = cv2.imwrite(output_path, marked_image)
        
        if success:
            print_with_color(f"按钮标注完成，图像大小: {marked_image.shape}", "green")
            return output_path
        else:
            print_with_color("保存标注图像失败", "red")
            return None
        
    except Exception as e:
        print_with_color(f"标注按钮失败: {str(e)}", "red")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main() 