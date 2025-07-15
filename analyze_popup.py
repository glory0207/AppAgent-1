#!/usr/bin/env python3
"""
弹窗内容分析脚本
直接分析用户提供的截图，理解弹窗内容并给出操作建议
"""

import os
import sys
import traceback
import cv2
import numpy as np
import re
import time
from typing import Dict, Tuple, Optional, List

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel
from scripts.utils import print_with_color
from scripts.and_controller import AndroidController, list_all_devices


def analyze_popup_content(image_path: str):
    """分析弹窗内容"""
    print_with_color("开始分析弹窗内容", "cyan")
    print_with_color("="*60, "yellow")
    
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
        
        # 构建分析提示词
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

**按钮位置**：
[如果能识别到按钮，请描述其大致位置]

请特别注意文本的语义含义，准确判断这是需要同意的协议还是需要关闭的广告。
"""
        
        print_with_color("正在调用大模型分析弹窗内容...", "yellow")
        
        # 调用模型分析
        success, response = model.get_model_response(analysis_prompt, [image_path])
        
        if not success:
            print_with_color("模型调用失败", "red")
            print_with_color(f"错误信息: {response}", "red")
            return
        
        # 显示分析结果
        print_with_color("\n" + "="*80, "yellow")
        print_with_color("大模型分析结果", "cyan")
        print_with_color("="*80, "yellow")
        print_with_color(response, "white")
        print_with_color("="*80, "yellow")
        
        # 提取关键信息
        print_with_color("\n分析完成！", "green")
        print_with_color("基于以上分析，您可以了解：", "cyan")
        print_with_color("1. 弹窗的具体文本内容", "white")
        print_with_color("2. 弹窗的类型和用途", "white")
        print_with_color("3. 建议执行的操作", "white")
        print_with_color("4. 操作的具体理由", "white")
        
        # 询问是否需要进一步的按钮定位分析和标注
        print_with_color("\n是否需要进一步分析按钮位置并生成标注图像？(y/n)", "yellow")
        if input().strip().lower() == 'y':
            button_info = analyze_button_positions_enhanced(model, image_path, response)
            if button_info:
                marked_image_path = mark_popup_buttons(image_path, button_info, response)
                if marked_image_path:
                    print_with_color(f"\n按钮标注图像已保存至: {marked_image_path}", "green")
                    print_with_color("标注说明:", "cyan")
                    print_with_color("  - 绿色框: 推荐操作的按钮", "green")
                    print_with_color("  - 红色框: 需要避免的按钮", "red")
                    print_with_color("  - 蓝色箭头: 建议点击位置", "blue")
                    
                    # 询问是否执行自动点击操作
                    execute_action(button_info, image_path)
        
    except Exception as e:
        print_with_color(f"分析过程中发生错误: {str(e)}", "red")
        traceback.print_exc()


def analyze_button_positions_enhanced(model, image_path: str, previous_analysis: str) -> Optional[Dict]:
    """增强的按钮位置分析，返回详细的按钮信息用于标注"""
    print_with_color("\n分析按钮位置...", "cyan")
    
    # 根据之前的分析结果决定查找什么按钮
    if "协议" in previous_analysis or "同意" in previous_analysis:
        target_buttons = ["同意", "确认", "我知道了", "接受", "好的", "确定"]
        avoid_buttons = ["拒绝", "取消", "不同意"]
        button_type = "协议类"
    elif "广告" in previous_analysis or "关闭" in previous_analysis:
        target_buttons = ["关闭", "×", "跳过", "取消", "不了", "稍后再说"]
        avoid_buttons = ["立即下载", "马上体验", "了解更多", "去看看"]
        button_type = "广告类"
    else:
        target_buttons = ["确定", "确认", "好的", "关闭", "×"]
        avoid_buttons = ["取消"]
        button_type = "通用"
    
    button_prompt = f"""
请仔细分析这个弹窗截图中的所有按钮，我需要你精确识别按钮的位置信息。

基于之前的分析，这是一个{button_type}弹窗。

请找到以下类型的按钮：

**推荐操作按钮**：{', '.join(target_buttons)}
**避免操作按钮**：{', '.join(avoid_buttons)}

对于每个找到的按钮，请提供以下信息：

**按钮1信息：**
- 按钮文字：[具体的按钮文字]
- 按钮类型：[推荐操作/避免操作]
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

请仔细观察按钮的边框、颜色、文字，确保坐标准确。
"""
    
    try:
        success, button_response = model.get_model_response(button_prompt, [image_path])
        
        if not success:
            print_with_color("按钮位置分析失败", "red")
            return None
        
        print_with_color("\n按钮位置分析结果:", "green")
        print_with_color(button_response, "white")
        
        # 解析按钮信息
        button_info = parse_button_response(button_response, target_buttons, avoid_buttons)
        
        if button_info and button_info['buttons']:
            print_with_color("\n解析后的按钮信息:", "cyan")
            for i, btn in enumerate(button_info['buttons']):
                print_with_color(f"按钮{i+1}: {btn['text']} ({btn['type']})", "green" if btn['type'] == "推荐操作" else "red")
                if 'x1' in btn:
                    print_with_color(f"  - 边界: ({btn['x1']},{btn['y1']}) -> ({btn['x2']},{btn['y2']})", "white")
                if 'center_x' in btn:
                    print_with_color(f"  - 中心: ({btn['center_x']},{btn['center_y']})", "white")
                if 'width' in btn:
                    print_with_color(f"  - 尺寸: {btn['width']}x{btn['height']}", "white")
            return button_info
        else:
            print_with_color("无法解析按钮位置信息，使用备用方法", "yellow")
            return create_fallback_button_info(button_response, button_type)
    
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
                continue
            
            # 判断按钮类型
            button_text = button['text']
            if any(target in button_text for target in target_buttons):
                button['type'] = "推荐操作"
            elif any(avoid in button_text for avoid in avoid_buttons):
                button['type'] = "避免操作"
            else:
                # 根据语义判断
                if "同意" in button_text or "确认" in button_text or "好" in button_text or "接受" in button_text:
                    button['type'] = "推荐操作"
                elif "取消" in button_text or "拒绝" in button_text or "不" in button_text:
                    button['type'] = "避免操作"
                else:
                    button['type'] = "推荐操作"
            
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
                button['center_x'] = (button['x1'] + button['x2']) // 2
                button['center_y'] = (button['y1'] + button['y2']) // 2
            
            # 如果没有边界坐标，尝试只从中心坐标推算
            if 'center_x' in button and 'x1' not in button:
                # 假设按钮大小
                w, h = 80, 40
                button.update({
                    'x1': button['center_x'] - w//2, 'y1': button['center_y'] - h//2,
                    'x2': button['center_x'] + w//2, 'y2': button['center_y'] + h//2,
                    'width': w, 'height': h
                })
            
            if 'text' in button and ('x1' in button or 'center_x' in button):
                button_info['buttons'].append(button)
        
        # 如果没有找到标准格式，尝试简单解析
        if not button_info['buttons']:
            button_info['buttons'] = extract_buttons_simple(response, target_buttons, avoid_buttons)
        
        return button_info if button_info['buttons'] else None
        
    except Exception as e:
        print_with_color(f"解析按钮响应失败: {str(e)}", "red")
        return None


def extract_buttons_simple(response: str, target_buttons: List[str], avoid_buttons: List[str]) -> List[Dict]:
    """简单提取按钮信息"""
    buttons = []
    
    try:
        # 查找坐标
        coordinate_matches = re.findall(r'\((\d+),\s*(\d+)\)', response)
        coordinates = [(int(x), int(y)) for x, y in coordinate_matches]
        
        # 查找按钮文字
        button_texts = []
        for target in target_buttons + avoid_buttons:
            if target in response:
                button_texts.append(target)
        
        # 如果没有找到精确匹配，查找常见按钮词
        if not button_texts:
            common_buttons = ["确定", "确认", "同意", "关闭", "取消", "×", "好的"]
            for btn in common_buttons:
                if btn in response:
                    button_texts.append(btn)
        
        # 匹配文字和坐标
        for i, text in enumerate(button_texts):
            button = {'text': text}
            
            # 判断类型
            if text in target_buttons or any(t in text for t in ["同意", "确认", "好", "关闭"]):
                button['type'] = "推荐操作"
            else:
                button['type'] = "避免操作"
            
            # 如果有坐标，使用坐标
            if i < len(coordinates):
                cx, cy = coordinates[i]
                button.update({
                    'center_x': cx, 'center_y': cy,
                    'x1': cx - 40, 'y1': cy - 20,
                    'x2': cx + 40, 'y2': cy + 20,
                    'width': 80, 'height': 40
                })
            else:
                # 使用默认位置
                button.update({
                    'center_x': 400, 'center_y': 600 + i * 60,
                    'x1': 360, 'y1': 580 + i * 60,
                    'x2': 440, 'y2': 620 + i * 60,
                    'width': 80, 'height': 40
                })
            
            buttons.append(button)
    
    except Exception as e:
        print_with_color(f"简单提取按钮信息失败: {str(e)}", "red")
    
    return buttons


def create_fallback_button_info(response: str, button_type: str) -> Dict:
    """创建备用按钮信息"""
    print_with_color("创建备用按钮信息...", "yellow")
    
    if "协议" in button_type:
        buttons = [
            {
                'text': '同意',
                'type': '推荐操作',
                'x1': 300, 'y1': 600, 'x2': 400, 'y2': 650,
                'width': 100, 'height': 50,
                'center_x': 350, 'center_y': 625
            }
        ]
    elif "广告" in button_type:
        buttons = [
            {
                'text': '关闭',
                'type': '推荐操作',
                'x1': 450, 'y1': 150, 'x2': 480, 'y2': 180,
                'width': 30, 'height': 30,
                'center_x': 465, 'center_y': 165
            }
        ]
    else:
        buttons = [
            {
                'text': '确定',
                'type': '推荐操作',
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


def mark_popup_buttons(image_path: str, button_info: Dict, popup_analysis: str) -> Optional[str]:
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
        
        # 创建标注图像
        marked_image = image.copy()
        
        # 标注每个按钮
        for i, button in enumerate(button_info['buttons']):
            # 确定颜色
            if button['type'] == "推荐操作":
                color = (0, 255, 0)  # 绿色
                thick = 3
            else:
                color = (0, 0, 255)  # 红色
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
                
                # 内边框
                cv2.rectangle(
                    marked_image,
                    (button['x1'] + 2, button['y1'] + 2),
                    (button['x2'] - 2, button['y2'] - 2),
                    (color[0] // 2, color[1] // 2, color[2] // 2),
                    1
                )
            
            # 标注中心点
            if 'center_x' in button:
                cv2.circle(
                    marked_image,
                    (button['center_x'], button['center_y']),
                    8,
                    color,
                    -1
                )
                
                # 推荐操作按钮添加箭头
                if button['type'] == "推荐操作":
                    start_point = (button['center_x'] - 60, button['center_y'] - 60)
                    end_point = (button['center_x'] - 15, button['center_y'] - 15)
                    cv2.arrowedLine(
                        marked_image,
                        start_point,
                        end_point,
                        (255, 0, 0),  # 蓝色
                        3,
                        tipLength=0.3
                    )
                    
                    cv2.putText(
                        marked_image,
                        "Click Here",
                        (start_point[0] - 30, start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )
            
            # 按钮标签
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
                    color,
                    2
                )
            
            # 坐标信息
            if 'center_x' in button and 'x1' in button:
                coord_text = f"({button['center_x']},{button['center_y']})"
                cv2.putText(
                    marked_image,
                    coord_text,
                    (button['x1'], button['y2'] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # 添加信息
        info_lines = [
            f"Buttons Found: {len(button_info['buttons'])}",
            f"Recommended Actions: {len([b for b in button_info['buttons'] if b['type'] == '推荐操作'])}"
        ]
        
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(
                marked_image,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                marked_image,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
        
        # 确保临时目录存在
        os.makedirs("./temp", exist_ok=True)
        
        # 保存标注图像
        output_path = os.path.join("./temp", "marked_popup_buttons.png")
        success = cv2.imwrite(output_path, marked_image)
        
        if success:
            print_with_color(f"按钮标注完成", "green")
            return output_path
        else:
            print_with_color("保存标注图像失败", "red")
            return None
        
    except Exception as e:
        print_with_color(f"标注按钮失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print_with_color("弹窗内容分析和按钮标注工具", "cyan")
    print_with_color("="*60, "yellow")
    
    # 选择输入模式
    print_with_color("请选择工作模式:", "blue")
    print_with_color("1. 全自动模式 - 自动识别并关闭弹窗广告", "green")
    print_with_color("2. 分析手机当前屏幕(交互模式)", "white")
    print_with_color("3. 分析本地图片文件", "white")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 全自动模式
        auto_mode()
        return
    elif choice == "2":
        # 实时手机屏幕模式
        image_path = get_phone_screenshot()
        if not image_path:
            return
    else:
        # 本地图片模式
        default_image = "page/_20250715_163040.png"
        print_with_color(f"默认分析图片: {default_image}", "yellow")
        
        custom_path = input("使用其他图片路径？(直接按Enter使用默认图片): ").strip()
        
        if custom_path:
            image_path = custom_path
        else:
            image_path = default_image
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        print_with_color(f"图片文件不存在: {image_path}", "red")
        return
    
    analyze_popup_content(image_path)


def get_phone_screenshot() -> Optional[str]:
    """获取手机当前屏幕截图"""
    try:
        print_with_color("正在连接Android设备...", "yellow")
        
        # 获取设备列表
        device_list = list_all_devices()
        if not device_list:
            print_with_color("未找到Android设备！", "red")
            print_with_color("请确保:", "yellow")
            print_with_color("1. 手机已通过USB连接到电脑", "white")
            print_with_color("2. 已开启USB调试模式", "white")
            print_with_color("3. 已安装ADB工具", "white")
            return None
        
        print_with_color(f"发现设备: {device_list}", "green")
        
        # 过滤掉离线设备
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("没有在线设备可用！", "red")
            print_with_color("请检查设备连接状态和USB调试权限", "yellow")
            return None
        
        # 如果有多个设备，让用户选择
        if len(online_devices) > 1:
            print_with_color("发现多个设备，请选择:", "blue")
            for i, device in enumerate(online_devices):
                print_with_color(f"{i+1}. {device}", "white")
            
            while True:
                try:
                    choice = int(input("请输入设备编号: ").strip()) - 1
                    if 0 <= choice < len(online_devices):
                        selected_device = online_devices[choice]
                        break
                    else:
                        print_with_color("输入的编号无效，请重新选择", "red")
                except ValueError:
                    print_with_color("请输入有效的数字", "red")
        else:
            selected_device = online_devices[0]
        
        print_with_color(f"已选择设备: {selected_device}", "green")
        
        # 创建控制器
        controller = AndroidController(selected_device)
        
        # 显示设备屏幕尺寸
        width, height = controller.get_device_size()
        print_with_color(f"设备屏幕尺寸: {width}x{height}", "green")
        
        # 确保临时目录存在
        os.makedirs("./temp", exist_ok=True)
        
        # 获取屏幕截图
        print_with_color("正在截取当前屏幕...", "yellow")
        screenshot_path = controller.get_screenshot("popup_analysis", "./temp")
        
        if screenshot_path == "ERROR":
            print_with_color("无法获取屏幕截图", "red")
            return None
        
        print_with_color(f"屏幕截图已保存至: {screenshot_path}", "green")
        return screenshot_path
        
    except Exception as e:
        print_with_color(f"获取手机截图失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def execute_action(button_info: Dict, image_path: str):
    """询问用户是否执行自动点击操作"""
    try:
        # 找到推荐操作的按钮
        recommended_buttons = [btn for btn in button_info['buttons'] if btn['type'] == '推荐操作']
        
        if not recommended_buttons:
            print_with_color("没有找到推荐操作的按钮", "yellow")
            return
        
        print_with_color("\n是否要自动点击推荐的按钮？", "yellow")
        for i, btn in enumerate(recommended_buttons):
            print_with_color(f"{i+1}. {btn['text']} - 坐标({btn.get('center_x', 'N/A')},{btn.get('center_y', 'N/A')})", "white")
        
        print_with_color("0. 不执行任何操作", "white")
        
        choice = input("请选择要点击的按钮编号 (0-{}): ".format(len(recommended_buttons))).strip()
        
        try:
            choice_idx = int(choice)
            if choice_idx == 0:
                print_with_color("跳过自动操作", "yellow")
                return
            elif 1 <= choice_idx <= len(recommended_buttons):
                selected_button = recommended_buttons[choice_idx - 1]
                
                # 检查是否是从手机获取的截图（说明有设备连接）
                if "temp" in image_path and "popup_analysis" in image_path:
                    # 从图片路径推断设备连接状态
                    perform_click_action(selected_button)
                else:
                    print_with_color("这是静态图片分析，无法执行实际点击操作", "yellow")
                    print_with_color(f"建议手动点击坐标: ({selected_button.get('center_x', 'N/A')}, {selected_button.get('center_y', 'N/A')})", "cyan")
            else:
                print_with_color("无效的选择", "red")
        except ValueError:
            print_with_color("请输入有效的数字", "red")
    
    except Exception as e:
        print_with_color(f"执行操作时出错: {str(e)}", "red")
        traceback.print_exc()


def perform_click_action(button: Dict):
    """执行实际的点击操作"""
    try:
        if 'center_x' not in button or 'center_y' not in button:
            print_with_color("按钮坐标信息不完整，无法执行点击", "red")
            return
        
        # 重新连接设备（简化版，实际项目中应该保持连接）
        print_with_color("正在连接设备执行点击操作...", "yellow")
        
        device_list = list_all_devices()
        if not device_list:
            print_with_color("未找到设备，无法执行点击操作", "red")
            return
        
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("没有在线设备，无法执行点击操作", "red")
            return
        
        # 使用第一个可用设备
        controller = AndroidController(online_devices[0])
        
        click_x = button['center_x']
        click_y = button['center_y']
        
        print_with_color(f"即将点击按钮: {button['text']}", "cyan")
        print_with_color(f"点击坐标: ({click_x}, {click_y})", "cyan")
        
        # 确认执行
        confirm = input("确认执行点击操作？(y/n): ").strip().lower()
        if confirm == 'y':
            # 执行点击
            result = controller.tap(click_x, click_y)
            if result == "SUCCESS":
                print_with_color(f"✓ 成功执行点击操作: {button['text']}", "green")
                
                # 等待界面更新
                print_with_color("等待界面更新...", "yellow")
                time.sleep(2)
                
                # 获取点击后截图
                print_with_color("获取点击后的屏幕截图进行验证...", "yellow")
                after_screenshot = controller.get_screenshot("after_click", "./temp")
                if after_screenshot != "ERROR":
                    print_with_color(f"点击后截图已保存: {after_screenshot}", "green")
                    
                    # 创建简化的弹窗分析用于验证
                    popup_info = {
                        'type': '交互模式弹窗',
                        'content': f'用户手动点击的{button["text"]}按钮',
                        'button_text': button['text']
                    }
                    
                    # 验证操作是否成功
                    print_with_color("正在验证操作结果...", "yellow")
                    success = verify_popup_removed(after_screenshot, popup_info)
                    
                    if success:
                        print_with_color("✓ 验证成功：弹窗已被成功处理", "green")
                    else:
                        print_with_color("✗ 验证失败：弹窗可能仍然存在，建议检查操作效果", "yellow")
                else:
                    print_with_color("无法获取验证截图", "red")
            else:
                print_with_color(f"点击操作失败: {result}", "red")
        else:
            print_with_color("取消点击操作", "yellow")
    
    except Exception as e:
        print_with_color(f"执行点击操作时出错: {str(e)}", "red")
        traceback.print_exc()


def auto_mode():
    """全自动弹窗检测和关闭模式"""
    print_with_color(" 启动全自动弹窗关闭模式", "green")
    print_with_color("="*60, "yellow")
    
    try:
        # 连接设备
        controller = connect_device_auto()
        if controller is None:
            return
        
        # 获取当前截图
        image_path = controller.get_screenshot("auto_popup_detection", "./temp")
        if image_path == "ERROR":
            print_with_color(" 无法获取屏幕截图", "red")
            return
        
        print_with_color(f" 已获取屏幕截图: {image_path}", "blue")
        
        # 使用图像预处理检测弹窗区域
        popup_region = detect_bright_popup_region(image_path)
        
        if not popup_region:
            print_with_color("ℹ 未检测到明显的弹窗区域", "blue")
            return
        
        # 智能分析弹窗内容
        popup_analysis = analyze_popup_auto(image_path, popup_region)
        
        if not popup_analysis:
            print_with_color("ℹ 未识别到需要处理的弹窗内容", "blue")
            return
        
        # 自动执行操作
        success = execute_auto_action(popup_analysis, controller)
        
        if success:
            print_with_color(" 弹窗已自动处理完成", "green")
        else:
            print_with_color(" 自动处理失败", "red")
    
    except Exception as e:
        print_with_color(f" 自动模式执行失败: {str(e)}", "red")
        traceback.print_exc()


def connect_device_auto() -> Optional[AndroidController]:
    """自动连接设备，无交互"""
    try:
        print_with_color("🔌 正在自动连接Android设备...", "yellow")
        
        device_list = list_all_devices()
        if not device_list:
            print_with_color(" 未找到Android设备", "red")
            return None
        
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color(" 没有在线设备可用", "red")
            return None
        
        # 自动选择第一个可用设备
        selected_device = online_devices[0]
        print_with_color(f" 已连接设备: {selected_device}", "green")
        
        controller = AndroidController(selected_device)
        width, height = controller.get_device_size()
        print_with_color(f" 设备屏幕尺寸: {width}x{height}", "blue")
        
        return controller
        
    except Exception as e:
        print_with_color(f" 设备连接失败: {str(e)}", "red")
        return None


def detect_bright_popup_region(image_path: str) -> Optional[Dict]:
    """检测亮度较高的弹窗区域"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 计算图像整体亮度统计
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        print_with_color(f" 图像亮度统计 - 平均: {avg_brightness:.1f}, 标准差: {std_brightness:.1f}", "blue")
        
        # 动态阈值：找比平均亮度高的区域
        threshold_value = min(int(avg_brightness + std_brightness * 0.8), 200)
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 形态学处理：连接相近区域
        kernel_size = max(10, min(width, height) // 50)  # 根据屏幕大小动态调整
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print_with_color(" 未找到显著的亮区域", "yellow")
            return None
        
        # 筛选合适的轮廓作为弹窗候选
        popup_candidates = []
        total_area = width * height
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / total_area
            
            # 弹窗面积筛选：5%-70%之间
            if 0.05 < area_ratio < 0.7:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 形状筛选：不能太细长，长宽比要合理
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    
                    # 位置筛选：弹窗不应该在屏幕边缘
                    margin = min(width, height) * 0.05
                    if (x > margin and y > margin and 
                        x + w < width - margin and y + h < height - margin):
                        
                        # 计算该区域的平均亮度
                        if w > 0 and h > 0:
                            region_brightness = np.mean(gray[y:y+h, x:x+w])
                            
                            popup_candidates.append({
                                'x': x, 'y': y, 'width': w, 'height': h,
                                'center_x': x + w // 2, 'center_y': y + h // 2,
                                'area_ratio': area_ratio,
                                'brightness': region_brightness,
                                'aspect_ratio': aspect_ratio
                            })
        
        if not popup_candidates:
            print_with_color(" 未找到符合条件的弹窗区域", "yellow")
            return None
        
        # 选择最佳弹窗候选：综合考虑亮度、位置、大小
        best_popup = max(popup_candidates, key=lambda x: (
            x['brightness'] * 0.4 +  # 亮度权重40%
            x['area_ratio'] * 100 * 0.3 +  # 面积权重30%
            (1 - abs(x['center_x'] / width - 0.5)) * 0.3  # 居中度权重30%
        ))
        
        print_with_color(f" 检测到弹窗区域: {best_popup['width']}x{best_popup['height']} @ ({best_popup['x']},{best_popup['y']})", "green")
        print_with_color(f" 面积占比: {best_popup['area_ratio']:.2%}, 亮度: {best_popup['brightness']:.1f}", "cyan")
        
        return best_popup
        
    except Exception as e:
        print_with_color(f" 弹窗区域检测失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def analyze_popup_auto(image_path: str, popup_region: Dict) -> Optional[Dict]:
    """自动分析弹窗内容，专注于弹窗区域"""
    try:
        # 加载配置和模型
        config = load_config()
        
        if config["MODEL"] == "OpenAI":
            model = OpenAIModel(
                base_url=config["OPENAI_BASE_URL"],
                api_key=config["OPENAI_API_KEY"],
                model=config["OPENAI_MODEL"],
                temperature=0.0,
                max_tokens=800
            )
        else:
            model = QwenModel(
                api_key=config["DASHSCOPE_API_KEY"],
                model=config["QWEN_MODEL"]
            )
        
        # 智能分析提示词 - 类似交互模式的全面分析
        smart_prompt = f"""
请作为一个移动应用自动化专家，仔细分析这张移动应用截图，并制定最佳的操作策略。

## 检测到的可能弹窗区域
坐标区域：({popup_region['x']}, {popup_region['y']}) 到 ({popup_region['x'] + popup_region['width']}, {popup_region['y'] + popup_region['height']})
该区域比背景更亮，很可能包含弹窗内容。

## 分析任务
请完整分析整个截图（不仅仅是高亮区域），识别所有的交互元素，然后制定最合适的操作方案：

1. **弹窗内容理解**：
   - 完整读取弹窗中的所有文字内容
   - 判断弹窗的具体用途和类型
   - 分析用户看到这个弹窗时的意图

2. **操作选项识别**：
   - 找到所有可点击的按钮和选项
   - 识别每个按钮的文字和功能
   - 分析每个操作的后果

3. **最佳策略制定**：
   - 基于弹窗内容选择最合适的操作
   - 如果是广告：选择关闭、跳过、取消等
   - 如果是协议：选择同意、接受等
   - 如果是权限：根据合理性选择允许或拒绝
   - 如果是安装提示：根据应用安全性选择安装或取消

## 回答格式
请按以下格式提供详细分析：

弹窗类型：[广告弹窗/用户协议/权限请求/安装确认/通知提示/其他]
弹窗内容：[完整描述弹窗要表达的信息]
检测到的所有按钮：
  - 按钮1：文字"xxx" | 坐标(x1,y1) | 功能描述
  - 按钮2：文字"xxx" | 坐标(x2,y2) | 功能描述
  [如有更多按钮继续列出]

推荐操作：
  目标按钮：[选择的按钮文字]
  操作坐标：(x, y)
  操作方式：[点击/长按/滑动]
  选择理由：[为什么选择这个操作]

备选操作：
  [如果推荐操作失败，可以尝试的其他操作]

注意事项：
- 坐标必须精确，将用于实际的自动化操作
- 优先选择对用户最安全、最合理的操作
- 避免可能导致安装恶意软件或泄露隐私的操作
- 如果不确定，选择保守的取消/关闭操作
"""
        
        print_with_color(" AI正在智能分析弹窗内容...", "yellow")
        success, response = model.get_model_response(smart_prompt, [image_path])
        
        if not success:
            print_with_color(f" AI分析失败: {response}", "red")
            return None
        
        print_with_color(" AI分析结果:", "cyan")
        print_with_color(response, "white")
        
        # 解析AI响应
        analysis_result = parse_auto_response(response)
        
        if analysis_result:
            button_text = analysis_result.get('button_text', '未知操作')
            print_with_color(f" 解析成功 - 类型: {analysis_result['type']}, 操作: {button_text}", "green")
            return analysis_result
        else:
            print_with_color(" 无法解析AI响应", "red")
            return None
        
    except Exception as e:
        print_with_color(f" 自动分析失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def parse_auto_response(response: str) -> Optional[Dict]:
    """解析智能分析的响应"""
    try:
        result = {}
        
        # 提取弹窗类型 - 多种格式兼容
        type_patterns = [
            r'弹窗类型[：:]\s*([^\n]+)',
            r'类型[：:]\s*([^\n]+)', 
            r'弹窗[：:]\s*([^\n]+)',
            r'这是.{0,5}(.*?弹窗|.*?对话框|.*?提示)',
            r'(广告|协议|权限|安装|通知|提示).*?弹窗',
            r'弹窗.*?(广告|协议|权限|安装|通知|提示)'
        ]
        
        for pattern in type_patterns:
            type_match = re.search(pattern, response, re.IGNORECASE)
            if type_match:
                popup_type = type_match.group(1).strip()
                # 清理提取的类型文本
                popup_type = re.sub(r'[【】\[\]""]', '', popup_type)
                result['type'] = popup_type
                print_with_color(f"    找到弹窗类型: {popup_type} (模式: {pattern})", "blue")
                break
        
        # 如果还是没找到，使用智能推断
        if 'type' not in result:
            if any(word in response for word in ['广告', '推广', '营销']):
                result['type'] = '广告弹窗'
            elif any(word in response for word in ['协议', '条款', '政策', '同意']):
                result['type'] = '协议弹窗'
            elif any(word in response for word in ['权限', '访问', '允许']):
                result['type'] = '权限弹窗'
            elif any(word in response for word in ['安装', '下载', '更新']):
                result['type'] = '安装确认'
            elif any(word in response for word in ['通知', '提示', '消息']):
                result['type'] = '通知弹窗'
            else:
                result['type'] = '未知弹窗'
            print_with_color(f"    智能推断弹窗类型: {result['type']}", "yellow")
        
        # 提取弹窗内容 - 多种格式兼容
        content_patterns = [
            r'弹窗内容[：:]\s*([^\n]+)',
            r'内容[：:]\s*([^\n]+)',
            r'描述[：:]\s*([^\n]+)',
            r'说明[：:]\s*([^\n]+)'
        ]
        
        for pattern in content_patterns:
            content_match = re.search(pattern, response)
            if content_match:
                result['content'] = content_match.group(1).strip()
                break
        
        # 如果没有找到内容，使用前100个字符作为内容摘要
        if 'content' not in result:
            clean_response = re.sub(r'\n+', ' ', response).strip()
            result['content'] = clean_response[:100] + ('...' if len(clean_response) > 100 else '')
            print_with_color(f"    使用响应摘要作为内容", "yellow")
        
        # 提取推荐操作信息 - 多种格式兼容
        button_patterns = [
            r'目标按钮[：:]\s*([^\n]+)',
            r'推荐按钮[：:]\s*([^\n]+)',
            r'选择按钮[：:]\s*([^\n]+)',
            r'点击按钮[：:]\s*([^\n]+)',
            r'操作按钮[：:]\s*([^\n]+)',
            r'建议点击[：:]\s*([^\n]+)',
            r'应该点击[：:]\s*([^\n]+)',
            r'点击["""]([^"""]+)["""]',
            r'选择["""]([^"""]+)["""]',
            r'"([^"]*(?:确定|取消|同意|关闭|安装|跳过|允许|拒绝)[^"]*)"'
        ]
        
        for pattern in button_patterns:
            button_match = re.search(pattern, response, re.IGNORECASE)
            if button_match:
                button_text = button_match.group(1).strip()
                button_text = re.sub(r'[【】\[\]"""'']', '', button_text)
                result['button_text'] = button_text
                print_with_color(f"    找到目标按钮: {button_text} (模式: {pattern})", "blue")
                break
        
        # 提取操作坐标 - 多种格式兼容
        coord_patterns = [
            r'操作坐标[：:]\s*\((\d+),\s*(\d+)\)',
            r'坐标[：:]\s*\((\d+),\s*(\d+)\)',
            r'位置[：:]\s*\((\d+),\s*(\d+)\)',
            r'点击坐标[：:]\s*\((\d+),\s*(\d+)\)',
            r'按钮坐标[：:]\s*\((\d+),\s*(\d+)\)',
            r'\((\d+),\s*(\d+)\)'  # 任何坐标格式
        ]
        
        for pattern in coord_patterns:
            coord_match = re.search(pattern, response)
            if coord_match:
                result['x'] = int(coord_match.group(1))
                result['y'] = int(coord_match.group(2))
                print_with_color(f"    找到坐标: ({result['x']}, {result['y']}) (模式: {pattern})", "blue")
                break
        
        # 提取操作方式
        method_match = re.search(r'操作方式[：:]\s*([^\n]+)', response)
        if method_match:
            result['operation_method'] = method_match.group(1).strip().replace('[', '').replace(']', '')
        else:
            result['operation_method'] = '点击'  # 默认为点击
        
        # 提取选择理由
        reason_match = re.search(r'选择理由[：:]\s*([^\n]+)', response)
        if reason_match:
            result['reason'] = reason_match.group(1).strip().replace('[', '').replace(']', '')
        
        # 提取所有检测到的按钮（备用信息）
        buttons_section = re.search(r'检测到的所有按钮[：:](.*?)推荐操作[：:]', response, re.DOTALL)
        if buttons_section:
            buttons_text = buttons_section.group(1)
            result['all_buttons'] = extract_all_buttons(buttons_text)
        
        # 提取备选操作
        fallback_match = re.search(r'备选操作[：:]\s*([^\n]+)', response)
        if fallback_match:
            result['fallback'] = fallback_match.group(1).strip()
        
        # 兼容旧格式 - 如果没有找到新格式，尝试旧格式
        if 'x' not in result or 'button_text' not in result:
            # 尝试旧格式解析
            old_button_match = re.search(r'操作按钮[：:]\s*([^\n]+)', response)
            old_coord_match = re.search(r'按钮坐标[：:]\s*\((\d+),\s*(\d+)\)', response)
            
            if old_button_match:
                result['button_text'] = old_button_match.group(1).strip()
            if old_coord_match:
                result['x'] = int(old_coord_match.group(1))
                result['y'] = int(old_coord_match.group(2))
        
        # 验证必要字段
        if 'type' in result and 'x' in result and 'y' in result and 'button_text' in result:
            print_with_color(f"  解析成功!", "green")
            print_with_color(f"    弹窗类型: {result['type']}", "cyan")
            print_with_color(f"    目标按钮: {result['button_text']}", "cyan")
            print_with_color(f"    操作坐标: ({result['x']}, {result['y']})", "cyan")
            print_with_color(f"    操作方式: {result.get('operation_method', '点击')}", "cyan")
            if 'reason' in result:
                print_with_color(f"    选择理由: {result['reason']}", "cyan")
            return result
        else:
            print_with_color("  响应缺少必要字段，尝试备用解析...", "yellow")
            missing_fields = []
            for field in ['type', 'x', 'y', 'button_text']:
                if field not in result:
                    missing_fields.append(field)
            print_with_color(f"    缺少字段: {missing_fields}", "yellow")
            # 尝试更宽松的解析
            return try_fallback_parsing(response)
        
    except Exception as e:
        print_with_color(f" 解析响应失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def extract_all_buttons(buttons_text: str) -> List[Dict]:
    """从按钮描述文本中提取所有按钮信息"""
    buttons = []
    try:
        # 查找所有按钮行
        button_lines = re.findall(r'-\s*按钮\d+[：:]\s*文字["""]([^"""]+)["""][^|]*\|\s*坐标\((\d+),\s*(\d+)\)', buttons_text)
        
        for text, x, y in button_lines:
            buttons.append({
                'text': text.strip(),
                'x': int(x),
                'y': int(y)
            })
    except Exception as e:
        print_with_color(f" 提取按钮信息失败: {str(e)}", "yellow")
    
    return buttons


def try_fallback_parsing(response: str) -> Optional[Dict]:
    """备用解析方法，尝试从响应中提取任何可用的坐标和操作信息"""
    try:
        print_with_color("🔧 启动备用解析模式...", "yellow")
        result = {}
        
        # 智能提取弹窗类型
        if any(word in response for word in ['广告', '推广', '营销', 'AD', 'ad']):
            result['type'] = '广告弹窗'
        elif any(word in response for word in ['协议', '条款', '政策', '同意', '用户协议']):
            result['type'] = '协议弹窗'
        elif any(word in response for word in ['权限', '访问', '允许', '授权']):
            result['type'] = '权限弹窗'
        elif any(word in response for word in ['安装', '下载', '更新', 'APK']):
            result['type'] = '安装确认'
        elif any(word in response for word in ['通知', '提示', '消息', '对话框']):
            result['type'] = '通知弹窗'
        else:
            result['type'] = '未知弹窗'
        
        print_with_color(f"    类型推断: {result['type']}", "cyan")
        
        # 查找所有坐标
        all_coords = re.findall(r'\((\d+),\s*(\d+)\)', response)
        print_with_color(f"    发现坐标: {all_coords}", "cyan")
        
        # 智能查找按钮文字
        button_patterns = [
            r'["""]([^"""]*(?:确定|取消|同意|关闭|安装|跳过|允许|拒绝|好的|确认)[^"""]*)["""]',
            r'点击.*?["""]([^"""]+)["""]',
            r'选择.*?["""]([^"""]+)["""]',
            r'(确定|取消|同意|关闭|安装|跳过|允许|拒绝|好的|确认)(?:按钮)?',
        ]
        
        button_found = False
        for pattern in button_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                result['button_text'] = matches[0]
                print_with_color(f"    按钮文字: {result['button_text']} (模式: {pattern})", "cyan")
                button_found = True
                break
        
        if not button_found:
            # 最后的尝试：查找常见按钮关键词
            common_buttons = ['确定', '取消', '同意', '关闭', '安装', '跳过', '允许', '拒绝']
            for btn in common_buttons:
                if btn in response:
                    result['button_text'] = btn
                    print_with_color(f"    按钮关键词: {btn}", "cyan")
                    break
        
        # 使用坐标
        if all_coords:
            # 优先使用最后一个坐标（通常是操作坐标）
            result['x'] = int(all_coords[-1][0])
            result['y'] = int(all_coords[-1][1])
            print_with_color(f"    选择坐标: ({result['x']}, {result['y']})", "cyan")
        
        # 设置默认值
        result['operation_method'] = '点击'
        result['content'] = response[:100] + ('...' if len(response) > 100 else '')
        
        # 验证是否有足够信息
        if 'x' in result and 'button_text' in result:
            print_with_color(f" 备用解析成功!", "green")
            print_with_color(f"    操作: {result['button_text']} @ ({result['x']},{result['y']})", "green")
            return result
        else:
            missing = []
            if 'x' not in result: missing.append('坐标')
            if 'button_text' not in result: missing.append('按钮文字')
            print_with_color(f" 备用解析失败，缺少: {missing}", "red")
            print_with_color(" 建议检查AI响应格式是否包含坐标和按钮信息", "yellow")
            return None
            
    except Exception as e:
        print_with_color(f" 备用解析异常: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        return None


def execute_auto_action(popup_analysis: Dict, controller: AndroidController) -> bool:
    """自动执行弹窗操作并验证结果"""
    try:
        popup_type = popup_analysis.get('type', '')
        button_text = popup_analysis.get('button_text', '未知按钮')
        operation_method = popup_analysis.get('operation_method', '点击')
        reason = popup_analysis.get('reason', '')
        x = popup_analysis.get('x')
        y = popup_analysis.get('y')
        
        if x is None or y is None:
            print_with_color(" 坐标信息缺失", "red")
            return False
        
        print_with_color(f" 智能执行操作", "cyan")
        print_with_color(f"   弹窗类型: {popup_type}", "white")
        print_with_color(f"   目标按钮: {button_text}", "white")
        print_with_color(f"   操作方式: {operation_method}", "white")
        print_with_color(f"   操作坐标: ({x}, {y})", "white")
        if reason:
            print_with_color(f"   选择理由: {reason}", "white")
        
        # 根据操作方式执行不同的操作
        if '长按' in operation_method:
            print_with_color(" 执行长按操作...", "yellow")
            result = controller.long_press(x, y, 1500)  # 长按1.5秒
        elif '滑动' in operation_method:
            print_with_color(" 执行滑动操作...", "yellow")
            # 默认向右滑动
            result = controller.swipe(x, y, "right", "medium")
        else:
            # 默认点击操作
            print_with_color(" 执行点击操作...", "yellow")
            result = controller.tap(x, y)
        
        if result != "SUCCESS":
            print_with_color(f" 点击操作失败: {result}", "red")
            return False
        
        print_with_color(f" 成功执行点击操作", "green")
        
        # 等待界面响应
        print_with_color(" 等待界面更新...", "yellow")
        time.sleep(2)  # 给界面足够时间来响应
        
        # 获取操作后的截图
        print_with_color(" 获取操作后截图以验证结果...", "yellow")
        after_image = controller.get_screenshot("after_popup_action", "./temp")
        
        if after_image == "ERROR":
            print_with_color(" 无法获取操作后截图，无法验证结果", "red")
            return False
        
        # 使用AI验证操作是否成功
        success = verify_popup_removed(after_image, popup_analysis)
        
        if success:
            print_with_color(" ✓ 验证成功：弹窗已被成功处理", "green")
            return True
        else:
            print_with_color(" ✗ 验证失败：弹窗可能仍然存在", "red")
            return False
        
    except Exception as e:
        print_with_color(f" 执行自动操作失败: {str(e)}", "red")
        traceback.print_exc()
        return False


def verify_popup_removed(after_image_path: str, original_popup_analysis: Dict) -> bool:
    """验证弹窗是否已被成功移除"""
    try:
        # 加载配置和模型
        config = load_config()
        
        if config["MODEL"] == "OpenAI":
            model = OpenAIModel(
                base_url=config["OPENAI_BASE_URL"],
                api_key=config["OPENAI_API_KEY"],
                model=config["OPENAI_MODEL"],
                temperature=0.0,
                max_tokens=500
            )
        else:
            model = QwenModel(
                api_key=config["DASHSCOPE_API_KEY"],
                model=config["QWEN_MODEL"]
            )
        
        # 构建验证提示词
        original_type = original_popup_analysis.get('type', '未知')
        original_content = original_popup_analysis.get('content', '未知内容')
        original_button = original_popup_analysis.get('button_text', '未知按钮')
        
        verification_prompt = f"""
请分析这张移动应用截图，判断之前的弹窗是否已经被成功关闭。

背景信息：
- 之前检测到的弹窗类型：{original_type}
- 之前的弹窗内容：{original_content}  
- 刚才点击的按钮：{original_button}

请仔细观察当前截图，判断：
1. 是否还能看到之前的弹窗？
2. 界面是否已经回到正常状态？
3. 是否有新的弹窗出现？

请用以下格式简洁回答：

弹窗状态：[已移除/仍存在/出现新弹窗]
验证结果：[成功/失败]
当前界面：[简要描述当前界面状态]
备注：[如果失败，说明原因]

判断标准：
- 如果之前的弹窗完全消失，且界面回到正常应用界面，则为"成功"
- 如果之前的弹窗仍然存在，或出现了新的弹窗，则为"失败"
- 如果界面发生了明显变化但不确定，请在备注中说明

请确保判断准确，这关系到自动化操作的成功率。
"""
        
        print_with_color(" AI正在验证操作结果...", "yellow")
        success, response = model.get_model_response(verification_prompt, [after_image_path])
        
        if not success:
            print_with_color(f" AI验证失败: {response}", "red")
            return False
        
        print_with_color(" AI验证结果:", "cyan")
        print_with_color(response, "white")
        
        # 解析验证结果
        verification_result = parse_verification_response(response)
        
        return verification_result
        
    except Exception as e:
        print_with_color(f" 验证弹窗移除失败: {str(e)}", "red")
        traceback.print_exc()
        return False


def parse_verification_response(response: str) -> bool:
    """解析验证响应，返回是否成功"""
    try:
        # 查找验证结果
        result_match = re.search(r'验证结果[：:]\s*([^\n]+)', response)
        if result_match:
            result = result_match.group(1).strip()
            print_with_color(f" 解析验证结果: {result}", "blue")
            
            # 判断是否成功
            if "成功" in result:
                return True
            elif "失败" in result:
                return False
        
        # 备用判断：查找弹窗状态
        status_match = re.search(r'弹窗状态[：:]\s*([^\n]+)', response)
        if status_match:
            status = status_match.group(1).strip()
            print_with_color(f" 弹窗状态: {status}", "blue")
            
            if "已移除" in status:
                return True
            elif "仍存在" in status or "新弹窗" in status:
                return False
        
        # 如果无法明确解析，采用保守策略
        print_with_color(" 无法明确解析验证结果，采用保守判断", "yellow")
        
        # 关键词判断
        if "成功" in response or "已移除" in response or "消失" in response:
            if "失败" not in response and "仍存在" not in response and "新弹窗" not in response:
                return True
        
        return False
        
    except Exception as e:
        print_with_color(f" 解析验证响应失败: {str(e)}", "red")
        return False


if __name__ == "__main__":
    main() 