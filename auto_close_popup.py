#!/usr/bin/env python3
"""
全自动弹窗广告关闭工具
专注识别亮度较高的弹窗广告并自动关闭，忽略背景干扰
"""

import os
import sys
import cv2
import numpy as np
import traceback
import time
import re
from typing import Dict, Tuple, Optional, List

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel
from scripts.utils import print_with_color
from scripts.and_controller import AndroidController, list_all_devices


def auto_close_popup(image_path: str = None, controller: AndroidController = None):
    """自动分析并关闭弹窗广告"""
    print_with_color(" 启动全自动弹窗广告关闭功能", "cyan")
    
    try:
        # 如果没有提供截图路径，从手机获取当前屏幕
        if image_path is None:
            if controller is None:
                controller = connect_device()
                if controller is None:
                    return False
            
            print_with_color(" 获取当前屏幕截图...", "yellow")
            image_path = controller.get_screenshot("popup_detection", "./temp")
            if image_path == "ERROR":
                print_with_color(" 无法获取屏幕截图", "red")
                return False
        
        # 预处理图像，识别弹窗区域
        popup_region = detect_popup_region(image_path)
        
        # 分析弹窗内容和类型
        popup_analysis = analyze_popup_smart(image_path, popup_region)
        
        if not popup_analysis:
            print_with_color("ℹ未检测到需要处理的弹窗", "blue")
            return True
        
        print_with_color(f"检测到{popup_analysis['type']}弹窗", "green")
        print_with_color(f"内容: {popup_analysis['summary']}", "white")
        
        # 自动执行关闭操作
        if popup_analysis['type'] in ['广告弹窗', '推广弹窗', '营销弹窗']:
            success = auto_close_ad_popup(popup_analysis, controller)
        elif popup_analysis['type'] in ['协议弹窗', '权限弹窗']:
            success = auto_accept_agreement(popup_analysis, controller)
        else:
            success = auto_handle_generic_popup(popup_analysis, controller)
        
        if success:
            print_with_color("弹窗处理成功", "green")
            
            # 等待界面更新
            time.sleep(1)
            
            # 验证弹窗是否已关闭
            if controller:
                verify_popup_closed(controller)
        else:
            print_with_color("弹窗处理失败", "red")
        
        return success
        
    except Exception as e:
        print_with_color(f"自动关闭弹窗时出错: {str(e)}", "red")
        traceback.print_exc()
        return False


def connect_device() -> Optional[AndroidController]:
    """连接Android设备"""
    try:
        print_with_color("正在连接Android设备...", "yellow")
        
        device_list = list_all_devices()
        if not device_list:
            print_with_color("未找到Android设备", "red")
            return None
        
        online_devices = [d for d in device_list if "offline" not in d and "unauthorized" not in d]
        if not online_devices:
            print_with_color("没有在线设备可用", "red")
            return None
        
        # 自动选择第一个可用设备
        selected_device = online_devices[0]
        print_with_color(f"已连接设备: {selected_device}", "green")
        
        controller = AndroidController(selected_device)
        width, height = controller.get_device_size()
        print_with_color(f"设备屏幕尺寸: {width}x{height}", "blue")
        
        return controller
        
    except Exception as e:
        print_with_color(f"连接设备失败: {str(e)}", "red")
        return None


def detect_popup_region(image_path: str) -> Optional[Dict]:
    """检测弹窗区域，基于亮度和对比度"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 计算图像的平均亮度
        avg_brightness = np.mean(gray)
        
        # 使用自适应阈值找到亮区域
        # 弹窗通常比背景亮
        threshold_value = min(avg_brightness + 30, 200)
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 形态学操作，连接相近的亮区域
        kernel = np.ones((20, 20), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        # 找到轮廓
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的亮区域（可能是弹窗）
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 弹窗面积应该占屏幕的一定比例，但不能太大
        total_area = width * height
        area_ratio = area / total_area
        
        if 0.1 < area_ratio < 0.8:  # 弹窗面积在10%-80%之间
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 确保弹窗不是全屏
            if w < width * 0.95 and h < height * 0.95:
                popup_region = {
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'center_x': x + w // 2, 'center_y': y + h // 2,
                    'area_ratio': area_ratio,
                    'avg_brightness': np.mean(gray[y:y+h, x:x+w]) if h > 0 and w > 0 else 0
                }
                
                print_with_color(f"检测到可能的弹窗区域: {w}x{h} @ ({x},{y})", "cyan")
                print_with_color(f"面积占比: {area_ratio:.2%}, 亮度: {popup_region['avg_brightness']:.1f}", "blue")
                
                return popup_region
        
        return None
        
    except Exception as e:
        print_with_color(f"弹窗区域检测失败: {str(e)}", "red")
        return None


def analyze_popup_smart(image_path: str, popup_region: Optional[Dict]) -> Optional[Dict]:
    """智能分析弹窗内容，专注于弹窗区域"""
    try:
        # 加载配置和模型
        config = load_config()
        
        if config["MODEL"] == "OpenAI":
            model = OpenAIModel(
                base_url=config["OPENAI_BASE_URL"],
                api_key=config["OPENAI_API_KEY"],
                model=config["OPENAI_MODEL"],
                temperature=0.0,
                max_tokens=1000
            )
        else:
            model = QwenModel(
                api_key=config["DASHSCOPE_API_KEY"],
                model=config["QWEN_MODEL"]
            )
        
        # 构建智能分析提示词
        region_hint = ""
        if popup_region:
            region_hint = f"""
重点关注坐标区域 ({popup_region['x']}, {popup_region['y']}) 到 ({popup_region['x'] + popup_region['width']}, {popup_region['y'] + popup_region['height']}) 的内容，这个区域比背景更亮，很可能是弹窗。
"""
        
        smart_prompt = f"""
你是一个专业的移动应用弹窗识别专家。请分析这个移动应用截图中的弹窗内容。

{region_hint}

请忽略背景中的其他界面元素，专注于识别弹窗内容。弹窗通常具有以下特征：
- 比背景更亮的矩形区域
- 有明显的边框或阴影
- 包含按钮和文字
- 覆盖在其他内容之上

分析重点：
1. **弹窗类型识别**：
   - 广告弹窗：包含推广、下载、体验等营销内容
   - 协议弹窗：用户协议、隐私政策、服务条款
   - 权限弹窗：请求权限访问
   - 通知弹窗：一般信息提示

2. **关闭按钮定位**：
   - 找到"关闭"、"×"、"跳过"、"稍后再说"等关闭按钮
   - 或"同意"、"确认"等同意按钮（针对协议弹窗）

请按以下JSON格式回答：
{{
    "has_popup": true/false,
    "type": "广告弹窗/协议弹窗/权限弹窗/通知弹窗/其他",
    "summary": "弹窗内容简要描述",
    "action_button": {{
        "text": "按钮文字",
        "type": "关闭/同意/取消",
        "x": 坐标x,
        "y": 坐标y,
        "confidence": 0.0-1.0
    }},
    "avoid_buttons": [
        {{"text": "按钮文字", "x": 坐标x, "y": 坐标y}}
    ]
}}

如果没有检测到明显的弹窗，请返回 {{"has_popup": false}}
"""
        
        print_with_color("AI正在分析弹窗内容...", "yellow")
        success, response = model.get_model_response(smart_prompt, [image_path])
        
        if not success:
            print_with_color(f"AI分析失败: {response}", "red")
            return None
        
        # 解析JSON响应
        try:
            import json
            # 提取JSON部分
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis_result = json.loads(json_str)
                
                if not analysis_result.get('has_popup', False):
                    return None
                
                print_with_color("AI分析结果:", "cyan")
                print_with_color(f"   类型: {analysis_result['type']}", "white")
                print_with_color(f"   内容: {analysis_result['summary']}", "white")
                if 'action_button' in analysis_result:
                    btn = analysis_result['action_button']
                    print_with_color(f"   建议操作: {btn['text']} @ ({btn['x']}, {btn['y']})", "green")
                
                return analysis_result
            else:
                print_with_color("无法解析AI响应的JSON格式", "red")
                return None
                
        except json.JSONDecodeError as e:
            print_with_color(f"JSON解析失败: {str(e)}", "red")
            print_with_color(f"原始响应: {response[:200]}...", "yellow")
            return None
        
    except Exception as e:
        print_with_color(f"智能分析失败: {str(e)}", "red")
        traceback.print_exc()
        return None


def auto_close_ad_popup(popup_analysis: Dict, controller: AndroidController) -> bool:
    """自动关闭广告弹窗并验证结果"""
    try:
        print_with_color("执行广告弹窗关闭操作", "cyan")
        
        action_button = popup_analysis.get('action_button')
        if not action_button or 'x' not in action_button or 'y' not in action_button:
            print_with_color("未找到有效的关闭按钮", "red")
            return False
        
        x, y = action_button['x'], action_button['y']
        button_text = action_button.get('text', '未知按钮')
        
        print_with_color(f"点击关闭按钮: {button_text} @ ({x}, {y})", "green")
        
        # 执行点击
        result = controller.tap(x, y)
        
        if result != "SUCCESS":
            print_with_color(f"点击操作失败: {result}", "red")
            return False
        
        print_with_color("点击操作成功，正在验证结果...", "green")
        
        # 等待界面响应
        time.sleep(2)
        
        # 获取操作后截图进行验证
        after_image = controller.get_screenshot("after_close_ad", "./temp")
        if after_image == "ERROR":
            print_with_color("无法获取验证截图", "red")
            return False
        
        # 验证弹窗是否真的被关闭
        success = verify_popup_closed(after_image, popup_analysis, "广告弹窗关闭")
        
        if success:
            print_with_color("✓ 验证成功：广告弹窗已被关闭", "green")
            return True
        else:
            print_with_color("✗ 验证失败：广告弹窗可能仍然存在", "red")
            return False
        
    except Exception as e:
        print_with_color(f"关闭广告弹窗失败: {str(e)}", "red")
        traceback.print_exc()
        return False


def auto_accept_agreement(popup_analysis: Dict, controller: AndroidController) -> bool:
    """自动接受协议弹窗并验证结果"""
    try:
        print_with_color("执行协议弹窗同意操作", "cyan")
        
        action_button = popup_analysis.get('action_button')
        if not action_button or 'x' not in action_button or 'y' not in action_button:
            print_with_color("未找到有效的同意按钮", "red")
            return False
        
        x, y = action_button['x'], action_button['y']
        button_text = action_button.get('text', '未知按钮')
        
        print_with_color(f"点击同意按钮: {button_text} @ ({x}, {y})", "green")
        
        # 执行点击
        result = controller.tap(x, y)
        
        if result != "SUCCESS":
            print_with_color(f"点击操作失败: {result}", "red")
            return False
        
        print_with_color("点击操作成功，正在验证结果...", "green")
        
        # 等待界面响应
        time.sleep(2)
        
        # 获取操作后截图进行验证
        after_image = controller.get_screenshot("after_accept_agreement", "./temp")
        if after_image == "ERROR":
            print_with_color("无法获取验证截图", "red")
            return False
        
        # 验证弹窗是否真的被处理
        success = verify_popup_closed(after_image, popup_analysis, "协议弹窗同意")
        
        if success:
            print_with_color("✓ 验证成功：协议弹窗已被处理", "green")
            return True
        else:
            print_with_color("✗ 验证失败：协议弹窗可能仍然存在", "red")
            return False
        
    except Exception as e:
        print_with_color(f"接受协议失败: {str(e)}", "red")
        traceback.print_exc()
        return False


def auto_handle_generic_popup(popup_analysis: Dict, controller: AndroidController) -> bool:
    """处理通用弹窗"""
    try:
        print_with_color("执行通用弹窗处理操作", "cyan")
        
        action_button = popup_analysis.get('action_button')
        if not action_button or 'x' not in action_button or 'y' not in action_button:
            print_with_color("未找到有效的操作按钮", "red")
            return False
        
        x, y = action_button['x'], action_button['y']
        button_text = action_button.get('text', '未知按钮')
        
        print_with_color(f"点击按钮: {button_text} @ ({x}, {y})", "green")
        
        # 执行点击
        result = controller.tap(x, y)
        
        if result == "SUCCESS":
            print_with_color("按钮点击成功", "green")
            return True
        else:
            print_with_color(f"点击失败: {result}", "red")
            return False
        
    except Exception as e:
        print_with_color(f"处理通用弹窗失败: {str(e)}", "red")
        return False


def verify_popup_closed(controller: AndroidController):
    """验证弹窗是否已关闭"""
    try:
        print_with_color("验证弹窗关闭状态...", "yellow")
        
        # 获取操作后的截图
        after_screenshot = controller.get_screenshot("after_popup_close", "./temp")
        if after_screenshot != "ERROR":
            print_with_color(f"操作后截图已保存: {after_screenshot}", "blue")
            
            # 简单检测：看是否还有明显的弹窗区域
            popup_region = detect_popup_region(after_screenshot)
            if popup_region:
                print_with_color("检测到仍有弹窗存在，可能需要进一步处理", "yellow")
            else:
                print_with_color(" 弹窗已成功关闭", "green")
        
    except Exception as e:
        print_with_color(f"验证弹窗状态时出错: {str(e)}", "yellow")


def verify_popup_closed(after_image_path: str, original_popup_analysis: Dict, operation_type: str) -> bool:
    """验证弹窗是否已被成功关闭（专用于auto_close_popup.py）"""
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
        popup_type = original_popup_analysis.get('type', '未知')
        action_button = original_popup_analysis.get('action_button', {})
        button_text = action_button.get('text', '未知按钮')
        
        verification_prompt = f"""
请分析这张移动应用截图，判断弹窗操作是否成功。

操作背景：
- 弹窗类型：{popup_type}
- 执行的操作：{operation_type}
- 点击的按钮：{button_text}

请仔细观察当前截图，判断：
1. 之前的弹窗是否已经消失？
2. 界面是否已经回到正常的应用界面？
3. 是否有新的弹窗出现？

请用以下格式简洁回答：

弹窗状态：[已消失/仍存在/出现新弹窗]
操作结果：[成功/失败]
界面状态：[简要描述当前界面]
备注：[如果失败，说明可能的原因]

判断标准：
- 如果原弹窗完全消失且界面正常，则为"成功"
- 如果原弹窗仍然存在或出现新弹窗，则为"失败"
"""
        
        print_with_color("AI正在验证操作结果...", "yellow")
        success, response = model.get_model_response(verification_prompt, [after_image_path])
        
        if not success:
            print_with_color(f"AI验证失败: {response}", "red")
            return False
        
        print_with_color("AI验证结果:", "cyan")
        print_with_color(response, "white")
        
        # 解析验证结果
        verification_result = parse_verification_result(response)
        
        return verification_result
        
    except Exception as e:
        print_with_color(f"验证弹窗关闭失败: {str(e)}", "red")
        traceback.print_exc()
        return False


def parse_verification_result(response: str) -> bool:
    """解析验证响应，返回是否成功"""
    try:
        # 查找操作结果
        result_match = re.search(r'操作结果[：:]\s*([^\n]+)', response)
        if result_match:
            result = result_match.group(1).strip()
            print_with_color(f"解析操作结果: {result}", "blue")
            
            if "成功" in result:
                return True
            elif "失败" in result:
                return False
        
        # 备用判断：查找弹窗状态
        status_match = re.search(r'弹窗状态[：:]\s*([^\n]+)', response)
        if status_match:
            status = status_match.group(1).strip()
            print_with_color(f"弹窗状态: {status}", "blue")
            
            if "已消失" in status:
                return True
            elif "仍存在" in status or "新弹窗" in status:
                return False
        
        # 关键词判断
        if "成功" in response or "已消失" in response or "消失" in response:
            if "失败" not in response and "仍存在" not in response and "新弹窗" not in response:
                return True
        
        return False
        
    except Exception as e:
        print_with_color(f"解析验证响应失败: {str(e)}", "red")
        return False


def main():
    """主函数"""
    print_with_color("全自动弹窗广告关闭工具", "cyan")
    print_with_color("="*60, "yellow")
    
    # 确保临时目录存在
    os.makedirs("./temp", exist_ok=True)
    
    try:
        # 连接设备
        controller = connect_device()
        if controller is None:
            print_with_color(" 无法连接设备，程序退出", "red")
            return
        
        # 自动执行弹窗关闭
        success = auto_close_popup(controller=controller)
        
        if success:
            print_with_color(" 弹窗处理完成", "green")
        else:
            print_with_color(" 弹窗处理结束", "yellow")
    
    except KeyboardInterrupt:
        print_with_color("\n 用户中断，程序退出", "yellow")
    except Exception as e:
        print_with_color(f"程序执行出错: {str(e)}", "red")
        traceback.print_exc()


if __name__ == "__main__":
    main() 