#!/usr/bin/env python3
"""
弹窗内容分析脚本
直接分析用户提供的截图，理解弹窗内容并给出操作建议
"""

import os
import sys
import traceback

# 添加当前目录到Python路径
sys.path.append('.')

from scripts.config import load_config
from scripts.model import QwenModel, OpenAIModel
from scripts.utils import print_with_color


def analyze_popup_content(image_path: str):
    """分析弹窗内容"""
    print_with_color("弹窗内容分析工具", "cyan")
    print_with_color("="*60, "yellow")
    
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
        
        # 询问是否需要进一步的按钮定位分析
        print_with_color("\n是否需要进一步分析按钮位置？(y/n)", "yellow")
        if input().strip().lower() == 'y':
            analyze_button_positions(model, image_path, response)
        
    except Exception as e:
        print_with_color(f"分析过程中发生错误: {str(e)}", "red")
        traceback.print_exc()


def analyze_button_positions(model, image_path: str, previous_analysis: str):
    """分析按钮位置"""
    print_with_color("\n分析按钮位置...", "cyan")
    
    # 根据之前的分析结果决定查找什么按钮
    if "协议" in previous_analysis or "同意" in previous_analysis:
        button_type = "同意"
        button_prompt = """
基于之前的分析，这是一个协议类弹窗。请在截图中找到"同意"、"确认"、"我知道了"等类似的按钮。

请提供：
1. 按钮上的确切文字
2. 按钮的中心坐标 (x, y)
3. 按钮的大致位置描述

格式：
按钮文字：[具体文字]
坐标：(x, y)
位置：[位置描述，如"屏幕下方中央"]
"""
    else:
        button_type = "关闭"
        button_prompt = """
基于之前的分析，这是一个需要关闭的弹窗。请在截图中找到"关闭"、"×"、"跳过"等类似的按钮。

请提供：
1. 按钮上的确切文字或符号
2. 按钮的中心坐标 (x, y)  
3. 按钮的大致位置描述

格式：
按钮文字：[具体文字或符号]
坐标：(x, y)
位置：[位置描述，如"右上角"]
"""
    
    success, button_response = model.get_model_response(button_prompt, [image_path])
    
    if success:
        print_with_color(f"\n{button_type}按钮定位结果:", "green")
        print_with_color(button_response, "white")
    else:
        print_with_color("按钮定位失败", "red")


def main():
    """主函数"""
    # 默认使用用户提供的图片
    default_image = "page/_20250715_163040.png"
    
    print_with_color("弹窗内容分析工具", "cyan")
    print_with_color(f"默认分析图片: {default_image}", "yellow")
    
    # 询问是否使用其他图片
    custom_path = input("使用其他图片路径？(直接按Enter使用默认图片): ").strip()
    
    if custom_path:
        image_path = custom_path
    else:
        image_path = default_image
    
    analyze_popup_content(image_path)


if __name__ == "__main__":
    main() 