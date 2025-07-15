import cv2
import numpy as np
import re
import base64
from typing import Dict, List, Any, Optional, Tuple
import time
import json
import os

from scripts.base_agent import BaseAgent, AgentTask, AgentResult
from scripts.utils import print_with_color
import scripts.prompts as prompts
from scripts.captcha_prompts import (
    captcha_detection_prompt,
    text_captcha_prompt,
    slider_captcha_prompt,
    click_captcha_prompt,
    math_captcha_prompt,
    general_captcha_prompt,
    captcha_confirmation_prompt
)


class CaptchaAgent(BaseAgent):
    """验证码处理专业化Agent"""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__("captcha_agent", config_path)
        self.capabilities = [
            "text_captcha_recognition",      # 文本验证码识别
            "image_captcha_solving",         # 图像验证码解决
            "slider_captcha_solving",        # 滑块验证码解决
            "click_captcha_solving",         # 点击验证码解决
            "math_captcha_solving",          # 数学验证码解决
            "pattern_recognition"            # 模式识别
        ]
        
        # 验证码相关的知识库初始化
        if "captcha_patterns" not in self.knowledge_base:
            self.knowledge_base["captcha_patterns"] = {
                "text_patterns": [],
                "image_patterns": [],
                "slider_patterns": [],
                "success_indicators": []
            }
    
    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否为验证码相关任务"""
        keywords = ["验证码", "captcha", "验证", "识别", "滑块", "拖拽", "点击验证"]
        task_text = (task.description + " " + task.app_name).lower()
        return any(keyword in task_text for keyword in keywords)
    
    def execute_task(self, task: AgentTask) -> AgentResult:
        """执行验证码解决任务"""
        print_with_color(f"验证码Agent开始处理任务: {task.description}", "cyan")
        
        if not self.controller:
            return AgentResult(
                success=False,
                message="Android控制器未初始化",
                actions_taken=[]
            )
        
        try:
            # 获取当前屏幕截图
            screenshot_path = self.controller.get_screenshot("captcha_task", "./temp")
            if screenshot_path == "ERROR":
                return AgentResult(
                    success=False,
                    message="无法获取屏幕截图",
                    actions_taken=[]
                )
            
            # 分析验证码类型
            captcha_type = self._detect_captcha_type(screenshot_path)
            print_with_color(f"检测到验证码类型: {captcha_type}", "yellow")
            
            # 根据类型选择处理策略
            if captcha_type == "text":
                result = self._solve_text_captcha(screenshot_path)
            elif captcha_type == "slider":
                result = self._solve_slider_captcha(screenshot_path)
            elif captcha_type == "click":
                result = self._solve_click_captcha(screenshot_path)
            elif captcha_type == "math":
                result = self._solve_math_captcha(screenshot_path)
            else:
                result = self._solve_general_captcha(screenshot_path, task.description)
            
            # 验证结果确认
            if result.success:
                confirmation_result = self._confirm_captcha_result(screenshot_path)
                if not confirmation_result:
                    result.success = False
                    result.message = "验证码提交后未通过验证"
            
            return result
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"验证码处理过程中发生错误: {str(e)}",
                actions_taken=[]
            )
    
    def _detect_captcha_type(self, screenshot_path: str) -> str:
        """检测验证码类型"""
        # 使用优化后的提示词分析验证码类型
        prompt = captcha_detection_prompt
        if not self.model:
            return "general"
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if success:
            response_lower = response.lower()
            if "滑块" in response or "slider" in response_lower:
                return "slider"
            elif "点击" in response or "click" in response_lower:
                return "click"
            elif "数学" in response or "math" in response_lower or "计算" in response:
                return "math"
            elif "文字" in response or "text" in response_lower or "字符" in response:
                return "text"
        
        return "general"
    
    def _solve_text_captcha(self, screenshot_path: str) -> AgentResult:
        """解决文本验证码"""
        print_with_color("开始处理文本验证码...", "cyan")
        
        # 使用优化后的文本验证码提示词
        prompt = text_captcha_prompt
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        # 提取识别的文本
        captcha_text = self._extract_captcha_text(response)
        if not captcha_text:
            return AgentResult(False, "无法识别验证码文本", actions_taken=[])
        
        print_with_color(f"识别到验证码文本: {captcha_text}", "green")
        
        # 查找输入框并输入文本
        input_result = self._input_captcha_text(captcha_text)
        
        return AgentResult(
            success=input_result,
            message=f"验证码文本识别并输入: {captcha_text}" if input_result else "验证码输入失败",
            data={"captcha_text": captcha_text},
            actions_taken=["text_recognition", "text_input"]
        )
    
    def _solve_slider_captcha(self, screenshot_path: str) -> AgentResult:
        """解决滑块验证码"""
        print_with_color("开始处理滑块验证码...", "cyan")
        
        # 让LLM分析并描述验证码内容
        prompt = slider_captcha_prompt
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        print_with_color("=" * 60, "yellow")
        print_with_color("LLM对滑块验证码的分析结果:", "cyan")
        print_with_color("=" * 60, "yellow")
        print_with_color(response, "white")
        print_with_color("=" * 60, "yellow")
        
        # 解析LLM的响应，提取关键信息
        analysis_result = self._parse_slider_analysis(response)
        
        print_with_color("\n解析后的关键信息:", "cyan")
        for key, value in analysis_result.items():
            print_with_color(f"{key}: {value}", "white")
        
        # 询问用户是否基于这个分析执行操作
        print_with_color("\n基于以上分析，是否继续执行滑块操作？", "yellow")
        print_with_color("输入 'y' 继续自动操作，'m' 手动指定坐标，其他键跳过:", "white")
        user_input = input().strip().lower()
        
        if user_input == 'y':
            # 尝试基于分析结果自动执行
            success = self._execute_auto_slider_operation(analysis_result, screenshot_path)
            action_type = "auto_operation"
        elif user_input == 'm':
            # 手动指定坐标
            success = self._execute_manual_slider_operation()
            action_type = "manual_operation"
        else:
            print_with_color("跳过操作执行", "yellow")
            success = False
            action_type = "skipped"
        
        return AgentResult(
            success=success,
            message=f"滑块验证码分析完成，操作类型: {action_type}",
            data={
                "llm_analysis": response,
                "parsed_analysis": analysis_result,
                "action_type": action_type
            },
            actions_taken=["slider_analysis", action_type]
        )
    
    def _parse_slider_analysis(self, response: str) -> Dict[str, str]:
        """解析LLM的滑块验证码分析结果"""
        analysis = {}
        
        # 定义要提取的字段（更新为新的格式）
        fields = [
            "验证码类型",
            "主背景图描述", 
            "被拖动图片",
            "滑块按钮位置",
            "缺口位置",
            "操作说明"
        ]
        
        for field in fields:
            pattern = rf"{field}[：:]\s*(.+?)(?:\n\n|\n(?=[验主被滑缺操])|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                analysis[field] = match.group(1).strip()
            else:
                analysis[field] = "未找到相关描述"
        
        return analysis
    
    def _execute_auto_slider_operation(self, analysis: Dict[str, str], screenshot_path: str) -> bool:
        """基于分析结果尝试自动执行滑块操作"""
        if not self.controller:
            print_with_color("控制器未初始化", "red")
            return False
        
        try:
            # 基于简单的启发式规则确定坐标
            print_with_color("尝试基于分析结果自动确定操作坐标...", "cyan")
            
            # 加载图像获取尺寸
            import cv2
            image = cv2.imread(screenshot_path)
            if image is None:
                print_with_color("无法加载截图", "red")
                return False
            
            height, width = image.shape[:2]
            print_with_color(f"图像尺寸: {width}x{height}", "blue")
            
            # 解析LLM的描述
            slider_desc = analysis.get("滑块按钮位置", "").lower()
            gap_desc = analysis.get("缺口位置", "").lower()
            dragged_img_desc = analysis.get("被拖动图片", "").lower()
            
            print_with_color(f"滑块按钮描述: {slider_desc}", "cyan")
            print_with_color(f"被拖动图片描述: {dragged_img_desc}", "cyan")
            print_with_color(f"缺口描述: {gap_desc}", "cyan")
            
            # 滑块位置估算 - 使用智能解析
            slider_x, slider_y = self._parse_slider_position(slider_desc, dragged_img_desc, width, height)
            
            # 缺口位置估算 - 改进解析逻辑
            gap_x, gap_y = self._parse_gap_position(gap_desc, width, height)
            
            slider_pos = (slider_x, slider_y)
            gap_pos = (gap_x, gap_y)
            
            print_with_color(f"估算的滑块按钮位置: {slider_pos}", "green")
            print_with_color(f"估算的缺口位置: {gap_pos}", "green")
            
            # 执行滑动操作
            print_with_color("执行滑动操作...", "cyan")
            
            # 添加小延迟，模拟人类思考时间
            import time
            import random
            time.sleep(random.uniform(0.5, 1.0))
            
            # 使用人性化滑动方法
            print_with_color(f"开始人性化滑动: 从 {slider_pos} 到 {gap_pos}", "green")
            result = self.controller.swipe_human_like(slider_pos, gap_pos, total_duration=3000)
            
            if result == "SUCCESS":
                print_with_color("滑动操作完成", "green")
                return True
            else:
                print_with_color("滑动操作可能失败，尝试备用方法", "yellow")
                # 备用：使用普通的慢速滑动
                self.controller.swipe_precise(slider_pos, gap_pos, duration=2500)
                return True
            
        except Exception as e:
            print_with_color(f"自动操作执行失败: {e}", "red")
            return False
    
    def _parse_slider_position(self, slider_desc: str, dragged_img_desc: str, width: int, height: int) -> tuple:
        """
        智能解析滑块按钮位置描述，返回更准确的坐标
        """
        # 默认位置：基于滑块通常在左下的规律
        slider_x = width // 6  # 默认左侧1/6位置
        slider_y = height * 3 // 4  # 默认下方3/4位置
        
        # 解析滑块按钮的相对位置
        if "正下方" in slider_desc:
            # 滑块在被拖动图片正下方，根据被拖动图片位置推断
            if "左侧" in dragged_img_desc or "左边" in dragged_img_desc:
                slider_x = width // 6  # 左侧1/6
            elif "中间" in dragged_img_desc or "中央" in dragged_img_desc:
                slider_x = width // 2  # 中间
            else:
                slider_x = width // 6  # 默认左侧
            slider_y = height * 3 // 4  # 下方位置
            
        elif "下方" in slider_desc and "附近" in slider_desc:
            # 下方附近，稍微偏移
            slider_x = width // 6
            slider_y = height * 3 // 4
            
        elif "左侧" in slider_desc:
            if "下方" in slider_desc or "底部" in slider_desc:
                slider_x = width // 8  # 更左一些
                slider_y = height * 4 // 5  # 更下一些
            else:
                slider_x = width // 6
                slider_y = height * 2 // 3
                
        elif "中间" in slider_desc or "中央" in slider_desc:
            slider_x = width // 2
            if "下方" in slider_desc:
                slider_y = height * 3 // 4
            else:
                slider_y = height * 2 // 3
                
        elif "右侧" in slider_desc:
            slider_x = width * 2 // 3
            if "下方" in slider_desc:
                slider_y = height * 3 // 4
            else:
                slider_y = height * 2 // 3
        
        # 特殊位置描述解析
        if "屏幕左侧约1/6处" in slider_desc:
            slider_x = width // 6
        elif "屏幕左侧约1/8处" in slider_desc:
            slider_x = width // 8
        elif "屏幕左侧约1/4处" in slider_desc:
            slider_x = width // 4
            
        if "垂直位置约3/4处" in slider_desc:
            slider_y = height * 3 // 4
        elif "垂直位置约4/5处" in slider_desc:
            slider_y = height * 4 // 5
        elif "垂直位置约2/3处" in slider_desc:
            slider_y = height * 2 // 3
        
        # 数字比例识别
        import re
        percentage_pattern = r'(\d+)%|(\d+)/(\d+)'
        matches = re.findall(percentage_pattern, slider_desc)
        if matches:
            for match in matches:
                if match[0]:  # 百分比形式
                    percentage = int(match[0]) / 100
                    if "水平" in slider_desc or "横向" in slider_desc:
                        slider_x = int(width * percentage)
                    elif "垂直" in slider_desc or "纵向" in slider_desc:
                        slider_y = int(height * percentage)
                elif match[1] and match[2]:  # 分数形式
                    fraction = int(match[1]) / int(match[2])
                    if "水平" in slider_desc or "横向" in slider_desc:
                        slider_x = int(width * fraction)
                    elif "垂直" in slider_desc or "纵向" in slider_desc:
                        slider_y = int(height * fraction)
        
        print_with_color(f"滑块按钮位置解析: '{slider_desc}' -> ({slider_x}, {slider_y})", "cyan")
        
        # 边界检查
        slider_x = max(10, min(slider_x, width - 10))  # 确保在屏幕范围内
        slider_y = max(10, min(slider_y, height - 10))
        
        print_with_color(f"滑块按钮位置（边界修正后）: ({slider_x}, {slider_y})", "green")
        return slider_x, slider_y

    def _parse_gap_position(self, gap_desc: str, width: int, height: int) -> tuple:
        """
        智能解析缺口位置描述，返回更准确的坐标
        """
        # 默认位置
        gap_x = width * 2 // 3  # 默认右侧2/3位置
        gap_y = height // 2     # 默认中间位置
        
        # 解析水平位置
        if "右侧" in gap_desc or "右边" in gap_desc or "右半" in gap_desc:
            # 进一步细分右侧位置
            if "3/4" in gap_desc or "四分三" in gap_desc:
                gap_x = width * 3 // 4
            elif "2/3" in gap_desc or "三分二" in gap_desc:
                gap_x = width * 2 // 3
            elif "远右" in gap_desc or "最右" in gap_desc:
                gap_x = width * 4 // 5
            elif "偏右" in gap_desc:
                gap_x = width * 5 // 8
            else:
                gap_x = width * 2 // 3  # 默认右侧2/3
        elif "左侧" in gap_desc or "左边" in gap_desc or "左半" in gap_desc:
            if "1/3" in gap_desc or "三分一" in gap_desc:
                gap_x = width // 3
            elif "1/4" in gap_desc or "四分一" in gap_desc:
                gap_x = width // 4
            else:
                gap_x = width // 3  # 默认左侧1/3
        elif "中间" in gap_desc or "中央" in gap_desc or "中心" in gap_desc:
            if "偏右" in gap_desc:
                gap_x = width * 5 // 8
            elif "偏左" in gap_desc:
                gap_x = width * 3 // 8
            else:
                gap_x = width // 2
        
        # 解析垂直位置
        if "上方" in gap_desc or "上部" in gap_desc or "顶部" in gap_desc:
            if "1/3" in gap_desc or "三分一" in gap_desc:
                gap_y = height // 3
            elif "1/4" in gap_desc or "四分一" in gap_desc:
                gap_y = height // 4
            else:
                gap_y = height // 3  # 默认上方1/3
        elif "下方" in gap_desc or "下部" in gap_desc or "底部" in gap_desc:
            if "2/3" in gap_desc or "三分二" in gap_desc:
                gap_y = height * 2 // 3
            elif "3/4" in gap_desc or "四分三" in gap_desc:
                gap_y = height * 3 // 4
            else:
                gap_y = height * 2 // 3  # 默认下方2/3
        elif "中间" in gap_desc or "中央" in gap_desc or "中心" in gap_desc:
            if "偏上" in gap_desc:
                gap_y = height * 2 // 5
            elif "偏下" in gap_desc:
                gap_y = height * 3 // 5
            else:
                gap_y = height // 2
        
        # 特殊关键词识别
        if "右上" in gap_desc:
            gap_x = width * 3 // 4
            gap_y = height // 3
        elif "右下" in gap_desc:
            gap_x = width * 3 // 4
            gap_y = height * 2 // 3
        elif "左上" in gap_desc:
            gap_x = width // 3
            gap_y = height // 3
        elif "左下" in gap_desc:
            gap_x = width // 3
            gap_y = height * 2 // 3
        
        # 数字比例识别
        import re
        # 查找类似"约75%"、"大约3/4"这样的描述
        percentage_pattern = r'(\d+)%|(\d+)/(\d+)'
        matches = re.findall(percentage_pattern, gap_desc)
        if matches:
            for match in matches:
                if match[0]:  # 百分比形式
                    percentage = int(match[0]) / 100
                    if "水平" in gap_desc or "横向" in gap_desc:
                        gap_x = int(width * percentage)
                    elif "垂直" in gap_desc or "纵向" in gap_desc:
                        gap_y = int(height * percentage)
                elif match[1] and match[2]:  # 分数形式
                    fraction = int(match[1]) / int(match[2])
                    if "水平" in gap_desc or "横向" in gap_desc:
                        gap_x = int(width * fraction)
                    elif "垂直" in gap_desc or "纵向" in gap_desc:
                        gap_y = int(height * fraction)
        
        print_with_color(f"缺口位置解析: '{gap_desc}' -> ({gap_x}, {gap_y})", "cyan")
        
        # 边界检查
        gap_x = max(10, min(gap_x, width - 10))  # 确保在屏幕范围内
        gap_y = max(10, min(gap_y, height - 10))
        
        print_with_color(f"缺口位置（边界修正后）: ({gap_x}, {gap_y})", "green")
        return gap_x, gap_y

    def _execute_manual_slider_operation(self) -> bool:
        """手动指定坐标执行滑块操作"""
        if not self.controller:
            print_with_color("控制器未初始化", "red")
            return False
        
        try:
            print_with_color("请手动输入滑块和缺口的坐标:", "cyan")
            
            # 获取滑块坐标
            slider_input = input("滑块坐标 (格式: x,y): ").strip()
            slider_x, slider_y = map(int, slider_input.split(','))
            slider_pos = (slider_x, slider_y)
            
            # 获取缺口坐标
            gap_input = input("缺口坐标 (格式: x,y): ").strip()
            gap_x, gap_y = map(int, gap_input.split(','))
            gap_pos = (gap_x, gap_y)
            
            print_with_color(f"滑块位置: {slider_pos}", "green")
            print_with_color(f"缺口位置: {gap_pos}", "green")
            
            # 执行滑动操作
            print_with_color("执行滑动操作...", "cyan")
            
            # 添加小延迟，模拟人类思考时间
            import time
            import random
            time.sleep(random.uniform(0.3, 0.8))
            
            # 使用人性化滑动方法
            result = self.controller.swipe_human_like(slider_pos, gap_pos, total_duration=3000)
            
            if result != "SUCCESS":
                # 备用：使用普通的慢速滑动
                self.controller.swipe_precise(slider_pos, gap_pos, duration=2500)
            
            return True
            
        except Exception as e:
            print_with_color(f"手动操作执行失败: {e}", "red")
            return False
    
    def _solve_click_captcha(self, screenshot_path: str) -> AgentResult:
        """解决点击验证码（如：点击所有的汽车）"""
        print_with_color("开始处理点击验证码...", "cyan")
        
        # 使用优化后的点击验证码提示词
        prompt = click_captcha_prompt
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        # 解析需要点击的位置
        click_positions = self._parse_click_positions(response)
        
        # 执行点击操作
        actions_taken = []
        for pos in click_positions:
            if self.controller:
                self.controller.tap(pos[0], pos[1])
                actions_taken.append(f"tap({pos[0]}, {pos[1]})")
                time.sleep(0.5)  # 防止操作过快
        
        return AgentResult(
            success=len(click_positions) > 0,
            message=f"完成 {len(click_positions)} 个点击操作",
            data={"click_positions": click_positions},
            actions_taken=actions_taken
        )
    
    def _solve_math_captcha(self, screenshot_path: str) -> AgentResult:
        """解决数学验证码"""
        print_with_color("开始处理数学验证码...", "cyan")
        
        # 使用优化后的数学验证码提示词
        prompt = math_captcha_prompt
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        # 提取数学计算结果
        result = self._extract_math_result(response)
        if result is None:
            return AgentResult(False, "无法计算数学表达式", actions_taken=[])
        
        # 输入计算结果
        input_success = self._input_captcha_text(str(result))
        
        return AgentResult(
            success=input_success,
            message=f"数学验证码计算结果: {result}",
            data={"math_result": result},
            actions_taken=["math_calculation", "text_input"]
        )
    
    def _solve_general_captcha(self, screenshot_path: str, task_desc: str) -> AgentResult:
        """处理通用验证码"""
        print_with_color("使用通用方法处理验证码...", "cyan")
        
        # 使用优化后的通用验证码提示词
        prompt = general_captcha_prompt + f"\n\n任务描述：{task_desc}"
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        # 解析并执行LLM建议的操作
        actions = self._parse_general_actions(response)
        success = self._execute_actions(actions)
        
        return AgentResult(
            success=success,
            message="通用验证码处理完成",
            data={"actions": actions},
            actions_taken=actions
        )
    
    def _confirm_captcha_result(self, screenshot_path: str) -> bool:
        """确认验证码提交后的结果"""
        print_with_color("确认验证码结果...", "cyan")
        
        # 等待验证结果界面加载
        time.sleep(2)
        
        # 获取新的屏幕截图
        if not self.controller:
            return False
        new_screenshot_path = self.controller.get_screenshot("captcha_result", "./temp")
        if new_screenshot_path == "ERROR":
            return False
        
        # 使用验证码结果确认提示词
        prompt = captcha_confirmation_prompt
        if not self.model:
            return False
        success, response = self.model.get_model_response(prompt, [new_screenshot_path])
        
        if not success:
            return False
        
        # 解析验证结果
        response_lower = response.lower()
        if "成功" in response_lower or "通过" in response_lower:
            print_with_color("验证码验证成功！", "green")
            return True
        else:
            print_with_color("验证码验证失败或无法确认", "red")
            return False
    
    # 辅助方法
    def _extract_captcha_text(self, response: str) -> Optional[str]:
        """从LLM响应中提取验证码文本"""
        # 查找各种可能的文本格式
        patterns = [
            r"验证码[：:]?\s*([A-Za-z0-9]+)",
            r"文本[：:]?\s*([A-Za-z0-9]+)",
            r"识别结果[：:]?\s*([A-Za-z0-9]+)",
            r"答案[：:]?\s*([A-Za-z0-9]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 如果没有匹配到特定格式，尝试提取所有大写字母和数字组合
        alphanumeric_pattern = r'[A-Z0-9]{4,8}'
        matches = re.findall(alphanumeric_pattern, response)
        if matches:
            return matches[0]
        
        return None
    
    def _input_captcha_text(self, text: str) -> bool:
        """输入验证码文本到输入框"""
        if not self.controller:
            return False
        
        try:
            # 查找输入框 - 这里需要根据实际情况调整
            # 可以使用元素检测或固定位置
            # 暂时使用text()函数直接输入
            time.sleep(1)
            self.controller.text(text)
            
            # 点击提交按钮（通常在输入框附近或下方）
            # 这里可以添加查找提交按钮的逻辑
            # 暂时模拟点击屏幕中下方位置作为提交
            width, height = self.controller.width, self.controller.height
            if width and height:
                self.controller.tap(width // 2, int(height * 0.7))
                
            return True
        except Exception as e:
            print_with_color(f"输入文本失败: {e}", "red")
            return False
    
    def get_specialized_prompts(self) -> Dict[str, str]:
        """获取验证码专用提示词"""
        return {
            "captcha_detection": captcha_detection_prompt,
            "text_captcha": text_captcha_prompt,
            "slider_captcha": slider_captcha_prompt,
            "click_captcha": click_captcha_prompt,
            "math_captcha": math_captcha_prompt,
            "general_captcha": general_captcha_prompt,
            "captcha_confirmation": captcha_confirmation_prompt
        }
    
    def _get_captcha_detection_prompt(self) -> str:
        """获取验证码类型检测提示词"""
        return captcha_detection_prompt

    def _get_text_captcha_prompt(self) -> str:
        """获取文本验证码提示词"""
        return text_captcha_prompt

    def _get_slider_captcha_prompt(self) -> str:
        """获取滑块验证码提示词"""
        return slider_captcha_prompt

    def _get_click_captcha_prompt(self) -> str:
        """获取点击验证码提示词"""
        return click_captcha_prompt

    def _get_math_captcha_prompt(self) -> str:
        """获取数学验证码提示词"""
        return math_captcha_prompt

    def _get_general_captcha_prompt(self, task_desc: str) -> str:
        """获取通用验证码提示词"""
        return general_captcha_prompt + f"\n\n任务描述：{task_desc}"
    
    def _parse_slider_action(self, response: str) -> Optional[Dict]:
        """解析滑块操作指令"""
        try:
            # 匹配从(x1,y1)滑动到(x2,y2)的格式
            pattern = r"从\s*\((\d+),\s*(\d+)\)\s*滑动到\s*\((\d+),\s*(\d+)\)"
            match = re.search(pattern, response)
            
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return {
                    "start_x": x1,
                    "start_y": y1,
                    "end_x": x2,
                    "end_y": y2
                }
            
            # 匹配滑块当前位置和目标位置
            current_pattern = r"滑块当前位置[：:]\s*\((\d+),\s*(\d+)\)"
            target_pattern = r"目标位置[：:]\s*\((\d+),\s*(\d+)\)"
            
            current_match = re.search(current_pattern, response)
            target_match = re.search(target_pattern, response)
            
            if current_match and target_match:
                x1, y1 = map(int, current_match.groups())
                x2, y2 = map(int, target_match.groups())
                return {
                    "start_x": x1,
                    "start_y": y1,
                    "end_x": x2,
                    "end_y": y2
                }
            
            # 尝试其他可能的格式
            pattern = r"滑动距离[：:]\s*约?(\d+)"
            match = re.search(pattern, response)
            if match:
                distance = int(match.group(1))
                # 假设滑块在屏幕中间位置
                return {
                    "start_x": 100,
                    "start_y": 500,
                    "end_x": 100 + distance,
                    "end_y": 500
                }
                
            return None
        except Exception:
            return None
    
    def _parse_click_positions(self, response: str) -> List[Tuple[int, int]]:
        """解析点击位置"""
        positions = []
        try:
            # 匹配点击位置：(x1,y1), (x2,y2), ...格式
            position_pattern = r"点击位置[：:]\s*((?:\(\d+,\s*\d+\)[,\s]*)+)"
            position_match = re.search(position_pattern, response)
            
            if position_match:
                positions_str = position_match.group(1)
                coord_pattern = r"\((\d+),\s*(\d+)\)"
                coord_matches = re.findall(coord_pattern, positions_str)
                
                for match in coord_matches:
                    x, y = map(int, match)
                    positions.append((x, y))
                
                return positions
            
            # 匹配所有(x,y)格式的坐标
            pattern = r"\((\d+),\s*(\d+)\)"
            matches = re.findall(pattern, response)
            
            for match in matches:
                x, y = map(int, match)
                positions.append((x, y))
                
            return positions
        except Exception:
            return []
    
    def _extract_math_result(self, response: str) -> Optional[int]:
        """提取数学计算结果"""
        try:
            # 匹配计算结果：数字格式
            pattern = r"计算结果[：:]\s*(\d+)"
            match = re.search(pattern, response)
            
            if match:
                return int(match.group(1))
            
            # 尝试其他可能的格式
            pattern = r"结果[：:]\s*(\d+)"
            match = re.search(pattern, response)
            
            if match:
                return int(match.group(1))
            
            # 匹配表达式和结果
            pattern = r"表达式[：:]\s*(.*?)\n.*?结果[：:]\s*(\d+)"
            match = re.search(pattern, response, re.DOTALL)
            
            if match:
                return int(match.group(2))
                
            # 尝试直接提取数字
            pattern = r"(\d+)"
            match = re.search(pattern, response)
            
            if match:
                return int(match.group(1))
                
            return None
        except Exception:
            return None
    
    def _parse_general_actions(self, response: str) -> List[str]:
        """解析通用操作"""
        actions = []
        try:
            # 提取操作指令部分
            operation_pattern = r"操作指令[：:](.*?)(?:\n\n|$)"
            operation_match = re.search(operation_pattern, response, re.DOTALL)
            
            if operation_match:
                operation_text = operation_match.group(1)
                
                # 提取tap操作
                tap_pattern = r"tap\((\d+),\s*(\d+)\)"
                tap_matches = re.findall(tap_pattern, operation_text)
                for match in tap_matches:
                    x, y = map(int, match)
                    actions.append(f"tap({x}, {y})")
                
                # 提取text操作
                text_pattern = r'text\("([^"]*)"\)'
                text_matches = re.findall(text_pattern, operation_text)
                for match in text_matches:
                    actions.append(f'text("{match}")')
                
                # 提取swipe操作
                swipe_pattern = r"swipe\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
                swipe_matches = re.findall(swipe_pattern, operation_text)
                for match in swipe_matches:
                    x1, y1, x2, y2 = map(int, match)
                    actions.append(f"swipe({x1}, {y1}, {x2}, {y2})")
            
            # 如果没有找到操作指令部分，尝试从整个响应中提取
            if not actions:
                # 提取tap操作
                tap_pattern = r"tap\((\d+),\s*(\d+)\)"
                tap_matches = re.findall(tap_pattern, response)
                for match in tap_matches:
                    x, y = map(int, match)
                    actions.append(f"tap({x}, {y})")
                
                # 提取text操作
                text_pattern = r'text\("([^"]*)"\)'
                text_matches = re.findall(text_pattern, response)
                for match in text_matches:
                    actions.append(f'text("{match}")')
                
                # 提取swipe操作
                swipe_pattern = r"swipe\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
                swipe_matches = re.findall(swipe_pattern, response)
                for match in swipe_matches:
                    x1, y1, x2, y2 = map(int, match)
                    actions.append(f"swipe({x1}, {y1}, {x2}, {y2})")
                
            return actions
        except Exception:
            return []
    
    def _execute_slider_action(self, action: Dict) -> bool:
        """执行滑块操作"""
        if not self.controller or not action:
            return False
        
        try:
            start_x = action.get("start_x", 0)
            start_y = action.get("start_y", 0)
            end_x = action.get("end_x", 0)
            end_y = action.get("end_y", 0)
            
            # 计算距离
            pixel_distance = max(abs(end_x - start_x), abs(end_y - start_y))
            distance = "medium"
            if pixel_distance < 100:
                distance = "short"
            elif pixel_distance > 300:
                distance = "long"
                
            # 使用人性化滑动方法进行滑块操作
            import time
            import random
            time.sleep(random.uniform(0.2, 0.5))  # 小延迟
            
            start_pos = (start_x, start_y)
            end_pos = (end_x, end_y)
            
            print_with_color(f"执行滑块滑动: 从 {start_pos} 到 {end_pos}", "cyan")
            result = self.controller.swipe_human_like(start_pos, end_pos, total_duration=3000)
            
            if result != "SUCCESS":
                # 备用：使用普通的慢速滑动
                self.controller.swipe_precise(start_pos, end_pos, duration=2500)
            
            return True
        except Exception as e:
            print_with_color(f"滑块操作失败: {e}", "red")
            return False
    
    def _execute_actions(self, actions: List[str]) -> bool:
        """执行通用操作"""
        if not self.controller or not actions:
            return False
        
        try:
            for action in actions:
                if action.startswith("tap"):
                    # 提取点击坐标
                    match = re.search(r"tap\((\d+),\s*(\d+)\)", action)
                    if match:
                        x, y = map(int, match.groups())
                        self.controller.tap(x, y)
                        time.sleep(0.5)
                
                elif action.startswith("text"):
                    # 提取文本内容
                    match = re.search(r'text\("([^"]*)"\)', action)
                    if match:
                        text = match.group(1)
                        self.controller.text(text)
                        time.sleep(0.5)
                
                elif action.startswith("swipe"):
                    # 提取滑动坐标
                    match = re.search(r"swipe\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", action)
                    if match:
                        x1, y1, x2, y2 = map(int, match.groups())
                        # 计算方向和距离
                        direction = "right"
                        if x2 < x1:
                            direction = "left"
                        elif y2 < y1:
                            direction = "up"
                        elif y2 > y1:
                            direction = "down"
                            
                        # 计算距离
                        pixel_distance = max(abs(x2 - x1), abs(y2 - y1))
                        distance = "medium"
                        if pixel_distance < 100:
                            distance = "short"
                        elif pixel_distance > 300:
                            distance = "long"
                            
                        self.controller.swipe(x1, y1, direction, distance)
                        time.sleep(0.5)
            
            return True
        except Exception as e:
            print_with_color(f"执行操作失败: {e}", "red")
            return False 

    def mark_slider_contours(self, screenshot_path: str) -> Optional[str]:
        """使用大模型分析并标记滑块和缺口的轮廓"""
        print_with_color("开始使用大模型标记滑块轮廓...", "cyan")
        
        if not self.model:
            print_with_color("模型未初始化", "red")
            return None
        
        try:
            # 使用增强的轮廓分析提示词
            contour_prompt = """
请非常仔细地分析这个滑块验证码图像，我需要你精确识别滑块和缺口的轮廓边界信息。

请按照以下格式详细描述：

**滑块轮廓信息：**
- 滑块边界坐标：左上角(x1,y1), 右下角(x2,y2)
- 滑块中心坐标：(cx,cy)
- 滑块宽度：xx像素
- 滑块高度：xx像素
- 滑块形状：[圆形/方形/椭圆形/不规则等]

**缺口轮廓信息：**
- 缺口边界坐标：左上角(x1,y1), 右下角(x2,y2)
- 缺口中心坐标：(cx,cy)
- 缺口宽度：xx像素
- 缺口高度：xx像素
- 缺口形状：[圆形/方形/椭圆形/拼图状等]

**位置关系：**
- 滑动距离：xx像素
- 滑动方向：[水平向右/向左/垂直向上/向下]

重要要求：
1. 所有坐标必须是相对于整个截图的绝对像素坐标
2. 边界坐标要完整包围目标元素
3. 提供具体的数值，不要使用模糊描述
4. 如果元素不规则，给出能包围它的最小矩形
"""
            
            print_with_color("发送轮廓分析请求...", "yellow")
            success, response = self.model.get_model_response(contour_prompt, [screenshot_path])
            
            if not success:
                print_with_color(f"LLM调用失败: {response}", "red")
                return None
            
            print_with_color("收到大模型轮廓分析结果", "green")
            print_with_color("="*60, "yellow")
            print_with_color("轮廓分析详情:", "cyan")
            print_with_color("="*60, "yellow")
            print_with_color(response, "white")
            print_with_color("="*60, "yellow")
            
            # 解析轮廓信息
            contour_info = self._parse_contour_info(response)
            
            if not contour_info:
                print_with_color("无法解析轮廓信息", "red")
                return None
            
            # 在图像上标记轮廓
            marked_path = self._draw_contour_markers(screenshot_path, contour_info)
            
            return marked_path
            
        except Exception as e:
            print_with_color(f"轮廓标记过程出错: {str(e)}", "red")
            return None
    
    def _parse_contour_info(self, response: str) -> Optional[Dict]:
        """解析大模型返回的轮廓信息"""
        try:
            import re
            
            contour_data = {
                'slider': {},
                'gap': {},
                'meta': {}
            }
            
            # 解析滑块边界坐标
            slider_boundary = re.search(r"滑块边界坐标[：:]\s*左上角\((\d+),\s*(\d+)\)[，,]\s*右下角\((\d+),\s*(\d+)\)", response)
            if slider_boundary:
                x1, y1, x2, y2 = map(int, slider_boundary.groups())
                contour_data['slider'].update({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1
                })
            
            # 解析滑块中心坐标
            slider_center = re.search(r"滑块中心坐标[：:]\s*\((\d+),\s*(\d+)\)", response)
            if slider_center:
                cx, cy = map(int, slider_center.groups())
                contour_data['slider'].update({'center_x': cx, 'center_y': cy})
            
            # 解析滑块尺寸（备用）
            slider_width = re.search(r"滑块宽度[：:]\s*(\d+)", response)
            slider_height = re.search(r"滑块高度[：:]\s*(\d+)", response)
            if slider_width and slider_height and 'width' not in contour_data['slider']:
                contour_data['slider']['width'] = int(slider_width.group(1))
                contour_data['slider']['height'] = int(slider_height.group(1))
            
            # 解析缺口边界坐标
            gap_boundary = re.search(r"缺口边界坐标[：:]\s*左上角\((\d+),\s*(\d+)\)[，,]\s*右下角\((\d+),\s*(\d+)\)", response)
            if gap_boundary:
                x1, y1, x2, y2 = map(int, gap_boundary.groups())
                contour_data['gap'].update({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'width': x2 - x1, 'height': y2 - y1
                })
            
            # 解析缺口中心坐标
            gap_center = re.search(r"缺口中心坐标[：:]\s*\((\d+),\s*(\d+)\)", response)
            if gap_center:
                cx, cy = map(int, gap_center.groups())
                contour_data['gap'].update({'center_x': cx, 'center_y': cy})
            
            # 解析缺口尺寸（备用）
            gap_width = re.search(r"缺口宽度[：:]\s*(\d+)", response)
            gap_height = re.search(r"缺口高度[：:]\s*(\d+)", response)
            if gap_width and gap_height and 'width' not in contour_data['gap']:
                contour_data['gap']['width'] = int(gap_width.group(1))
                contour_data['gap']['height'] = int(gap_height.group(1))
            
            # 解析滑动距离
            distance_match = re.search(r"滑动距离[：:]\s*(\d+)", response)
            if distance_match:
                contour_data['meta']['distance'] = int(distance_match.group(1))
            
            # 补全缺失的数据
            self._complete_contour_data(contour_data)
            
            # 验证数据有效性
            if self._validate_contour_data(contour_data):
                print_with_color("轮廓信息解析成功", "green")
                print_with_color(f"滑块: ({contour_data['slider']['x1']},{contour_data['slider']['y1']}) -> ({contour_data['slider']['x2']},{contour_data['slider']['y2']})", "green")
                print_with_color(f"缺口: ({contour_data['gap']['x1']},{contour_data['gap']['y1']}) -> ({contour_data['gap']['x2']},{contour_data['gap']['y2']})", "red")
                return contour_data
            else:
                print_with_color("轮廓数据不完整，使用智能提取", "yellow")
                return self._extract_smart_contour(response)
                
        except Exception as e:
            print_with_color(f"解析轮廓信息失败: {str(e)}", "red")
            return None
    
    def _complete_contour_data(self, contour_data: Dict) -> None:
        """补全轮廓数据中的缺失信息"""
        # 补全滑块中心坐标
        if 'center_x' not in contour_data['slider'] and 'x1' in contour_data['slider']:
            contour_data['slider']['center_x'] = (contour_data['slider']['x1'] + contour_data['slider']['x2']) // 2
            contour_data['slider']['center_y'] = (contour_data['slider']['y1'] + contour_data['slider']['y2']) // 2
        
        # 补全缺口中心坐标
        if 'center_x' not in contour_data['gap'] and 'x1' in contour_data['gap']:
            contour_data['gap']['center_x'] = (contour_data['gap']['x1'] + contour_data['gap']['x2']) // 2
            contour_data['gap']['center_y'] = (contour_data['gap']['y1'] + contour_data['gap']['y2']) // 2
        
        # 补全滑动距离
        if 'distance' not in contour_data['meta'] and 'center_x' in contour_data['slider'] and 'center_x' in contour_data['gap']:
            dx = contour_data['gap']['center_x'] - contour_data['slider']['center_x']
            dy = contour_data['gap']['center_y'] - contour_data['slider']['center_y']
            contour_data['meta']['distance'] = int((dx**2 + dy**2)**0.5)
    
    def _validate_contour_data(self, contour_data: Dict) -> bool:
        """验证轮廓数据的有效性"""
        try:
            required_keys = ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
            slider_valid = all(key in contour_data['slider'] for key in required_keys)
            gap_valid = all(key in contour_data['gap'] for key in required_keys)
            return slider_valid and gap_valid
        except:
            return False
    
    def _extract_smart_contour(self, response: str) -> Dict:
        """智能提取轮廓信息（备用方法）"""
        import re
        
        print_with_color("使用智能提取方法...", "yellow")
        
        # 提取所有数字坐标对
        coordinates = re.findall(r"\((\d+),\s*(\d+)\)", response)
        
        if len(coordinates) >= 4:
            # 假设前两对是滑块，后两对是缺口
            slider_coords = [(int(x), int(y)) for x, y in coordinates[:2]]
            gap_coords = [(int(x), int(y)) for x, y in coordinates[-2:]]
            
            # 构建滑块信息
            sx1, sy1 = min(slider_coords, key=lambda p: p[0] + p[1])
            sx2, sy2 = max(slider_coords, key=lambda p: p[0] + p[1])
            
            # 构建缺口信息  
            gx1, gy1 = min(gap_coords, key=lambda p: p[0] + p[1])
            gx2, gy2 = max(gap_coords, key=lambda p: p[0] + p[1])
            
            return {
                'slider': {
                    'x1': sx1, 'y1': sy1, 'x2': sx2, 'y2': sy2,
                    'width': sx2 - sx1, 'height': sy2 - sy1,
                    'center_x': (sx1 + sx2) // 2, 'center_y': (sy1 + sy2) // 2
                },
                'gap': {
                    'x1': gx1, 'y1': gy1, 'x2': gx2, 'y2': gy2,
                    'width': gx2 - gx1, 'height': gy2 - gy1,
                    'center_x': (gx1 + gx2) // 2, 'center_y': (gy1 + gy2) // 2
                },
                'meta': {
                    'distance': int(((gx1 - sx1)**2 + (gy1 - sy1)**2)**0.5)
                }
            }
        else:
            # 使用默认值
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
                'meta': {
                    'distance': 340
                }
            }
    
    def _draw_contour_markers(self, image_path: str, contour_info: Dict) -> Optional[str]:
        """在图像上绘制轮廓标记"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print_with_color(f"无法读取图像: {image_path}", "red")
                return None
            
            height, width = image.shape[:2]
            marked_image = image.copy()
            
            slider = contour_info['slider']
            gap = contour_info['gap']
            meta = contour_info.get('meta', {})
            
            # 绘制滑块轮廓（绿色双边框）
            # 外边框 - 粗线
            cv2.rectangle(marked_image, (slider['x1'], slider['y1']), (slider['x2'], slider['y2']), (0, 255, 0), 4)
            # 内边框 - 细线
            cv2.rectangle(marked_image, (slider['x1']+2, slider['y1']+2), (slider['x2']-2, slider['y2']-2), (0, 200, 0), 2)
            
            # 绘制缺口轮廓（红色双边框）
            # 外边框 - 粗线
            cv2.rectangle(marked_image, (gap['x1'], gap['y1']), (gap['x2'], gap['y2']), (0, 0, 255), 4)
            # 内边框 - 细线
            cv2.rectangle(marked_image, (gap['x1']+2, gap['y1']+2), (gap['x2']-2, gap['y2']-2), (0, 0, 200), 2)
            
            # 标记中心点
            cv2.circle(marked_image, (slider['center_x'], slider['center_y']), 6, (0, 255, 0), -1)
            cv2.circle(marked_image, (gap['center_x'], gap['center_y']), 6, (0, 0, 255), -1)
            
            # 绘制滑动路径箭头
            cv2.arrowedLine(marked_image, 
                          (slider['center_x'], slider['center_y']),
                          (gap['center_x'], gap['center_y']),
                          (255, 0, 0), 4, tipLength=0.3)
            
            # 添加标签
            self._add_contour_labels(marked_image, slider, gap, meta)
            
            # 保存标记后的图像
            output_path = os.path.join("./temp", "marked_slider_contours.png")
            cv2.imwrite(output_path, marked_image)
            
            print_with_color(f"轮廓标记完成: {output_path}", "green")
            return output_path
            
        except Exception as e:
            print_with_color(f"绘制轮廓标记失败: {str(e)}", "red")
            return None
    
    def _add_contour_labels(self, image: np.ndarray, slider: Dict, gap: Dict, meta: Dict) -> None:
        """添加轮廓标签和信息"""
        # 滑块标签
        slider_label = f"Slider ({slider['width']}x{slider['height']})"
        label_y = max(20, slider['y1'] - 10)
        cv2.putText(image, slider_label, (slider['x1'], label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 缺口标签
        gap_label = f"Gap ({gap['width']}x{gap['height']})"
        label_y = max(20, gap['y1'] - 10)
        cv2.putText(image, gap_label, (gap['x1'], label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 距离信息
        if 'distance' in meta:
            dist_text = f"Distance: {meta['distance']}px"
            mid_x = (slider['center_x'] + gap['center_x']) // 2 - 80
            mid_y = min(slider['y1'], gap['y1']) - 40
            if mid_y < 30:
                mid_y = max(slider['y2'], gap['y2']) + 30
            cv2.putText(image, dist_text, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # 坐标信息（在图像底部）
        height = image.shape[0]
        info_lines = [
            f"Slider: ({slider['x1']},{slider['y1']}) -> ({slider['x2']},{slider['y2']})",
            f"Gap: ({gap['x1']},{gap['y1']}) -> ({gap['x2']},{gap['y2']})",
            f"Centers: S({slider['center_x']},{slider['center_y']}) G({gap['center_x']},{gap['center_y']})"
        ]
        
        y_start = height - 80
        for i, line in enumerate(info_lines):
            y_pos = y_start + i * 25
            # 白色文字，黑色描边
            cv2.putText(image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 