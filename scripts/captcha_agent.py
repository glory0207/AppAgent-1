import cv2
import numpy as np
import re
import base64
from typing import Dict, List, Any, Optional, Tuple
import time

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
        
        # 使用优化后的滑块验证码提示词
        prompt = slider_captcha_prompt
        if not self.model:
            return AgentResult(False, "模型未初始化", actions_taken=[])
        success, response = self.model.get_model_response(prompt, [screenshot_path])
        
        if not success:
            return AgentResult(False, "LLM调用失败", actions_taken=[])
        
        # 提取滑块操作指令
        slider_action = self._parse_slider_action(response)
        if not slider_action:
            return AgentResult(False, "无法解析滑块操作", actions_taken=[])
        
        # 执行滑块操作
        success = self._execute_slider_action(slider_action)
        
        return AgentResult(
            success=success,
            message="滑块验证码处理完成" if success else "滑块操作失败",
            data={"slider_action": slider_action},
            actions_taken=["slider_analysis", "swipe_action"]
        )
    
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
            
            # 计算方向和距离
            direction = "right"
            if end_x < start_x:
                direction = "left"
            elif end_y < start_y:
                direction = "up"
            elif end_y > start_y:
                direction = "down"
                
            # 计算距离
            pixel_distance = max(abs(end_x - start_x), abs(end_y - start_y))
            distance = "medium"
            if pixel_distance < 100:
                distance = "short"
            elif pixel_distance > 300:
                distance = "long"
                
            # 使用controller的swipe方法
            self.controller.swipe(start_x, start_y, direction, distance)
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