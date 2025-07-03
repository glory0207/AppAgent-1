import re
from abc import abstractmethod
from typing import List, Tuple, Optional
from http import HTTPStatus

import requests
import dashscope

from scripts.utils import print_with_color, encode_image


class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str, images: List[str]) -> Tuple[bool, str]:
        pass


class OpenAIModel(BaseModel):
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, max_tokens: int, user_id: Optional[str] = None, api_version: Optional[str] = None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_id = user_id
        self.api_version = api_version

    def get_model_response(self, prompt: str, images: List[str]) -> Tuple[bool, str]:
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 添加用户ID和API版本到请求头（如果提供）
        if self.user_id:
            headers["X-User-Id"] = self.user_id
        
        if self.api_version:
            headers["X-API-Version"] = self.api_version
            
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            print_with_color(f"发送请求到: {self.base_url}", "blue")
            print_with_color(f"使用模型: {self.model}", "blue")
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            # 打印HTTP状态码
            print_with_color(f"HTTP状态码: {response.status_code}", "yellow")
            
            # 尝试解析JSON响应
            try:
                response_json = response.json()
            except Exception as e:
                print_with_color(f"无法解析JSON响应: {str(e)}", "red")
                print_with_color(f"原始响应: {response.text}", "red")
                return False, f"API响应解析失败: {str(e)}"
            
            # 检查是否有错误
            if "error" in response_json:
                error_message = response_json["error"].get("message", "未知错误")
                print_with_color(f"API返回错误: {error_message}", "red")
                return False, error_message
            
            # 检查并处理usage信息
            if "usage" in response_json:
                usage = response_json["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                print_with_color(f"Request cost is "
                                f"${'{0:.2f}'.format(prompt_tokens / 1000 * 0.01 + completion_tokens / 1000 * 0.03)}",
                                "yellow")
            else:
                print_with_color("API响应中没有usage信息", "yellow")
            
            # 检查并处理choices信息
            if "choices" in response_json and len(response_json["choices"]) > 0:
                choice = response_json["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return True, choice["message"]["content"]
                else:
                    print_with_color("API响应中choices格式不正确", "red")
                    print_with_color(f"Choice内容: {choice}", "yellow")
                    return False, "API响应格式错误: choices中缺少message或content"
            else:
                print_with_color("API响应中没有choices信息或choices为空", "red")
                print_with_color(f"API响应内容: {response_json}", "yellow")
                return False, "API响应格式错误: 缺少choices字段"
                
        except Exception as e:
            import traceback
            print_with_color(f"API请求异常: {str(e)}", "red")
            traceback.print_exc()
            return False, f"API请求异常: {str(e)}"


class QwenModel(BaseModel):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.model = model
        dashscope.api_key = api_key
        print_with_color(f"初始化Qwen模型: {model}", "blue")
        # 检查API密钥是否有效
        if not api_key or api_key.startswith("sk-") and len(api_key) < 10:
            print_with_color("⚠️ 警告: API密钥可能无效", "yellow")

    def get_model_response(self, prompt: str, images: List[str]) -> (bool, str):
        try:
            print_with_color(f"准备调用Qwen模型，提示词长度: {len(prompt)}", "blue")
            print_with_color(f"图像数量: {len(images)}", "blue")
            
            content = [{
                "text": prompt
            }]
            
            for img in images:
                print_with_color(f"处理图像: {img}", "blue")
                img_path = f"file://{img}"
                content.append({
                    "image": img_path
                })
                
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            print_with_color("发送请求到Qwen API...", "blue")
            response = dashscope.MultiModalConversation.call(model=self.model, messages=messages)
            
            if response.status_code == HTTPStatus.OK:
                print_with_color("Qwen API请求成功", "green")
                try:
                    result_text = response.output.choices[0].message.content[0]["text"]
                    print_with_color(f"响应长度: {len(result_text)}", "blue")
                    return True, result_text
                except Exception as e:
                    print_with_color(f"解析响应失败: {str(e)}", "red")
                    print_with_color(f"响应内容: {response}", "yellow")
                    return False, f"解析响应失败: {str(e)}"
            else:
                print_with_color(f"Qwen API请求失败，状态码: {response.status_code}", "red")
                print_with_color(f"错误信息: {response.message}", "red")
                return False, f"API请求失败: {response.message}"
                
        except Exception as e:
            import traceback
            print_with_color(f"调用Qwen模型时发生异常: {str(e)}", "red")
            traceback.print_exc()
            return False, f"模型调用异常: {str(e)}"


def parse_explore_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            area = int(re.findall(r"tap\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "text":
            input_str = re.findall(r"text\((.*?)\)", act)[0][1:-1]
            return [act_name, input_str, last_act]
        elif act_name == "long_press":
            area = int(re.findall(r"long_press\((.*?)\)", act)[0])
            return [act_name, area, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0]
            area, swipe_dir, dist = params.split(",")
            area = int(area)
            swipe_dir = swipe_dir.strip()[1:-1]
            dist = dist.strip()[1:-1]
            return [act_name, area, swipe_dir, dist, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_grid_rsp(rsp):
    try:
        observation = re.findall(r"Observation: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        act = re.findall(r"Action: (.*?)$", rsp, re.MULTILINE)[0]
        last_act = re.findall(r"Summary: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Observation:", "yellow")
        print_with_color(observation, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        print_with_color("Action:", "yellow")
        print_with_color(act, "magenta")
        print_with_color("Summary:", "yellow")
        print_with_color(last_act, "magenta")
        if "FINISH" in act:
            return ["FINISH"]
        act_name = act.split("(")[0]
        if act_name == "tap":
            params = re.findall(r"tap\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "long_press":
            params = re.findall(r"long_press\((.*?)\)", act)[0].split(",")
            area = int(params[0].strip())
            subarea = params[1].strip()[1:-1]
            return [act_name + "_grid", area, subarea, last_act]
        elif act_name == "swipe":
            params = re.findall(r"swipe\((.*?)\)", act)[0].split(",")
            start_area = int(params[0].strip())
            start_subarea = params[1].strip()[1:-1]
            end_area = int(params[2].strip())
            end_subarea = params[3].strip()[1:-1]
            return [act_name + "_grid", start_area, start_subarea, end_area, end_subarea, last_act]
        elif act_name == "grid":
            return [act_name]
        else:
            print_with_color(f"ERROR: Undefined act {act_name}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]


def parse_reflect_rsp(rsp):
    try:
        decision = re.findall(r"Decision: (.*?)$", rsp, re.MULTILINE)[0]
        think = re.findall(r"Thought: (.*?)$", rsp, re.MULTILINE)[0]
        print_with_color("Decision:", "yellow")
        print_with_color(decision, "magenta")
        print_with_color("Thought:", "yellow")
        print_with_color(think, "magenta")
        if decision == "INEFFECTIVE":
            return [decision, think]
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            doc = re.findall(r"Documentation: (.*?)$", rsp, re.MULTILINE)[0]
            print_with_color("Documentation:", "yellow")
            print_with_color(doc, "magenta")
            return [decision, think, doc]
        else:
            print_with_color(f"ERROR: Undefined decision {decision}!", "red")
            return ["ERROR"]
    except Exception as e:
        print_with_color(f"ERROR: an exception occurs while parsing the model response: {e}", "red")
        print_with_color(rsp, "red")
        return ["ERROR"]
