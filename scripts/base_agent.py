from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os
from dataclasses import dataclass

from scripts.config import load_config
from scripts.model import BaseModel, OpenAIModel, QwenModel
from scripts.and_controller import AndroidController
from scripts.utils import print_with_color


@dataclass
class AgentTask:
    """代理任务数据结构"""
    task_id: str
    description: str
    app_name: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResult:
    """代理执行结果数据结构"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    actions_taken: Optional[List[str]] = None


class BaseAgent(ABC):
    """基础Agent类，所有专业化Agent的父类"""
    
    def __init__(self, agent_name: str, config_path: str = "config.yaml"):
        self.agent_name = agent_name
        self.config = load_config(config_path)
        self.controller: Optional[AndroidController] = None
        self.model: Optional[BaseModel] = None
        self.capabilities: List[str] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        # 初始化模型
        self._init_model()
        
        # 加载Agent特定的知识库
        self._load_knowledge_base()
    
    def _init_model(self):
        """初始化LLM模型"""
        if self.config["MODEL"] == "OpenAI":
            self.model = OpenAIModel(
                base_url=self.config["OPENAI_API_BASE"],
                api_key=self.config["OPENAI_API_KEY"],
                model=self.config["OPENAI_API_MODEL"],
                temperature=float(self.config["TEMPERATURE"]),
                max_tokens=int(self.config["MAX_TOKENS"])
            )
        elif self.config["MODEL"] == "Qwen":
            self.model = QwenModel(
                api_key=self.config["DASHSCOPE_API_KEY"],
                model=self.config["QWEN_MODEL"]
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config['MODEL']}")
    
    def _load_knowledge_base(self):
        """加载Agent特定的知识库"""
        kb_path = f"agents/{self.agent_name}/knowledge_base.json"
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
    
    def save_knowledge_base(self):
        """保存知识库"""
        kb_dir = f"agents/{self.agent_name}"
        os.makedirs(kb_dir, exist_ok=True)
        kb_path = f"{kb_dir}/knowledge_base.json"
        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
    
    def set_controller(self, controller: AndroidController):
        """设置Android控制器"""
        self.controller = controller
    
    @abstractmethod
    def can_handle_task(self, task: AgentTask) -> bool:
        """判断当前Agent是否可以处理指定任务"""
        pass
    
    @abstractmethod
    def execute_task(self, task: AgentTask) -> AgentResult:
        """执行任务的核心方法"""
        pass
    
    @abstractmethod
    def get_specialized_prompts(self) -> Dict[str, str]:
        """获取专业化的提示词模板"""
        pass
    
    def learn_from_demonstration(self, app_name: str, task_description: str):
        """从演示中学习（通用学习框架）"""
        print_with_color(f"开始为 {self.agent_name} 进行专业化学习...", "yellow")
        # 这里可以扩展特定的学习逻辑
        pass
    
    def get_capabilities(self) -> List[str]:
        """获取Agent的能力列表"""
        return self.capabilities.copy()
    
    def update_knowledge(self, key: str, value: Any):
        """更新知识库"""
        self.knowledge_base[key] = value
        self.save_knowledge_base()
    
    def get_knowledge(self, key: str, default=None):
        """获取知识库中的信息"""
        return self.knowledge_base.get(key, default) 