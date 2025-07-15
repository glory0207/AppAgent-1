from typing import Dict, List, Type, Optional
import importlib
import os

from scripts.base_agent import BaseAgent, AgentTask, AgentResult
from scripts.and_controller import AndroidController
from scripts.utils import print_with_color


class AgentManager:
    """Agent管理器，负责协调多个专业化Agent"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.agents: Dict[str, BaseAgent] = {}
        self.controller: Optional[AndroidController] = None
        
        # 自动加载所有可用的Agent
        self._load_agents()
    
    def _load_agents(self):
        """自动加载所有可用的Agent"""
        agent_modules = [
            ("scripts.captcha_agent", "CaptchaAgent"),
            # 可以在这里添加更多的Agent
            # ("shopping_agent", "ShoppingAgent"),
            # ("social_media_agent", "SocialMediaAgent"),
        ]
        
        for module_name, class_name in agent_modules:
            try:
                module = importlib.import_module(module_name)
                agent_class = getattr(module, class_name)
                agent = agent_class(self.config_path)
                self.agents[agent.agent_name] = agent
                print_with_color(f"成功加载Agent: {agent.agent_name}", "green")
            except Exception as e:
                print_with_color(f"无法导入Agent模块 {module_name}: {str(e)}", "red")
    
    def set_controller(self, controller: AndroidController):
        """设置Android控制器"""
        self.controller = controller
        # 为所有Agent设置控制器
        for agent in self.agents.values():
            agent.controller = controller
    
    def get_available_agents(self) -> Dict[str, List[str]]:
        """获取所有可用的Agent及其能力"""
        agent_info = {}
        for name, agent in self.agents.items():
            agent_info[name] = agent.get_capabilities()
        return agent_info
    
    def execute_task(self, task: AgentTask) -> AgentResult:
        """执行任务，自动选择合适的Agent"""
        print_with_color(f"开始执行任务: {task.description}", "cyan")
        
        # 查找能处理该任务的Agent
        suitable_agent = None
        for agent in self.agents.values():
            if agent.can_handle_task(task):
                suitable_agent = agent
                break
        
        if suitable_agent:
            print_with_color(f"选择Agent: {suitable_agent.agent_name}", "green")
            return suitable_agent.execute_task(task)
        else:
            print_with_color(f"没有找到能处理任务的Agent: {task.description}", "red")
            return AgentResult(
                success=False,
                message=f"没有找到能处理此任务的Agent",
                actions_taken=[]
            )
    
    def train_agent(self, agent_name: str, training_data: Dict) -> bool:
        """训练特定的Agent"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            if hasattr(agent, 'train'):
                return agent.train(training_data)
            else:
                print_with_color(f"Agent {agent_name} 不支持训练", "red")
                return False
        else:
            print_with_color(f"Agent {agent_name} 不存在", "red")
            return False


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.tasks: List[AgentTask] = []
    
    def add_task(self, task: AgentTask):
        """添加任务到队列"""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: t.priority, reverse=True)  # 按优先级排序
    
    def process_all(self) -> List[AgentResult]:
        """处理所有任务"""
        results = []
        for task in self.tasks:
            result = self.agent_manager.execute_task(task)
            results.append(result)
        self.tasks = []  # 清空队列
        return results 