# 🤖 多功能AppAgent扩展系统

基于原有的AppAgent项目，我们扩展了一个支持多种专业化Agent的系统架构，每个Agent专注于特定领域的任务。

## 🏗️ 系统架构

### 核心组件

- **BaseAgent**: 所有专业化Agent的基类
- **AgentManager**: Agent管理器，负责协调多个Agent
- **TaskQueue**: 任务队列管理器，支持批量处理
- **CaptchaAgent**: 验证码处理专业Agent（示例实现）

### 架构优势

✅ **模块化设计**: 每个Agent独立开发和维护  
✅ **可扩展性**: 轻松添加新的专业化Agent  
✅ **智能调度**: 自动选择最适合的Agent处理任务  
✅ **知识积累**: 每个Agent维护自己的专业知识库  
✅ **批量处理**: 支持任务队列和批量执行  

## 🚀 快速开始

### 1. 运行多功能Agent系统

```bash
# 交互式模式（推荐）
python run_multi_agent.py

# 单任务模式
python run_multi_agent.py --mode single --app "微信" --task "识别登录验证码"

# 批量任务模式
python run_multi_agent.py --mode batch
```

### 2. 交互式命令

在交互式模式下可以使用以下命令：

```
AppAgent> help          # 查看帮助
AppAgent> agents        # 显示所有可用Agent
AppAgent> task          # 执行单个任务
AppAgent> train captcha_agent  # 训练验证码Agent
AppAgent> clear         # 清屏
AppAgent> quit          # 退出
```

## 🎯 验证码Agent示例

### 支持的验证码类型

- **文本验证码**: 字母数字组合识别
- **滑块验证码**: 拖拽滑块到指定位置
- **点击验证码**: 按要求点击特定图像
- **数学验证码**: 计算数学表达式
- **通用验证码**: 使用LLM分析处理

### 使用示例

```python
# 创建验证码处理任务
task = AgentTask(
    task_id="captcha_001",
    description="识别并输入登录页面的验证码",
    app_name="支付宝"
)

# 执行任务
result = agent_manager.execute_task(task)
```

## 🔧 创建新的专业化Agent

### 步骤1: 继承BaseAgent

```python
from base_agent import BaseAgent, AgentTask, AgentResult

class ShoppingAgent(BaseAgent):
    """购物助手Agent"""
    
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__("shopping_agent", config_path)
        self.capabilities = [
            "product_search",
            "price_comparison", 
            "cart_management",
            "order_tracking"
        ]
    
    def can_handle_task(self, task: AgentTask) -> bool:
        """判断是否为购物相关任务"""
        keywords = ["购物", "商品", "价格", "订单", "购买"]
        task_text = (task.description + " " + task.app_name).lower()
        return any(keyword in task_text for keyword in keywords)
    
    def execute_task(self, task: AgentTask) -> AgentResult:
        """执行购物任务"""
        # 实现具体的购物逻辑
        pass
    
    def get_specialized_prompts(self) -> Dict[str, str]:
        """获取购物专用提示词"""
        return {
            "product_search": "...",
            "price_comparison": "..."
        }
```

### 步骤2: 注册到AgentManager

在 `scripts/agent_manager.py` 中添加：

```python
agent_modules = [
    ("captcha_agent", "CaptchaAgent"),
    ("shopping_agent", "ShoppingAgent"),  # 新增
    ("social_media_agent", "SocialMediaAgent"),
]
```

### 步骤3: 实现专业化功能

```python
def _search_product(self, product_name: str) -> Dict:
    """搜索商品"""
    prompt = f"请在当前页面搜索商品：{product_name}"
    success, response = self.model.get_model_response(prompt, [screenshot])
    # 解析响应并执行操作
    
def _compare_prices(self, products: List[str]) -> Dict:
    """比较价格"""
    # 实现价格比较逻辑
```

## 🎨 更多Agent创意

### 1. 社交媒体Agent
- 自动回复消息
- 内容发布管理
- 好友互动分析

### 2. 办公助手Agent
- 会议管理
- 文档处理
- 邮件管理

### 3. 学习助手Agent
- 在线课程学习
- 笔记整理
- 练习题解答

### 4. 金融助手Agent
- 账单管理
- 投资分析
- 支付操作

### 5. 游戏助手Agent
- 自动任务完成
- 资源管理
- 战略分析

## 🔄 学习和优化

### 自主学习模式

```bash
# 让Agent通过自主探索学习
python learn_multi_agent.py --agent captcha_agent --app "淘宝" --mode autonomous
```

### 人类演示学习

```bash
# 通过人类演示教Agent新技能
python learn_multi_agent.py --agent shopping_agent --app "京东" --mode demonstration
```

### 知识库管理

每个Agent维护自己的知识库：

```
agents/
├── captcha_agent/
│   ├── knowledge_base.json
│   ├── training_data/
│   └── prompts/
├── shopping_agent/
│   ├── knowledge_base.json
│   └── product_database.json
```

## 📊 性能监控

### 任务执行统计

```python
# 获取Agent性能统计
stats = agent_manager.get_performance_stats()
print(f"成功率: {stats['success_rate']}%")
print(f"平均执行时间: {stats['avg_time']}秒")
```

### 实时监控

```bash
# 启动监控面板
python monitor_agents.py
```

## 🛠️ 高级配置

### 自定义提示词

```yaml
# config.yaml
custom_prompts:
  captcha_agent:
    text_recognition: "你是验证码识别专家..."
    slider_solving: "分析滑块验证码..."
  
  shopping_agent:
    product_search: "你是购物助手..."
```

### Agent协作

```python
# Agent间协作示例
class CompositeTask:
    def execute(self):
        # 1. 验证码Agent处理登录
        login_result = captcha_agent.execute_task(login_task)
        
        # 2. 购物Agent执行购买
        if login_result.success:
            shopping_result = shopping_agent.execute_task(buy_task)
```

## 🤝 贡献指南

1. **Fork项目**
2. **创建新的Agent类**
3. **编写测试用例**
4. **提交Pull Request**

### Agent质量标准

- ✅ 继承BaseAgent
- ✅ 实现所有抽象方法
- ✅ 包含错误处理
- ✅ 提供详细文档
- ✅ 通过测试用例

## 🐛 故障排除

### 常见问题

**Q: Agent无法加载？**
A: 检查模块导入路径和依赖项

**Q: 任务执行失败？**
A: 查看日志文件，检查LLM配置

**Q: 验证码识别不准确？**
A: 增加训练样本，优化提示词

## 📞 获取帮助

- 📖 查看完整文档
- 💬 加入讨论群
- 🐛 提交Issue
- 📧 联系开发者

---

**祝你使用愉快！🎉** 