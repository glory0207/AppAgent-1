#!/usr/bin/env python3
"""
测试真实AI响应的解析工具
将真实的AI响应粘贴到这里进行解析测试
"""

import sys
sys.path.append('.')

from analyze_popup import parse_auto_response
from scripts.utils import print_with_color

def test_real_response():
    """在这里粘贴真实的AI响应进行测试"""
    
    # 请将您收到的真实AI响应粘贴到这里
    real_ai_response = """
请在这里粘贴您的真实AI响应...
    
例如：
这是一个应用安装确认对话框。我可以看到有两个按钮，左边是"取消"按钮，坐标在(180, 580)，右边是"安装"按钮，坐标在(420, 580)。

基于安全考虑，我建议点击"取消"按钮来拒绝安装未知应用。
"""
    
    print_with_color(" 测试真实AI响应解析", "cyan")
    print_with_color("="*80, "yellow")
    
    if "请在这里粘贴" in real_ai_response:
        print_with_color(" 请先将真实的AI响应粘贴到 real_ai_response 变量中", "red")
        print_with_color("然后重新运行此脚本", "yellow")
        return
    
    print_with_color("原始AI响应:", "blue")
    print(real_ai_response)
    print_with_color("="*80, "yellow")
    
    # 尝试解析
    result = parse_auto_response(real_ai_response)
    
    print_with_color("="*80, "yellow")
    if result:
        print_with_color(" 解析成功！准备执行以下操作：", "green")
        print_with_color(f"   弹窗类型: {result.get('type', 'N/A')}", "cyan")
        print_with_color(f"   目标按钮: {result.get('button_text', 'N/A')}", "cyan")
        print_with_color(f"   操作坐标: ({result.get('x', 'N/A')}, {result.get('y', 'N/A')})", "cyan")
        print_with_color(f"   操作方式: {result.get('operation_method', 'N/A')}", "cyan")
        if 'reason' in result:
            print_with_color(f"   选择理由: {result['reason']}", "cyan")
    else:
        print_with_color(" 解析失败！", "red")
        print_with_color("可能的解决方案：", "yellow")
        print_with_color("1. 检查AI响应是否包含坐标信息 (x, y)", "white")
        print_with_color("2. 检查AI响应是否包含按钮文字", "white")
        print_with_color("3. 尝试修改AI提示词，要求更严格的格式", "white")

def test_various_formats():
    """测试各种可能的AI响应格式"""
    
    test_cases = [
        {
            "name": "标准格式",
            "response": """
弹窗类型：安装确认
弹窗内容：应用安装确认对话框
推荐操作：
  目标按钮：取消
  操作坐标：(200, 600)
  操作方式：点击
"""
        },
        {
            "name": "描述性格式",
            "response": """
这是一个安装确认对话框。我看到有两个按钮：
- 左边的"取消"按钮，位置在(180, 580)
- 右边的"安装"按钮，位置在(420, 580)

我建议点击"取消"按钮，因为安装未知应用存在安全风险。
"""
        },
        {
            "name": "简单格式",
            "response": """
广告弹窗，建议点击右上角的"×"关闭按钮，坐标(450, 150)。
"""
        },
        {
            "name": "混合格式",
            "response": """
类型：权限请求
这个弹窗是在请求访问位置权限。
有两个选项："允许"(320, 500) 和 "拒绝"(180, 500)
建议选择"拒绝"来保护隐私。
"""
        }
    ]
    
    print_with_color(" 测试各种AI响应格式", "cyan")
    print_with_color("="*80, "yellow")
    
    for i, test_case in enumerate(test_cases, 1):
        print_with_color(f"\n测试 {i}: {test_case['name']}", "blue")
        print_with_color("-" * 40, "white")
        print(test_case['response'])
        print_with_color("-" * 40, "white")
        
        result = parse_auto_response(test_case['response'])
        
        if result:
            print_with_color(" 解析成功", "green")
            print_with_color(f"   类型: {result.get('type')}", "cyan")
            print_with_color(f"   按钮: {result.get('button_text')}", "cyan")
            print_with_color(f"   坐标: ({result.get('x')}, {result.get('y')})", "cyan")
        else:
            print_with_color(" 解析失败", "red")
        
        print_with_color("=" * 80, "yellow")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_various_formats()
    else:
        test_real_response()
        print_with_color("\n💡 提示: 运行 'python test_ai_response.py test' 可以测试各种格式示例", "yellow") 