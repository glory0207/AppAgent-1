#!/usr/bin/env python3
"""
调试AI响应解析工具
用于测试和调试弹窗分析AI响应的解析过程
"""

import re
import sys
from typing import Dict, Optional, List

def test_parse_auto_response(response: str) -> Optional[Dict]:
    """测试解析智能分析的响应"""
    print("="*80)
    print("开始解析AI响应")
    print("="*80)
    print("原始响应:")
    print(response)
    print("="*80)
    
    try:
        result = {}
        
        # 提取弹窗类型
        type_match = re.search(r'弹窗类型[：:]\s*([^\n]+)', response)
        if type_match:
            result['type'] = type_match.group(1).strip()
            print(f"✓ 弹窗类型: {result['type']}")
        else:
            print("✗ 未找到弹窗类型")
        
        # 提取弹窗内容
        content_match = re.search(r'弹窗内容[：:]\s*([^\n]+)', response)
        if content_match:
            result['content'] = content_match.group(1).strip()
            print(f"✓ 弹窗内容: {result['content']}")
        else:
            print("✗ 未找到弹窗内容")
        
        # 提取推荐操作信息
        target_button_match = re.search(r'目标按钮[：:]\s*([^\n]+)', response)
        if target_button_match:
            result['button_text'] = target_button_match.group(1).strip().replace('[', '').replace(']', '')
            print(f"✓ 目标按钮: {result['button_text']}")
        else:
            print("✗ 未找到目标按钮")
        
        # 提取操作坐标
        coord_match = re.search(r'操作坐标[：:]\s*\((\d+),\s*(\d+)\)', response)
        if coord_match:
            result['x'] = int(coord_match.group(1))
            result['y'] = int(coord_match.group(2))
            print(f"✓ 操作坐标: ({result['x']}, {result['y']})")
        else:
            print("✗ 未找到操作坐标")
        
        # 提取操作方式
        method_match = re.search(r'操作方式[：:]\s*([^\n]+)', response)
        if method_match:
            result['operation_method'] = method_match.group(1).strip().replace('[', '').replace(']', '')
            print(f"✓ 操作方式: {result['operation_method']}")
        else:
            result['operation_method'] = '点击'  # 默认为点击
            print(f"- 操作方式 (默认): {result['operation_method']}")
        
        # 提取选择理由
        reason_match = re.search(r'选择理由[：:]\s*([^\n]+)', response)
        if reason_match:
            result['reason'] = reason_match.group(1).strip().replace('[', '').replace(']', '')
            print(f"✓ 选择理由: {result['reason']}")
        else:
            print("- 未找到选择理由")
        
        # 尝试旧格式解析
        print("\n尝试兼容旧格式...")
        if 'x' not in result or 'button_text' not in result:
            old_button_match = re.search(r'操作按钮[：:]\s*([^\n]+)', response)
            old_coord_match = re.search(r'按钮坐标[：:]\s*\((\d+),\s*(\d+)\)', response)
            
            if old_button_match:
                result['button_text'] = old_button_match.group(1).strip()
                print(f"✓ 操作按钮 (旧格式): {result['button_text']}")
            if old_coord_match:
                result['x'] = int(old_coord_match.group(1))
                result['y'] = int(old_coord_match.group(2))
                print(f"✓ 按钮坐标 (旧格式): ({result['x']}, {result['y']})")
        
        # 备用解析：查找任何坐标
        print("\n备用解析：查找所有坐标...")
        all_coords = re.findall(r'\((\d+),\s*(\d+)\)', response)
        if all_coords:
            print(f"找到的所有坐标: {all_coords}")
            if 'x' not in result and all_coords:
                result['x'] = int(all_coords[0][0])
                result['y'] = int(all_coords[0][1])
                print(f"✓ 使用第一个坐标: ({result['x']}, {result['y']})")
        
        # 备用解析：查找按钮相关文字
        print("\n备用解析：查找按钮关键词...")
        if 'button_text' not in result:
            button_keywords = ['按钮', '点击', '确定', '取消', '同意', '关闭', '安装', '跳过', '允许', '拒绝']
            for keyword in button_keywords:
                if keyword in response:
                    result['button_text'] = keyword
                    print(f"✓ 找到按钮关键词: {keyword}")
                    break
        
        print("\n="*80)
        print("解析结果汇总:")
        print("="*80)
        
        # 验证必要字段
        required_fields = ['type', 'x', 'y', 'button_text']
        missing_fields = []
        
        for field in required_fields:
            if field in result:
                print(f"✓ {field}: {result[field]}")
            else:
                print(f"✗ {field}: 缺失")
                missing_fields.append(field)
        
        if missing_fields:
            print(f"\n❌ 解析失败，缺少必要字段: {missing_fields}")
            return None
        else:
            print(f"\n✅ 解析成功！")
            return result
        
    except Exception as e:
        print(f"\n❌ 解析过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_sample_responses():
    """测试一些示例响应"""
    
    # 测试新格式响应
    print("测试新格式响应...")
    sample_new_format = """
弹窗类型：安装确认
弹窗内容：应用安装确认对话框，询问是否安装某个应用
检测到的所有按钮：
  - 按钮1：文字"取消" | 坐标(200,600) | 功能描述：取消安装
  - 按钮2：文字"安装" | 坐标(400,600) | 功能描述：确认安装

推荐操作：
  目标按钮：取消
  操作坐标：(200, 600)
  操作方式：点击
  选择理由：出于安全考虑，建议取消安装未知应用

备选操作：
  如果用户确认该应用安全，可以点击"安装"按钮
"""
    
    result1 = test_parse_auto_response(sample_new_format)
    
    print("\n" + "="*100 + "\n")
    
    # 测试旧格式响应
    print("测试旧格式响应...")
    sample_old_format = """
弹窗类型：广告弹窗
弹窗内容：应用推广广告
操作按钮：关闭
按钮坐标：(450, 150)
操作类型：关闭
"""
    
    result2 = test_parse_auto_response(sample_old_format)
    
    print("\n" + "="*100 + "\n")
    
    # 测试不规范响应
    print("测试不规范响应...")
    sample_irregular = """
这是一个安装对话框，有两个按钮。
左边是"取消"按钮，坐标大概在(180, 580)
右边是"安装"按钮，坐标在(420, 580)
建议点击取消按钮比较安全。
"""
    
    result3 = test_parse_auto_response(sample_irregular)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果提供了参数，解析该响应文本
        response_text = " ".join(sys.argv[1:])
        result = test_parse_auto_response(response_text)
    else:
        # 运行测试示例
        test_sample_responses() 