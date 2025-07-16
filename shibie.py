import cv2
import easyocr
import time
import numpy as np
import os

class TextDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'])
    
    def find_buttons(self, image_path, keywords=None):
        """
        在图片中查找按钮坐标
        
        Args:
            image_path: 图片路径
            keywords: 关键字列表，可以是字符串或列表
                     默认: ["关闭", "同意", "取消", "确定", "确认"]
        
        Returns:
            dict: 包含坐标信息的字典
        """
        # 设置默认关键字
        if keywords is None:
            keywords = [
                "关闭", "同意", "取消", "确定", "确认", 
                "登录", "注册", "提交", "下一步", "继续",
                "允许", "拒绝", "接受", "退出", "返回"
            ]
        elif isinstance(keywords, str):
            keywords = [keywords]
        
        start_time = time.time()
        
        try:
            # OCR识别
            results = self.reader.readtext(image_path)
            
            if not results:
                return {"error": "未识别到任何文字"}
            
            # 查找匹配的按钮
            found_buttons = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 置信度过滤
                    # 检查是否包含任何关键字
                    for keyword in keywords:
                        if keyword in text:
                            # 计算坐标
                            bbox = np.array(bbox)
                            center_x = int(np.mean(bbox[:, 0]))
                            center_y = int(np.mean(bbox[:, 1]))
                            
                            # 计算边界框
                            min_x = int(np.min(bbox[:, 0]))
                            max_x = int(np.max(bbox[:, 0]))
                            min_y = int(np.min(bbox[:, 1]))
                            max_y = int(np.max(bbox[:, 1]))
                            
                            found_buttons.append({
                                "keyword": keyword,
                                "full_text": text,
                                "confidence": round(confidence, 3),
                                "center": {"x": center_x, "y": center_y},
                                "bbox": {
                                    "x": min_x, "y": min_y,
                                    "width": max_x - min_x,
                                    "height": max_y - min_y
                                },
                                "corners": bbox.tolist()
                            })
                            break  # 找到就跳出，避免重复匹配
            
            total_time = time.time() - start_time
            
            if found_buttons:
                # 返回第一个找到的按钮（通常是最显眼的）
                best_button = found_buttons[0]
                return {
                    "found": True,
                    "matched_keyword": best_button["keyword"],
                    "detected_text": best_button["full_text"],
                    "confidence": best_button["confidence"],
                    "center": best_button["center"],
                    "bbox": best_button["bbox"],
                    "corners": best_button["corners"],
                    "all_buttons": found_buttons,  # 所有找到的按钮
                    "processing_time": f"{total_time:.3f}s"
                }
            else:
                # 没找到任何匹配的按钮
                all_texts = [text for (_, text, conf) in results if conf > 0.5]
                return {
                    "found": False,
                    "target_keywords": keywords,
                    "all_detected_texts": all_texts,
                    "processing_time": f"{total_time:.3f}s"
                }
        
        except Exception as e:
            return {"error": f"处理失败: {str(e)}"}


def quick_find_button(image_path, keywords=["关闭", "同意", "取消"]):
    """
    快速查找按钮 - 一行代码版本
    
    Args:
        image_path: 图片路径
        keywords: 关键字列表
    
    Returns:
        tuple: (是否找到, 坐标, 匹配的关键字, 完整文字)
    """
    try:
        reader = easyocr.Reader(['ch_sim', 'en'])
        results = reader.readtext(image_path)
        
        for (bbox, text, conf) in results:
            if conf > 0.5:
                for keyword in keywords:
                    if keyword in text:
                        bbox = np.array(bbox)
                        center_x = int(np.mean(bbox[:, 0]))
                        center_y = int(np.mean(bbox[:, 1]))
                        return True, (center_x, center_y), keyword, text
        
        return False, None, None, None
    
    except Exception as e:
        print(f"错误: {e}")
        return False, None, None, None

# 主程序
def main():
    
    # 检查当前目录的图片文件
    image_files = []
    for file in os.listdir("."):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_files.append(file)
            print(f"   {file}")
    
    # 获取图片路径
    if image_files:
        print(f"\n自动选择第一个图片: {image_files[0]}")
        image_path = image_files[0]
    else:
        print("\n未找到图片文件，请手动输入图片路径:")
        image_path = input("图片路径: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f" 图片不存在: {image_path}")
        return
    
    print(f"\n 准备处理图片: {image_path}")
    
    # 初始化检测器
    detector = TextDetector()
    
    # 测试不同的关键字组合
    test_cases = [
        {
            "name": "默认按钮",
            "keywords": None  # 使用默认关键字
        },
        {
            "name": "弹窗按钮", 
            "keywords": ["关闭", "同意", "取消", "确定", "允许"]
        },
        {
            "name": "操作按钮",
            "keywords": ["登录", "注册", "提交", "发送", "下载"]
        }
    ]
    
    for case in test_cases:
        print(f"\n 测试场景: {case['name']}")
        result = detector.find_buttons(image_path, case['keywords'])
        
        if result.get("found"):
            print(f" 找到按钮: '{result['detected_text']}'")
            print(f" 匹配关键字: '{result['matched_keyword']}'")
            print(f" 中心坐标: ({result['center']['x']}, {result['center']['y']})")
            print(f" 置信度: {result['confidence']}")
            print(f" 处理时间: {result['processing_time']}")
            
            # 如果找到了多个按钮，显示所有的
            if len(result.get('all_buttons', [])) > 1:
                print(f" 总共找到 {len(result['all_buttons'])} 个按钮:")
                for i, btn in enumerate(result['all_buttons']):
                    print(f"   {i+1}. '{btn['full_text']}' -> {btn['keyword']} 坐标:({btn['center']['x']}, {btn['center']['y']})")
            break  # 找到就停止
        else:
            print(f" 未找到相关按钮")
            if result.get("all_detected_texts"):
                print(f" 识别到的文字: {result['all_detected_texts'][:5]}...")  # 只显示前5个
    
    # 演示快速查找
    print(f"\n 快速查找演示:")
    found, center, keyword, text = quick_find_button(image_path, ["关闭", "同意", "取消"])
    if found:
        print(f" 快速找到: '{text}' -> 关键字:'{keyword}' 坐标:{center}")
    else:
        print(" 快速查找未找到")

# 使用示例和测试
# def demo_usage():
   
    
#     # 方法1: 使用类（推荐，功能最完整）
#     detector = TextDetector()
#     result = detector.find_buttons("screenshot.png", ["关闭", "同意"])
    
#     # 方法2: 快速函数（简单直接）
#     found, pos, keyword, text = quick_find_button("screenshot.png", ["关闭"])
    
#     # 方法3: 自定义场景
#     login_keywords = ["登录", "注册", "忘记密码"]
#     popup_keywords = ["确定", "取消", "关闭", "同意"]
    
#     print("示例代码:")
#     print("""
#     # 查找登录按钮
#     result = detector.find_buttons("app.png", ["登录", "注册"])
#     if result["found"]:
#         x, y = result["center"]["x"], result["center"]["y"]
#         print(f"登录按钮坐标: ({x}, {y})")
    
#     # 快速查找
#     found, pos, keyword, text = quick_find_button("popup.png", ["确定", "取消"])
#     if found:
#         print(f"找到 {keyword} 按钮: {pos}")
#     """)

if __name__ == "__main__":
   
    main()
    
   