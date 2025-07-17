"""
EasyOCR按钮查找工具 - 轻度压缩

- 压缩到800px
- 单例模式，避免重复加载模型
- 坐标自动还原到原图尺寸
- 支持中文按钮识别

- 约1-2秒
"""

import easyocr
import numpy as np
import time
import os
import cv2

class OptimizedButtonFinder:
    """单例模式避免重复加载模型"""
    
    def __init__(self):
        """初始化OCR模型"""
        start_time = time.time()
        self.reader = easyocr.Reader(['ch_sim'], gpu=False, verbose=False)
        
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成，耗时: {load_time:.2f}s")
        
        # 目标关键字
        self.keywords = ["关闭", "同意", "取消", "确定", "确认", "OK"]
    
    def preprocess_image(self, image_path):
        """
        轻度压缩预处理 - 固定800px压缩
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            original_size = max(height, width)
            
            # 轻度压缩：固定800px
            max_size = 800
            if original_size > max_size:
                scale_factor = max_size / original_size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # 使用INTER_CUBIC高质量插值
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f" 轻度压缩: {width}x{height} → {new_width}x{new_height} (缩放比: {scale_factor:.3f})")
                
                # 保存缩放比例，用于还原真实坐标
                self.scale_factor = 1.0 / scale_factor
            else:
                print(f" 图片尺寸合适，无需压缩: {width}x{height}")
                self.scale_factor = 1.0
            
            return img
            
        except Exception as e:
            print(f" 图片预处理失败: {e}")
            return None
    
    def find_button(self, image_path):
        """
        优化版按钮查找函数
        """
        start_time = time.time()
        
        # 1. 图片预处理
        preprocess_start = time.time()
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return {
                "found": False,
                "error": "图片预处理失败",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        preprocess_time = time.time() - preprocess_start
        
        # 2. OCR文字识别
        ocr_start = time.time()
        try:
            results = self.reader.readtext(processed_img)
        except Exception as e:
            return {
                "found": False,
                "error": f"OCR识别失败: {str(e)}",
                "preprocess_time": f"{preprocess_time:.3f}s",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        
        ocr_time = time.time() - ocr_start
        
        # 3. 检查是否有识别结果
        if not results:
            return {
                "found": False,
                "error": "未识别到任何文字",
                "preprocess_time": f"{preprocess_time:.3f}s",
                "ocr_time": f"{ocr_time:.3f}s",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        
        # 4. 查找目标按钮
        all_detected_texts = []
        
        for (bbox, text, confidence) in results:
            # 置信度过滤
            if confidence > 0.5:
                all_detected_texts.append(f"{text}({confidence:.2f})")
                
                # 检查是否包含目标关键字
                for keyword in self.keywords:
                    if keyword in text:
                        # 计算按钮中心坐标（原图尺寸）
                        bbox = np.array(bbox)
                        center_x = int(np.mean(bbox[:, 0]) * self.scale_factor)
                        center_y = int(np.mean(bbox[:, 1]) * self.scale_factor)
                        
                        # 计算原图的边界框坐标
                        original_bbox = (bbox * self.scale_factor).astype(int).tolist()
                        
                        total_time = time.time() - start_time
                        
                        return {
                            "found": True,
                            "keyword": keyword,
                            "full_text": text,
                            "center": {"x": center_x, "y": center_y},
                            "bbox": original_bbox,  # 原图坐标系的边界框
                            "confidence": round(confidence, 3),
                            "preprocess_time": f"{preprocess_time:.3f}s",
                            "ocr_time": f"{ocr_time:.3f}s",
                            "total_time": f"{total_time:.3f}s"
                        }
        
        # 没有找到目标按钮
        total_time = time.time() - start_time
        return {
            "found": False,
            "keywords_searched": self.keywords,
            "all_detected": all_detected_texts,
            "preprocess_time": f"{preprocess_time:.3f}s",
            "ocr_time": f"{ocr_time:.3f}s",
            "total_time": f"{total_time:.3f}s"
        }

# 全局单例实例
_button_finder_instance = None

def get_button_finder():
    global _button_finder_instance
    if _button_finder_instance is None:
        _button_finder_instance = OptimizedButtonFinder()
    return _button_finder_instance

def find_button_optimized(image_path):
    
    finder = get_button_finder()
    return finder.find_button(image_path)

def main():
 
    
    image_path = "screenshot.png"  
    
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return
    
    print("-" * 50)
    
    # 执行按钮查找
    result = find_button_optimized(image_path)
    
    # 显示结果
    
    if result["found"]:
     
        print(f" 关键字: {result['keyword']}")
        print(f" 完整文字: {result['full_text']}")
        print(f" 中心坐标: ({result['center']['x']}, {result['center']['y']})")
        print(f" 置信度: {result['confidence']}")
        print(f" 边界框坐标: {result['bbox']}")
        print(f"   中心点击: ({result['center']['x']}, {result['center']['y']})")
        print(f"   # pyautogui.click({result['center']['x']}, {result['center']['y']})")
    else:
        print(f" 未找到目标按钮")
        if "keywords_searched" in result:
            print(f" 搜索的关键字: {', '.join(result['keywords_searched'])}")
        if "all_detected" in result and result["all_detected"]:
            print(f" 检测到的所有文字:")
            for text in result["all_detected"]:
                print(f"   • {text}")
        if "error" in result:
            print(f" 错误信息: {result['error']}")
    
    # 性能统计
    if "preprocess_time" in result:
        print(f"    图片预处理: {result['preprocess_time']}")
    if "ocr_time" in result:
        print(f"    文字识别: {result['ocr_time']}")
    print(f"    总耗时: {result['total_time']}")
    print("=" * 50)

# 简化版函数 - 只返回坐标
def get_button_coordinates(image_path, target_keywords=None):
    """
    简化版函数：直接返回按钮坐标
    
    Args:
        image_path: 图片路径
        target_keywords: 目标关键字列表，如果不指定则使用默认
    
    Returns:
        tuple: (x, y) 坐标，如果未找到返回 None
    """
    if target_keywords:
        finder = get_button_finder()
        finder.keywords = target_keywords
    
    result = find_button_optimized(image_path)
    
    if result["found"]:
        return (result['center']['x'], result['center']['y'])
    else:
        return None

# 使用示例
def example_usage():
    
    # 示例1：详细结果
    image_path = "screenshot.png"  # 你的图片路径
    result = find_button_optimized(image_path)
    print("详细结果:", result)
    
    # 示例2：只获取坐标
    coordinates = get_button_coordinates(image_path)
    if coordinates:
        x, y = coordinates
        print(f"按钮坐标: ({x}, {y})")
        # 可以用于自动化点击
        # pyautogui.click(x, y)
    else:
        print("未找到按钮")
    
    # 示例3：自定义关键字
    custom_coords = get_button_coordinates(image_path, ["提交", "发送", "保存"])
    if custom_coords:
        print(f"自定义按钮坐标: {custom_coords}")

# 快速验证坐标的辅助函数
def verify_coordinates(image_path, x, y):
    """快速验证坐标是否正确"""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            # 在指定坐标画红点
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(img, f"({x},{y})", (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            verify_path = f"verify_{x}_{y}.png"
            cv2.imwrite(verify_path, img)
            print(f" 坐标验证图片已保存: {verify_path}")
            return True
    except:
        pass
    return False

if __name__ == "__main__":
    main()
    
    # 如果想验证特定坐标
    verify_coordinates("screenshot.png", 668, 1281)