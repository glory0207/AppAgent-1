import easyocr
import numpy as np
import time
import os

def find_button(image_path):
    """
    在图片中快速查找 关闭/同意/取消 按钮
    
    Args:
        image_path: 图片路径
    
    Returns:
        dict: 包含结果和坐标信息
    """
    # 目标关键字 - 只要这三个
    keywords = ["关闭", "同意", "取消"]
    
    start_time = time.time()
    
    try:
        # 初始化OCR
        reader = easyocr.Reader(['ch_sim', 'en'])
        model_time = time.time() - start_time
        print(f"模型加载完成！耗时: {model_time:.3f}s")
        
        # 开始识别
        ocr_start = time.time()
        results = reader.readtext(image_path)
        ocr_time = time.time() - ocr_start
        
        if not results:
            return {
                "found": False,
                "error": "未识别到任何文字",
                "model_load_time": f"{model_time:.3f}s",
                "ocr_time": f"{ocr_time:.3f}s",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        
        # 查找目标按钮
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # 置信度过滤
                # 检查是否包含目标关键字
                for keyword in keywords:
                    if keyword in text:
                        # 计算中心坐标
                        bbox = np.array(bbox)
                        center_x = int(np.mean(bbox[:, 0]))
                        center_y = int(np.mean(bbox[:, 1]))
                        
                        total_time = time.time() - start_time
                        
                        return {
                            "found": True,
                            "keyword": keyword,
                            "full_text": text,
                            "center": {"x": center_x, "y": center_y},
                            "confidence": round(confidence, 3),
                            "model_load_time": f"{model_time:.3f}s",
                            "ocr_time": f"{ocr_time:.3f}s", 
                            "total_time": f"{total_time:.3f}s"
                        }
        
        # 没找到目标按钮
        all_texts = [text for (_, text, conf) in results if conf > 0.5]
        total_time = time.time() - start_time
        
        return {
            "found": False,
            "keywords": keywords,
            "all_detected": all_texts,
            "model_load_time": f"{model_time:.3f}s",
            "ocr_time": f"{ocr_time:.3f}s",
            "total_time": f"{total_time:.3f}s"
        }
        
    except Exception as e:
        return {
            "found": False,
            "error": f"处理失败: {str(e)}",
            "total_time": f"{time.time() - start_time:.3f}s"
        }

def main():
    
    # 查找图片文件
    image_files = [f for f in os.listdir(".") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
        image_path = image_files[0]
    else:
        print("当前目录没有图片文件，请输入图片路径:")
        image_path = input("图片路径: ").strip().strip('"')
    
    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        return
    
    
    # 查找按钮
    result = find_button(image_path)
    
    # 显示结果
    print("\n" + "="*50)
    if result["found"]:
        print(f"关键字: {result['keyword']}")
        print(f"完整文字: {result['full_text']}")
        print(f"中心坐标: ({result['center']['x']}, {result['center']['y']})")
        print(f"置信度: {result['confidence']}")
    else:
        print(f"未找到目标按钮")
        if "all_detected" in result:
            print(f"识别到的文字: {result['all_detected']}")
        if "error" in result:
            print(f"错误信息: {result['error']}")
    
    # 显示时间统计
    if "model_load_time" in result:
        print(f"   模型加载: {result['model_load_time']}")
    if "ocr_time" in result:
        print(f"   文字识别: {result['ocr_time']}")
    print(f"   总耗时: {result['total_time']}")
    print("="*50)


if __name__ == "__main__":
    main()
    
   