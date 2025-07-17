"""
EasyOCR按钮查找工具 - 去掉缓存功能

- 压缩到800px
- 单例模式，避免重复加载模型
- 坐标自动还原到原图尺寸
- 中文按钮识别

- 约1-2秒（首次）
"""

import easyocr
import numpy as np
import time
import os
import cv2

def detect_close_button_by_shape(image_path):
    """
    通过形状识别关闭按钮（×号）
    改进版：支持不同位置和样式，提高准确性
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 方法1：检测圆形关闭按钮（右上角）- 提高检测精度
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=40,  # 提高阈值，减少误检测
            minRadius=15,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 检查是否在右上角区域 - 更严格的位置要求
                if x > img.shape[1] * 0.75 and y < img.shape[0] * 0.25:
                    return {
                        "found": True,
                        "center": {"x": x, "y": y},
                        "radius": r,
                        "type": "circular_close_button"
                    }
        
        # 方法2：检测角落的×号（左上角或右上角）- 提高精度
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 计算轮廓面积和边界框
            area = cv2.contourArea(contour)
            if 100 < area < 800:  # 更严格的面积范围
                x, y, w, h = cv2.boundingRect(contour)
                
                # 检查是否在角落区域（左上角或右上角）- 更严格的位置要求
                in_top_left = x < img.shape[1] * 0.08 and y < img.shape[0] * 0.08
                in_top_right = x > img.shape[1] * 0.85 and y < img.shape[0] * 0.08
                
                if (in_top_left or in_top_right) and abs(w - h) < min(w, h) * 0.3:  # 更严格的正方形要求
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    return {
                        "found": True,
                        "center": {"x": center_x, "y": center_y},
                        "area": area,
                        "type": "corner_close_button"
                    }
        
        return None
        
    except Exception as e:
        print(f"形状识别失败: {e}")
        return None

def detect_close_button_by_template(image_path):
    """
    通过模板匹配识别×号关闭按钮
    提高准确性，减少误识别
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 创建多个不同大小的×号模板
        results = []
        
        for template_size in [16, 20, 24, 30, 36]:
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            
            # 画×号 - 修复颜色参数
            thickness = max(1, template_size // 10)
            margin = template_size // 4
            cv2.line(template, (margin, margin), (template_size-margin, template_size-margin), (255,), thickness)
            cv2.line(template, (template_size-margin, margin), (margin, template_size-margin), (255,), thickness)
            
            # 模板匹配
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 记录结果
            results.append({
                'max_val': max_val,
                'location': max_loc,
                'template_size': template_size
            })
        
        # 找到最佳匹配
        best_result = max(results, key=lambda x: x['max_val'])
        
        # 提高匹配阈值，减少误识别
        if best_result['max_val'] > 0.5:  # 提高阈值从0.3到0.5
            x = best_result['location'][0] + best_result['template_size'] // 2
            y = best_result['location'][1] + best_result['template_size'] // 2
            
            # 额外检查：确保在合理的位置（角落区域）
            h, w = gray.shape
            in_corner = (x < w * 0.2 and y < h * 0.2) or (x > w * 0.8 and y < h * 0.2)
            
            if in_corner:
                return {
                    "found": True,
                    "center": {"x": x, "y": y},
                    "confidence": best_result['max_val'],
                    "template_size": best_result['template_size'],
                    "type": "template_match"
                }
        
        return None
        
    except Exception as e:
        print(f"模板匹配失败: {e}")
        return None

def detect_simple_x_button(image_path):
    """
    检测简单的×号按钮（如左上角、右上角的×号）
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 在左上角和右上角区域寻找×号
        h, w = gray.shape
        
        # 定义搜索区域 - 扩大搜索范围
        search_regions = [
            (0, 0, w//3, h//3),  # 左上角 - 扩大范围
            (w*2//3, 0, w//3, h//3),  # 右上角 - 扩大范围
            (w*3//4, 0, w//4, h//4),  # 右上角小区域
        ]
        
        for region_x, region_y, region_w, region_h in search_regions:
            # 提取搜索区域
            roi = gray[region_y:region_y+region_h, region_x:region_x+region_w]
            
            # 尝试多种二值化方法
            binary_methods = []
            
            # 方法1：自适应阈值
            try:
                adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                binary_methods.append(adaptive_binary)
            except:
                pass
            
            # 方法2：OTSU二值化
            try:
                _, otsu_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_methods.append(otsu_binary)
            except:
                pass
            
            # 方法3：反向二值化（针对浅色×号）
            try:
                _, inv_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                binary_methods.append(inv_binary)
            except:
                pass
            
            # 方法4：固定阈值
            for threshold in [100, 127, 150, 200]:
                try:
                    _, fixed_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
                    binary_methods.append(fixed_binary)
                    _, fixed_inv_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
                    binary_methods.append(fixed_inv_binary)
                except:
                    pass
            
            # 对每种二值化方法查找轮廓
            for binary in binary_methods:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 10 < area < 1000:  # 扩大面积范围
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        
                        # 检查是否接近正方形（×号通常是正方形的）
                        aspect_ratio = w_box / h_box
                        if 0.3 < aspect_ratio < 3.0:  # 放宽宽高比限制
                            # 计算在原图中的坐标
                            center_x = region_x + x + w_box // 2
                            center_y = region_y + y + h_box // 2
                            
                            # 检查是否在合理位置（避免误检测）
                            if region_x > w//2:  # 右侧区域
                                if center_y < h//2:  # 上半部分
                                    return {
                                        "found": True,
                                        "center": {"x": center_x, "y": center_y},
                                        "area": area,
                                        "region": f"right_top_{region_x}_{region_y}",
                                        "type": "simple_x_button"
                                    }
                            else:  # 左侧区域
                                if center_y < h//2:  # 上半部分
                                    return {
                                        "found": True,
                                        "center": {"x": center_x, "y": center_y},
                                        "area": area,
                                        "region": f"left_top_{region_x}_{region_y}",
                                        "type": "simple_x_button"
                                    }
        
        return None
        
    except Exception as e:
        print(f"简单×号检测失败: {e}")
        return None

def detect_bottom_center_close_button(image_path):
    """
    检测底部中间位置的圆形关闭按钮
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 定义底部中间区域
        bottom_region_y = int(h * 0.7)  # 底部30%区域
        center_margin = int(w * 0.2)    # 中间60%区域
        
        # 提取底部中间区域
        roi = gray[bottom_region_y:h, center_margin:w-center_margin]
        
        if roi.size == 0:
            return None
        
        # 使用霍夫圆变换检测圆形
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=25,  # 降低阈值
            minRadius=10,
            maxRadius=40
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # 找到最靠近底部中间的圆
            best_circle = None
            best_score = float('inf')
            
            for (x, y, r) in circles:
                # 计算到底部中间的距离
                actual_x = center_margin + x
                actual_y = bottom_region_y + y
                
                # 距离底部中间的分数（越小越好）
                center_x = w // 2
                bottom_y = h - 50  # 距离底部50像素
                
                distance = ((actual_x - center_x) ** 2 + (actual_y - bottom_y) ** 2) ** 0.5
                
                if distance < best_score:
                    best_score = distance
                    best_circle = (actual_x, actual_y, r)
            
            if best_circle and best_score < 100:  # 距离阈值
                return {
                    "found": True,
                    "center": {"x": best_circle[0], "y": best_circle[1]},
                    "radius": best_circle[2],
                    "distance_score": best_score,
                    "type": "bottom_center_close_button"
                }
        
        # 如果圆形检测失败，尝试轮廓检测
        # 二值化
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # 适合关闭按钮的面积范围
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # 检查是否接近圆形
                aspect_ratio = w_box / h_box
                if 0.7 < aspect_ratio < 1.3:  # 接近正方形
                    # 计算在原图中的坐标
                    actual_x = center_margin + x + w_box // 2
                    actual_y = bottom_region_y + y + h_box // 2
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "area": area,
                        "type": "bottom_center_contour"
                    }
        
        return None
        
    except Exception as e:
        print(f"底部中间关闭按钮检测失败: {e}")
        return None

def detect_top_right_x_button(image_path):
    """
    专门检测右上角的×号按钮
    使用边缘检测和线条检测
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 定义右上角区域
        top_right_x = int(w * 0.7)
        top_right_y = 0
        top_right_w = w - top_right_x
        top_right_h = int(h * 0.3)
        
        # 提取右上角区域
        roi = gray[top_right_y:top_right_y + top_right_h, top_right_x:top_right_x + top_right_w]
        
        if roi.size == 0:
            return None
        
        # 边缘检测
        edges = cv2.Canny(roi, 50, 150)
        
        # 使用霍夫线变换检测线条
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            # 分析线条，寻找×形状
            diagonal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线条角度
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # 寻找对角线（接近45度或-45度）
                if abs(abs(angle) - 45) < 20 or abs(abs(angle) - 135) < 20:
                    diagonal_lines.append(line[0])
            
            if len(diagonal_lines) >= 2:
                # 找到可能的×号中心
                centers = []
                for i, line1 in enumerate(diagonal_lines):
                    for j, line2 in enumerate(diagonal_lines[i+1:], i+1):
                        # 计算两条线的交点
                        x1, y1, x2, y2 = line1
                        x3, y3, x4, y4 = line2
                        
                        # 计算交点
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if abs(denom) > 1e-10:
                            px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                            py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                            
                            # 检查交点是否在ROI内
                            if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
                                centers.append((px, py))
                
                if centers:
                    # 选择最靠近右上角的交点
                    best_center = min(centers, key=lambda p: p[0]**2 + p[1]**2)
                    
                    # 转换为原图坐标
                    actual_x = int(top_right_x + best_center[0])
                    actual_y = int(top_right_y + best_center[1])
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "line_count": len(diagonal_lines),
                        "type": "top_right_x_lines"
                    }
        
        # 如果线条检测失败，尝试轮廓检测
        # 多种二值化方法
        binary_methods = []
        
        # 尝试不同的阈值
        for threshold in [80, 100, 120, 140, 160, 180, 200]:
            _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
            binary_methods.append(binary)
            _, inv_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
            binary_methods.append(inv_binary)
        
        for binary in binary_methods:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 800:  # 适合×号的面积范围
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # 检查宽高比
                    aspect_ratio = w_box / h_box
                    if 0.5 < aspect_ratio < 2.0:
                        # 转换为原图坐标
                        actual_x = top_right_x + x + w_box // 2
                        actual_y = top_right_y + y + h_box // 2
                        
                        return {
                            "found": True,
                            "center": {"x": actual_x, "y": actual_y},
                            "area": area,
                            "type": "top_right_x_contour"
                        }
        
        return None
        
    except Exception as e:
        print(f"右上角×号检测失败: {e}")
        return None

class OptimizedButtonFinder:
    """单例模式按钮识别器"""
    
    def __init__(self):
        """初始化OCR模型"""
        start_time = time.time()
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}s")
        
        # 目标关键字
        self.keywords = ["关闭","同意",  "确定","更新微信","继续访问","我知道了","同意并继续","放弃","取消", "重试"]
    
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
        按钮查找函数
        修改执行顺序：优先OCR文字识别，图像识别作为备选
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
        
        # 2. 进行OCR文字识别
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
        
        # 3. 检查OCR结果中是否有目标关键字
        all_detected_texts = []
        
        if results:
            for (bbox, text, confidence) in results:
                # 置信度过滤
                try:
                    conf_float = float(confidence) if isinstance(confidence, str) else confidence
                    if conf_float > 0.5:
                        all_detected_texts.append(f"{text}({conf_float:.2f})")
                        
                        # 检查是否包含目标关键字
                        for keyword in self.keywords:
                            if keyword in text:
                                # 计算按钮中心坐标（原图尺寸）
                                bbox = np.array(bbox)
                                center_x = int(np.mean(bbox[:, 0]) * self.scale_factor)
                                center_y = int(np.mean(bbox[:, 1]) * self.scale_factor)
                                
                                # 计算原图的边界框坐标
                                original_bbox = (bbox * self.scale_factor).astype(int).tolist()
                                
                                result = {
                                    "found": True,
                                    "keyword": keyword,
                                    "full_text": text,
                                    "center": {"x": center_x, "y": center_y},
                                    "bbox": original_bbox,
                                    "confidence": round(conf_float, 3),
                                    "detection_method": "ocr",
                                    "preprocess_time": f"{preprocess_time:.3f}s",
                                    "ocr_time": f"{ocr_time:.3f}s",
                                    "shape_time": "0.000s",
                                    "total_time": f"{time.time() - start_time:.3f}s"
                                }
                                
                                return result
                except (ValueError, TypeError):
                    # 跳过无法转换的confidence值
                    continue
        
        # 4. 如果OCR未找到关键字，再尝试图像识别关闭按钮（针对图标化的×号）
        shape_start = time.time()
        
        # 尝试多种图像识别方法
        image_detection_result = None
        
        # 方法1：形状识别（圆形关闭按钮）
        shape_result = detect_close_button_by_shape(image_path)
        if shape_result and shape_result["found"]:
            image_detection_result = shape_result
        
        # 方法2：简单×号检测（左上角、右上角的×号）
        if not image_detection_result:
            simple_x_result = detect_simple_x_button(image_path)
            if simple_x_result and simple_x_result["found"]:
                image_detection_result = simple_x_result
        
        # 方法3：模板匹配
        if not image_detection_result:
            template_result = detect_close_button_by_template(image_path)
            if template_result and template_result["found"]:
                image_detection_result = template_result
        
        # 方法4：底部中间圆形关闭按钮
        if not image_detection_result:
            bottom_center_result = detect_bottom_center_close_button(image_path)
            if bottom_center_result and bottom_center_result["found"]:
                image_detection_result = bottom_center_result
        
        # 方法5：右上角×号检测
        if not image_detection_result:
            top_right_result = detect_top_right_x_button(image_path)
            if top_right_result and top_right_result["found"]:
                image_detection_result = top_right_result
        
        shape_time = time.time() - shape_start
        
        # 5. 如果图像识别找到了×号，返回结果
        if image_detection_result:
            result = {
                "found": True,
                "keyword": "×",
                "full_text": f"×({image_detection_result['type']})",
                "center": image_detection_result["center"],
                "confidence": image_detection_result.get("confidence", 0.8),
                "detection_method": "image_recognition",
                "detection_type": image_detection_result["type"],
                "preprocess_time": f"{preprocess_time:.3f}s",
                "ocr_time": f"{ocr_time:.3f}s",
                "shape_time": f"{shape_time:.3f}s",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
            
            return result
        
        # 6. 没有找到目标按钮
        result = {
            "found": False,
            "keywords_searched": self.keywords,
            "all_detected": all_detected_texts,
            "preprocess_time": f"{preprocess_time:.3f}s",
            "ocr_time": f"{ocr_time:.3f}s",
            "shape_time": f"{shape_time:.3f}s",
            "total_time": f"{time.time() - start_time:.3f}s"
        }
        
        return result

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
    image_path = "page/27.png"  
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return
    
    print(f"开始识别按钮: {image_path}")
    
    # 执行按钮查找
    result = find_button_optimized(image_path)
    
    # 显示识别结果
    if result["found"]:
        print(f"✓ 找到目标按钮!")
        print(f"    关键字: {result['keyword']}")
        print(f"    完整文字: {result['full_text']}")
        print(f"    中心坐标: ({result['center']['x']}, {result['center']['y']})")
        print(f"    置信度: {result['confidence']}")
        print(f"    点击代码: pyautogui.click({result['center']['x']}, {result['center']['y']})")
    else:
        print(f"✗ 未找到目标按钮")
        if "keywords_searched" in result:
            print(f"    搜索的关键字: {', '.join(result['keywords_searched'])}")
        if "all_detected" in result and result["all_detected"]:
            print(f"    检测到的所有文字:")
            for text in result["all_detected"]:
                print(f"      • {text}")
        if "error" in result:
            print(f"    错误信息: {result['error']}")
    
    # 性能统计
    print(f"性能统计:")
    if "preprocess_time" in result:
        print(f"    图片预处理: {result['preprocess_time']}")
    if "ocr_time" in result:
        print(f"    文字识别: {result['ocr_time']}")
    if "shape_time" in result:
        print(f"    图像识别: {result['shape_time']}")
    print(f"    总耗时: {result['total_time']}")
    print("=" * 60)

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

def verify_coordinates(image_path, x, y):
    """验证坐标是否正确"""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            # 在指定坐标画红点
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(img, f"({x},{y})", (x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            verify_path = f"verify_{x}_{y}.png"
            cv2.imwrite(verify_path, img)
            print(f"坐标验证图片已保存: {verify_path}")
            return True
    except:
        pass
    return False

def test_close_button_recognition(image_path):
    """
    专门测试×号等关闭按钮的识别效果
    """
    print(f"测试×号识别: {image_path}")
    print("=" * 50)
    
    # 临时设置只识别关闭相关的符号
    finder = get_button_finder()
    original_keywords = finder.keywords.copy()
    
    # 设置专门的关闭符号关键字
    close_keywords = ["×", "✕", "x", "X", "Close", "close", "关闭"]
    finder.keywords = close_keywords
    
    print(f"搜索关键字: {', '.join(close_keywords)}")
    
    try:
        result = find_button_optimized(image_path)
        
        if result["found"]:
            print(f"  ✓ 找到关闭按钮!")
            print(f"    匹配关键字: {result['keyword']}")
            print(f"    完整识别文字: {result['full_text']}")
            print(f"    中心坐标: ({result['center']['x']}, {result['center']['y']})")
            print(f"    置信度: {result['confidence']}")
            print(f"    点击代码: pyautogui.click({result['center']['x']}, {result['center']['y']})")
            
            # 生成验证图片
            if verify_coordinates(image_path, result['center']['x'], result['center']['y']):
                print(f"    验证图片已生成")
        else:
            print(f"  ✗ 未找到关闭按钮")
            if "all_detected" in result and result["all_detected"]:
                print(f"    但识别到了这些文字:")
                for text in result["all_detected"]:
                    print(f"      • {text}")
            if "error" in result:
                print(f"    错误: {result['error']}")
        
        print(f"  识别耗时: {result['total_time']}")
        
    finally:
        # 恢复原始关键字
        finder.keywords = original_keywords
    
    print("=" * 50)
    return result

# 批量测试多个图片的×号识别
def batch_test_close_buttons(image_dir="page"):
    """批量测试目录中所有图片的×号识别"""
    import glob
    
    if not os.path.exists(image_dir):
        print(f"目录不存在: {image_dir}")
        return
    
    # 支持的图片格式
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_files:
        print(f"在 {image_dir} 目录中未找到图片文件")
        return
    
    print(f"批量测试×号识别 - 共 {len(image_files)} 个图片")
    print("=" * 60)
    
    found_count = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 测试: {os.path.basename(image_path)}")
        result = test_close_button_recognition(image_path)
        if result["found"]:
            found_count += 1
        print()
    
    print(f"总结: {found_count}/{len(image_files)} 个图片找到了关闭按钮")
    print("=" * 60)

def test_icon_close_button(image_path):
    """
    专门测试图标化关闭按钮的识别
    """
    print(f"测试图标化×号识别: {image_path}")
    print("=" * 50)
    
    # 1. 尝试形状识别
    print("1. 形状识别测试（圆形按钮）...")
    shape_result = detect_close_button_by_shape(image_path)
    if shape_result and shape_result["found"]:
        print(f"  ✓ 形状识别成功!")
        print(f"    类型: {shape_result['type']}")
        print(f"    中心坐标: ({shape_result['center']['x']}, {shape_result['center']['y']})")
        if 'radius' in shape_result:
            print(f"    半径: {shape_result['radius']}")
        verify_coordinates(image_path, shape_result['center']['x'], shape_result['center']['y'])
    else:
        print(f"  ✗ 形状识别失败")
    
    # 2. 尝试简单×号检测
    print("2. 简单×号检测测试...")
    simple_result = detect_simple_x_button(image_path)
    if simple_result and simple_result["found"]:
        print(f"  ✓ 简单×号检测成功!")
        print(f"    类型: {simple_result['type']}")
        print(f"    中心坐标: ({simple_result['center']['x']}, {simple_result['center']['y']})")
        print(f"    面积: {simple_result['area']}")
        print(f"    区域: {simple_result['region']}")
        verify_coordinates(image_path, simple_result['center']['x'], simple_result['center']['y'])
    else:
        print(f"  ✗ 简单×号检测失败")
    
    # 3. 尝试模板匹配
    print("3. 模板匹配测试...")
    template_result = detect_close_button_by_template(image_path)
    if template_result and template_result["found"]:
        print(f"  ✓ 模板匹配成功!")
        print(f"    类型: {template_result['type']}")
        print(f"    中心坐标: ({template_result['center']['x']}, {template_result['center']['y']})")
        print(f"    置信度: {template_result['confidence']:.3f}")
        print(f"    模板大小: {template_result['template_size']}")
        verify_coordinates(image_path, template_result['center']['x'], template_result['center']['y'])
    else:
        print(f"  ✗ 模板匹配失败")
    
    # 4. 尝试底部中间关闭按钮检测
    print("4. 底部中间关闭按钮检测...")
    bottom_result = detect_bottom_center_close_button(image_path)
    if bottom_result and bottom_result["found"]:
        print(f"  ✓ 底部中间关闭按钮检测成功!")
        print(f"    类型: {bottom_result['type']}")
        print(f"    中心坐标: ({bottom_result['center']['x']}, {bottom_result['center']['y']})")
        if 'radius' in bottom_result:
            print(f"    半径: {bottom_result['radius']}")
        if 'distance_score' in bottom_result:
            print(f"    距离分数: {bottom_result['distance_score']:.1f}")
        verify_coordinates(image_path, bottom_result['center']['x'], bottom_result['center']['y'])
    else:
        print(f"  ✗ 底部中间关闭按钮检测失败")
    
    # 5. 尝试右上角×号检测
    print("5. 右上角×号检测...")
    top_right_result = detect_top_right_x_button(image_path)
    if top_right_result and top_right_result["found"]:
        print(f"  ✓ 右上角×号检测成功!")
        print(f"    类型: {top_right_result['type']}")
        print(f"    中心坐标: ({top_right_result['center']['x']}, {top_right_result['center']['y']})")
        if 'line_count' in top_right_result:
            print(f"    检测到线条数: {top_right_result['line_count']}")
        if 'area' in top_right_result:
            print(f"    面积: {top_right_result['area']}")
        verify_coordinates(image_path, top_right_result['center']['x'], top_right_result['center']['y'])
    else:
        print(f"  ✗ 右上角×号检测失败")
    
    # 6. 尝试完整的按钮识别（包含新的图像识别）
    print("6. 完整识别测试...")
    result = find_button_optimized(image_path)
    if result["found"]:
        print(f"  ✓ 完整识别成功!")
        print(f"    识别方法: {result.get('detection_method', 'unknown')}")
        if 'detection_type' in result:
            print(f"    检测类型: {result['detection_type']}")
        print(f"    关键字: {result['keyword']}")
        print(f"    完整文字: {result['full_text']}")
        print(f"    中心坐标: ({result['center']['x']}, {result['center']['y']})")
        print(f"    置信度: {result['confidence']}")
        print(f"    总耗时: {result['total_time']}")
    else:
        print(f"  ✗ 完整识别失败")
        if "all_detected" in result:
            print(f"    检测到的文字: {result['all_detected']}")
        if "error" in result:
            print(f"    错误: {result['error']}")
    
    print("=" * 50)
    return result

def test_all_methods(image_path):
    """测试所有识别方法"""
    print(f"综合测试: {image_path}")
    print("=" * 60)
    
    # 保存一个测试图片到临时目录
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    test_icon_close_button(image_path)
    
    print("=" * 60)

if __name__ == "__main__":
    main()
    verify_coordinates("page/27.png", 782, 132)