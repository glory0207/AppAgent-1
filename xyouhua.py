"""
批量EasyOCR按钮查找工具 - 支持URL版本 + 图片标注功能

功能：
- 从txt文件读取图片路径列表（支持本地路径和URL）
- 批量识别图片中的按钮
- 自动在识别成功的图片上标注按钮位置
- 保存带标注的图片到指定目录
- 输出JSON和TXT格式的识别结果
- 程序持续运行，支持多次处理

新增功能：
- 图片标注：识别成功后自动在原图上标注按钮位置
- 标注样式：红色圆圈+十字线标注中心点，白色文字标签显示关键字和置信度
- 自动保存：标注图片保存到带时间戳的目录中

最新优化 (v2.0)：
- 大幅提高×号检测精度，减少误报
- 增加形状验证：检查圆形度、实体度等几何特征
- 提高匹配阈值：模板匹配从0.5提升到0.65
- 缩小搜索范围：只在角落的小区域搜索×号
- 增加交叉验证：检查对角线交叉点的有效性
- 优化检测优先级：优先使用更精确的模板匹配
- 添加内容复杂度检查：避免识别纯色区域
"""

import easyocr
import numpy as np
import time
import os
import cv2
import json
from datetime import datetime
import sys
import requests
import tempfile
from urllib.parse import urlparse

# 保留所有原有的检测函数
def detect_close_button_by_shape(image_path):
    """通过形状识别关闭按钮（×号）- 更严格的检测"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 只检测圆形按钮，提高阈值，只在角落区域查找
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # 增加最小距离
            param1=60,   # 提高边缘检测阈值
            param2=50,   # 提高圆形检测阈值
            minRadius=18,  # 增加最小半径
            maxRadius=45   # 减少最大半径
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 更严格的位置检查：只在右上角很小的区域
                if x > w * 0.85 and y < h * 0.15:
                    # 检查圆形区域内是否有×形状的特征
                    roi = gray[max(0, y-r):min(h, y+r), max(0, x-r):min(w, x+r)]
                    
                    if roi.size > 0:
                        # 检查是否有交叉线条
                        edges = cv2.Canny(roi, 50, 150)
                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=r//2, maxLineGap=3)
                        
                        if lines is not None and len(lines) >= 2:
                            # 检查是否有交叉的对角线
                            diagonal_count = 0
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                                if abs(abs(angle) - 45) < 25 or abs(abs(angle) - 135) < 25:
                                    diagonal_count += 1
                            
                            if diagonal_count >= 2:
                                return {
                                    "found": True,
                                    "center": {"x": x, "y": y},
                                    "radius": r,
                                    "type": "circular_close_button",
                                    "diagonal_lines": diagonal_count
                                }
        
        return None
        
    except Exception as e:
        print(f"形状识别失败: {e}")
        return None

def detect_close_button_by_template(image_path):
    """通过模板匹配识别×号关闭按钮 - 更严格的检测"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 只在角落区域搜索
        corner_regions = [
            (0, 0, w//4, h//4),           # 左上角
            (w*3//4, 0, w//4, h//4),      # 右上角
        ]
        
        best_match = None
        best_score = 0
        
        for region_x, region_y, region_w, region_h in corner_regions:
            roi = gray[region_y:region_y+region_h, region_x:region_x+region_w]
            
            if roi.size == 0:
                continue
            
            results = []
            
            # 测试不同大小的×号模板
            for template_size in [20, 24, 28, 32]:
                template = np.zeros((template_size, template_size), dtype=np.uint8)
                
                # 创建更清晰的×号模板
                thickness = max(2, template_size // 12)
                margin = template_size // 5
                
                cv2.line(template, (margin, margin), (template_size-margin, template_size-margin), (255,), thickness)
                cv2.line(template, (template_size-margin, margin), (margin, template_size-margin), (255,), thickness)
                
                # 模板匹配
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = {
                        'max_val': max_val,
                        'location': (region_x + max_loc[0], region_y + max_loc[1]),
                        'template_size': template_size,
                        'region': (region_x, region_y)
                    }
        
        # 提高匹配阈值，减少误报
        if best_match and best_score > 0.65:  # 从0.5提高到0.65
            x = best_match['location'][0] + best_match['template_size'] // 2
            y = best_match['location'][1] + best_match['template_size'] // 2
            
            # 验证位置是否合理（确实在角落）
            if ((x < w * 0.25 and y < h * 0.25) or (x > w * 0.75 and y < h * 0.25)):
                return {
                    "found": True,
                    "center": {"x": x, "y": y},
                    "confidence": best_score,
                    "template_size": best_match['template_size'],
                    "type": "template_match"
                }
        
        return None
        
    except Exception as e:
        print(f"模板匹配失败: {e}")
        return None

def detect_simple_x_button(image_path):
    """检测简单的×号按钮 - 更严格的检测"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 只在角落的小区域搜索
        search_regions = [
            (0, 0, w//5, h//5),           # 左上角，缩小搜索范围
            (w*4//5, 0, w//5, h//5),      # 右上角，缩小搜索范围
        ]
        
        for region_x, region_y, region_w, region_h in search_regions:
            roi = gray[region_y:region_y+region_h, region_x:region_x+region_w]
            
            if roi.size == 0:
                continue
            
            # 减少二值化方法，只使用最可靠的几种
            binary_methods = []
            
            # 自适应阈值
            try:
                adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                binary_methods.append(adaptive_binary)
            except:
                pass
            
            # OTSU阈值
            try:
                _, otsu_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_methods.append(otsu_binary)
            except:
                pass
            
            for binary in binary_methods:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # 提高面积要求，减少小噪点
                    if 50 < area < 500:  # 从10-1000改为50-500
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        
                        # 更严格的长宽比检查
                        aspect_ratio = w_box / h_box
                        if 0.6 < aspect_ratio < 1.6:  # 从0.3-3.0改为0.6-1.6，更接近正方形
                            center_x = region_x + x + w_box // 2
                            center_y = region_y + y + h_box // 2
                            
                            # 验证轮廓的复杂性，×号应该有一定的复杂度
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                # ×号不应该太圆，但也不应该太不规则
                                if 0.2 < circularity < 0.8:
                                    # 检查轮廓的凸包特征
                                    hull = cv2.convexHull(contour)
                                    hull_area = cv2.contourArea(hull)
                                    if hull_area > 0:
                                        solidity = area / hull_area
                                        # ×号的实体度应该在合理范围内
                                        if 0.4 < solidity < 0.9:
                                            return {
                                                "found": True,
                                                "center": {"x": center_x, "y": center_y},
                                                "area": area,
                                                "circularity": circularity,
                                                "solidity": solidity,
                                                "region": f"corner_{region_x}_{region_y}",
                                                "type": "simple_x_button"
                                            }
        
        return None
        
    except Exception as e:
        print(f"简单×号检测失败: {e}")
        return None

def detect_bottom_center_close_button(image_path):
    """检测底部中间位置的圆形关闭按钮 - 更严格的检测"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 缩小搜索范围，更精确定位底部中心区域
        bottom_region_y = int(h * 0.75)  # 从0.7改为0.75
        center_margin = int(w * 0.3)     # 从0.2改为0.3，缩小水平搜索范围
        
        roi = gray[bottom_region_y:h, center_margin:w-center_margin]
        
        if roi.size == 0:
            return None
        
        # 方法1：圆形检测 - 提高要求
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,      # 增加最小距离
            param1=60,       # 提高边缘检测阈值
            param2=35,       # 提高圆形检测阈值
            minRadius=15,    # 增加最小半径
            maxRadius=35     # 减少最大半径
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            best_circle = None
            best_score = float('inf')
            
            for (x, y, r) in circles:
                actual_x = center_margin + x
                actual_y = bottom_region_y + y
                
                # 更严格的位置检查 - 必须在底部中心很小的范围内
                center_x = w // 2
                bottom_y = h - 40  # 从h-50改为h-40
                
                distance = ((actual_x - center_x) ** 2 + (actual_y - bottom_y) ** 2) ** 0.5
                
                # 距离要求更严格
                if distance < 60 and distance < best_score:  # 从100改为60
                    # 检查圆形区域内是否有合适的内容
                    circle_roi = roi[max(0, y-r):min(roi.shape[0], y+r), max(0, x-r):min(roi.shape[1], x+r)]
                    
                    if circle_roi.size > 0:
                        # 检查圆形区域的内容复杂度
                        mean_intensity = np.mean(circle_roi)
                        std_intensity = np.std(circle_roi)
                        
                        # 应该有一定的对比度（不是纯色）
                        if std_intensity > 15:
                            best_score = distance
                            best_circle = (actual_x, actual_y, r)
            
            if best_circle:
                return {
                    "found": True,
                    "center": {"x": best_circle[0], "y": best_circle[1]},
                    "radius": best_circle[2],
                    "distance_score": best_score,
                    "type": "bottom_center_close_button"
                }
        
        # 方法2：轮廓检测 - 更严格的条件
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 1200:  # 提高最小面积要求
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # 更严格的长宽比，应该接近正方形或圆形
                aspect_ratio = w_box / h_box
                if 0.75 < aspect_ratio < 1.25:  # 从0.7-1.3改为0.75-1.25
                    
                    actual_x = center_margin + x + w_box // 2
                    actual_y = bottom_region_y + y + h_box // 2
                    
                    # 检查是否真的在底部中心
                    center_x = w // 2
                    distance_to_center = abs(actual_x - center_x)
                    
                    if distance_to_center < w * 0.15:  # 距离中心不超过15%宽度
                        # 检查轮廓的复杂性
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            # 应该相对较圆（但不是完美圆形）
                            if 0.4 < circularity < 0.9:
                                return {
                                    "found": True,
                                    "center": {"x": actual_x, "y": actual_y},
                                    "area": area,
                                    "circularity": circularity,
                                    "distance_to_center": distance_to_center,
                                    "type": "bottom_center_contour"
                                }
        
        return None
        
    except Exception as e:
        print(f"底部中间关闭按钮检测失败: {e}")
        return None

def detect_top_right_x_button(image_path):
    """专门检测右上角的×号按钮 - 更严格的检测"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 缩小搜索范围，只在右上角很小的区域
        top_right_x = int(w * 0.8)    # 从0.7改为0.8
        top_right_y = 0
        top_right_w = w - top_right_x
        top_right_h = int(h * 0.2)    # 从0.3改为0.2
        
        roi = gray[top_right_y:top_right_y + top_right_h, top_right_x:top_right_x + top_right_w]
        
        if roi.size == 0:
            return None
        
        # 方法1：线条检测 - 提高要求
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=12, maxLineGap=3)  # 提高阈值
        
        if lines is not None:
            diagonal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算线条长度
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 8:  # 过滤太短的线条
                    continue
                    
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # 更严格的角度检查
                if abs(abs(angle) - 45) < 15 or abs(abs(angle) - 135) < 15:  # 从20改为15
                    diagonal_lines.append((line[0], length))
            
            # 需要至少2条对角线，且长度相似
            if len(diagonal_lines) >= 2:
                # 检查是否有交叉点
                valid_intersections = []
                for i, (line1, len1) in enumerate(diagonal_lines):
                    for j, (line2, len2) in enumerate(diagonal_lines[i+1:], i+1):
                        x1, y1, x2, y2 = line1
                        x3, y3, x4, y4 = line2
                        
                        # 检查长度是否相似
                        if abs(len1 - len2) / max(len1, len2) > 0.5:
                            continue
                        
                        # 计算交点
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if abs(denom) > 1e-10:
                            px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                            py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                            
                            # 交点应该在ROI内且不在边缘
                            if 5 <= px < roi.shape[1]-5 and 5 <= py < roi.shape[0]-5:
                                valid_intersections.append((px, py))
                
                if valid_intersections:
                    # 选择最佳交点（最接近中心的）
                    best_intersection = min(valid_intersections, 
                                          key=lambda p: (p[0] - roi.shape[1]/2)**2 + (p[1] - roi.shape[0]/2)**2)
                    
                    actual_x = int(top_right_x + best_intersection[0])
                    actual_y = int(top_right_y + best_intersection[1])
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "line_count": len(diagonal_lines),
                        "intersection_count": len(valid_intersections),
                        "type": "top_right_x_lines"
                    }
        
        # 方法2：轮廓检测 - 更严格的条件
        for threshold in [120, 140, 160]:  # 减少阈值数量，使用中等阈值
            _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 80 < area < 400:  # 提高最小面积要求
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # 更严格的长宽比
                    aspect_ratio = w_box / h_box
                    if 0.7 < aspect_ratio < 1.4:  # 从0.5-2.0改为0.7-1.4
                        
                        # 检查轮廓复杂性
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if 0.1 < circularity < 0.7:  # ×号不应该太圆
                                
                                # 检查轮廓的凸包特征
                                hull = cv2.convexHull(contour)
                                hull_area = cv2.contourArea(hull)
                                if hull_area > 0:
                                    solidity = area / hull_area
                                    if 0.3 < solidity < 0.8:  # ×号的实体度
                                        
                                        actual_x = top_right_x + x + w_box // 2
                                        actual_y = top_right_y + y + h_box // 2
                                        
                                        return {
                                            "found": True,
                                            "center": {"x": actual_x, "y": actual_y},
                                            "area": area,
                                            "circularity": circularity,
                                            "solidity": solidity,
                                            "type": "top_right_x_contour"
                                        }
        
        return None
        
    except Exception as e:
        print(f"右上角×号检测失败: {e}")
        return None

def is_url(path):
    """检查路径是否为URL"""
    return path.startswith(('http://', 'https://'))

def download_image_from_url(url, timeout=30):
    """从URL下载图片到临时文件"""
    try:
        print(f"  正在下载图片...")
        
        # 设置请求头，模拟浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 下载图片
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 从URL或Content-Type获取文件扩展名
        content_type = response.headers.get('Content-Type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            suffix = '.jpg'
        elif 'png' in content_type:
            suffix = '.png'
        elif 'gif' in content_type:
            suffix = '.gif'
        elif 'webp' in content_type:
            suffix = '.webp'
        else:
            # 从URL路径获取扩展名
            parsed_url = urlparse(url)
            path = parsed_url.path
            if '.' in path:
                suffix = '.' + path.split('.')[-1].lower()
                if suffix not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    suffix = '.jpg'  # 默认扩展名
            else:
                suffix = '.jpg'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # 写入图片数据
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        
        print(f"  图片下载完成: {temp_file.name}")
        return temp_file.name
        
    except requests.exceptions.RequestException as e:
        print(f"  下载失败: {e}")
        return None
    except Exception as e:
        print(f"  下载图片时发生错误: {e}")
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
        self.keywords = ["关闭","同意", "确定","更新微信","继续访问","我知道了","同意并继续","放弃","取消", "重试"]
    
    def preprocess_image(self, image_path):
        """轻度压缩预处理 - 固定800px压缩"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            original_size = max(height, width)
            
            max_size = 800
            if original_size > max_size:
                scale_factor = max_size / original_size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                self.scale_factor = 1.0 / scale_factor
            else:
                self.scale_factor = 1.0
            
            return img
            
        except Exception as e:
            print(f" 图片预处理失败: {e}")
            return None

    def find_button(self, image_path):
        """按钮查找函数"""
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
                try:
                    conf_float = float(confidence) if isinstance(confidence, str) else confidence
                    if conf_float > 0.5:
                        all_detected_texts.append(f"{text}({conf_float:.2f})")
                        
                        for keyword in self.keywords:
                            if keyword in text:
                                bbox = np.array(bbox)
                                center_x = int(np.mean(bbox[:, 0]) * self.scale_factor)
                                center_y = int(np.mean(bbox[:, 1]) * self.scale_factor)
                                
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
                    continue
        
        # 4. 如果OCR未找到关键字，再尝试图像识别关闭按钮
        # 调整检测优先级，优先使用更精确的方法
        shape_start = time.time()
        
        image_detection_result = None
        
        # 优先使用模板匹配，它通常更准确
        template_result = detect_close_button_by_template(image_path)
        if template_result and template_result["found"]:
            image_detection_result = template_result
        
        # 如果模板匹配失败，尝试形状识别
        if not image_detection_result:
            shape_result = detect_close_button_by_shape(image_path)
            if shape_result and shape_result["found"]:
                image_detection_result = shape_result
        
        # 如果还是没找到，尝试右上角专门检测
        if not image_detection_result:
            top_right_result = detect_top_right_x_button(image_path)
            if top_right_result and top_right_result["found"]:
                image_detection_result = top_right_result
        
        # 最后尝试简单×号检测（最宽松的）
        if not image_detection_result:
            simple_x_result = detect_simple_x_button(image_path)
            if simple_x_result and simple_x_result["found"]:
                image_detection_result = simple_x_result
        
        # 底部中心按钮检测（通常用于特定类型的弹窗）
        if not image_detection_result:
            bottom_center_result = detect_bottom_center_close_button(image_path)
            if bottom_center_result and bottom_center_result["found"]:
                image_detection_result = bottom_center_result
        
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

def create_annotated_image(image_path, center_x, center_y, keyword, confidence, output_path):
    """
    创建带标注的图片
    
    Args:
        image_path: 原图片路径
        center_x, center_y: 按钮中心坐标
        keyword: 识别的关键字
        confidence: 置信度
        output_path: 输出路径
    """
    try:
        # 读取原图
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # 绘制标注
        # 1. 绘制红色圆圈标注按钮位置
        cv2.circle(img, (center_x, center_y), 20, (0, 0, 255), 3)
        
        # 2. 绘制十字线精确标注中心点
        cv2.line(img, (center_x-10, center_y), (center_x+10, center_y), (0, 0, 255), 2)
        cv2.line(img, (center_x, center_y-10), (center_x, center_y+10), (0, 0, 255), 2)
        
        # 3. 添加文字标签
        label = f"{keyword} ({confidence:.2f})"
        
        # 计算文字位置（避免超出图片边界）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # 获取文字尺寸
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # 计算文字背景矩形位置
        text_x = center_x - text_width // 2
        text_y = center_y - 40  # 文字显示在圆圈上方
        
        # 确保文字不超出图片边界
        if text_x < 0:
            text_x = 5
        if text_x + text_width > img.shape[1]:
            text_x = img.shape[1] - text_width - 5
        if text_y < text_height:
            text_y = center_y + 50  # 如果上方空间不够，显示在下方
        
        # 绘制文字背景（半透明黑色矩形）
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + baseline + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 绘制白色文字
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # 保存标注图片
        cv2.imwrite(output_path, img)
        return True
        
    except Exception as e:
        print(f"    标注图片失败: {e}")
        return False

def batch_process_images(txt_file_path, output_file="recognition_results.json"):
    """
    批量处理txt文件中的图片路径（支持URL和本地路径）
    
    Args:
        txt_file_path: 包含图片路径的txt文件
        output_file: 输出结果的json文件名
    """
    print(f"开始批量处理图片...")
    print(f"读取路径文件: {txt_file_path}")
    print("=" * 60)
    
    # 检查txt文件是否存在
    if not os.path.exists(txt_file_path):
        print(f"错误: 路径文件不存在 - {txt_file_path}")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"annotated_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"标注图片将保存到: {output_dir}")
    
    # 读取图片路径
    image_paths = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 忽略空行和注释行
                    image_paths.append(line)
    except Exception as e:
        print(f"读取路径文件失败: {e}")
        return
    
    if not image_paths:
        print("未找到有效的图片路径")
        return
    
    print(f"找到 {len(image_paths)} 个图片路径")
    print("=" * 60)
    
    # 初始化结果列表
    results = []
    success_count = 0
    annotated_count = 0
    downloaded_files = []  # 记录下载的临时文件，用于后续清理
    
    # 处理每个图片
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] 处理: {image_path}")
        
        temp_file_path = None
        actual_image_path = image_path
        
        # 检查是否为URL
        if is_url(image_path):
            print(f"  检测到URL，正在下载...")
            temp_file_path = download_image_from_url(image_path)
            
            if temp_file_path is None:
                print(f"  ✗ URL图片下载失败")
                result_item = {
                    "index": idx,
                    "image_path": image_path,
                    "status": "failed",
                    "error": "URL图片下载失败",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result_item)
                continue
            
            actual_image_path = temp_file_path
            downloaded_files.append(temp_file_path)
        
        else:
            # 检查本地文件是否存在
            if not os.path.exists(image_path):
                print(f"  ✗ 本地文件不存在")
                result_item = {
                    "index": idx,
                    "image_path": image_path,
                    "status": "failed",
                    "error": "本地文件不存在",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result_item)
                continue
        
        # 执行识别
        try:
            recognition_result = find_button_optimized(actual_image_path)
            
            # 构建结果项
            result_item = {
                "index": idx,
                "image_path": image_path,
                "status": "success" if recognition_result["found"] else "not_found",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_url": is_url(image_path)
            }
            
            if recognition_result["found"]:
                success_count += 1
                
                # 生成标注图片文件名
                base_name = f"annotated_{idx:04d}"
                if is_url(image_path):
                    # URL图片，从URL中提取文件名或使用默认扩展名
                    parsed_url = urlparse(image_path)
                    url_path = parsed_url.path
                    if '.' in url_path:
                        extension = '.' + url_path.split('.')[-1].lower()
                        if extension not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                            extension = '.jpg'
                    else:
                        extension = '.jpg'
                else:
                    # 本地文件，保持原扩展名
                    _, extension = os.path.splitext(image_path)
                    if not extension:
                        extension = '.jpg'
                
                annotated_filename = base_name + extension
                annotated_path = os.path.join(output_dir, annotated_filename)
                
                # 创建标注图片
                annotation_success = create_annotated_image(
                    actual_image_path,
                    recognition_result["center"]["x"],
                    recognition_result["center"]["y"],
                    recognition_result["keyword"],
                    recognition_result["confidence"],
                    annotated_path
                )
                
                if annotation_success:
                    annotated_count += 1
                    print(f"  ✓ 找到按钮: {recognition_result['keyword']} at ({recognition_result['center']['x']}, {recognition_result['center']['y']})")
                    print(f"    标注图片已保存: {annotated_path}")
                else:
                    print(f"  ✓ 找到按钮: {recognition_result['keyword']} at ({recognition_result['center']['x']}, {recognition_result['center']['y']})")
                    print(f"    ✗ 标注图片保存失败")
                
                result_item.update({
                    "keyword": recognition_result["keyword"],
                    "full_text": recognition_result["full_text"],
                    "coordinates": {
                        "x": recognition_result["center"]["x"],
                        "y": recognition_result["center"]["y"]
                    },
                    "confidence": recognition_result["confidence"],
                    "detection_method": recognition_result["detection_method"],
                    "annotated_image": annotated_path if annotation_success else None,
                    "annotation_success": annotation_success
                })
            else:
                result_item["detected_texts"] = recognition_result.get("all_detected", [])
                print(f"  ✗ 未找到目标按钮")
                if recognition_result.get("all_detected"):
                    print(f"    检测到的文字: {', '.join(recognition_result['all_detected'][:5])}")
            
            # 添加耗时信息
            result_item["time_stats"] = {
                "preprocess": recognition_result.get("preprocess_time", "N/A"),
                "ocr": recognition_result.get("ocr_time", "N/A"),
                "shape": recognition_result.get("shape_time", "N/A"),
                "total": recognition_result.get("total_time", "N/A")
            }
            
        except Exception as e:
            print(f"  ✗ 识别过程出错: {e}")
            result_item = {
                "index": idx,
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_url": is_url(image_path)
            }
        
        results.append(result_item)
    
    # 清理临时文件
    print(f"\n清理临时文件...")
    for temp_file in downloaded_files:
        try:
            os.unlink(temp_file)
            print(f"  已删除临时文件: {temp_file}")
        except Exception as e:
            print(f"  删除临时文件失败: {temp_file} - {e}")
    
    # 保存结果到JSON文件
    try:
        output_data = {
            "process_info": {
                "total_images": len(image_paths),
                "success_count": success_count,
                "annotated_count": annotated_count,
                "failed_count": len(image_paths) - success_count,
                "url_count": sum(1 for path in image_paths if is_url(path)),
                "local_count": sum(1 for path in image_paths if not is_url(path)),
                "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": txt_file_path,
                "output_directory": output_dir
            },
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print(f"处理完成!")
        print(f"  总计: {len(image_paths)} 个图片")
        print(f"  URL: {sum(1 for path in image_paths if is_url(path))} 个")
        print(f"  本地: {sum(1 for path in image_paths if not is_url(path))} 个")
        print(f"  成功: {success_count} 个")
        print(f"  标注: {annotated_count} 个")
        print(f"  失败: {len(image_paths) - success_count} 个")
        print(f"  结果已保存到: {output_file}")
        print(f"  标注图片保存到: {output_dir}")
        
    except Exception as e:
        print(f"\n保存结果文件失败: {e}")
    
    return results

def save_results_as_txt(results, txt_output_file="recognition_results.txt"):
    """
    将结果保存为txt格式
    """
    try:
        with open(txt_output_file, 'w', encoding='utf-8') as f:
            f.write("图片按钮识别结果\n")
            f.write("=" * 60 + "\n\n")
            
            for result in results:
                f.write(f"[{result['index']}] {result['image_path']}\n")
                f.write(f"类型: {'URL' if result.get('is_url') else '本地文件'}\n")
                f.write(f"状态: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"关键字: {result['keyword']}\n")
                    f.write(f"完整文字: {result['full_text']}\n")
                    f.write(f"坐标: ({result['coordinates']['x']}, {result['coordinates']['y']})\n")
                    f.write(f"置信度: {result['confidence']}\n")
                    f.write(f"检测方法: {result['detection_method']}\n")
                    
                    if result.get('annotated_image'):
                        f.write(f"标注图片: {result['annotated_image']}\n")
                    else:
                        f.write(f"标注图片: 保存失败\n")
                elif result['status'] == 'not_found':
                    if result.get('detected_texts'):
                        f.write(f"检测到的文字: {', '.join(result['detected_texts'][:5])}\n")
                else:
                    f.write(f"错误: {result.get('error', 'Unknown')}\n")
                
                f.write(f"耗时: {result['time_stats']['total']}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"TXT格式结果已保存到: {txt_output_file}")
        
    except Exception as e:
        print(f"保存TXT文件失败: {e}")

def monitor_and_process():
    """
    主程序：持续运行，监控和处理图片
    """
    print("=" * 60)
    print("批量OCR按钮识别工具 - 支持URL版本")
    print("=" * 60)
    print("功能:")
    print("- 批量识别图片中的按钮")
    print("- 支持本地文件和URL")
    print("- 自动标注识别结果并保存标注图片")
    print("- 输出详细的JSON和TXT格式结果")
    print("=" * 60)
    print("使用方法:")
    print("请将图片路径写入到 image_paths.txt 文件中")
    print("每行一个图片路径")
    print("支持相对路径、绝对路径和URL")
    print("以#开头的行将被忽略（注释）")
    print("=" * 60)
    
    # 默认的路径文件名
    default_txt_file = "image_paths.txt"
    
    while True:
        print("\n选择操作:")
        print("1. 处理 image_paths.txt 中的图片")
        print("2. 指定其他txt文件")
        print("3. 退出程序")
        
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            if os.path.exists(default_txt_file):
                # 生成带时间戳的输出文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_output = f"results_{timestamp}.json"
                txt_output = f"results_{timestamp}.txt"
                
                print(f"\n开始处理，标注图片将保存到: annotated_images_{timestamp}/")
                
                # 处理图片
                results = batch_process_images(default_txt_file, json_output)
                
                if results:
                    # 同时保存为txt格式
                    save_results_as_txt(results, txt_output)
                
            else:
                print(f"\n错误: {default_txt_file} 文件不存在")
                print("请创建该文件并添加图片路径")
                
        elif choice == '2':
            txt_file = input("请输入txt文件路径: ").strip()
            if os.path.exists(txt_file):
                # 生成带时间戳的输出文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_output = f"results_{timestamp}.json"
                txt_output = f"results_{timestamp}.txt"
                
                print(f"\n开始处理，标注图片将保存到: annotated_images_{timestamp}/")
                
                # 处理图片
                results = batch_process_images(txt_file, json_output)
                
                if results:
                    # 同时保存为txt格式
                    save_results_as_txt(results, txt_output)
            else:
                print(f"\n错误: {txt_file} 文件不存在")
                
        elif choice == '3':
            print("\n程序退出")
            break
        else:
            print("\n无效选项，请重新选择")
        
        # 询问是否继续
        print("\n" + "-" * 40)
        continue_choice = input("是否继续处理其他文件? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n程序退出")
            break

if __name__ == "__main__":
    # 创建示例txt文件（如果不存在）
    example_txt = "image_paths.txt"
    if not os.path.exists(example_txt):
        with open(example_txt, 'w', encoding='utf-8') as f:
            f.write("# 图片路径列表\n")
            f.write("# 每行一个路径，支持相对路径、绝对路径和URL\n")
            f.write("# 以#开头的行会被忽略\n")
            f.write("# 识别成功的图片会自动生成标注图片\n\n")
            f.write("# 本地文件示例:\n")
            f.write("# page/1.png\n")
            f.write("# page/2.png\n")
            f.write("# C:/Users/xxx/Pictures/test.jpg\n\n")
            f.write("# URL示例:\n")
            f.write("# https://example.com/image1.jpg\n")
            f.write("# https://example.com/image2.png\n")
        print(f"已创建示例文件: {example_txt}")
        print("请编辑该文件，添加要识别的图片路径或URL")
        print("识别成功的图片会自动生成带标注的图片保存到输出目录\n")
    
    # 启动主程序
    monitor_and_process()