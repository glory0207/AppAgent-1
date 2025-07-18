"""
批量EasyOCR按钮查找工具 - 支持URL版本

功能：
- 从txt文件读取图片路径列表（本地路径和URL）
- 批量识别图片中的按钮
- 输出JSON格式的识别结果
- 程序持续运行，支持多次处理
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
    """通过形状识别关闭按钮（×号） - 优先检测右上角"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 重点搜索右上角区域 (右侧30%，顶部30%)
        roi_x = int(w * 0.7)
        roi_y = 0
        roi_w = int(w * 0.3)
        roi_h = int(h * 0.3)
        roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 在ROI中查找圆形
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=40,
            minRadius=15,
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 转换回原图坐标
                actual_x = roi_x + x
                actual_y = roi_y + y
                
                return {
                    "found": True,
                    "center": {"x": actual_x, "y": actual_y},
                    "radius": r,
                    "type": "circular_close_button_top_right"
                }
        
        # 在ROI中查找轮廓
        edges = cv2.Canny(roi, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 800:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # 检查是否接近正方形
                if abs(w_box - h_box) < min(w_box, h_box) * 0.3:
                    actual_x = roi_x + x + w_box // 2
                    actual_y = roi_y + y + h_box // 2
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "area": area,
                        "type": "corner_close_button_top_right"
                    }
        
        return None
        
    except Exception as e:
        print(f"形状识别失败: {e}")
        return None

def detect_close_button_by_template(image_path):
    """通过模板匹配识别×号关闭按钮"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = []
        
        for template_size in [16, 20, 24, 30, 36]:
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            
            thickness = max(1, template_size // 10)
            margin = template_size // 4
            cv2.line(template, (margin, margin), (template_size-margin, template_size-margin), (255,), thickness)
            cv2.line(template, (template_size-margin, margin), (margin, template_size-margin), (255,), thickness)
            
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            results.append({
                'max_val': max_val,
                'location': max_loc,
                'template_size': template_size
            })
        
        best_result = max(results, key=lambda x: x['max_val'])
        
        if best_result['max_val'] > 0.5:
            x = best_result['location'][0] + best_result['template_size'] // 2
            y = best_result['location'][1] + best_result['template_size'] // 2
            
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
    """检测简单的×号按钮"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        search_regions = [
            (0, 0, w//3, h//3),
            (w*2//3, 0, w//3, h//3),
            (w*3//4, 0, w//4, h//4),
        ]
        
        for region_x, region_y, region_w, region_h in search_regions:
            roi = gray[region_y:region_y+region_h, region_x:region_x+region_w]
            
            binary_methods = []
            
            try:
                adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
                binary_methods.append(adaptive_binary)
            except:
                pass
            
            try:
                _, otsu_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary_methods.append(otsu_binary)
            except:
                pass
            
            try:
                _, inv_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                binary_methods.append(inv_binary)
            except:
                pass
            
            for threshold in [100, 127, 150, 200]:
                try:
                    _, fixed_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
                    binary_methods.append(fixed_binary)
                    _, fixed_inv_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
                    binary_methods.append(fixed_inv_binary)
                except:
                    pass
            
            for binary in binary_methods:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 10 < area < 1000:
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        
                        aspect_ratio = w_box / h_box
                        if 0.3 < aspect_ratio < 3.0:
                            center_x = region_x + x + w_box // 2
                            center_y = region_y + y + h_box // 2
                            
                            if region_x > w//2:
                                if center_y < h//2:
                                    return {
                                        "found": True,
                                        "center": {"x": center_x, "y": center_y},
                                        "area": area,
                                        "region": f"right_top_{region_x}_{region_y}",
                                        "type": "simple_x_button"
                                    }
                            else:
                                if center_y < h//2:
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
    """检测底部中间位置的圆形关闭按钮"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        bottom_region_y = int(h * 0.7)
        center_margin = int(w * 0.2)
        
        roi = gray[bottom_region_y:h, center_margin:w-center_margin]
        
        if roi.size == 0:
            return None
        
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=25,
            minRadius=10,
            maxRadius=40
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            best_circle = None
            best_score = float('inf')
            
            for (x, y, r) in circles:
                actual_x = center_margin + x
                actual_y = bottom_region_y + y
                
                center_x = w // 2
                bottom_y = h - 50
                
                distance = ((actual_x - center_x) ** 2 + (actual_y - bottom_y) ** 2) ** 0.5
                
                if distance < best_score:
                    best_score = distance
                    best_circle = (actual_x, actual_y, r)
            
            if best_circle and best_score < 100:
                return {
                    "found": True,
                    "center": {"x": best_circle[0], "y": best_circle[1]},
                    "radius": best_circle[2],
                    "distance_score": best_score,
                    "type": "bottom_center_close_button"
                }
        
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                aspect_ratio = w_box / h_box
                if 0.7 < aspect_ratio < 1.3:
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
    """专门检测右上角的×号按钮"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        top_right_x = int(w * 0.7)
        top_right_y = 0
        top_right_w = w - top_right_x
        top_right_h = int(h * 0.3)
        
        roi = gray[top_right_y:top_right_y + top_right_h, top_right_x:top_right_x + top_right_w]
        
        if roi.size == 0:
            return None
        
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
        
        if lines is not None:
            diagonal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(abs(angle) - 45) < 20 or abs(abs(angle) - 135) < 20:
                    diagonal_lines.append(line[0])
            
            if len(diagonal_lines) >= 2:
                centers = []
                for i, line1 in enumerate(diagonal_lines):
                    for j, line2 in enumerate(diagonal_lines[i+1:], i+1):
                        x1, y1, x2, y2 = line1
                        x3, y3, x4, y4 = line2
                        
                        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if abs(denom) > 1e-10:
                            px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                            py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                            
                            if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
                                centers.append((px, py))
                
                if centers:
                    best_center = min(centers, key=lambda p: p[0]**2 + p[1]**2)
                    
                    actual_x = int(top_right_x + best_center[0])
                    actual_y = int(top_right_y + best_center[1])
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "line_count": len(diagonal_lines),
                        "type": "top_right_x_lines"
                    }
        
        binary_methods = []
        
        for threshold in [80, 100, 120, 140, 160, 180, 200]:
            _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
            binary_methods.append(binary)
            _, inv_binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
            binary_methods.append(inv_binary)
        
        for binary in binary_methods:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 800:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    aspect_ratio = w_box / h_box
                    if 0.5 < aspect_ratio < 2.0:
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

def detect_back_arrow_button(image_path):
    """检测左上角的返回箭头（<形状）- 更严格的版本"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 更精确地限定左上角区域 (左侧20%，顶部20%)
        roi_x = 0
        roi_y = 0
        roi_w = int(w * 0.2)
        roi_h = int(h * 0.2)
        roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if roi.size == 0:
            return None
        
        # 方法1：使用模板匹配检测 < 形状的箭头
        results = []
        
        for template_size in [16, 20, 24, 30, 36]:
            # 创建 < 形状的模板
            template = np.zeros((template_size, template_size), dtype=np.uint8)
            
            # 画一个更精确的 < 形状
            thickness = max(2, template_size // 10)
            center_y = template_size // 2
            left_x = template_size // 4  # 更靠左
            right_x = template_size * 3 // 4  # 更靠右
            
            # 上半部分线条 (角度约45度)
            cv2.line(template, 
                    (right_x, template_size // 3), 
                    (left_x, center_y), 
                    (255,), thickness)
            
            # 下半部分线条 (角度约-45度)
            cv2.line(template, 
                    (left_x, center_y), 
                    (right_x, template_size * 2 // 3), 
                    (255,), thickness)
            
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            results.append({
                'max_val': max_val,
                'location': max_loc,
                'template_size': template_size
            })
        
        best_result = max(results, key=lambda x: x['max_val'])
        
        # 提高阈值到0.6，减少误判
        if best_result['max_val'] > 0.6:
            x = best_result['location'][0] + best_result['template_size'] // 2
            y = best_result['location'][1] + best_result['template_size'] // 2
            
            # 额外验证：检查周围是否有其他干扰元素
            box_size = best_result['template_size']
            box_x = best_result['location'][0]
            box_y = best_result['location'][1]
            
            if box_x + box_size < roi.shape[1] and box_y + box_size < roi.shape[0]:
                roi_box = roi[box_y:box_y+box_size, box_x:box_x+box_size]
                
                # 计算箭头区域的像素密度，应该相对较低（不是实心的）
                white_pixels = np.sum(roi_box > 200)
                total_pixels = box_size * box_size
                density = white_pixels / total_pixels
                
                # 箭头的像素密度应该在合理范围内
                if 0.05 < density < 0.4:
                    return {
                        "found": True,
                        "center": {"x": roi_x + x, "y": roi_y + y},
                        "confidence": best_result['max_val'],
                        "template_size": best_result['template_size'],
                        "type": "back_arrow_template"
                    }
        
        # 方法2：检测角度线条 - 更严格的条件
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=3)
        
        if lines is not None and len(lines) >= 2:
            # 查找形成 < 形状的线条对
            valid_arrow_pairs = []
            
            for i, line1 in enumerate(lines):
                x1, y1, x2, y2 = line1[0]
                angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                for j, line2 in enumerate(lines[i+1:], i+1):
                    x3, y3, x4, y4 = line2[0]
                    angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
                    length2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
                    
                    # 更严格的角度检查：一条约45度向下，一条约-45度向上
                    if ((-50 < angle1 < -30 and 30 < angle2 < 50) or 
                        (30 < angle1 < 50 and -50 < angle2 < -30)):
                        
                        # 检查两条线长度是否相近
                        length_ratio = min(length1, length2) / max(length1, length2)
                        if length_ratio > 0.5:  # 长度不能相差太大
                            
                            # 计算两条线的交点
                            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                            if abs(denom) > 1e-10:
                                px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
                                py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
                                
                                # 交点必须在ROI内
                                if 5 <= px < roi.shape[1]-5 and 5 <= py < roi.shape[0]-5:
                                    # 验证两条线确实相交（不是延长线相交）
                                    # 检查交点是否在两条线段的范围内
                                    on_line1 = (min(x1,x2) <= px <= max(x1,x2)) and (min(y1,y2) <= py <= max(y1,y2))
                                    on_line2 = (min(x3,x4) <= px <= max(x3,x4)) and (min(y3,y4) <= py <= max(y3,y4))
                                    
                                    if on_line1 or on_line2:  # 至少在一条线段上
                                        valid_arrow_pairs.append({
                                            'point': (px, py),
                                            'confidence': length_ratio,
                                            'angle_diff': abs(abs(angle1) + abs(angle2) - 90)
                                        })
            
            if valid_arrow_pairs:
                # 选择最符合箭头特征的
                best_arrow = min(valid_arrow_pairs, key=lambda x: x['angle_diff'])
                
                # 角度差必须小于20度（接近90度的夹角）
                if best_arrow['angle_diff'] < 20:
                    actual_x = int(roi_x + best_arrow['point'][0])
                    actual_y = int(roi_y + best_arrow['point'][1])
                    
                    return {
                        "found": True,
                        "center": {"x": actual_x, "y": actual_y},
                        "confidence": best_arrow['confidence'],
                        "type": "back_arrow_lines"
                    }
        
        # 方法3：轮廓检测 - 更严格的验证
        binary_methods = []
        
        # 使用更少但更可靠的二值化方法
        _, otsu_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_methods.append(otsu_binary)
        _, otsu_inv = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_methods.append(otsu_inv)
        
        try:
            adaptive_binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            binary_methods.append(adaptive_binary)
        except:
            pass
        
        for binary in binary_methods:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 800:  # 提高最小面积要求
                    # 使用多边形近似
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = 0.03 * perimeter  # 稍微提高近似精度
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # 箭头形状应该是3-4个顶点
                    if len(approx) == 3 or len(approx) == 4:
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        
                        # 检查宽高比
                        aspect_ratio = w_box / h_box
                        if 0.5 < aspect_ratio < 1.0:  # 箭头通常宽度小于高度
                            
                            # 检查质心位置
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # 质心必须明显偏左
                                centroid_offset = (cx - x) / w_box
                                if centroid_offset < 0.45:  # 质心在左侧45%以内
                                    
                                    # 额外验证：检查凸包
                                    hull = cv2.convexHull(contour)
                                    hull_area = cv2.contourArea(hull)
                                    solidity = area / hull_area if hull_area > 0 else 0
                                    
                                    # 箭头的凸实性应该较高（0.7-0.95）
                                    if 0.7 < solidity < 0.95:
                                        actual_x = roi_x + cx
                                        actual_y = roi_y + cy
                                        
                                        return {
                                            "found": True,
                                            "center": {"x": actual_x, "y": actual_y},
                                            "area": area,
                                            "solidity": solidity,
                                            "type": "back_arrow_contour"
                                        }
        
        return None
        
    except Exception as e:
        print(f"返回箭头检测失败: {e}")
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
        
        # 目标关键字 - 添加"返回"、"后退"等关键字
        self.keywords = ["关闭", "同意", "确定", "更新微信", "继续访问", "我知道了", 
                        "同意并继续", "取消", "前往京东APP", "返回", "后退", "back"]
        
        # 图片处理参数
        self.scale_factor = 1.0  # 缩放因子
        self.crop_applied = False  # 是否进行了截取
        self.crop_offset_y = 0  # 截取的Y偏移（长截图情况下为0）
    
    def preprocess_image(self, image_path):
        """轻度压缩预处理 - 固定800px压缩，并处理长截图"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            height, width = img.shape[:2]
            original_height = height  # 保存原始高度
            
            # 重置处理参数
            self.crop_applied = False
            self.crop_offset_y = 0
            
            # 处理长截图 - 判断是否为手机长截图
            aspect_ratio = height / width
            
            # 如果高宽比大于2.5，认为是长截图（一般手机屏幕高宽比在1.5-2.3之间）
            if aspect_ratio > 2.5:
                print(f"  检测到长截图 (高宽比: {aspect_ratio:.2f})，截取第一屏...")
                
                # 计算第一屏的高度
                # 常见手机屏幕比例：
                # 16:9 = 1.78
                # 18:9 = 2.0
                # 19.5:9 = 2.17
                # 21:9 = 2.33
                # 取一个合理的最大值2.3作为第一屏的高宽比
                first_screen_height = int(width * 2.3)
                
                # 确保不超过原图高度
                if first_screen_height > height:
                    first_screen_height = height
                
                # 截取第一屏
                img = img[:first_screen_height, :, :]
                height = first_screen_height
                self.crop_applied = True
                # 从顶部截取，所以Y偏移为0
                self.crop_offset_y = 0
                
                print(f"  已截取第一屏: {width}x{height} (原图: {width}x{original_height})")
            
            # 继续原有的缩放处理
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
        """按钮查找函数 - 优先OCR，然后比较形状检测的置信度
        
        检测优先级：
        1. OCR文字识别（置信度>0.5）
        2. 同时检测箭头和×号，返回置信度更高的
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
        
        # 2. 优先进行OCR文字识别
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
        
        # 3. 检查OCR结果中是否有目标关键字（提高阈值到0.5）
        all_detected_texts = []
        
        if results:
            for (bbox, text, confidence) in results:
                try:
                    conf_float = float(confidence) if isinstance(confidence, str) else confidence
                    if conf_float > 0.5:  # 记录所有文本用于调试
                        all_detected_texts.append(f"{text}({conf_float:.2f})")
                        
                        # OCR置信度阈值提高到0.5
                        if conf_float > 0.5:
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
                                    
                                    print(f"  OCR检测到高置信度按钮: {keyword} (置信度: {conf_float:.3f})")
                                    return result
                except (ValueError, TypeError):
                    continue
        
        # 4. OCR未找到高置信度关键字，同时检测箭头和×号
        shape_start = time.time()
        
        # 收集所有检测结果
        detection_candidates = []
        
        # 检测返回箭头（左上角）
        back_arrow_result = detect_back_arrow_button(image_path)
        if back_arrow_result and back_arrow_result["found"]:
            back_arrow_result["priority"] = 0.9  # 箭头基础优先级
            detection_candidates.append(back_arrow_result)
            print(f"  检测到返回箭头，置信度: {back_arrow_result.get('confidence', 0.8):.3f}")
        
        # 检测×号（右半屏幕）- 调用多个检测方法
        # 扩展的形状检测（右半屏）
        x_shape_result = self.detect_x_button_extended(image_path)
        if x_shape_result and x_shape_result["found"]:
            x_shape_result["priority"] = 0.85  # ×号基础优先级
            detection_candidates.append(x_shape_result)
            print(f"  检测到×号(形状)，置信度: {x_shape_result.get('confidence', 0.8):.3f}")
        
        # 模板匹配（右半屏）
        x_template_result = self.detect_x_by_template_extended(image_path)
        if x_template_result and x_template_result["found"]:
            x_template_result["priority"] = 0.85
            detection_candidates.append(x_template_result)
            print(f"  检测到×号(模板)，置信度: {x_template_result.get('confidence', 0.8):.3f}")
        
        # 简单×检测
        simple_x_result = detect_simple_x_button(image_path)
        if simple_x_result and simple_x_result["found"]:
            simple_x_result["priority"] = 0.8
            detection_candidates.append(simple_x_result)
            print(f"  检测到×号(简单)，置信度: {simple_x_result.get('confidence', 0.8):.3f}")
        
        # 右上角专门检测
        top_right_result = detect_top_right_x_button(image_path)
        if top_right_result and top_right_result["found"]:
            top_right_result["priority"] = 0.85
            detection_candidates.append(top_right_result)
            print(f"  检测到×号(右上角)，置信度: {top_right_result.get('confidence', 0.8):.3f}")
        
        # 底部中间检测
        bottom_center_result = detect_bottom_center_close_button(image_path)
        if bottom_center_result and bottom_center_result["found"]:
            bottom_center_result["priority"] = 0.75  # 底部按钮优先级较低
            detection_candidates.append(bottom_center_result)
            print(f"  检测到底部按钮，置信度: {bottom_center_result.get('confidence', 0.8):.3f}")
        
        shape_time = time.time() - shape_start
        
        # 5. 选择置信度最高的结果
        if detection_candidates:
            # 计算综合得分 = 置信度 * 优先级
            for candidate in detection_candidates:
                confidence = candidate.get("confidence", 0.8)
                priority = candidate.get("priority", 0.8)
                candidate["score"] = confidence * priority
            
            # 选择得分最高的
            best_result = max(detection_candidates, key=lambda x: x["score"])
            
            # 根据检测类型设置显示文本
            if "back_arrow" in best_result["type"]:
                keyword = "<"
                full_text = f"<({best_result['type']})"
            else:
                keyword = "×"
                full_text = f"×({best_result['type']})"
            
            result = {
                "found": True,
                "keyword": keyword,
                "full_text": full_text,
                "center": best_result["center"],
                "confidence": best_result.get("confidence", 0.8),
                "detection_method": "image_recognition",
                "detection_type": best_result["type"],
                "score": best_result["score"],
                "preprocess_time": f"{preprocess_time:.3f}s",
                "ocr_time": f"{ocr_time:.3f}s",
                "shape_time": f"{shape_time:.3f}s",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
            
            print(f"  最终选择: {keyword} (得分: {best_result['score']:.3f})")
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
    
    def detect_x_button_extended(self, image_path):
        """扩展的×号检测 - 搜索右半屏幕"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 搜索右半屏幕
            roi_x = w // 2
            roi_y = 0
            roi_w = w - roi_x
            roi_h = h
            roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # 在ROI中查找圆形（可能的圆形×按钮）
            circles = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=35,
                minRadius=10,
                maxRadius=50
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # 优先选择右上角的圆形
                best_circle = None
                best_score = float('inf')
                
                for (x, y, r) in circles:
                    # 计算到右上角的距离
                    distance_to_corner = ((roi_w - x) ** 2 + y ** 2) ** 0.5
                    
                    # 优先选择靠近右上角的
                    if y < roi_h * 0.3:  # 在上部30%
                        score = distance_to_corner * 0.5  # 权重更高
                    else:
                        score = distance_to_corner
                    
                    if score < best_score:
                        best_score = score
                        best_circle = (roi_x + x, roi_y + y, r)
                
                if best_circle:
                    return {
                        "found": True,
                        "center": {"x": best_circle[0], "y": best_circle[1]},
                        "radius": best_circle[2],
                        "confidence": 0.85,
                        "type": "x_button_circle_extended"
                    }
            
            # 在ROI中查找轮廓
            edges = cv2.Canny(roi, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            potential_x_buttons = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 1000:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    
                    # 检查是否接近正方形
                    aspect_ratio = w_box / h_box
                    if 0.7 < aspect_ratio < 1.3:
                        actual_x = roi_x + x + w_box // 2
                        actual_y = roi_y + y + h_box // 2
                        
                        # 计算置信度（基于位置）
                        if actual_y < h * 0.3:  # 上部30%
                            confidence = 0.9
                        elif actual_y < h * 0.5:  # 上半部分
                            confidence = 0.8
                        else:
                            confidence = 0.7
                        
                        potential_x_buttons.append({
                            "center": {"x": actual_x, "y": actual_y},
                            "area": area,
                            "confidence": confidence,
                            "y": actual_y
                        })
            
            # 选择最靠上的×按钮
            if potential_x_buttons:
                best_button = min(potential_x_buttons, key=lambda x: x["y"])
                
                return {
                    "found": True,
                    "center": best_button["center"],
                    "area": best_button["area"],
                    "confidence": best_button["confidence"],
                    "type": "x_button_contour_extended"
                }
            
            return None
            
        except Exception as e:
            print(f"扩展×号检测失败: {e}")
            return None
    
    def detect_x_by_template_extended(self, image_path):
        """扩展的模板匹配×号检测 - 搜索右半屏幕"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 搜索右半屏幕
            roi_x = w // 2
            roi_y = 0
            roi_w = w - roi_x
            roi_h = h
            roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            results = []
            
            for template_size in [16, 20, 24, 30, 36, 42]:
                template = np.zeros((template_size, template_size), dtype=np.uint8)
                
                thickness = max(2, template_size // 8)
                margin = template_size // 4
                
                # 画×形状
                cv2.line(template, (margin, margin), (template_size-margin, template_size-margin), (255,), thickness)
                cv2.line(template, (template_size-margin, margin), (margin, template_size-margin), (255,), thickness)
                
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                results.append({
                    'max_val': max_val,
                    'location': max_loc,
                    'template_size': template_size
                })
            
            # 找到最佳匹配
            best_result = max(results, key=lambda x: x['max_val'])
            
            if best_result['max_val'] > 0.5:
                x = roi_x + best_result['location'][0] + best_result['template_size'] // 2
                y = roi_y + best_result['location'][1] + best_result['template_size'] // 2
                
                # 根据位置调整置信度
                position_confidence = 1.0
                if y < h * 0.3:  # 上部30%
                    position_confidence = 1.1
                elif y > h * 0.7:  # 下部30%
                    position_confidence = 0.9
                
                final_confidence = min(best_result['max_val'] * position_confidence, 1.0)
                
                return {
                    "found": True,
                    "center": {"x": x, "y": y},
                    "confidence": final_confidence,
                    "template_size": best_result['template_size'],
                    "type": "x_template_extended"
                }
            
            return None
            
        except Exception as e:
            print(f"扩展模板匹配失败: {e}")
            return None

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
                result_item.update({
                    "keyword": recognition_result["keyword"],
                    "full_text": recognition_result["full_text"],
                    "coordinates": {
                        "x": recognition_result["center"]["x"],
                        "y": recognition_result["center"]["y"]
                    },
                    "confidence": recognition_result["confidence"],
                    "detection_method": recognition_result["detection_method"]
                })
                print(f"  ✓ 找到按钮: {recognition_result['keyword']} at ({recognition_result['center']['x']}, {recognition_result['center']['y']})")
                
                # 创建标注图片
                if is_url(image_path):
                    # URL图片使用索引作为文件名
                    annotated_filename = f"annotated_{idx}.jpg"
                else:
                    # 本地图片使用原文件名
                    base_name = os.path.basename(image_path)
                    name_without_ext = os.path.splitext(base_name)[0]
                    annotated_filename = f"annotated_{name_without_ext}.jpg"
                
                annotated_path = os.path.join(output_dir, annotated_filename)
                
                if create_annotated_image(
                    actual_image_path,
                    recognition_result["center"]["x"],
                    recognition_result["center"]["y"],
                    recognition_result["keyword"],
                    recognition_result["confidence"],
                    annotated_path
                ):
                    annotated_count += 1
                    result_item["annotated_image"] = annotated_path
                    print(f"    已创建标注图片: {annotated_filename}")
                
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
                "failed_count": len(image_paths) - success_count,
                "annotated_count": annotated_count,
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
        print(f"  失败: {len(image_paths) - success_count} 个")
        print(f"  标注: {annotated_count} 个")
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
                    if 'annotated_image' in result:
                        f.write(f"标注图片: {result['annotated_image']}\n")
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
    print("批量OCR按钮识别工具 - 增强版")
    print("支持识别：关闭按钮(×)、返回箭头(<)、文字按钮等")
    print("=" * 60)
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
    # 创建示例txt文件
    example_txt = "image_paths.txt"
    if not os.path.exists(example_txt):
        with open(example_txt, 'w', encoding='utf-8') as f:
            f.write("# 图片路径列表\n")
            f.write("# 每行一个路径，支持相对路径、绝对路径和URL\n")
            f.write("# 以#开头的行会被忽略\n\n")
            f.write("# 本地文件示例:\n")
            f.write("# page/1.png\n")
            f.write("# page/2.png\n")
            f.write("# C:/Users/xxx/Pictures/test.jpg\n\n")
            f.write("# URL示例:\n")
            f.write("# https://example.com/image1.jpg\n")
            f.write("# https://example.com/image2.png\n")
        print(f"已创建示例文件: {example_txt}")
        print("请编辑该文件，添加要识别的图片路径或URL\n")
    
    # 启动主程序
    monitor_and_process()