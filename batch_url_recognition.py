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
import glob

class OptimizedButtonFinder:
    """按钮识别器"""
    
    def __init__(self):
        """初始化OCR模型"""
        start_time = time.time()
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}s")
        
        # 目标关键字
        self.keywords = ["关闭", "同意", "确定", "我知道了", "同意并继续","放弃","取消","继续逛逛"]
        
        # 文字置信度阈值
        self.text_confidence_threshold = 0.6
        
        # ×号置信度阈值 
        self.x_confidence_threshold = 0.75
        
        # 图片缩放因子
        self.scale_factor = 1.0
        
        # Y坐标限制（Y<140的区域不检测×号）
        self.y_threshold = 140
    
    def preprocess_image(self, image_path):
        """图片预处理 - 固定800px压缩 + 图像增强"""
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
    
    def enhance_image_contrast(self, image):
        """对比度增强 - 专门用于提升文字识别"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 使用CLAHE (Contrast Limited Adaptive Histogram Equalization) 进行对比度增强
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 转换回彩色图像（如果原图是彩色的）
            if len(image.shape) == 3:
                enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                return enhanced_color
            else:
                return enhanced
        except Exception as e:
            print(f"    对比度增强失败: {e}")
            return image
    
    def ocr_detect_text(self, img, start_y=0):
        """OCR文字检测 - 支持图像增强
        
        Args:
            img: 图片
            start_y: Y坐标偏移量（用于长截图）
        
        Returns:
            dict: 检测结果，包含最佳文字结果和所有检测到的文字
        """
        # 先尝试原图识别
        try:
            results = self.reader.readtext(img)
        except Exception as e:
            print(f"    OCR识别失败: {e}")
            return {"best_result": None, "all_texts": []}
        
        best_result = None
        all_texts = []
        
        # 处理原图识别结果
        for (bbox, text, confidence) in results:
            try:
                conf_float = float(confidence) if isinstance(confidence, str) else confidence
                
                if conf_float > 0.5:
                    all_texts.append(f"{text}({conf_float:.2f})")
                    
                    # 检查是否包含目标关键字且置信度高于阈值
                    if conf_float > self.text_confidence_threshold:
                        for keyword in self.keywords:
                            if keyword in text:
                                bbox = np.array(bbox)
                                center_x = int(np.mean(bbox[:, 0]) * self.scale_factor)
                                center_y = int((start_y + np.mean(bbox[:, 1])) * self.scale_factor)
                                
                                # 更新最佳结果（选择置信度最高的）
                                if best_result is None or conf_float > best_result['confidence']:
                                    best_result = {
                                        "found": True,
                                        "keyword": keyword,
                                        "full_text": text,
                                        "center": {"x": center_x, "y": center_y},
                                        "confidence": round(conf_float, 3),
                                        "detection_method": "ocr"
                                    }
            except (ValueError, TypeError):
                continue
        
        # 如果原图没有找到结果，或者找到的是"取消"，尝试对比度增强寻找"同意"
        should_enhance = (best_result is None or 
                         (best_result and best_result.get('keyword') == '取消'))
        
        if should_enhance:
            try:
                enhanced_img = self.enhance_image_contrast(img)
                enhanced_results = self.reader.readtext(enhanced_img)
                
                for (bbox, text, confidence) in enhanced_results:
                    try:
                        conf_float = float(confidence) if isinstance(confidence, str) else confidence
                        
                        if conf_float > 0.5:
                            if f"{text}({conf_float:.2f})" not in all_texts:
                                all_texts.append(f"{text}({conf_float:.2f})[增强]")
                            
                            # 检查是否包含目标关键字且置信度高于阈值
                            if conf_float > self.text_confidence_threshold:
                                for keyword in self.keywords:
                                    if keyword in text:
                                        bbox = np.array(bbox)
                                        center_x = int(np.mean(bbox[:, 0]) * self.scale_factor)
                                        center_y = int((start_y + np.mean(bbox[:, 1])) * self.scale_factor)
                                        
                                        # 如果找到了"同意"，优先使用；否则更新最佳结果
                                        if keyword == "同意":
                                            print(f"    对比度增强发现'同意'按钮 (置信度: {conf_float:.3f})")
                                            best_result = {
                                                "found": True,
                                                "keyword": keyword,
                                                "full_text": text,
                                                "center": {"x": center_x, "y": center_y},
                                                "confidence": round(conf_float, 3),
                                                "detection_method": "ocr_enhanced"
                                            }
                                            break  # 找到同意就退出
                                        elif best_result is None or conf_float > best_result['confidence']:
                                            best_result = {
                                                "found": True,
                                                "keyword": keyword,
                                                "full_text": text,
                                                "center": {"x": center_x, "y": center_y},
                                                "confidence": round(conf_float, 3),
                                                "detection_method": "ocr_enhanced"
                                            }
                    except (ValueError, TypeError):
                        continue
                        
            except Exception as e:
                print(f"    增强图像OCR识别失败: {e}")
        
        return {"best_result": best_result, "all_texts": all_texts}
    
    def detect_x_in_region(self, img, region_name, roi_x, roi_y, roi_w, roi_h):
        """在指定区域检测×号
        
        Args:
            img: 原始图片
            region_name: 区域名称
            roi_x, roi_y, roi_w, roi_h: 区域坐标和大小
        
        Returns:
            dict or None: 检测结果
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 确保ROI在图片范围内
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, w - roi_x)
            roi_h = min(roi_h, h - roi_y)
            
            if roi_w <= 0 or roi_h <= 0:
                return None
            
            roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # 使用模板匹配检测×号
            best_match = None
            best_confidence = 0
            
            for template_size in [16, 20, 24, 30, 36, 42]:
                template = np.zeros((template_size, template_size), dtype=np.uint8)
                
                thickness = max(2, template_size // 8)
                margin = template_size // 4
                
                # 画×形状
                cv2.line(template, (margin, margin), (template_size-margin, template_size-margin), (255,), thickness)
                cv2.line(template, (template_size-margin, margin), (margin, template_size-margin), (255,), thickness)
                
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = {
                        'location': max_loc,
                        'template_size': template_size
                    }
            
            # 使用更高的置信度阈值来过滤×号检测结果
            if best_confidence > self.x_confidence_threshold:
                x = roi_x + best_match['location'][0] + best_match['template_size'] // 2
                y = roi_y + best_match['location'][1] + best_match['template_size'] // 2
                
                # 排除Y<140的检测结果
                if y < self.y_threshold:
                    return None
                
                return {
                    "found": True,
                    "center": {"x": x, "y": y},
                    "confidence": best_confidence,
                    "type": f"x_button_{region_name}",
                    "detection_method": "template_matching"
                }
            
            return None
            
        except Exception as e:
            print(f"  {region_name}区域×号检测失败: {e}")
            return None
    
    def detect_all_x_buttons(self, image_path):
        """检测所有区域的×号按钮"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            h, w = img.shape[:2]
            x_candidates = []
            
            # 1. 右侧中上部（右侧30%，上部50%，但Y>140）
            roi_y = self.y_threshold
            roi_h = max(int(h * 0.5) - self.y_threshold, 50)
            result = self.detect_x_in_region(
                img, "right_top",
                int(w * 0.7), roi_y, int(w * 0.3), roi_h
            )
            if result:
                x_candidates.append(result)
            
            # 2. 右侧中间（右侧20%，中间40%）
            result = self.detect_x_in_region(
                img, "right_middle",
                int(w * 0.8), int(h * 0.3), int(w * 0.2), int(h * 0.4)
            )
            if result:
                x_candidates.append(result)
            
            # 3. 左侧中下部（左侧30%，下方50%）
            result = self.detect_x_in_region(
                img, "left_bottom",
                0, int(h * 0.5), int(w * 0.3), int(h * 0.5)
            )
            if result:
                x_candidates.append(result)
            
            # 4. 底部中间（中间60%，底部30%）
            result = self.detect_x_in_region(
                img, "bottom_center",
                int(w * 0.2), int(h * 0.7), int(w * 0.6), int(h * 0.3)
            )
            if result:
                x_candidates.append(result)
            
            return x_candidates
            
        except Exception as e:
            print(f"×号检测失败: {e}")
            return []
    
    def process_single_screen(self, img, image_path, start_y=0, screen_index=1):
        """处理单个屏幕的识别
        
        Args:
            img: 要处理的图片区域
            image_path: 原始图片路径（用于×号检测）
            start_y: Y坐标偏移量
            screen_index: 屏幕索引
        
        Returns:
            dict: 包含所有候选结果的字典
        """
        candidates = []
        all_texts = []
        
        # 1. 首先进行OCR文字检测
        ocr_result = self.ocr_detect_text(img, start_y)
        
        # 2. 检查是否找到"放弃"关键字
        found_abandon = False
        if ocr_result["best_result"] and ocr_result["best_result"]["keyword"] == "放弃":
            found_abandon = True
            print(f"    第{screen_index}屏检测到'放弃'关键字，继续检测×号...")
        
        # 3. 根据不同情况处理
        if ocr_result["best_result"] and not found_abandon:
            # 如果找到高置信度的文字且不是"放弃"，直接使用
            result = ocr_result["best_result"]
            result["screen"] = screen_index
            candidates.append(result)
            print(f"    第{screen_index}屏检测到文字按钮: {result['keyword']} (置信度: {result['confidence']})")
        else:
            # 如果没有找到文字或找到的是"放弃"，进行×号检测
            x_candidates = self.detect_all_x_buttons(image_path)
            
            # 过滤出在当前屏幕范围内的×号
            screen_height = img.shape[0]
            found_x_in_screen = False
            
            for x_result in x_candidates:
                y = x_result["center"]["y"]
                # 检查×号是否在当前屏幕范围内
                if start_y <= y < start_y + screen_height * self.scale_factor:
                    x_result["screen"] = screen_index
                    x_result["keyword"] = "×"
                    x_result["full_text"] = f"×({x_result['type']})"
                    candidates.append(x_result)
                    found_x_in_screen = True
                    print(f"    第{screen_index}屏检测到×号: {x_result['type']} (置信度: {x_result['confidence']:.3f})")
            
            # 如果是"放弃"但没找到×号，仍然使用"放弃"的结果
            if found_abandon and not found_x_in_screen and ocr_result["best_result"]:
                result = ocr_result["best_result"]
                result["screen"] = screen_index
                candidates.append(result)
                print(f"    第{screen_index}屏未找到×号，使用'放弃'按钮 (置信度: {result['confidence']})")
        
        all_texts.extend(ocr_result["all_texts"])
        
        return {
            "candidates": candidates,
            "all_texts": all_texts
        }
    
    def find_button(self, image_path):
        """主要的按钮查找函数"""
        start_time = time.time()
        
        # 1. 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            return {
                "found": False,
                "error": "图片读取失败",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        
        original_height, original_width = img.shape[:2]
        aspect_ratio = original_height / original_width
        
        # 2. 图片预处理
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return {
                "found": False,
                "error": "图片预处理失败",
                "total_time": f"{time.time() - start_time:.3f}s"
            }
        
        # 3. 判断是否为长截图
        is_long_screenshot = aspect_ratio > 2.5
        
        all_candidates = []
        all_detected_texts = []
        
        if is_long_screenshot:
            print(f"  检测到长截图 (高宽比: {aspect_ratio:.2f})，将逐屏识别...")
            
            # 计算屏幕参数
            scaled_height, scaled_width = processed_img.shape[:2]
            screen_height = int(scaled_width * 2.3)  # 一屏的高度
            num_screens = max(1, (scaled_height + screen_height - 1) // screen_height)
            
            print(f"  共{num_screens}屏")
            
            # 逐屏处理
            for i in range(num_screens):
                start_y = i * screen_height
                end_y = min((i + 1) * screen_height, scaled_height)
                
                print(f"    处理第{i+1}/{num_screens}屏...")
                
                # 获取当前屏幕的图像
                screen_img = processed_img[start_y:end_y, :]
                
                # 处理当前屏幕
                screen_result = self.process_single_screen(
                    screen_img, image_path, 
                    start_y * self.scale_factor, i + 1
                )
                
                all_candidates.extend(screen_result["candidates"])
                all_detected_texts.extend(screen_result["all_texts"])
        
        else:
            # 非长截图，直接处理整张图片
            screen_result = self.process_single_screen(processed_img, image_path)
            all_candidates.extend(screen_result["candidates"])
            all_detected_texts.extend(screen_result["all_texts"])
        
        # 4. 选择最佳结果
        if all_candidates:
            # 特殊处理：如果同时有"放弃"和×号，优先选择×号
            abandon_candidates = [c for c in all_candidates if c.get('keyword') == '放弃']
            x_candidates = [c for c in all_candidates if c.get('keyword') == '×']
            other_text_candidates = [c for c in all_candidates if c.get('detection_method') == 'ocr' and c.get('keyword') != '放弃']
            
            # 特殊处理：如果同时有"同意"和"取消"，优先选择"同意"
            agree_candidates = [c for c in other_text_candidates if c.get('keyword') == '同意']
            cancel_candidates = [c for c in other_text_candidates if c.get('keyword') == '取消']
            
            if abandon_candidates and x_candidates:
                # 同时存在"放弃"和×号，选择置信度最高的×号
                best_result = max(x_candidates, key=lambda x: x['confidence'])
                print(f"\n  同时检测到'放弃'和×号，优先选择×号 (置信度: {best_result['confidence']:.3f})")
            elif agree_candidates and cancel_candidates:
                # 同时存在"同意"和"取消"，优先选择置信度最高的"同意"
                best_result = max(agree_candidates, key=lambda x: x['confidence'])
                print(f"\n  同时检测到'同意'和'取消'，优先选择'同意' (置信度: {best_result['confidence']:.3f})")
            elif other_text_candidates:
                # 选择置信度最高的其他文字结果
                best_result = max(other_text_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: 文字'{best_result['keyword']}' (置信度: {best_result['confidence']:.3f})")
            elif abandon_candidates:
                # 只有"放弃"，选择置信度最高的"放弃"
                best_result = max(abandon_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: 文字'{best_result['keyword']}' (置信度: {best_result['confidence']:.3f})")
            elif x_candidates:
                # 只有×号，选择置信度最高的×号
                best_result = max(x_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: ×号'{best_result['type']}' (置信度: {best_result['confidence']:.3f})")
            else:
                # 其他情况，选择置信度最高的结果
                best_result = max(all_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: {best_result.get('keyword', 'unknown')} (置信度: {best_result['confidence']:.3f})")
            
            best_result["total_time"] = f"{time.time() - start_time:.3f}s"
            return best_result
        
        # 5. 没有找到任何按钮
        return {
            "found": False,
            "keywords_searched": self.keywords,
            "all_detected": all_detected_texts,
            "total_time": f"{time.time() - start_time:.3f}s"
        }


# 辅助函数
def is_url(path):
    """检查路径是否为URL"""
    return path.startswith(('http://', 'https://'))


def download_image_from_url(url, timeout=30):
    """从URL下载图片到临时文件"""
    try:
        print(f"  正在下载图片...")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 获取文件扩展名
        content_type = response.headers.get('Content-Type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            suffix = '.jpg'
        elif 'png' in content_type:
            suffix = '.png'
        else:
            suffix = '.jpg'
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        print(f"  图片下载完成: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"  下载失败: {e}")
        return None


def create_annotated_image(image_path, center_x, center_y, keyword, confidence, output_path):
    """创建带标注的图片"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # 绘制标注
        cv2.circle(img, (center_x, center_y), 20, (0, 0, 255), 3)
        cv2.line(img, (center_x-10, center_y), (center_x+10, center_y), (0, 0, 255), 2)
        cv2.line(img, (center_x, center_y-10), (center_x, center_y+10), (0, 0, 255), 2)
        
        # 添加文字标签
        label = f"{keyword} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        text_x = center_x - text_width // 2
        text_y = center_y - 40
        
        # 确保文字不超出边界
        if text_x < 0:
            text_x = 5
        if text_x + text_width > img.shape[1]:
            text_x = img.shape[1] - text_width - 5
        if text_y < text_height:
            text_y = center_y + 50
        
        # 绘制文字背景
        overlay = img.copy()
        cv2.rectangle(overlay, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + baseline + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 绘制文字
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        cv2.imwrite(output_path, img)
        return True
        
    except Exception as e:
        print(f"    标注图片失败: {e}")
        return False


def get_images_from_folder(folder_path):
    """获取文件夹中的所有图片文件
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        list: 图片文件路径列表
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    image_files = []
    
    # 遍历所有支持的格式
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # 大写扩展名
        pattern_upper = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # 去重并排序
    image_files = list(set(image_files))
    image_files.sort()
    
    return image_files


def batch_process_images(txt_file_path=None, folder_path=None, output_file="recognition_results.json"):
    """批量处理图片
    
    Args:
        txt_file_path: txt文件路径（包含图片路径列表）
        folder_path: 文件夹路径（直接读取文件夹中的图片）
        output_file: 输出文件名
    """
    print(f"开始批量处理图片...")
    
    image_paths = []
    source_info = ""
    
    # 根据输入类型获取图片路径列表
    if txt_file_path:
        print(f"读取路径文件: {txt_file_path}")
        source_info = f"文本文件: {txt_file_path}"
        
        if not os.path.exists(txt_file_path):
            print(f"错误: 路径文件不存在 - {txt_file_path}")
            return
        
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    image_paths.append(line)
                    
    elif folder_path:
        print(f"读取文件夹: {folder_path}")
        source_info = f"文件夹: {folder_path}"
        
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹不存在 - {folder_path}")
            return
        
        if not os.path.isdir(folder_path):
            print(f"错误: {folder_path} 不是一个文件夹")
            return
        
        image_paths = get_images_from_folder(folder_path)
    
    print("=" * 60)
    
    if not image_paths:
        print("未找到有效的图片路径")
        return
    
    print(f"找到 {len(image_paths)} 个图片路径")
    print("=" * 60)
    
    # 初始化按钮识别器
    finder = OptimizedButtonFinder()
    
    results = []
    success_count = 0
    downloaded_files = []
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"annotated_images_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图片
    for idx, image_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] 处理: {image_path}")
        
        temp_file_path = None
        actual_image_path = image_path
        
        # 处理URL图片
        if is_url(image_path):
            temp_file_path = download_image_from_url(image_path)
            if temp_file_path is None:
                results.append({
                    "index": idx,
                    "image_path": image_path,
                    "status": "failed",
                    "error": "URL图片下载失败",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            actual_image_path = temp_file_path
            downloaded_files.append(temp_file_path)
        else:
            # 检查本地文件
            if not os.path.exists(image_path):
                results.append({
                    "index": idx,
                    "image_path": image_path,
                    "status": "failed",
                    "error": "本地文件不存在",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
        
        # 执行识别
        try:
            recognition_result = finder.find_button(actual_image_path)
            
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
                    "full_text": recognition_result.get("full_text", ""),
                    "coordinates": {
                        "x": recognition_result["center"]["x"],
                        "y": recognition_result["center"]["y"]
                    },
                    "confidence": recognition_result["confidence"],
                    "detection_method": recognition_result["detection_method"]
                })
                
                if "screen" in recognition_result:
                    result_item["screen"] = recognition_result["screen"]
                
                print(f"  ✓ 找到按钮: {recognition_result['keyword']} at ({recognition_result['center']['x']}, {recognition_result['center']['y']})")
                
                # 创建标注图片
                if is_url(image_path):
                    annotated_filename = f"annotated_{idx}.jpg"
                else:
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
                    result_item["annotated_image"] = annotated_path
                
            else:
                result_item["detected_texts"] = recognition_result.get("all_detected", [])
                print(f"  ✗ 未找到目标按钮")
            
            result_item["total_time"] = recognition_result.get("total_time", "N/A")
            
        except Exception as e:
            print(f"  ✗ 识别过程出错: {e}")
            result_item = {
                "index": idx,
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        results.append(result_item)
    
    # 清理临时文件
    for temp_file in downloaded_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # 保存结果
    output_data = {
        "process_info": {
            "total_images": len(image_paths),
            "success_count": success_count,
            "failed_count": len(image_paths) - success_count,
            "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_info,
            "output_directory": output_dir,
            "x_confidence_threshold": 0.6
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"  总计: {len(image_paths)} 个图片")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {len(image_paths) - success_count} 个")
    print(f"  结果已保存到: {output_file}")
    print(f"  标注图片保存到: {output_dir}")
    print(f"  ×号置信度阈值: 0.6")
    
    return results


def main():
    example_txt = "image_paths.txt"
    if not os.path.exists(example_txt):
        with open(example_txt, 'w', encoding='utf-8') as f:
            f.write("# 图片路径列表\n")
            f.write("# 每行一个路径，支持本地路径和URL\n")
            f.write("# 示例:\n")
            f.write("# page/1.png\n")
            f.write("# https://example.com/image.jpg\n")
        print(f"已创建示例文件: {example_txt}")
    
    while True:
        print("\n选择操作:")
        print("1. 处理 image_paths.txt")
        print("2. 指定其他txt文件")
        print("3. 处理文件夹中的图片")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1/2/3/4): ").strip()
        
        if choice == '1':
            if os.path.exists(example_txt):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"results_{timestamp}.json"
                batch_process_images(txt_file_path=example_txt, output_file=output_file)
            else:
                print(f"\n错误: {example_txt} 文件不存在")
                
        elif choice == '2':
            txt_file = input("请输入txt文件路径: ").strip()
            if os.path.exists(txt_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"results_{timestamp}.json"
                batch_process_images(txt_file_path=txt_file, output_file=output_file)
            else:
                print(f"\n错误: {txt_file} 文件不存在")
                
        elif choice == '3':
            folder_path = input("请输入文件夹路径: ").strip()
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"results_folder_{timestamp}.json"
                batch_process_images(folder_path=folder_path, output_file=output_file)
            else:
                print(f"\n错误: {folder_path} 不是有效的文件夹路径")
                
        elif choice == '4':
            print("\n程序退出")
            break
        else:
            print("\n无效选项")


if __name__ == "__main__":
    main()