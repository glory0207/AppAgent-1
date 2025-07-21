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
import gc  # 添加垃圾回收模块
import psutil  # 用于内存监控
import threading  # 添加线程支持，用于监控内存
import logging  # 添加日志支持

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_recognition.log"),
        logging.StreamHandler()
    ]
)

# 内存阈值设置 (MB)
MEM_WARNING_THRESHOLD = 1024  # 1GB
MEM_CRITICAL_THRESHOLD = 1536  # 1.5GB
MEM_LIMIT_THRESHOLD = 2048  # 2GB

class MemoryMonitor:
    """内存监控和管理类"""
    
    def __init__(self):
        """初始化内存监控器"""
        self.process = psutil.Process(os.getpid())
        self.stop_flag = False
        self.monitor_thread = None
        self.max_memory_used = 0
        self.last_check_time = 0
        self.check_interval = 5  # 检查间隔(秒)
    
    def get_memory_usage(self):
        """获取当前内存使用量(MB)"""
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except:
            return 0
    
    def monitor_memory(self):
        """内存监控线程函数"""
        logging.info("内存监控线程已启动")
        while not self.stop_flag:
            current_mem = self.get_memory_usage()
            self.max_memory_used = max(self.max_memory_used, current_mem)
            
            if current_mem > MEM_CRITICAL_THRESHOLD:
                logging.warning(f"[警告] 内存使用接近阈值: {current_mem:.1f}MB/{MEM_LIMIT_THRESHOLD}MB")
                gc.collect()
            elif current_mem > MEM_WARNING_THRESHOLD:
                logging.info(f"[注意] 内存使用较高: {current_mem:.1f}MB")
            
            # 强制执行内存限制
            if current_mem > MEM_LIMIT_THRESHOLD:
                logging.critical(f"[危险] 内存使用超过阈值 ({current_mem:.1f}MB)，执行强制垃圾回收")
                gc.collect()
                
                # 如果垃圾回收后仍然超过阈值，可能需要暂停处理
                if self.get_memory_usage() > MEM_LIMIT_THRESHOLD:
                    logging.critical("[危险] 内存使用仍然过高，暂停10秒等待系统回收资源")
                    time.sleep(10)
            
            time.sleep(self.check_interval)
            
    def start_monitoring(self):
        """启动内存监控"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_flag = False
            self.monitor_thread = threading.Thread(target=self.monitor_memory)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.stop_flag = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)
        logging.info(f"内存监控已停止。最大内存使用: {self.max_memory_used:.1f}MB")
    
    def check_memory_now(self, force_collect=False):
        """立即检查内存使用情况"""
        current_time = time.time()
        # 避免频繁检查
        if current_time - self.last_check_time < 2 and not force_collect:
            return
            
        self.last_check_time = current_time
        current_mem = self.get_memory_usage()
        
        if force_collect or current_mem > MEM_WARNING_THRESHOLD:
            logging.info(f"执行内存回收，当前: {current_mem:.1f}MB")
            gc.collect()
            new_mem = self.get_memory_usage()
            logging.info(f"内存回收后: {new_mem:.1f}MB (释放: {current_mem - new_mem:.1f}MB)")
        
        return current_mem

def release_ocr_resources():
    """释放EasyOCR资源，减少内存占用"""
    try:
        # 强制调用垃圾回收
        gc.collect()
        
        # 如果有CUDA支持，清理GPU内存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("已清理GPU缓存")
        except ImportError:
            pass
        
        return True
    except Exception as e:
        logging.error(f"释放资源失败: {e}")
        return False

class OptimizedButtonFinder:
    """按钮识别器"""
    
    def __init__(self):
        """初始化OCR模型"""
        start_time = time.time()
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}s")
        
        # 目标关键字
        self.keywords = ["关闭", "同意", "确定", "我知道了"]
        
        # 文字置信度阈值
        self.text_confidence_threshold = 0.6
        
        # ×号置信度阈值 
        self.x_confidence_threshold = 0.75
        
        # 图片缩放因子
        self.scale_factor = 1.0
        
        # Y坐标限制（Y<140的区域不检测×号）
        self.y_threshold = 140
    
    def preprocess_image(self, image_path):
        """图片预处理 - 固定800px压缩"""
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
            logging.info(f" 图片预处理失败: {e}")
            return None
    
    def ocr_detect_text(self, img, start_y=0):
        """OCR文字检测
        
        Args:
            img: 图片
            start_y: Y坐标偏移量（用于长截图）
        
        Returns:
            dict: 检测结果，包含最佳文字结果和所有检测到的文字
        """
        try:
            results = self.reader.readtext(img)
        except Exception as e:
            print(f"    OCR识别失败: {e}")
            return {"best_result": None, "all_texts": []}
        
        best_result = None
        all_texts = []
        
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
        
        if ocr_result["best_result"]:
            # 如果找到高置信度的文字，直接使用
            result = ocr_result["best_result"]
            result["screen"] = screen_index
            candidates.append(result)
            print(f"    第{screen_index}屏检测到文字按钮: {result['keyword']} (置信度: {result['confidence']})")
        
        all_texts.extend(ocr_result["all_texts"])
        
        # 2. 如果没有找到高置信度文字，进行×号检测
        if not ocr_result["best_result"]:
            # 使用原始图片进行×号检测
            x_candidates = self.detect_all_x_buttons(image_path)
            
            # 过滤出在当前屏幕范围内的×号
            screen_height = img.shape[0]
            for x_result in x_candidates:
                y = x_result["center"]["y"]
                # 检查×号是否在当前屏幕范围内
                if start_y <= y < start_y + screen_height * self.scale_factor:
                    x_result["screen"] = screen_index
                    x_result["keyword"] = "×"
                    x_result["full_text"] = f"×({x_result['type']})"
                    candidates.append(x_result)
                    print(f"    第{screen_index}屏检测到×号: {x_result['type']} (置信度: {x_result['confidence']:.3f})")
        
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
            # 文字结果优先
            text_candidates = [c for c in all_candidates if c.get('detection_method') == 'ocr']
            
            if text_candidates:
                # 选择置信度最高的文字结果
                best_result = max(text_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: 文字'{best_result['keyword']}' (置信度: {best_result['confidence']})")
            else:
                # 没有文字结果，选择置信度最高的×号
                best_result = max(all_candidates, key=lambda x: x['confidence'])
                print(f"\n  最终选择: ×号'{best_result['type']}' (置信度: {best_result['confidence']:.3f})")
            
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
    logging.info(f"开始批量处理图片...")
    
    # 启动内存监控器
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()
    
    # 记录开始时间
    start_time = time.time()
    
    image_paths = []
    source_info = ""
    
    # 根据输入类型获取图片路径列表
    if txt_file_path:
        logging.info(f"读取路径文件: {txt_file_path}")
        source_info = f"文本文件: {txt_file_path}"
        
        if not os.path.exists(txt_file_path):
            logging.error(f"错误: 路径文件不存在 - {txt_file_path}")
            return
        
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    image_paths.append(line)
                    
    elif folder_path:
        logging.info(f"读取文件夹: {folder_path}")
        source_info = f"文件夹: {folder_path}"
        
        if not os.path.exists(folder_path):
            logging.error(f"错误: 文件夹不存在 - {folder_path}")
            memory_monitor.stop_monitoring()
            return
        
        if not os.path.isdir(folder_path):
            logging.error(f"错误: {folder_path} 不是一个文件夹")
            memory_monitor.stop_monitoring()
            return
        
        image_paths = get_images_from_folder(folder_path)
    
    print("=" * 60)
    
    if not image_paths:
        logging.warning("未找到有效的图片路径")
        memory_monitor.stop_monitoring()
        return
    
    total_images = len(image_paths)
    logging.info(f"找到 {total_images} 个图片路径")
    print("=" * 60)
    
    # 初始化按钮识别器
    finder = OptimizedButtonFinder()
    
    results = []
    success_count = 0
    failed_count = 0
    downloaded_files = []
    
    # 注释掉创建输出目录的部分，因为不再生成标注图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None  # 不再需要输出目录
    
    # API模式，不进行分批处理和检查点恢复
    start_index = 0
    
    # 处理每个图片
    for idx, image_path in enumerate(image_paths[start_index:], start_index + 1):
        try:
            logging.info(f"\n[{idx}/{total_images}] 处理: {image_path}")
            
            # 检查内存状态
            current_mem = memory_monitor.check_memory_now()
            if current_mem and current_mem > MEM_CRITICAL_THRESHOLD:
                logging.warning(f"内存使用较高 ({current_mem:.1f}MB)，执行强制垃圾回收")
                gc.collect()
                # 如果垃圾回收后仍然内存过高，暂停一下
                if memory_monitor.get_memory_usage() > MEM_CRITICAL_THRESHOLD:
                    pause_time = 5
                    logging.warning(f"暂停处理 {pause_time} 秒，等待内存释放")
                    time.sleep(pause_time)
            
            temp_file_path = None
            actual_image_path = image_path
            
            # 处理URL图片
            if is_url(image_path):
                temp_file_path = download_image_from_url(image_path)
                if temp_file_path is None:
                    result_item = {
                        "index": idx,
                        "image_path": image_path,
                        "status": "failed",
                        "error": "URL图片下载失败",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    results.append(result_item)
                    failed_count += 1
                    continue
                actual_image_path = temp_file_path
                downloaded_files.append(temp_file_path)
            else:
                # 检查本地文件
                if not os.path.exists(image_path):
                    result_item = {
                        "index": idx,
                        "image_path": image_path,
                        "status": "failed",
                        "error": "本地文件不存在",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    results.append(result_item)
                    failed_count += 1
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
                    
                    logging.info(f"  [成功] 找到按钮: {recognition_result['keyword']} at ({recognition_result['center']['x']}, {recognition_result['center']['y']})")
                    
                    # 注释掉标注图片的功能，只输出坐标
                    """
                    if is_url(image_path):
                        annotated_filename = f"annotated_{idx:04d}.jpg"
                    else:
                        base_name = os.path.basename(image_path)
                        name_without_ext = os.path.splitext(base_name)[0]
                        annotated_filename = f"annotated_{idx:04d}_{name_without_ext}.jpg"
                    
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
                    """
                    # 只记录坐标信息，不生成标注图片
                    
                else:
                    failed_count += 1
                    result_item["detected_texts"] = recognition_result.get("all_detected", [])
                    logging.info(f"  [未找到] 未找到目标按钮")
                
                result_item["total_time"] = recognition_result.get("total_time", "N/A")
                
            except Exception as e:
                failed_count += 1
                logging.error(f"  [错误] 识别过程出错: {e}")
                result_item = {
                    "index": idx,
                    "image_path": image_path,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # 清理临时URL文件
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                    downloaded_files.remove(temp_file_path)
                except:
                    pass
            
            results.append(result_item)
            
            # 定期进行内存回收（每处理10张图片）
            if idx % 10 == 0:
                # 强制内存回收和资源释放
                memory_monitor.check_memory_now(force_collect=True)
                release_ocr_resources()
                
                # 当内存使用过高时，重新初始化OCR引擎
                if memory_monitor.get_memory_usage() > MEM_WARNING_THRESHOLD:
                    logging.info("重新初始化OCR引擎以减少内存占用...")
                    finder = OptimizedButtonFinder()  # 重新创建OCR实例
                
        except Exception as e:
            logging.error(f"处理图片时发生异常: {e}")
            continue
    
    # 确保所有临时文件都被清理
    for temp_file in downloaded_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # 保存最终结果
    output_data = {
        "process_info": {
            "total_images": total_images,
            "success_count": success_count,
            "failed_count": failed_count,
            "process_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_time": f"{(time.time() - start_time)/60:.2f} 分钟",
            "source": source_info,
            "x_confidence_threshold": 0.6,
            "max_memory_used_mb": memory_monitor.max_memory_used
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 停止内存监控
    memory_monitor.stop_monitoring()
    
    print("\n" + "=" * 60)
    logging.info(f"处理完成!")
    logging.info(f"  总计: {total_images} 个图片")
    logging.info(f"  成功: {success_count} 个")
    logging.info(f"  失败: {failed_count} 个")
    logging.info(f"  总耗时: {(time.time() - start_time)/60:.2f} 分钟")
    logging.info(f"  结果已保存到: {output_file}")
    logging.info(f"  最大内存使用: {memory_monitor.max_memory_used:.1f} MB")
    
    return results


def main():
    # 显示系统信息和内存阈值
    logging.info("-" * 80)
    logging.info(f"批量按钮识别工具启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示内存限制设置
    logging.info(f"内存阈值设置: 警告={MEM_WARNING_THRESHOLD}MB, 关键={MEM_CRITICAL_THRESHOLD}MB, 限制={MEM_LIMIT_THRESHOLD}MB")
    
    # 检查psutil是否可用
    if not 'psutil' in sys.modules:
        logging.warning("警告: psutil模块未安装，内存监控将不可用。建议安装: pip install psutil")
    
    # 创建示例文件
    example_txt = "image_paths.txt"
    if not os.path.exists(example_txt):
        with open(example_txt, 'w', encoding='utf-8') as f:
            f.write("# 图片路径列表\n")
            f.write("# 每行一个路径，支持本地路径和URL\n")
            f.write("# 示例:\n")
            f.write("# page/1.png\n")
            f.write("# https://example.com/image.jpg\n")
        logging.info(f"已创建示例文件: {example_txt}")
    
    # 主循环
    try:
        while True:
            print("\n" + "=" * 40)
            print("【批量按钮识别工具】")
            print("=" * 40)
            print("选择操作:")
            print("1. 处理 image_paths.txt")
            print("2. 指定其他txt文件")
            print("3. 处理文件夹中的图片")
            print("4. 系统状态和内存使用")
            print("5. 退出")
            
            choice = input("\n请输入选项 (1-5): ").strip()
            
            if choice == '1':
                if os.path.exists(example_txt):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"results_{timestamp}.json"
                    try:
                        batch_process_images(txt_file_path=example_txt, output_file=output_file)
                    except Exception as e:
                        logging.error(f"处理失败: {str(e)}")
                        print(f"\n处理过程中出错: {e}")
                else:
                    logging.error(f"{example_txt} 文件不存在")
                    print(f"\n错误: {example_txt} 文件不存在")
                    
            elif choice == '2':
                txt_file = input("请输入txt文件路径: ").strip()
                if os.path.exists(txt_file):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"results_{timestamp}.json"
                    try:
                        batch_process_images(txt_file_path=txt_file, output_file=output_file)
                    except Exception as e:
                        logging.error(f"处理失败: {str(e)}")
                        print(f"\n处理过程中出错: {e}")
                else:
                    logging.error(f"{txt_file} 文件不存在")
                    print(f"\n错误: {txt_file} 文件不存在")
                    
            elif choice == '3':
                folder_path = input("请输入文件夹路径: ").strip()
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"results_folder_{timestamp}.json"
                    try:
                        batch_process_images(folder_path=folder_path, output_file=output_file)
                    except Exception as e:
                        logging.error(f"处理失败: {str(e)}")
                        print(f"\n处理过程中出错: {e}")
                else:
                    logging.error(f"{folder_path} 不是有效的文件夹路径")
                    print(f"\n错误: {folder_path} 不是有效的文件夹路径")
            
            elif choice == '4':
                # 显示系统状态和内存使用情况
                print("\n系统状态:")
                if 'psutil' in sys.modules:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    memory_mb = mem_info.rss / (1024 * 1024)
                    
                    print(f"  当前内存使用: {memory_mb:.2f} MB")
                    print(f"  CPU使用率: {process.cpu_percent(interval=1):.1f}%")
                    
                    # 系统内存信息
                    system_memory = psutil.virtual_memory()
                    print(f"  系统总内存: {system_memory.total / (1024**3):.2f} GB")
                    print(f"  系统可用内存: {system_memory.available / (1024**3):.2f} GB")
                    print(f"  系统内存使用率: {system_memory.percent:.1f}%")
                    
                    # 内存警告
                    if memory_mb > MEM_WARNING_THRESHOLD:
                        print(f"\n当前内存使用较高。如果继续处理大批量图片，建议：")
                        print("  1. 减少每批处理的图片数量")
                        print("  2. 重启应用程序后再继续")
                    else:
                        print("\n 当前内存使用正常，可以继续处理图片")
                    
                else:
                    print("  psutil模块未安装，无法获取详细内存信息")
                    print("  建议安装psutil: pip install psutil")
                
                # 强制垃圾回收
                print("\n执行垃圾回收...")
                gc.collect()
                print("垃圾回收完成")
                
            elif choice == '5':
                logging.info("程序正常退出")
                print("\n程序退出")
                break
            else:
                print("\n无效选项，请重新输入")
    
    except KeyboardInterrupt:
        logging.info("用户中断程序")
        print("\n程序被用户中断")
    except Exception as e:
        logging.critical(f"程序发生严重错误: {str(e)}")
        print(f"\n程序发生错误: {e}")
    
    finally:
        # 最终清理
        logging.info("执行最终清理...")
        gc.collect()
        logging.info("程序结束")
        print("\n程序已结束")

if __name__ == "__main__":
    main()