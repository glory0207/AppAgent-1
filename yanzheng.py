import cv2
import numpy as np
import requests
import re
import os
from urllib.parse import urlparse, parse_qs
import tempfile

def parse_text_entry(text):
    """
    解析文本条目，提取URL和坐标信息
    """
    lines = text.strip().split('\n')
    
    entry = {}
    for line in lines:
        line = line.strip()
        
        # 解析URL行
        if line.startswith('[') and '] http' in line:
            # 提取URL
            url_match = re.search(r'\] (http[s]?://[^\s]+)', line)
            if url_match:
                entry['url'] = url_match.group(1)
        
        # 解析其他字段
        elif ': ' in line:
            key, value = line.split(': ', 1)
            
            # 特殊处理坐标
            if key == '坐标':
                coord_match = re.search(r'\((\d+),\s*(\d+)\)', value)
                if coord_match:
                    entry['x'] = int(coord_match.group(1))
                    entry['y'] = int(coord_match.group(2))
            else:
                entry[key] = value
    
    return entry

def download_image(url, save_path=None):
    """
    下载图片到本地
    """
    try:
        # 创建临时文件
        if save_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            save_path = temp_file.name
            temp_file.close()
        
        # 下载图片
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    except Exception as e:
        print(f"下载图片失败: {e}")
        return None

def mark_coordinate(image_path, x, y, label=None, color=(0, 0, 255)):
    """
    在图片上标记坐标点
    
    Args:
        image_path: 图片路径
        x, y: 坐标
        label: 标签文字
        color: 标记颜色 (B, G, R)
    
    Returns:
        标记后的图片
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 画圆点
    cv2.circle(img, (x, y), 10, color, -1)
    
    # 画十字线
    cv2.line(img, (x-20, y), (x+20, y), color, 2)
    cv2.line(img, (x, y-20), (x, y+20), color, 2)
    
    # 添加坐标文字
    coord_text = f"({x},{y})"
    cv2.putText(img, coord_text, (x+15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 如果有标签，在坐标下方显示
    if label:
        cv2.putText(img, label, (x+15, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img

def process_single_entry(text_entry, output_dir="marked_images"):
    """
    处理单个文本条目
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 解析文本
    entry = parse_text_entry(text_entry)
    
    if 'url' not in entry or 'x' not in entry or 'y' not in entry:
        print("无法解析完整信息")
        return None
    
    print(f"处理图片: {entry.get('关键字', 'unknown')} at ({entry['x']}, {entry['y']})")
    
    # 下载图片
    temp_image_path = download_image(entry['url'])
    if not temp_image_path:
        return None
    
    # 标记坐标
    label = f"{entry.get('关键字', '')} {entry.get('检测方法', '')}"
    marked_img = mark_coordinate(temp_image_path, entry['x'], entry['y'], label)
    
    if marked_img is not None:
        # 保存标记后的图片
        output_filename = f"marked_{entry['x']}_{entry['y']}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, marked_img)
        print(f"已保存到: {output_path}")
        
        # 清理临时文件
        os.remove(temp_image_path)
        
        return output_path
    
    # 清理临时文件
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    return None

def process_text_file(file_path, output_dir="marked_images"):
    """
    处理包含多个条目的文本文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割条目（假设每个条目以 [数字] 开头）
    entries = re.split(r'\n(?=\[\d+\])', content.strip())
    
    results = []
    for i, entry in enumerate(entries):
        if entry.strip():
            print(f"\n处理第 {i+1} 个条目...")
            result = process_single_entry(entry, output_dir)
            if result:
                results.append(result)
    
    print(f"\n处理完成，共处理 {len(results)} 个图片")
    return results

def mark_multiple_coordinates(image_path, coordinates, output_path=None):
    """
    在一张图片上标记多个坐标点
    
    Args:
        image_path: 图片路径
        coordinates: 坐标列表 [(x1, y1, label1), (x2, y2, label2), ...]
        output_path: 输出路径
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    # 使用不同颜色标记不同的点
    colors = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 紫色
        (255, 255, 0),  # 青色
    ]
    
    for i, coord_info in enumerate(coordinates):
        if len(coord_info) >= 2:
            x, y = coord_info[0], coord_info[1]
            label = coord_info[2] if len(coord_info) > 2 else None
            color = colors[i % len(colors)]
            
            # 画圆点
            cv2.circle(img, (x, y), 10, color, -1)
            
            # 画十字线
            cv2.line(img, (x-20, y), (x+20, y), color, 2)
            cv2.line(img, (x, y-20), (x, y+20), color, 2)
            
            # 添加坐标文字
            coord_text = f"({x},{y})"
            cv2.putText(img, coord_text, (x+15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 如果有标签
            if label:
                cv2.putText(img, label, (x+15, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 保存图片
    if output_path is None:
        output_path = "marked_multiple.jpg"
    
    cv2.imwrite(output_path, img)
    print(f"已保存多点标记图片到: {output_path}")
    
    return img

# 
if __name__ == "__main__":
    # 示例1: 处理单个文本条目
   
    
    # 处理单个条目
    #process_single_entry(sample_text)
    
    # 示例2: 处理文本文件（如果有的话）
    process_text_file("results_20250718_111527.txt")
    
    # 示例3: 在本地图片上标记坐标（用于测试）
    # 如果你有本地图片，可以这样测试
    # marked_img = mark_coordinate("test.jpg", 526, 168, "× circular_close_button")
    # if marked_img is not None:
    #     cv2.imwrite("marked_test.jpg", marked_img)