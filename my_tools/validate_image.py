"""
图像验证工具模块

用于验证和清理损坏的图像文件。
"""

import glob
import os
from PIL import Image
import tqdm

# GSV图像所在文件夹
img_root_dir = "/root/autodl-tmp/GSV"

# 利用glob模块获取所有png文件的路径
file_paths = glob.glob(f"{img_root_dir}\\clip\\**\\*.png")
print("文件路径数量:", len(file_paths))

def validate_image(image_path):
    """
    验证图像文件的有效性。
    
    参数：
        image_path (str): 图像文件路径
        
    返回：
        bool: 如果图像有效返回True，否则返回False
    """
    try:
        img = Image.open(image_path)  # 尝试打开图像
        img.verify()  # 验证该文件是一个有效的PNG
        return True
    except (IOError, SyntaxError) as e:
        print(f"无效文件: {image_path}, 错误: {e}")
        return False

# 验证所有图像文件并删除损坏的文件
for file_path in tqdm.tqdm(file_paths):
    if not validate_image(file_path):
        os.remove(file_path)
        print(f"已删除 {file_path}")