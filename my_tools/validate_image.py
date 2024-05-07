#%%
import glob
import os
from PIL import Image
import tqdm

# GSV图像所在文件夹
img_root_dir = "/root/autodl-tmp/GSV"

# 利用glob模块获取所有png文件的路径
file_paths = glob.glob(f"{img_root_dir}\\clip\\**\\*.png")
print("file_paths length:", len(file_paths))

## 1验证文件有效性
def validate_image(image_path):
    try:
        img = Image.open(image_path)  # 尝试打开图像
        img.verify()  # 验证该文件是一个有效的PNG
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid file: {image_path}, error: {e}")
        return False

for file_path in tqdm.tqdm(file_paths):
    if not validate_image(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")