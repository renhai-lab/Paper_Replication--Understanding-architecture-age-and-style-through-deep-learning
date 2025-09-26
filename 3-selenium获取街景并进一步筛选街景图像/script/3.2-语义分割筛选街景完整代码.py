"""
语义分割筛选街景图像完整代码

使用MIT ADE20K预训练模型对街景图像进行语义分割，
筛选出包含建筑物facade的高质量图像。
"""

import csv
import glob
import os
import shutil
import time

import numpy as np
import PIL.Image
import scipy.io
import torch
import torchvision.transforms as transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from pypushdeer import PushDeer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 常量定义
DEFAULT_IMAGE_PATH = r'../data/GSV/default_image.png'
BUILDING_INDEX = 1
BUILDING_RATIO_THRESHOLD = 0.4
BATCH_SIZE = 12
NUM_WORKERS = 12
START_BATCH_NUM = 2200
NOTIFICATION_INTERVAL = 200
MAX_RETRIES = 5
RETRY_DELAY = 5


def load_segmentation_model(seg_repo_dir):
    """加载语义分割模型。
    
    Args:
        seg_repo_dir (str): 语义分割仓库目录路径
        
    Returns:
        tuple: (分割模块, 颜色映射, 类别名称映射)
    """
    # 加载颜色映射表
    colors = scipy.io.loadmat(f'{seg_repo_dir}/data/color150.mat')['colors']

    # 加载类别名称映射表
    names = {}
    with open(f'{seg_repo_dir}/data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    # 加载编码器模型和权重
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights=f'{seg_repo_dir}/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')

    # 加载解码器模型和权重
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights=f'{seg_repo_dir}/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    # 创建损失函数和分割模块
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()
    return segmentation_module, colors, names


def safe_delete_file(file_path):
    """安全删除文件，支持重试机制。
    
    Args:
        file_path (str): 要删除的文件路径
    """
    # 最多尝试指定次数
    for attempt in range(MAX_RETRIES):
        try:
            os.remove(file_path)
            break
        except PermissionError:
            print(f"删除文件 {file_path} 时权限被拒绝。正在重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"无法删除文件 {file_path}: {e}")
            break
# 定义数据集类
class ImageDataset(Dataset):
    """图像数据集类，用于加载和预处理街景图像。"""
    
    def __init__(self, file_paths):
        """初始化图像数据集。
        
        Args:
            file_paths (list): 图像文件路径列表
        """
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """返回数据集大小。"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """获取指定索引的数据项。
        
        Args:
            idx (int): 数据索引
            
        Returns:
            tuple: (图像数据, 文件路径)
        """
        try:
            pil_image = PIL.Image.open(self.file_paths[idx]).convert('RGB')
            img_data = self.transform(pil_image)
        except Exception as e:
            print(f"Error processing file {self.file_paths[idx]}: {e}")
            # 删除损坏的图像
            safe_delete_file(self.file_paths[idx])

            # 如果发生错误，返回默认图像
            pil_image = PIL.Image.open(DEFAULT_IMAGE_PATH).convert('RGB')
            img_data = self.transform(pil_image)
            
        return img_data, self.file_paths[idx]



def process_prediction(pred):
    """对预测结果进行处理（计算各类别比率并排序）。
    
    Args:
        pred (numpy.ndarray): 分割预测结果
        
    Returns:
        tuple: (类别像素数, 排序后的类别索引)
    """
    # 计算每个类别的像素数，并获取从多到少的排序
    class_counts = np.bincount(pred.flatten())
    sorted_classes = class_counts.argsort()[::-1]
    return class_counts, sorted_classes


def judge_prediction(pred, building_index=BUILDING_INDEX):
    """
    根据给定的分割预测结果，判断“建筑物”类别是否是图像中的主导类别，
    并且其比例是否超过40%。

    参数:
        pred: numpy array, 分割预测结果
        building_index: int, “建筑物”类别的索引

    返回:
        bool: 如果“建筑物”是主导类别并且其比例超过40%，则返回True，否则返回False。
    """
    # 处理预测结果
    class_counts, sorted_classes = process_prediction(pred)

    # 判断
    # 检查“建筑物”是否是最常见的类别
    if sorted_classes[0] == building_index:
        # 计算“建筑物”类别的像素数占比
        building_ratio = class_counts[building_index] / pred.size
        # print(f'“建筑物”类别的像素数占比为{building_ratio:.2%}')

        # 如果“建筑物”的比例超过40%，执行相应的操作
        if building_ratio > 0.4:
            # 执行你想要的操作，例如可视化或保存图像
            return True
        else:
            # print("照片不符合要求：建筑占比不超过40%")
            return False
    else:
        # print(f"照片不符合要求：占比最大的类别不是建筑")

        # 打印前4个最常见的类别，方便检查
        # for i, c in enumerate(sorted_classes[:4]):
        #     print(f'排序后（占比多的在前）第{i + 1}个类别名称是：{names[c + 1]}，预测的类别代号：{c}')

        return False


def create_directory(path):
    """创建文件夹。
    
    Args:
        path (str): 文件夹路径
    """
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    """
    主函数：执行语义分割筛选街景图像的完整流程。
    """
    start_time = time.time()

    # 语义分割模型仓库目录
    seg_repo_dir = "../semantic-segmentation-pytorch-master"

    # 街景图像根目录
    img_root_dir = r"/data/GSV"

    # 创建不符合要求的图像分类文件夹
    folders = [
        f"{img_root_dir}\\unqualified"  # 不符合要求的图像根目录
    ]
    
    # 建筑年代分类：分为9个文件夹加1个未知年代
    facade_photos = [
        "pre-1652",
        "1653–1705", 
        "1706–1764",
        "1765–1845",
        "1846–1910",
        "1911–1943",
        "1944–1977",
        "1978–1994",
        "1995–2023",
        "未知年代"
    ]

    # 将建筑年代标签添加到输出文件夹列表中
    folders.extend([os.path.join(img_root_dir, "unqualified", label) for label in facade_photos])

    # 创建所有必要的文件夹
    for folder in folders:
        create_directory(folder)

    # 加载语义分割模型
    print("正在加载语义分割模型...")
    segmentation_module, colors, names = load_segmentation_model(seg_repo_dir)

    # 创建图像数据集
    # 获取所有PNG文件的路径
    file_paths = glob.glob(f"{img_root_dir}\\clip\\**\\*.png")
    print(f"找到图像文件数量: {len(file_paths)}")

    dataset = ImageDataset(file_paths)
    print(f"数据集大小: {len(dataset)}")

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 初始化推送通知（可选）
    try:
        pushdeer = PushDeer(pushkey="PDU22018TBKAygHi6CfrjI99HYdp6H2U4JVRVkOXQ")
    except Exception as e:
        print(f"初始化通知服务失败: {e}")
        pushdeer = None

    # 创建进度条
    pbar = tqdm(total=len(dataloader) - START_BATCH_NUM, 
               desc="处理图像批次", ncols=100)

    # 开始语义分割和筛选过程
    with torch.inference_mode():
        for current_batch_num, (img_data, file_paths_batch) in enumerate(dataloader, start=1):
            # 跳过指定批次之前的数据（用于断点续传）
            if current_batch_num < START_BATCH_NUM:
                continue

            # 记录当前批次的合格率
            qualified_rate = []

            # 执行语义分割预测
            img_data = img_data.cuda()
            scores = segmentation_module({'img_data': img_data}, segSize=img_data.shape[2:])
            _, pred = torch.max(scores, dim=1)
            pred = pred.cpu().numpy()

            # 处理批次中的每张图像
            for idx, single_pred in enumerate(pred):
                # 判断图片是否合格（建筑物占比是否超过阈值）
                if judge_prediction(single_pred):
                    qualified_rate.append(1)  # 合格
                else:
                    # 将不合格的图片移动到指定文件夹
                    old_path = file_paths_batch[idx]
                    new_path = old_path.replace("clip", "unqualified")
                    try:
                        shutil.move(old_path, new_path)
                        qualified_rate.append(0)  # 不合格
                    except Exception as e:
                        print(f"移动文件失败 {old_path}: {e}")

            # 更新进度条
            pbar.update()

            # 计算并打印当前批次合格率
            qualified_rate = np.array(qualified_rate)
            print(f"批次 {current_batch_num} 合格率: {qualified_rate.mean():.2%}")

            # 定期发送进度通知
            if pushdeer and current_batch_num % NOTIFICATION_INTERVAL == 0:
                try:
                    pushdeer.send_text(f"持续筛选街景图片中", desp=f"已处理批次: {current_batch_num}")
                except Exception as e:
                    print(f"发送通知失败: {e}")

    # 完成处理
    pbar.close()

    # 计算并输出总执行时间
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"执行时间: {int(minutes)} 分钟 {seconds:.2f} 秒")

    # 发送完成通知
    if pushdeer:
        try:
            pushdeer.send_text(f"街景图片筛选完成!", 
                             desp=f"执行时间: {int(minutes)} 分钟 {seconds:.2f} 秒")
        except Exception as e:
            print(f"发送完成通知失败: {e}")
