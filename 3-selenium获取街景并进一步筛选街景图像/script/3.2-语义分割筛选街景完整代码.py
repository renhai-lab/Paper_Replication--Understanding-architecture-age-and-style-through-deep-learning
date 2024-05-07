import shutil
import time
import os
import csv
# 导入必要的库
import glob
from pypushdeer import PushDeer
import PIL.Image
import numpy as np
import scipy.io
import torch
import torchvision.transforms as transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_seg_model(seg_repo_dir):
    # 加载颜色映射表
    colors = scipy.io.loadmat(f'{seg_repo_dir}/data/color150.mat')['colors']

    # 加载类别名称映射表
    names = {}
    with open(f'{seg_repo_dir}/data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    # 加载模型和权重
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights=f'{seg_repo_dir}/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')

    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights=f'{seg_repo_dir}/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()
    return segmentation_module, colors, names

def safe_delete(file_path):
    # 最多尝试几次
    for _ in range(5):
        try:
            os.remove(file_path)
            break
        except PermissionError:
            print(f"Permission denied when deleting file {file_path}. Retrying...")
            time.sleep(5)  # 稍等一会儿再重试
        except Exception as e:
            print(f"Unable to delete file {file_path}: {e}")
            break
# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        default_image = r'../data/GSV/default_image.png'  # 指定一个默认图像或默认图像的路径，以便在发生错误时返回
        try:
            pil_image = PIL.Image.open(self.file_paths[idx]).convert('RGB')
            img_data = self.transform(pil_image)
        except Exception as e:
            print(f"Error processing file {self.file_paths[idx]}: {e}")
            # 删除损坏的图像
            safe_delete(self.file_paths[idx])

            # 如果发生错误，返回默认图像
            pil_image = PIL.Image.open(default_image).convert('RGB')  # 加载默认图像
            img_data = self.transform(pil_image)
        return img_data, self.file_paths[idx]



def process_pred(pred):
    """对预测结果进行处理（计算各类别比率并排序）"""
    # 计算每个类别的像素数，并获取从多到少的排序
    class_counts = np.bincount(pred.flatten())
    sorted_classes = class_counts.argsort()[::-1]
    return class_counts, sorted_classes  # 返回类别像素数和排序后的类别


def judge_pred(pred, building_index):
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
    class_counts, sorted_classes = process_pred(pred)

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


def mkdir(path):
    """创建文件夹"""
    os.makedirs(path, exist_ok=True)  # exist_ok=True表示如果文件夹已经存在，就不要再创建了
    # print(f"创建文件夹{path}成功")


if __name__ == '__main__':
    start_time = time.time()

    seg_repo_dir = "../semantic-segmentation-pytorch-master"

    # GSV图像所在文件夹
    img_root_dir = r"/data/GSV"

    # 创建不符合要求的文件夹
    # 要创建的文件夹列表
    folders = [
        f"{img_root_dir}\\unqualified" # 不符合要求的图像
        ]
    # 建筑年代：分为9个文件夹
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

    # 将facade_photos中的每个标签添加到输出文件夹中
    folders.extend([os.path.join(img_root_dir, "unqualified", label) for label in
                    facade_photos])  # extend 是列表的一个方法，它允许你添加另一个列表的所有元素到当前列表中。

    # 创建所有文件夹
    for folder in folders:
        mkdir(folder)

    # if not os.path.exists(new_path):
    #     print(f"new_path文件不存在: {new_path}")
    #     continue
    # 加载模型
    segmentation_module, colors, names = load_seg_model(seg_repo_dir)

    # 创建数据集
    # 利用glob模块获取所有png文件的路径
    file_paths = glob.glob(f"{img_root_dir}\\clip\\**\\*.png")
    print("file_paths length:", len(file_paths))

    dataset = ImageDataset(file_paths)
    print("dataset length:", len(dataset))

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False, num_workers=12)

    # 设置开始的批次号
    start_batch_num = 2200

    # 初始化通知
    pushdeer = PushDeer(pushkey="PDU22018TBKAygHi6CfrjI99HYdp6H2U4JVRVkOXQ")

    # 用于跳过批次直到指定的批次号的标志
    begin_processing = start_batch_num == 0

    # 用tqdm显示进度，设置总批次数减去要跳过的批次数
    # 创建一个进度条考虑到总批次数
    pbar = tqdm(total=len(dataloader) - start_batch_num, desc="Processing each dataloaders", ncols=100)

    # 进行分割并可视化结果
    with torch.inference_mode():
        # 遍历数据加载器，返回图像数据和文件路径
        for current_batch_num, (img_data, file_paths_batch) in enumerate(dataloader, start=1):
            # 如果当前批次号小于我们想要开始的批次号，则跳过
            if current_batch_num < start_batch_num:
                continue

            # 对每批次的图像的合格率进行记录
            qualified_rate = []

            # 预测
            # 将图像数据移动到GPU上
            img_data = img_data.cuda()
            scores = segmentation_module({'img_data': img_data}, segSize=img_data.shape[2:])
            _, pred = torch.max(scores, dim=1)
            pred = pred.cpu().numpy()

            # 遍历批次中的每张图像的预测结果
            for idx, single_pred in enumerate(pred):
                # 判断图片是否合格
                if judge_pred(single_pred, building_index=1):
                    qualified_rate.append(1)  # 合格
                else:
                    # 不合格的图片移动到指定文件夹
                    old_path = file_paths_batch[idx]
                    new_path = old_path.replace("clip", "unqualified")
                    shutil.move(old_path, new_path)
                    qualified_rate.append(0)  # 不合格

            # 更新进度条
            pbar.update()

            # 记录合格率
            qualified_rate = np.array(qualified_rate)
            print(f"此批次合格率：{qualified_rate.mean():.2%}")

            # 每隔200个批次通知
            if current_batch_num % 200 == 0:
                pushdeer.send_text(f"持续筛选街景图片中", desp=f"{current_batch_num}")

    # 完成后关闭进度条
    pbar.close()

    # 计算并输出执行时间
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"Execution time: {minutes} minutes {seconds:.2f} seconds")

    pushdeer.send_text(f"！街景图片筛选完成", desp=f"Execution time: {minutes} minutes {seconds:.2f} seconds")
#%%
