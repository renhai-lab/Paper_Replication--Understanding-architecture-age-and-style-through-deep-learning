
"""
建筑年代深度学习训练脚本

使用DenseNet121模型对建筑年代进行分类训练，
支持数据增强、类别平衡和模型保存等功能。
"""

import os
import torch
import numpy as np
from torch import manual_seed, cuda
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import densenet121
from torchvision.models.densenet import DenseNet121_Weights
import time
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR


def safe_delete(file_path):
    """
    安全删除文件，支持重试机制。
    
    参数：
        file_path (str): 要删除的文件路径
    """
    # 最多尝试5次
    for _ in range(5):
        try:
            os.remove(file_path)
            break
        except PermissionError:
            print(f"删除文件 {file_path} 时权限被拒绝。正在重试...")
            time.sleep(5)  # 稍等一会儿再重试
        except Exception as e:
            print(f"无法删除文件 {file_path}: {e}")
            break


class CustomDataset(Dataset):
    """
    自定义数据集类，包装PyTorch数据集以应用特定的转换。
    
    该类允许对已拆分的数据子集应用不同的数据转换。
    """
    
    def __init__(self, subset, transform=None):
        """
        初始化自定义数据集。
        
        参数：
            subset: PyTorch数据集的子集
            transform: 要应用的数据转换
        """
        self.subset = subset
        self.transform = transform
        self.imgs = subset.dataset.imgs

    def __getitem__(self, index):
        """
        获取指定索引的数据项。
        
        参数：
            index (int): 数据项索引
            
        返回：
            tuple: (图像, 标签)
        """
        img, y = self.subset[index]

        if self.transform:
            img = self.transform(img)

        return img, y

    def __len__(self):
        """返回数据集大小。"""
        return len(self.subset)


if __name__ == "__main__":
    """
    主训练流程：
    1. 加载和配置DenseNet121模型
    2. 设置数据转换和增强
    3. 处理数据集的类别不平衡问题
    4. 配置训练参数和优化器
    5. 执行训练过程
    """
    
    print("开始建筑年代分类模型训练...")
    
    # 1. 模型配置
    print("1. 配置DenseNet121模型...")
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    
    # 修改最后一层的输出特征数为9个建筑年代类别
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 9)
    print(f"模型输出层已修改为 {num_features} -> 9 个类别")

    # 2. 数据转换配置
    print("2. 配置数据转换和增强...")
    
    # 训练数据转换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize(size=(300, 300), antialias=True),
        transforms.RandomHorizontalFlip(p=0.2),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.2),    # 随机垂直翻转
        transforms.RandomRotation(degrees=45),    # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试数据转换（无数据增强）
    test_transform = transforms.Compose([
        transforms.Resize(size=(300, 300), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. 数据集加载和拆分
    print("3. 加载数据集...")
    img_root = r"..\..\data\GSV\clip"  # 图像根目录
    all_data = datasets.ImageFolder(root=img_root)
    
    print(f"数据集总大小: {len(all_data)}")
    print(f"类别数量: {len(all_data.classes)}")
    print(f"类别名称: {all_data.classes}")
    
    # 拆分训练集和测试集（8:2比例）
    train_size = int(0.8 * len(all_data))
    test_size = len(all_data) - train_size
    
    # 设置随机种子以确保可重现性
    manual_seed(8)
    train_data_raw, test_data_raw = random_split(all_data, [train_size, test_size])
    print(f"训练集大小: {len(train_data_raw)}, 测试集大小: {len(test_data_raw)}")

    # 4. 处理类别不平衡问题
    print("4. 处理类别不平衡...")
    
    # 获取所有样本的标签
    all_labels = [label for _, label in all_data.samples]
    class_counts = np.bincount(all_labels)
    total_count = len(all_data)
    
    # 计算原始权重（反比于类别频率）
    original_weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    print(f"类别样本数量: {class_counts}")
    print(f"原始权重: {[f'{w:.2f}' for w in original_weights]}")

    # 调整权重以减少极端不平衡
    max_weight = max(original_weights)
    min_weight = min(original_weights)
    diff_weight = max_weight - min_weight
    increment = diff_weight * 0.1  # 为较小权重增加10%的增量

    adjusted_weights = [
        weight + increment if weight + increment <= max_weight else weight 
        for weight in original_weights
    ]
    print(f"调整后权重: {[f'{w:.2f}' for w in adjusted_weights]}")

    # 5. 创建加权采样器
    print("5. 创建数据加载器...")
    
    # 获取训练集索引和对应标签
    train_indices = train_data_raw.indices
    train_labels = [all_labels[idx] for idx in train_indices]
    
    # 计算训练样本权重
    train_sample_weights = [adjusted_weights[label] for label in train_labels]
    
    # 创建加权随机采样器
    train_sampler = WeightedRandomSampler(
        train_sample_weights, 
        num_samples=len(train_sample_weights), 
        replacement=True
    )

    # 应用数据转换
    train_data = CustomDataset(train_data_raw, transform=train_transform)
    test_data = CustomDataset(test_data_raw, transform=test_transform)
    
    # 创建数据加载器
    BATCH_SIZE = 96
    print(f"批次大小: {BATCH_SIZE}")
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=12
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=12
    )

    # 6. 训练配置
    print("6. 配置训练参数...")
    
    # 导入训练引擎
    from engine import train
    
    # 配置优化器和损失函数
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    # 配置学习率调度器（每10个epoch降低90%）
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    print("优化器: SGD (lr=0.001, momentum=0.9)")
    print("损失函数: CrossEntropyLoss")
    print("调度器: StepLR (step_size=10, gamma=0.1)")

    # 7. 开始训练
    print("7. 开始训练...")
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        cuda.empty_cache()
        print("CUDA缓存已清空")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 开始计时
    from timeit import default_timer as timer
    start_time = timer()

    # 执行训练
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        epochs=40,
        device=device,
        model_save_path="models/weights_7",
        save_interval=1,
        logs_path='/root/tf-logs/train_experiment_7'
    )

    # 计算总训练时间
    end_time = timer()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n训练完成!")
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print("模型已保存到 models/weights_7 目录")
    print("训练日志已保存到 /root/tf-logs/train_experiment_7 目录")
