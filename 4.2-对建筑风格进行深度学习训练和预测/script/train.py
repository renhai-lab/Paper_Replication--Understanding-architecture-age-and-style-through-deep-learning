
import numpy as np
import torch
from torch import manual_seed, cuda, load
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import densenet121
from torchvision.models.densenet import DenseNet121_Weights
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
class CustomDataset(Dataset):
    """包装PyTorch数据集以应用转换。"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.imgs = subset.dataset.imgs

    def __getitem__(self, index):
        img, y = self.subset[index]

        if self.transform:
            img = self.transform(img)

        return img, y

    def __len__(self):
        return len(self.subset)

"""
包含用于训练和测试PyTorch模型的函数。
"""

if __name__ == "__main__":
    # 加载预训练的DenseNet121模型
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    ## 修改最后一层的输出特征数
    num_features = model.classifier.in_features
    # 修改为9个类别的输出特征数
    model.classifier = nn.Linear(num_features, 9)

    # 加载建筑年代的模型
    model.load_state_dict(load('../../4.1-对建筑年代进行深度学习训练和预测/weights_6/best_model.pth'))

    # 修改最后一层为7类
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 7)

    # 定义训练数据的转换函数
    train_transform = transforms.Compose([transforms.Resize(size=(300, 300), antialias=True),
                                          transforms.RandomHorizontalFlip(p=0.2),
                                          transforms.RandomVerticalFlip(p=0.2),
                                          transforms.RandomRotation(degrees=45),
                                          transforms.ToTensor(),
                                          ])

    # 定义测试数据的转换（如果需要的话，这里我们不应用任何转换）
    test_transform = transforms.Compose([
        transforms.Resize(size=(300, 300), antialias=True),
        transforms.ToTensor()
        # 可以添加其他转换，但在这个例子中我们不添加
    ])

    # 加载图片
    img_root = "/root/autodl-tmp/cambridge" # 修改为你的图片地址
    all_data = datasets.ImageFolder(root=img_root)  # 不要应用tranform
    # 拆分数据
    train_size = int(0.8 * len(all_data))
    test_size = len(all_data) - train_size
    # 固定随机种子
    manual_seed(2) # 固定cpu的随机种子
    cuda.manual_seed(2) # 固定cuda的随机种子
    train_data_raw, test_data_raw = random_split(all_data, [train_size, test_size])

    # 处理不平衡的数据集
    # 获取原始标签列表
    all_labels = [label for _, label in all_data.samples]
    class_counts = np.bincount(all_labels) # 获取每个类的样本数，以便进一步处理
    total_count = len(all_data) # 获取总样本数
    original_weights = [total_count / class_counts[i] for i in range(len(class_counts))] # 计算每个类的权重 [95.86867469879518, 43.69632070291049, 60.7412213740458, 5.552368990300747, 2.46502478314746, 7.819477201257862, 11.38029176201373, 7.482697009591875, 66.47535505430243]
    print("original_weights：", original_weights )

    # 计算最大最小的权重
    max_weight = max(original_weights)
    min_weight = min(original_weights)

    # 计算最大权重和最小权重之间的差异
    diff_weight = max_weight - min_weight

    # 计算要添加到每个较小权重的增量（例如，差异的一部分）
    increment = diff_weight * 0.01

    # 调整权重，为较小的权重增加增量
    adjusted_weights = [weight + increment if weight + increment <= max_weight else weight for weight in
                        original_weights]

    print("adjusted_weights：", adjusted_weights)

    # random_split返回的是Subset对象，我们可以通过.indices属性来获取原始数据集中的索引
    train_indices = train_data_raw.indices

    # 现在，我们使用这些索引来从全部标签列表中提取训练集标签
    train_labels = [all_labels[idx] for idx in train_indices]

    # 计算训练样本权重
    train_sample_weights = [adjusted_weights[label] for label in train_labels]

    # 创建加权随机采样器以进行重采样
    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    print("Size of training data:", len(train_data_raw))
    print("Number of sample weights:", len(train_sample_weights))
    assert len(train_sample_weights) == len(train_data_raw)

    # 使用自定义数据集类应用转换
    train_data = CustomDataset(train_data_raw, transform=train_transform)
    test_data = CustomDataset(test_data_raw, transform=test_transform)

    # 创建DataLoader
    BATCH_SIZE = 96
    print("BATCH_SIZE", BATCH_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=10)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    # 获取类名列表
    class_names = all_data.classes

    # 数据集的类别的字典形式
    class_dict = all_data.class_to_idx
    print(class_dict)

    ## 进行训练
    # 定义训练函数
    from engine import *
    # 定义损失函数和优化器
    optimizer = SGD(model.parameters(), lr=0.002, momentum=0.9)  # 使用更高的初始学习率
    loss_fn = nn.CrossEntropyLoss()
    # 引入学习率调度器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epochs降低当前学习率的10%


    # 清空cuda缓存
    cuda.empty_cache()

    # 开始计时器
    from timeit import default_timer as timer

    start_time = timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 开始训练并且保存结果
    results = train(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    scheduler=scheduler,  #  引入学习率调度器
                    epochs=70,
                    device=device,
                    model_save_path="../models/building_style_weights_2",
                    save_interval=1,
                    logs_path='/root/tf-logs/building_style_train_experiment_2_4090'  # autodl的tensorboard路径日志
                    )

    # 返回时间
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")
