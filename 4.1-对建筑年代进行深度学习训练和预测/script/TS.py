### 1.读取图片文件并分割出测试集数据

# 重新加载
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


### 3.自定义加载dataset并加载

# 我们需要进一步修改CustomDataset类以返回文件名中的建筑id。然后，在预测循环中收集文件名，并在所有预测完成后将它们与预测结果一起保存到CSV文件中。以下是如何实现它的步骤：

class CustomDataset(Dataset):
    """包装PyTorch数据集以应用转换。"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.imgs = subset.dataset.imgs

    def __getitem__(self, index):
        img, y = self.subset[index]  # 这里的y是类别的索引

        # 获取文件名
        file_name = self.imgs[self.subset.indices[index]][0]  # 修改这里以匹配您的文件名和路径结构
        # 获取文件名中的id
        id = file_name.split('--')[-2]

        if self.transform:
            img = self.transform(img)

        return img, y, id

    def __len__(self):
        return len(self.subset)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # 获取到BatchNorm2d (norm5)层的所有层
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        # 对空间维度进行平均池化
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


if __name__ == '__main__':  # 以尝试将启动训练过程的代码放入此保护块中。这有助于防止 multiprocessing 在它不应该这样做的时候启动新进程。
    img_root = "../../data/GSV/clip"  # '/root/autodl-tmp/GSV/clip'
    all_data = datasets.ImageFolder(root=img_root)  # 不要应用tranform
    # 拆分数据
    train_size = int(0.8 * len(all_data))
    test_size = len(all_data) - train_size
    # 固定随机种子
    torch.manual_seed(8)
    train_data_raw, test_data_raw = random_split(all_data, [train_size, test_size])

    len(test_data_raw)

    ### 2.获取类名列表

    class_names = all_data.classes

    # 数据集的类别的字典形式
    class_dict = all_data.class_to_idx
    print(class_dict)

    ### 4.定义transform并加载测试集

    # 只需要调整尺寸和转换为张量
    test_transform = transforms.Compose([
        transforms.Resize(size=(300, 300), antialias=True),
        transforms.ToTensor()

    ])

    test_data = CustomDataset(test_data_raw, transform=test_transform)

    ### 5.加载模型

    from torchvision.models import densenet121
    from torchvision.models.densenet import DenseNet121_Weights
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载预训练的DenseNet121模型
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)

    ## 修改最后一层的输出特征数
    num_features = model.classifier.in_features
    # 修改为9个类别的输出特征数
    model.classifier = nn.Linear(num_features, 9)

    # 加载建筑年代的数据集
    model_path = '../models/weights_6/model_epoch_32.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 调整到eval评估模式
    model.eval()

    model.to(device)

    # 创建DataLoader
    BATCH_SIZE = 8
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_data_iter = iter(test_loader)
    test_samples, test_labels, ids = next(test_data_iter)  # next() 函数是用来获取迭代器的下一个批次的数据
    print(test_samples.shape, test_labels.shape, len(ids))

    # 6.进行预测并且提取深度特征
    # 提取特征
    features_list = []
    labels_list = []

    with torch.inference_mode():
        # for inputs, labels, _ in tqdm(test_data_iter, desc="Predicting"): # 遍历test_loader会使用__getitem__方法,返回img, y, id
        # 转移到gpu
        inputs, labels = test_samples.to(device), test_labels.to(device)

        # 提取特征
        feature_extractor = FeatureExtractor(model).to(device)
        features = feature_extractor(inputs)
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)