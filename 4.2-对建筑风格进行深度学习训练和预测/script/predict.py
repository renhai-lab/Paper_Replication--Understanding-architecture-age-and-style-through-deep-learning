import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import densenet121
from torchvision.models.densenet import DenseNet121_Weights
from tqdm import tqdm


# 当使用t-SNE进行特征可视化时，最常用的做法是选择模型中的最后一个全连接层之前的输出作为特征。
# Densenst121应该在BatchNorm2d (norm5)之后、Linear (classifier)之前提取特征。
# 具体地说，应该提取BatchNorm2d (norm5)层的输出作为特征。
# 从模型summary的特征上看，BatchNorm2d (norm5)层将输出一个形状为[32, 1024, 9, 9]的特征张量，但通常我们会进行平均池化或者直接reshape来将其转换为[32, 1024]的二维张量，
# 然后再用这些特征进行t-SNE可视化。
class FeatureExtractor(torch.nn.Module):
    """用于提取DenseNet121模型中BatchNorm2d (norm5)层的输出作为特征的模块。"""

    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()  # 调用FeatureExtractor类的父类（torch.nn.Module）的构造函数，以确保所有基础功能都被正确初始化。
        # 获取到BatchNorm2d (norm5)层的所有层
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])  # 从原始模型original_model中提取除最后一层外的所有层
        # original_model.children(): 这会返回一个迭代器，包含original_model的所有直接子模块。在许多情况下，例如当使用预训练的模型时，这些子模块通常是网络的主要层或层组。
        # list(original_model.children()): 这将迭代器转换为一个列表，这样我们就可以对其进行索引和切片。
        # list(original_model.children())[:-1]: 通过这种方式进行切片，我们可以获得除最后一层外的所有层。假设original_model是一个预训练的分类网络，最后一层可能是一个全连接层，用于输出类别分数。通过排除这一层，我们可以得到一个仅进行特征提取的模型。
        # torch.nn.Sequential(*...): 这创建了一个新的Sequential模块，它包含我们从original_model中提取出来的所有层。*操作符是Python中的解包操作符，它将列表中的所有项解包并传递给Sequential构造函数。

    def forward(self, x):
        x = self.features(x)
        # 对空间维度进行平均池化
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class CustomSubset(Dataset):
    """Custom dataset to apply transforms Subset and extract building id from filenames."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.imgs = subset.dataset.imgs

    def __getitem__(self, index):
        img, y = self.subset[index]
        file_name = self.imgs[self.subset.indices[index]][0]
        id = os.path.basename(file_name).split("--")[-2]
        if self.transform:
            img = self.transform(img)
        return img, y, id

    def __len__(self):
        return len(self.subset)


class CustomDataset(Dataset):
    """Custom dataset to apply transforms Dataset and extract building id from filenames."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.imgs = data.imgs

    def __getitem__(self, index):
        img, y = self.data[index]
        file_name = self.imgs[index][0]
        id = os.path.basename(file_name).split("_")[0]
        if self.transform:
            img = self.transform(img)
        return img, y, id

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # 1. 读取图片文件并分割出测试集数据
    img_root = "/root/autodl-tmp/GSV/clip" # '../../data/GSV/clip'
    all_data = datasets.ImageFolder(root=img_root)

    # 2. 获取类名列表及数据集的类别的字典形式
    # class_names = all_data.classes
    # class_dict = all_data.class_to_idx
    # print(class_dict)

    # 3. 自定义加载dataset
    # 4. 定义transform并加载测试集
    transform = transforms.Compose([
        transforms.Resize(size=(300, 300), antialias=True),
        transforms.ToTensor()
    ])
    all_data_transformed = CustomDataset(all_data, transform=transform)

    # 5. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 7)
    model_path = 'model_epoch_32.pth' # r'C:\Users\hncdj\Documents\Python_\Python辅助城市研究\建筑风格和年代机器学习\4.2-对建筑风格进行深度学习训练和预测\models\building_style_weights_2\model_epoch_32.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # 6. 定义DataLoader并进行预测
    BATCH_SIZE = 400
    dataloader = DataLoader(all_data_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    # 6.1 定义变量
    pred_labels_list = []
    ids_list = []
    features_list = []
    labels_list = []

    # 初始化特征提取器
    feature_extractor = FeatureExtractor(model).to(device)

    with torch.inference_mode():
        for images, _, img_id in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)

            # 获取模型输出和预测标签
            outputs = model(images)
            pred_labels = outputs.argmax(dim=1)
            pred_labels_list.extend(pred_labels.cpu().numpy())

            # 提取特征
            features = feature_extractor(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(pred_labels.cpu().numpy())

            # 收集图像ID
            ids_list.extend(img_id)

    # 使用np.concatenate将所有batch的特征和标签连接成一个大的Numpy数组。
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    # 7. 保存预测结果到CSV文件
    df_predictions = pd.DataFrame({
        'id': ids_list,
        'prediction': pred_labels_list,
    })
    df_predictions.to_csv('predictions_with_building_style_model_2_on_all_data.csv', index=False)
    print("Predictions saved to predictions_with_building_style_model_2_on_all_data.csv")

    # %%
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)  # , perplexity=5
    transformed_features = tsne.fit_transform(features_array)
    dict = {0: '18th-century_style', 1: 'contemporary', 2: 'early_19th-century_style', 3: 'interwar', 4: 'late_19th-century_style', 5: 'postwar', 6: 'revival'}
    # 更换一下标签
    # 使用vectorize函数替换labels_array中的值
    vfunc = np.vectorize(dict.get)
    labels_updated = vfunc(labels_array)


    pd.DataFrame({'1-t': transformed_features[:, 0],
                  '2-t': transformed_features[:, 1],
                  'label': labels_updated,
                  'id': ids_list,
                  }).to_csv('TSNE-styele-output-on-all-datasets.csv', index=False)