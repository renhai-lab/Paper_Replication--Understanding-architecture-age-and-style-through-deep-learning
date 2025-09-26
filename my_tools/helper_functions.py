"""
贯穿整个项目使用的一系列辅助函数集合。

如果一个函数被定义一次并且可以重复使用，它将放在这里。
"""

import os
import zipfile
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
from torch import nn

def walk_through_directory(dir_path):
    """遍历目录并返回其内容的详细信息。
    
    Args:
        dir_path (str): 目标目录路径
    
    Returns:
        None: 打印输出以下信息：
            - dir_path中子目录的数量
            - 每个子目录中图像（文件）的数量
            - 每个子目录的名称
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"在目录 '{dirpath}' 中有 {len(dirnames)} 个子目录和 {len(filenames)} 个图像。")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    绘制模型在X上预测结果与y的对比决策边界。

    来源 - https://madewithml.com/courses/foundations/neural-networks/ (有修改)
    
    参数：
        model (torch.nn.Module): 训练好的PyTorch模型
        X (torch.Tensor): 输入特征数据 
        y (torch.Tensor): 真实标签数据
    """
    # 将所有内容放到CPU上（与NumPy + Matplotlib配合更好）
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # 设置预测边界和网格
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # 创建特征
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # 进行预测
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # 测试是否为多类别或二分类，并将logits调整为预测标签
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # 多类别分类
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # 二分类

    # 重塑预测结果并绘图
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_linear_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """绘制线性训练数据和测试数据并比较预测结果。
    
    Args:
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据  
        test_labels: 测试标签
        predictions: 可选的预测结果
    """
    plt.figure(figsize=(10, 7))

    # 用蓝色绘制训练数据
    plt.scatter(train_data, train_labels, c="b", s=4, label="训练数据")

    # 用绿色绘制测试数据
    plt.scatter(test_data, test_labels, c="g", s=4, label="测试数据")

    if predictions is not None:
        # 用红色绘制预测结果（预测是在测试数据上进行的）
        plt.scatter(test_data, predictions, c="r", s=4, label="预测结果")

    # 显示图例
    plt.legend(prop={"size": 14})


# 计算准确率（分类指标）
def calculate_accuracy(y_true, y_pred):
    """计算真实标签和预测结果之间的准确率。

    Args:
        y_true (torch.Tensor): 预测的真实标签
        y_pred (torch.Tensor): 要与真实标签比较的预测结果

    Returns:
        float: y_true和y_pred之间的准确率值，例如78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_training_time(start, end, device=None):
    """打印开始时间和结束时间之间的差值。

    Args:
        start (float): 计算的开始时间（建议使用timeit格式）
        end (float): 计算的结束时间
        device (str, optional): 运行计算的设备。默认为None

    Returns:
        float: 开始和结束之间的时间（以秒为单位，数值越大表示时间越长）
    """
    total_time = end - start
    print(f"\n在设备 {device} 上的训练时间: {total_time:.3f} 秒")
    return total_time


# 绘制模型的损失曲线
def plot_loss_curves(results):
    """
    绘制结果字典的训练曲线。

    参数：
        results (dict): 包含值列表的字典，例如：
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="训练损失")
    plt.plot(epochs, test_loss, label="测试损失")
    plt.title("损失")
    plt.xlabel("轮次")
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="训练准确率")
    plt.plot(epochs, test_accuracy, label="测试准确率")
    plt.title("准确率")
    plt.xlabel("轮次")
    plt.legend()


# 来自notebook 04的预测和绘图图像功能
# 参见创建过程: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def predict_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """使用训练好的模型对目标图像进行预测并绘制图像。

    Args:
        model (torch.nn.Module): 训练好的PyTorch图像分类模型
        image_path (str): 目标图像的文件路径
        class_names (List[str], optional): 目标图像的不同类别名称。默认为None
        transform: 目标图像的变换。默认为None
        device (torch.device, optional): 用于计算的目标设备。
                                       默认为"cuda"（如果torch.cuda.is_available()）否则为"cpu"
    
    Returns:
        None: 显示目标图像的Matplotlib图和以模型预测为标题的图像。

    Example:
        predict_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["类别_1", "类别_2", "类别_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. 加载图像并将张量值转换为float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. 将图像像素值除以255以使其在[0, 1]之间
    target_image = target_image / 255.0

    # 3. 如有必要进行变换
    if transform:
        target_image = transform(target_image)

    # 4. 确保模型在目标设备上
    model.to(device)

    # 5. 启用模型评估模式和推理模式
    model.eval()
    with torch.inference_mode():
        # 为图像添加额外的维度
        target_image = target_image.unsqueeze(dim=0)

        # 对带有额外维度的图像进行预测并将其发送到目标设备
        target_image_pred = model(target_image.to(device))

    # 6. 将logits转换为预测概率（对于多类别分类使用torch.softmax()）
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. 将预测概率转换为预测标签
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. 绘制图像以及预测结果和预测概率
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # 确保尺寸适合matplotlib
    if class_names:
        title = f"预测: {class_names[target_image_pred_label.cpu()]} | 概率: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"预测: {target_image_pred_label} | 概率: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_random_seeds(seed: int = 42):
    """为torch操作设置随机种子。

    Args:
        seed (int, optional): 要设置的随机种子。默认为42
    """
    # 为一般torch操作设置种子
    torch.manual_seed(seed)
    # 为CUDA torch操作（在GPU上发生的操作）设置种子
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """
    从源下载压缩数据集并解压到目标位置。

    参数：
        source (str): 包含数据的压缩文件链接
        destination (str): 解压数据的目标目录
        remove_source (bool): 是否在下载和提取后删除源文件
    
    返回：
        pathlib.Path: 下载数据的路径
    
    使用示例：
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # 设置数据文件夹路径
    data_path = Path("data/")
    image_path = data_path / destination

    # 如果图像文件夹不存在，则下载并准备...
    if image_path.is_dir():
        print(f"[信息] {image_path} 目录已存在，跳过下载。")
    else:
        print(f"[信息] 未找到 {image_path} 目录，正在创建...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # 下载数据
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[信息] 正在从 {source} 下载 {target_file}...")
            f.write(request.content)

        # 解压数据
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[信息] 正在解压 {target_file} 数据...") 
            zip_ref.extractall(image_path)

        # 删除.zip文件
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
