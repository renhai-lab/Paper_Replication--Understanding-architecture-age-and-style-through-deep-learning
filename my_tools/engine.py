import copy
import os
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm.auto import tqdm


# %%


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """对PyTorch模型进行单个epoch的训练。

    将目标PyTorch模型设置为训练模式，然后执行所有必要的训练步骤（前向传播、损失计算、优化器步骤）。

    参数：
    model：要训练的PyTorch模型。
    dataloader：用于训练模型的DataLoader实例。
    loss_fn：要最小化的PyTorch损失函数。
    optimizer：帮助最小化损失函数的PyTorch优化器。
    device：计算设备（例如："cuda"或"cpu"）。

    返回：
    训练损失和训练准确率的元组。
    格式为（train_loss，train_accuracy）。例如：

    (0.1112，0.8743)
    """
    # 将模型设置为训练模式
    model.train()

    # 设置训练损失和训练准确率的初始值
    train_loss, train_acc = 0, 0

    # 遍历数据加载器中的数据批次
    for batch, (X, y) in enumerate(dataloader):
        # 将数据发送到目标设备
        X, y = X.to(device), y.to(device)

        # 1. 前向传播
        y_pred = model(X)

        # 2. 计算并累加损失
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. 优化器梯度清零
        optimizer.zero_grad()

        # 4. 反向传播计算梯度
        loss.backward()

        # 5. 优化器更新参数
        optimizer.step()

        # 计算并累加准确率指标
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)



    # 调整指标以获得每个批次的平均损失和准确率
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """对PyTorch模型进行单个epoch的测试。

    将目标PyTorch模型设置为“eval”模式，然后在测试数据集上执行前向传播。

    参数：
    model：要测试的PyTorch模型。
    dataloader：用于测试模型的DataLoader实例。
    loss_fn：用于计算测试数据上的损失的PyTorch损失函数。
    device：计算设备（例如："cuda"或"cpu"）。

    返回：
    测试损失和测试准确率的元组。
    格式为（test_loss，test_accuracy）。例如：

    (0.0223，0.8985)
    """
    # 将模型设置为评估模式
    model.eval()

    # 设置测试损失和测试准确率的初始值
    test_loss, test_acc = 0, 0

    # 打开推理上下文管理器
    with torch.inference_mode():
        # 遍历DataLoader中的数据批次
        for batch, (X, y) in enumerate(dataloader):
            # 将数据发送到目标设备
            X, y = X.to(device), y.to(device)

            # 1. 前向传播
            test_pred_logits = model(X)

            # 2. 计算并累加损失
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 计算并累加准确率
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # 调整指标以获得每个批次的平均损失和准确率
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,  # 新增：学习率调度器
        epochs: int,
        device: torch.device,
        model_save_path: str,  # 新增：模型保存路径
        save_interval: int = 5,  # 新增：保存间隔（以epochs为单位）
        logs_path: str = '/root/tb-logs/train_experiment',
) -> Dict[str, List]:
    """训练和测试PyTorch模型。

    将目标PyTorch模型通过train_step()和test_step()函数进行多个epoch的训练和测试，
    在同一个epoch循环中训练和测试模型。

    计算、打印和存储评估指标。

    参数：
    model：要训练和测试的PyTorch模型。
    train_dataloader：用于训练模型的DataLoader实例。
    test_dataloader：用于测试模型的DataLoader实例。
    optimizer：帮助最小化损失函数的PyTorch优化器。
    loss_fn：用于计算两个数据集上的损失的PyTorch损失函数。
    epochs：要训练的epoch数。
    device：计算设备（例如："cuda"或"cpu"）。
    model_save_path：模型保存路径。
    save_interval：保存模型的间隔epoch数。

    返回：
    包含训练和测试损失以及训练和测试准确率指标的字典。
    每个指标都有一个列表值，表示每个epoch的指标。
    格式为：{train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    例如，如果训练2个epochs：
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """

    writer = SummaryWriter(logs_path)

    # 确保保存路径存在
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # 初始化最佳准确度

    # 创建空的结果字典
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 确保模型在目标设备上
    model.to(device)

    # 循环进行训练和测试步骤，直到达到指定的epoch数
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        # 记录训练的计算平均损失和准确度
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # 记录测试的计算平均损失和平均准确度
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        # 打印当前进度
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 性能提升时保存模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

        # 定期保存模型
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))

        # 更新结果字典
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # 在每个epoch结束时步进调度器
        scheduler.step()
        # 可选：打印当前学习率
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch + 1}: Current learning rate: {current_lr}")

    # 在训练结束后，我们也可以选择加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 在训练结束时返回填充的结果字典
    return results
