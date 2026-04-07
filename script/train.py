import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR

# 导入你昨天写的模块
from preprocess import load_all_batteries
from dataset import BatteryDataset
from model import BatterySOHTransformer


def train_model():
    # 1. 路径准备
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')

    # 2. 加载并准备数据 (这里会用到你昨天的归一化逻辑)
    print("正在加载数据...")
    features, labels = load_all_batteries(data_dir)

    labels = labels - 0.8

    # 实例化 Dataset 和 DataLoader
    dataset = BatteryDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 检测并定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 3. 初始化模型、损失函数和优化器
    model = BatterySOHTransformer().to(device) # 将模型推送到显卡

    criterion = nn.MSELoss()  # 均方误差，最适合回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率0.001
    # 每隔 20 个 epoch，将学习率乘以 0.1
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # 4. 开启训练循环
    epochs = 200  # 训练200遍
    print(f"开始训练，共计 {epochs} 个 Epoch...")

    model.train()  # 切换到训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            # 将每一批次的数据推送到显卡
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_features)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播与权重更新
            loss.backward()

            # 梯度裁剪 (防止模型 “炸掉” )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        # 每 10 个 epoch 打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.6f}")

        scheduler.step()

    # 5. 保存训练好的模型权重
    model_path = os.path.join(current_dir, '..', 'battery_transformer.pth')
    torch.save(model.state_dict(), model_path)
    print(f"训练完成！模型已保存至: {model_path}")


if __name__ == "__main__":
    train_model()