import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

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

    # 实例化 Dataset 和 DataLoader
    dataset = BatteryDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)  # 暂时用小batch测试

    # 3. 初始化模型、损失函数和优化器
    model = BatterySOHTransformer()
    criterion = nn.MSELoss()  # 均方误差，最适合回归任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率0.001

    # 4. 开启训练循环
    epochs = 50  # 初始练50遍试试
    print(f"开始训练，共计 {epochs} 个 Epoch...")

    model.train()  # 切换到训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_features)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播与权重更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 每 10 个 epoch 打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.6f}")

    # 5. 保存训练好的模型权重
    model_path = os.path.join(current_dir, '..', 'battery_transformer.pth')
    torch.save(model.state_dict(), model_path)
    print(f"训练完成！模型已保存至: {model_path}")


if __name__ == "__main__":
    train_model()