import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# 导入你之前的模块
from model import BatterySOHTransformer
from preprocess import load_all_batteries


def run_inference():
    # 1. 路径与环境准备
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    model_path = os.path.join(current_dir, '..', 'battery_transformer.pth')
    scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')

    # 2. 加载数据与预处理工具
    print("正在加载测试数据...")
    features, labels = load_all_batteries(data_dir)
    scaler = joblib.load(scaler_path)

    # 对特征进行归一化（必须使用训练时的同一个 scaler）
    N, T, D = features.shape
    features_scaled = scaler.transform(features.reshape(-1, D)).reshape(N, T, D)
    input_tensor = torch.FloatTensor(features_scaled)

    # 3. 加载训练好的“大脑”
    model = BatterySOHTransformer()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # 切换到评估模式（关闭 Dropout 等）

    # 4. 执行预测
    with torch.no_grad():  # 推理阶段不需要计算梯度
        predictions = model(input_tensor).numpy()

    # 5. 可视化对比
    plt.figure(figsize=(10, 6))
    plt.plot(labels, 'bo-', label='True SOH (Ground Truth)', markersize=8)
    plt.plot(predictions, 'rx--', label='Predicted SOH (Transformer)', markersize=8)

    plt.title('Battery SOH Prediction: Transformer vs Reality', fontsize=14)
    plt.xlabel('Sample Index (Battery Cycles)', fontsize=12)
    plt.ylabel('SOH Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 打印数值对比
    print("\n--- 预测结果对比 ---")
    for i in range(len(labels)):
        error = abs(labels[i] - predictions[i])
        print(f"样本 {i + 1}: 真实值={labels[i]:.4f}, 预测值={predictions[i]:.4f}, 误差={error:.4f}")

    plt.show()


if __name__ == "__main__":
    run_inference()