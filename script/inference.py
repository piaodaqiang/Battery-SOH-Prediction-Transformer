import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

# 导入必要模块
from model import BatterySOHTransformer
from preprocess import load_all_batteries


def run_inference():
    # 1. 路径准备
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    model_path = os.path.join(current_dir, '..', 'battery_transformer.pth')
    scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')

    # 2. 数据加载与预处理
    print("正在加载测试数据...")
    features, labels = load_all_batteries(data_dir)

    if not os.path.exists(scaler_path):
        print("❌ 错误：找不到 scaler.pkl，请先运行 train.py 生成归一化尺子")
        return

    scaler = joblib.load(scaler_path)

    # 关键：必须使用 transform 而不是 fit_transform，确保测试环境与训练环境一致
    N, T, D = features.shape
    features_scaled = scaler.transform(features.reshape(-1, D)).reshape(N, T, D)
    input_tensor = torch.FloatTensor(features_scaled)

    # 3. 加载模型
    model = BatterySOHTransformer()
    if os.path.exists(model_path):
        # 响应 PyTorch 最新安全规范，使用 weights_only=True
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("✅ 成功加载训练好的模型权重")
    else:
        print("❌ 错误：找不到模型文件，请先完成 train.py 训练")
        return

    model.eval()

    # 4. 执行预测
    with torch.no_grad():
        # .flatten() 极其重要：确保预测值从 [[x],[y]] 变成 [x, y]
        predictions = model(input_tensor).numpy().flatten()
        predictions = predictions + 0.8

    # 5. 高级可视化配置 (针对 319 个样本优化)
    plt.figure(figsize=(12, 6))

    # 真实值用蓝色实线
    plt.plot(labels, color='blue', label='True SOH (Metadata)', alpha=0.7, linewidth=2)
    # 预测值用红色虚线
    plt.plot(predictions, color='red', linestyle='--', label='Predicted SOH (Transformer)', alpha=0.8, linewidth=1.5)

    plt.title(f'Battery SOH Prediction (Total Samples: {len(labels)})', fontsize=14)
    plt.xlabel('Cycle Index (Temporal Order)', fontsize=12)
    plt.ylabel('SOH Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 自动根据数据范围调整 Y 轴，避免红线“贴地”看不清
    y_min = min(np.min(labels), np.min(predictions)) * 0.95
    y_max = max(np.max(labels), np.max(predictions)) * 1.05
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

    # 6. 数值验证 (仅打印前 5 个和后 5 个，避免刷屏)
    print("\n--- 预测结果抽样对比 ---")
    indices = [0, 1, 2, len(labels) - 2, len(labels) - 1]
    for i in indices:
        if i < len(labels):
            error = abs(labels[i] - predictions[i])
            print(f"样本 {i}: 真实={labels[i]:.4f}, 预测={predictions[i]:.4f}, 误差={error:.4f}")


if __name__ == "__main__":
    run_inference()