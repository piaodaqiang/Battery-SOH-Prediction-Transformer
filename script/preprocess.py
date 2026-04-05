import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib # 用于保存归一化模型


def load_all_batteries(data_dir):
    all_features = []
    all_labels = []
    target_length = 400

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))

        # 1. 提取特征
        feature = df[['Voltage_measured', 'Current_measured', 'Temperature_measured']].values

        if len(feature) >= target_length:
            feature = feature[:target_length, :]
            all_features.append(feature)

            # 2. 计算真实 SOH：使用 Capacity 列的平均值（或最大值）
            # NASA数据集中，Capacity代表该次循环的总容量，额定容量通常为 2.0
            current_capacity = df['Capacity'].iloc[0]
            soh = current_capacity / 2.0
            all_labels.append(soh)

    return np.array(all_features), np.array(all_labels)

if __name__ == "__main__":
    # 1. 确定数据文件夹路径 (根据你的项目结构)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')

    # 2. 调用函数
    features, labels = load_all_batteries(data_dir)

    # 3. 打印关键信息进行验证
    print("--- 数据预处理验证 ---")
    print(f"成功读取的电池样本数: {features.shape[0]}")
    print(f"每个样本的时间点数: {features.shape[1]}")
    print(f"特征维度 (压/流/温): {features.shape[2]}")
    print(f"特征张量形状: {features.shape}")
    print(f"标签张量形状: {labels.shape}")
    print("前5个标签示例:", labels[:5])

    # 1. 初始化归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 2. 变换形状以适配归一化器: (3, 400, 3) -> (3*400, 3)
    N, T, D = features.shape
    features_reshaped = features.reshape(-1, D)

    # 3. 执行归一化
    features_scaled = scaler.fit_transform(features_reshaped)

    # 4. 还原形状: (1200, 3) -> (3, 400, 3)
    features_final = features_scaled.reshape(N, T, D)

    # 5. 保存归一化模型（非常重要！预测时要用同样的参数）
    joblib.dump(scaler, os.path.join(current_dir, '..', 'scaler.pkl'))

    print("--- 归一化验证 ---")
    print(f"归一化后的最大值: {features_final.max()}")
    print(f"归一化后的最小值: {features_final.min()}")
    print("归一化模型已保存为 scaler.pkl")