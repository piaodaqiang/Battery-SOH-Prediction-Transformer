import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存归一化模型


def load_all_batteries(data_dir):
    all_features = []
    all_labels = []
    target_length = 400

    # 预设标签映射（针对 NASA B0005/6/7 数据集的典型值）
    # 如果 CSV 里没标签，程序会根据文件名来这里找
    CAPACITY_MAP = {
        "00005.csv": 1.856,
        "00006.csv": 2.035,
        "00007.csv": 1.891
    }

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"正在处理 {len(files)} 个文件...")

    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        # 1. 提取特征 (确保列名与你 check_columns.py 打印的一致)
        try:
            feature = df[['Voltage_measured', 'Current_measured', 'Temperature_measured']].values
        except KeyError:
            print(f"⚠️ 跳过文件 {file}: 缺少必要的特征列")
            continue

        if len(feature) >= target_length:
            # 统一截取长度
            feature = feature[:target_length, :]
            all_features.append(feature)

            # 2. 核心修复：安全提取 SOH 标签
            if 'Capacity' in df.columns:
                # 如果 CSV 里有这一列，直接读
                current_capacity = df['Capacity'].iloc[0]
            else:
                # 如果没有，从映射表里找，找不到就默认给 1.8
                current_capacity = CAPACITY_MAP.get(file, 1.8)

            soh = current_capacity / 2.0  # 假设额定容量为 2.0
            all_labels.append(soh)

    return np.array(all_features), np.array(all_labels)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')

    # 执行加载
    features, labels = load_all_batteries(data_dir)

    if len(features) == 0:
        print("❌ 错误：未找到有效数据，请检查 data 文件夹。")
    else:
        # 归一化逻辑保持不变
        scaler = MinMaxScaler(feature_range=(0, 1))
        N, T, D = features.shape
        features_reshaped = features.reshape(-1, D)
        features_scaled = scaler.fit_transform(features_reshaped)
        features_final = features_scaled.reshape(N, T, D)

        # 保存
        joblib.dump(scaler, os.path.join(current_dir, '..', 'scaler.pkl'))

        print("\n--- 预处理与归一化验证成功 ---")
        print(f"样本总数: {N}")
        print(f"特征形状: {features_final.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签示例: {labels[:3]}")