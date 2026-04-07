import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存归一化模型


def load_all_batteries(data_dir):
    all_features = []
    all_labels = []
    target_length = 400

    # 1. 加载 metadata.csv
    metadata_path = os.path.join(os.path.dirname(data_dir), 'metadata.csv')
    meta_df = pd.read_csv(metadata_path)
    meta_df.set_index('filename', inplace=True)

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"正在匹配 metadata 并处理 {len(files)} 个文件...")

    for file in files:
        try:
            # --- 核心修复部分 ---
            raw_capacity = meta_df.loc[file, 'Capacity']

            # 强制转换为数字，如果转换失败则设为 NaN
            current_capacity = pd.to_numeric(raw_capacity, errors='coerce')

            # 检查：如果是空的或者是无效数据，直接跳过
            if pd.isna(current_capacity) or current_capacity <= 0:
                continue

        except (KeyError, TypeError):
            continue

        # 3. 读取特征数据逻辑保持不变
        df = pd.read_csv(os.path.join(data_dir, file))
        required_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured']

        if all(col in df.columns for col in required_cols):
            feature = df[required_cols].values
            if len(feature) >= target_length:
                all_features.append(feature[:target_length, :])
                all_labels.append(float(current_capacity) / 2.0)  # 确保是 float 计算

    print(f"✅ 匹配完成！成功提取出 {len(all_labels)} 个高质量样本。")
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