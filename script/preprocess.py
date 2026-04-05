import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def load_all_batteries(data_dir):
    all_features = []
    all_labels = []

    # 定义统一的序列长度（根据你数据集的最小行数调整，先设为400确保安全）
    target_length = 400

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        # 提取特征
        feature = df[['Voltage_measured', 'Current_measured', 'Temperature_measured']].values

        # --- 核心修复：对齐长度 ---
        if len(feature) >= target_length:
            # 如果够长，截取前 target_length 个点
            feature = feature[:target_length, :]
            all_features.append(feature)

            # 临时标签 (等我们写完安时积分再替换真实的)
            all_labels.append(0.95)
        else:
            # 如果该文件太短，直接跳过或者报错提醒
            print(f"警告: 文件 {file} 长度不足 {target_length}, 已跳过")

    # 此时所有元素形状都是 (400, 3)，可以安全转换了
    return np.array(all_features), np.array(all_labels)

# 4. 执行归一化并保存
# ... 使用 MinMaxScaler ...

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