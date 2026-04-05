import joblib
import os

# 1. 定位文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')

# 2. 加载归一化模型
scaler = joblib.load(scaler_path)

# 3. 打印关键参数
print("--- 归一化模型参数详情 ---")
print(f"训练时看到的特征总数: {scaler.n_features_in_}")
print(f"每个特征的原始最大值: {scaler.data_max_}")
print(f"每个特征的原始最小值: {scaler.data_min_}")
print(f"每个特征的缩放幅度 (Scale): {scaler.scale_}")