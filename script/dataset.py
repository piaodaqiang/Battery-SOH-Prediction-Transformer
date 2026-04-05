import torch
from torch.utils.data import Dataset, DataLoader


class BatteryDataset(Dataset):
    def __init__(self, features, labels):
        # 将 numpy 转换为 pytorch 张量
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    # 模拟数据验证逻辑
    import numpy as np

    mock_features = np.random.rand(3, 400, 3)
    mock_labels = np.array([0.95, 0.92, 0.90])

    dataset = BatteryDataset(mock_features, mock_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for feat, lbl in dataloader:
        print(f"Batch 特征形状: {feat.shape}")  # 预期: torch.Size([2, 400, 3])
        print(f"Batch 标签形状: {lbl.shape}")  # 预期: torch.Size([2])
        break