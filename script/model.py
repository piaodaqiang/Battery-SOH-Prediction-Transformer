import torch
import torch.nn as nn
import math


# 1. 位置编码：赋予 Transformer 对序列顺序的感知能力
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x 形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


# 2. 电池 SOH 预测 Transformer 模型
class BatterySOHTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super(BatterySOHTransformer, self).__init__()

        # 将原始特征 (3维) 投射到模型维度 (64维)
        self.input_projection = nn.Linear(input_dim, model_dim)

        # 添加位置编码
        self.pos_encoder = PositionalEncoding(model_dim)

        # Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 回归层：将特征转化为 SOH 标量
        self.regressor = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 调试：检查输入形状
        # print(f"[Debug] 输入形状: {x.shape}")

        # 1. 特征投射: (B, 400, 3) -> (B, 400, 64)
        x = self.input_projection(x)

        # 2. 注入位置信息
        x = self.pos_encoder(x)

        # 3. Transformer 编码
        x = self.transformer_encoder(x)

        # 4. 全局特征聚合 (取序列均值作为该循环的特征概括)
        x = x.mean(dim=1)

        # 5. 输出预测值: (B, 1) -> (B)
        x = self.regressor(x)
        return x.squeeze(-1)


# --- 检查代码 (Verification Logic) ---
if __name__ == "__main__":
    print("--- 启动模型架构自检 ---")

    # 模拟你之前生成的 Batch 数据: 2个样本, 400个点, 3个特征
    mock_input = torch.randn(2, 400, 3)
    print(f"1. 模拟输入数据形状: {mock_input.shape}")

    # 实例化模型
    model = BatterySOHTransformer()
    print("2. 模型实例化成功！")

    # 尝试一次前向传播 (Forward Pass)
    try:
        output = model(mock_input)
        print(f"3. 前向传播成功！")
        print(f"4. 输出预测值形状: {output.shape}")  # 预期应为 torch.Size([2])
        print(f"5. 输出 SOH 示例: {output.detach().numpy()}")

        # 统计参数量，看看模型规模
        total_params = sum(p.numel() for p in model.parameters())
        print(f"6. 模型总参数量: {total_params:,}")
        print("--- 自检完成：模型逻辑完全正确！ ---")

    except Exception as e:
        print(f"自检失败，错误原因: {e}")