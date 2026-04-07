import pandas as pd

# 只要路径是对的就行
test_file = r"D:\Battery-SOH-Prediction-Transformer\data\00005.csv"
try:
    df = pd.read_csv(test_file)
    print("\n✅ 读取成功！")
    print("你的 CSV 文件里所有的列名有：")
    print(df.columns.tolist())
    print("\n前两行数据预览：")
    print(df.head(2))
except Exception as e:
    print(f"❌ 读取失败，错误原因: {e}")