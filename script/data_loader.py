import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 动态获取路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', '00005.csv')

# 2. 读取 CSV 数据
df = pd.read_csv(data_path)

# 打印前几行看看数据长什么样，方便我们确认
print("成功读取数据，前5行如下：")
print(df.head())

# 3. 绘图验证：观察电压随时间的变化 (Voltage vs Time)
# 我们取前 500 个数据点来观察一次典型的放电行为
plt.figure(figsize=(10, 6))
plt.plot(df['Time'][:500], df['Voltage_measured'][:500], 'g-', label='Voltage')

plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Battery #05: Voltage Discharge Characteristic')
plt.legend()
plt.show()

print("绘图成功！如果你看到了下降的电压曲线，说明原始数据采集完全正确。")