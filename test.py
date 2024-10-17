import numpy as np
import matplotlib.pyplot as plt

# 1. 读取txt文件数据
data = np.loadtxt('testSet3.txt', delimiter='\t')

# 2. 分离数据的两列，x 和 y
x = data[:, 0]  # 第一列数据作为x轴
y = data[:, 1]  # 第二列数据作为y轴

# 3. 绘制散点图
plt.figure(figsize=(8, 6))  # 设置图形尺寸
plt.scatter(x, y, c='b', marker='o', label='Data Points')

# 4. 添加图形细节
plt.title('Scatter plot of data points(testSet3)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

# 5. 显示图形
plt.show()
