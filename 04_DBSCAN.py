import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score


# 读取数据集
def loadDataSet(fileName):
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))  # 映射所有的元素为 float 类型
            dataSet.append(fltLine)
    return np.array(dataSet)

# 加载数据集
x = loadDataSet('testSet3.txt')

# 实例化 DBSCAN 模型
dbscan = DBSCAN(eps=0.5, min_samples=4)

# 训练模型并预测簇
y_pre = dbscan.fit_predict(x)
# 计算 DB 指数
db_index = davies_bouldin_score(x, y_pre)
print(f"DBSCAN Davies-Bouldin Index: {db_index}")

# 可视化聚类结果
colors = "rgbycmk"  # 添加更多颜色
plt.figure()
# 处理噪声点 (-1 标签) 用黑色表示
plt.scatter(x[:, 0], x[:, 1], c=[colors[i % len(colors)] if i != -1 else 'w' for i in y_pre], s=30)
plt.title("DBSCAN", fontsize=24)
plt.xlabel("Feature 1", fontsize=20)  # Increase font size for the x-axis label
plt.ylabel("Feature 2", fontsize=20)  # Increase font size for the y-axis label
plt.show()
