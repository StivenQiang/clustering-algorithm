import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
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

# 实例化 AP 聚类模型
ap = AffinityPropagation()

# 训练模型并预测簇
y_pre = ap.fit_predict(x)
# 计算 DB 指数
db_index = davies_bouldin_score(x, y_pre)
print(f"AP Davies-Bouldin Index: {db_index}")

# 可视化聚类结果
colors = "rgbycmk"  # 添加更多颜色
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=[colors[i % len(colors)] for i in y_pre], s=30)
plt.title("AP Clustering", fontsize=24)
plt.xlabel("Feature 1", fontsize=20)  # Increase font size for the x-axis label
plt.ylabel("Feature 2", fontsize=20)  # Increase font size for the y-axis label

plt.show()
