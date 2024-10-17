import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


# 加载数据集函数
def loadDataSet(fileName):
    '''
    加载数据集
    '''
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))  # 将数据映射为浮点数类型
            dataSet.append(fltLine)
    return np.array(dataSet)


# 主程序
if __name__ == '__main__':
    # 加载数据
    data = loadDataSet('testSet3.txt')

    # K-Means 聚类算法，指定聚类数量为 4
    kmeans = KMeans(n_clusters=7, random_state=42)
    y_pre = kmeans.fit_predict(data)
    # 计算 DB 指数
    db_index = davies_bouldin_score(data, y_pre)
    print(f"Davies-Bouldin Index: {db_index}")
    # 可视化聚类结果
    colors = "rgbycmk"
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=[colors[i % len(colors)] for i in y_pre], s=20)

    # 标记聚类中心
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='*', s=100,
                label='Centroids')
    plt.xlabel("Feature 1", fontsize=20)  # Increase font size for the x-axis label
    plt.ylabel("Feature 2", fontsize=20)  # Increase font size for the y-axis label
    plt.title("K-Means", fontsize=24)  # Increase font size for the title
    plt.legend()
    plt.show()
