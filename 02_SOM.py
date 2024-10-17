import numpy as np
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
np.random.seed(22)

class CyrusSOM(object):
    def __init__(self, net=[[1, 1], [1, 1]], epochs=50, r_t=[None, None], eps=1e-6):
        """
        :param net: 竞争层的拓扑结构，支持一维及二维，1表示该输出节点存在，0表示不存在该输出节点
        :param epochs: 最大迭代次数
        :param r_t:   [C,B]    领域半径参数，r = C*e**(-B*t/eoochs),其中t表示当前迭代次数
        :param eps: learning rate的阈值
        """

        self.epochs = epochs
        self.C = r_t[0]
        self.B = r_t[1]
        self.eps = eps
        self.output_net = np.array(net)
        if len(self.output_net.shape) == 1:
            self.output_net = self.output_net.reshape([-1, 1])
        self.coord = np.zeros([self.output_net.shape[0], self.output_net.shape[1], 2])
        for i in range(self.output_net.shape[0]):
            for j in range(self.output_net.shape[1]):
                self.coord[i, j] = [i, j]

    def __r_t(self, t):
        if not self.C:
            return 0.5
        else:
            return self.C * np.exp(-self.B * t / self.epochs)

    def __lr(self, t, distance):
        return (self.epochs - t) / self.epochs * np.exp(-distance)

    def standard_x(self, x):
        x = np.array(x)
        for i in range(x.shape[0]):
            x[i, :] = [value / (((x[i, :]) ** 2).sum() ** 0.5) for value in x[i, :]]
        return x

    def standard_w(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i, j, :] = [value / (((w[i, j, :]) ** 2).sum() ** 0.5) for value in w[i, j, :]]
        return w

    def cal_similar(self, x, w):
        similar = (x * w).sum(axis=2)
        coord = np.where(similar == similar.max())
        return [coord[0][0], coord[1][0]]

    def update_w(self, center_coord, x, step):
        for i in range(self.coord.shape[0]):
            for j in range(self.coord.shape[1]):
                distance = (((center_coord - self.coord[i, j]) ** 2).sum()) ** 0.5
                if distance <= self.__r_t(step):
                    self.W[i, j] = self.W[i, j] + self.__lr(step, distance) * (x - self.W[i, j])

    def transform_fit(self, x):
        self.train_x = self.standard_x(x)
        self.W = np.zeros([self.output_net.shape[0], self.output_net.shape[1], self.train_x.shape[1]])
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j, :] = self.train_x[random.choice(range(self.train_x.shape[0])), :]
        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step, 0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                center_coord = self.cal_similar(self.train_x[index, :], self.W)
                self.update_w(center_coord, self.train_x[index, :], step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[0])
        return label

def load_dataset(filename):
    """
    Load dataset from a text file.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(list(map(float, line.strip().split())))
    return np.array(data)

if __name__ == '__main__':
    # Load dataset from the text file
    data = load_dataset('testSet3.txt')

    # Initialize SOM
    SOM = CyrusSOM(epochs=5)

    # Train SOM on the loaded data
    x = data
    y_pre = SOM.transform_fit(x)
    # 计算 DB 指数
    db_index = davies_bouldin_score(x, y_pre)
    print(f"Davies-Bouldin Index: {db_index}")
    # print(y_pre)
    # Adjust plot settings
    #plt.rcParams.update({'font.size': 18})  # Set global font size for the plot

    # Plot the result with larger points
    colors = "rgbycmk"
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=[colors[i % len(colors)] for i in y_pre], s=30)  # Increase 's' for bigger points
    plt.xlabel("Feature 1", fontsize=20)  # Increase font size for the x-axis label
    plt.ylabel("Feature 2", fontsize=20)  # Increase font size for the y-axis label
    plt.title("SOM Clustering", fontsize=24)  # Increase font size for the title
    plt.show()
