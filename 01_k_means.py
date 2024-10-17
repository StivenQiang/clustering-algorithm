from numpy import *
from time import sleep
import matplotlib
from matplotlib import pyplot as plt


def loadDataSet(fileName):
    '''
    加载数据集
    '''
    # 初始化一个空列表
    dataSet = []
    # 读取文件
    fr = open(fileName)
    # 循环遍历文件所有行
    for line in fr.readlines():
        # 切割每一行的数据
        curLine = line.strip().split('\t')
        # 将数据转换为浮点类型,便于后面的计算
        # fltLine = [float(x) for x in curLine]
        # 将数据追加到dataMat
        fltLine = list(map(float, curLine))  # 映射所有的元素为 float（浮点数）类型
        dataSet.append(fltLine)
    # 返回dataMat
    return dataSet


def distEclud(vecA, vecB):
    '''
    欧氏距离计算函数
    :param vecA:
    :param vecB:
    :return:
    '''
    return sqrt(sum(power(vecA - vecB, 2)))


def randCent(dataMat, k):
    '''
    为给定数据集构建一个包含K个随机质心的集合,
    随机质心必须要在整个数据集的边界之内,这可以通过找到数据集每一维的最小和最大值来完成
    然后生成0到1.0之间的随机数并通过取值范围和最小值,以便确保随机点在数据的边界之内
    :param dataMat:
    :param k:
    :return:
    '''
    # 获取样本数与特征值
    m, n = shape(dataMat)
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = mat(zeros((k, n)))
    # 循环遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataMat[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        # 计算每一列的质心,并将值赋给centroids
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    # 返回质心
    return centroids


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    '''
    创建K个质心,然后将每个店分配到最近的质心,再重新计算质心。
    这个过程重复数次,直到数据点的簇分配结果不再改变为止
    :param dataMat: 数据集
    :param k: 簇的数目
    :param distMeans: 计算距离
    :param createCent: 创建初始质心
    :return:
    '''
    # 获取样本数和特征数
    m, n = shape(dataMat)
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = mat(zeros((m, 2)))
    # 创建质心,随机K个质心
    centroids = createCent(dataMat, k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有数据找到距离每个点最近的质心,
        # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distJI = distMeas(centroids[j, :], dataMat[i, :])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)的平方
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        # 遍历所有质心并更新它们的取值
        for cent in range(k):
            # 通过数据过滤来获得给定簇的所有点
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
    # 返回所有的类质心与点分配结果
    return centroids, clusterAssment


def plotClusters(dataMat, k, centroids, clusterAssment):
    """
    可视化聚类结果
    :param dataMat: 数据集
    :param k: 聚类的个数
    :param centroids: 质心
    :param clusterAssment: 每个点的簇分配结果
    """
    # 颜色列表，用于绘制不同簇的颜色
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # 创建散点图窗口
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 遍历每个簇
    for cent in range(k):
        # 获取当前簇的数据点
        ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
        # 绘制当前簇的数据点
        ax.scatter(ptsInClust[:, 0].flatten().A[0], ptsInClust[:, 1].flatten().A[0],
                   s=30, c=colors[cent % len(colors)], marker='o')

    # 绘制质心
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0],
               s=100, c='black', marker='*', label='Centroids')
    plt.xlabel("Feature 1", fontsize=20)  # Increase font size for the x-axis label
    plt.ylabel("Feature 2", fontsize=20)  # Increase font size for the y-axis label
    plt.title("k-means", fontsize=24)  # Increase font size for the title
    # 显示图例
    plt.legend()
    plt.show()

def testBasicFunc():
    # 加载测试数据集
    dataMat = mat(loadDataSet('testSet.txt'))

    # 测试 randCent() 函数是否正常运行。
    # 首先，先看一下矩阵中的最大值与最小值
    print('min(dataMat[:, 0])=', min(dataMat[:, 0]))
    print('min(dataMat[:, 1])=', min(dataMat[:, 1]))
    print('max(dataMat[:, 1])=', max(dataMat[:, 1]))
    print('max(dataMat[:, 0])=', max(dataMat[:, 0]))

    # 然后看看 randCent() 函数能否生成 min 到 max 之间的值
    print('randCent(dataMat, 2)=', randCent(dataMat, 2))

    # 最后测试一下距离计算方法
    print(' distEclud(dataMat[0], dataMat[1])=', distEclud(dataMat[0], dataMat[1]))


def testKMeans():
    # 加载测试数据集
    dataMat = mat(loadDataSet('testSet.txt'))
    print(dataMat)
    print(type(dataMat))
    # 运行 K-means 聚类算法
    myCentroids, clustAssing = kMeans(dataMat, 4)

    # 打印质心
    print('centroids=', myCentroids)

    # 绘制聚类结果
    plotClusters(dataMat, 4, myCentroids, clustAssing)


if __name__ == "__main__":
    # # 测试基础的函数
    # testBasicFunc()

    # 测试 kMeans 函数
    testKMeans()