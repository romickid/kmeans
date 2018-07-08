from math import sqrt
from time import time
import json


# 点对象的数量
NUMBER_OF_POINTS = 100000

# 聚类数量
NUMBER_OF_CENTROIDS = 10

# 实验重复次数
REPEAT = 10

# k-means算法迭代次数
ITERATIONS = 15


'''
    点对象类
'''
class Point(object):
    x = None
    y = None
    cluster = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cluster = -1


'''
    聚类对象类
'''
class Centroids(object):
    x = None
    y = None
    x_sum = None
    y_sum = None
    num_points = None

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_sum = 0
        self.y_sum = 0
        self.num_points = 0


'''
    使用点对象中的前k个点生成聚类
'''
def getListCentroids(listPoints):
    listCentroids = list()
    for d in listPoints:
        listCentroids.append(Centroids(d.x, d.y))

    return listCentroids


'''
    计算点对象与聚类均值之间的距离
'''
def distance(point, centroid):
    dx = point.x - centroid.x
    dy = point.y - centroid.y
    return sqrt(dx * dx + dy * dy)


'''
    打印聚类信息
'''
def printCentroid(listCentroids):
    for d in listCentroids:
        print("[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
            d.x, d.y, d.x_sum, d.y_sum, d.num_points)
        )

    print('--------------------------------------------------\n')


'''
    分配聚类
'''
def groupByCluster(listPoints, listCentroids):
    for i0, _ in enumerate(listPoints):
        # 使用负数初始化当前聚类的最短距离
        minor_distance = -1

        for i1, centroid in enumerate(listCentroids):
            # 计算当前聚类均值点与点对象的距离
            my_distance = distance(listPoints[i0], centroid)
            # 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                listPoints[i0].cluster = i1
    return listPoints


'''
    计算聚类总值
'''
def calCentroidsSum(listPoints, listCentroids):
    # 初始化聚类的总值信息
    for i in range(NUMBER_OF_CENTROIDS):
        listCentroids[i].x_sum = 0
        listCentroids[i].y_sum = 0
        listCentroids[i].num_points = 0

    # 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    for point in listPoints:
        i = point.cluster
        listCentroids[i].x_sum += point.x
        listCentroids[i].y_sum += point.y
        listCentroids[i].num_points += 1

    return listCentroids


'''
    计算聚类均值
'''
def updateCentroids(listCentroids):
    # 对已经计算好总值信息的聚类，计算其均值信息
    for i, centroid in enumerate(listCentroids):
        listCentroids[i].x = centroid.x_sum / centroid.num_points
        listCentroids[i].y = centroid.y_sum / centroid.num_points
    return listCentroids


'''
    kmeans辅助代码
'''
def kmeans(listPoints, listCentroids):
    for i in range(ITERATIONS):
        listPoints = groupByCluster(listPoints, listCentroids)
        listCentroids = calCentroidsSum(listPoints, listCentroids)
        listCentroids = updateCentroids(listCentroids)

    return listCentroids


'''
    实验代码，用于计时与重复实验
'''
def runKmeans(listPoints):
    # 开始计时
    start = time()

    listCentroids = None
    for i in range(REPEAT):
        listCentroids = getListCentroids(listPoints[:NUMBER_OF_CENTROIDS])
        listCentroids = kmeans(listPoints, listCentroids)
        if i+1 == REPEAT:
            printCentroid(listCentroids)

    # 结束计时
    end = time()
    total = (end - start) * 1000 / REPEAT

    print("Iterations: {:d}".format(ITERATIONS))
    print("Average Time: {:.4f} ms".format(total))



if __name__ == "__main__":
    # 从json文件中读取数据集
    with open("../points.json") as f:
        listPoints = list(map(lambda x: Point(x[0], x[1]), json.loads(f.read())))

    runKmeans(listPoints)
