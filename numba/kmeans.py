from math import sqrt
from config import *
import numba


'''
    分配聚类
'''
@numba.jit(nopython=True)
def groupByCluster(arrayP, arrayPcluster,
                   arrayC,
                   num_points, num_centroids):
    for i0 in range(num_points):
        # 使用负数初始化当前聚类的最短距离
        minor_distance = -1
        for i1 in range(num_centroids):
            # 计算当前聚类均值点与点对象的距离
            dx = arrayP[i0, 0] - arrayC[i1, 0]
            dy = arrayP[i0, 1] - arrayC[i1, 1]
            my_distance = sqrt(dx * dx + dy * dy)
            # 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[i0] = i1
    return arrayPcluster


'''
    计算聚类总值
'''
@numba.jit(nopython=True)
def calCentroidsSum(arrayP, arrayPcluster,
                    arrayCsum, arrayCnumpoint,
                    num_points, num_centroids):
    # 初始化聚类的总值信息
    for i in range(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

    # 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint


'''
    计算聚类均值
'''
@numba.jit(nopython=True)
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint,
                    num_centroids):
    for i in range(num_centroids):
        # 对已经计算好总值信息的聚类，计算其均值信息
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]


'''
    kmeans辅助代码
'''
def kmeans(arrayP, arrayPcluster,
           arrayC, arrayCsum, arrayCnumpoint,
           num_points, num_centroids):

    for i in range(ITERATIONS):
        groupByCluster(
            arrayP, arrayPcluster,
            arrayC,
            num_points, num_centroids
        )

        calCentroidsSum(
            arrayP, arrayPcluster,
            arrayCsum, arrayCnumpoint,
            num_points, num_centroids
        )

        updateCentroids(
            arrayC, arrayCsum, arrayCnumpoint,
            num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint
