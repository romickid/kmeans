from device import *
from config import *
from math import ceil
from numba import int32, float32


'''
    分配聚类的核函数
'''
@cuda.jit('void(float32[:,:], int32[:], '
          'float32[:,:], '
          'int32, int32)',
          target='gpu')
def groupByCluster(arrayP, arrayPcluster,
                   arrayC,
                   num_points, num_centroids):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # 序号不能超过点对象的数量大小
    if idx < num_points:
        # 使用负数初始化当前聚类的最短距离
        minor_distance = -1

        for i in range(num_centroids):
            # 计算当前聚类均值点与点对象的距离
            my_distance = distance(arrayP[idx, 0], arrayP[idx, 1], arrayC[i, 0], arrayC[i, 1])
            # 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
            if minor_distance > my_distance or minor_distance == -1:
                minor_distance = my_distance
                arrayPcluster[idx] = i


'''
    计算聚类总值的核函数
'''
@cuda.jit('void(float32[:,:], int32[:], '
          'float32[:,:], int32[:], '
          'int32, int32)',
          target='gpu')
def calCentroidsSum(arrayP, arrayPcluster,
                    arrayCsum, arrayCnumpoint,
                    num_points, num_centroids):
    # 为线程块定义共享的聚类变量
    s_arrayCsum = cuda.shared.array(shape=(10, 2), dtype=float32)
    s_arrayCnumpoint = cuda.shared.array(shape=10, dtype=int32)

    tdx = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # 初始化聚类的总值信息
    if idx < num_centroids:
        arrayCsum[idx, 0] = 0.0
        arrayCsum[idx, 1] = 0.0
        arrayCnumpoint[idx] = 0

    # 初始化共享的聚类变量的总值信息
    if tdx < num_centroids:
        s_arrayCsum[tdx, 0] = 0.0
        s_arrayCsum[tdx, 1] = 0.0
        s_arrayCnumpoint[tdx] = 0.0

    cuda.syncthreads()

    # 线程块内的每个线程各自完成
    # 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    if idx < num_points:
        i = arrayPcluster[idx]
        cuda.atomic.add(s_arrayCsum[i], 0, arrayP[idx, 0]);
        cuda.atomic.add(s_arrayCsum[i], 1, arrayP[idx, 1]);
        cuda.atomic.add(s_arrayCnumpoint, i, 1);

    cuda.syncthreads()

    # 线程块间数据的汇总
    # 将各个线程块计算的聚类总值信息累加至函数调用时参数的变量
    if tdx < num_centroids:
        cuda.atomic.add(arrayCsum[tdx], 0, s_arrayCsum[tdx, 0]);
        cuda.atomic.add(arrayCsum[tdx], 1, s_arrayCsum[tdx, 1]);
        cuda.atomic.add(arrayCnumpoint, tdx, s_arrayCnumpoint[tdx]);


'''
    计算聚类均值的核函数
'''
@cuda.jit('void(float32[:,:], float32[:,:], int32[:], '
          'int32)',
          target='gpu')
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint,
                    num_centroids):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx < num_centroids:
        # 对已经计算好总值信息的聚类，计算其均值信息
        if arrayCnumpoint[idx] > 0:
            arrayC[idx, 0] = arrayCsum[idx, 0] / arrayCnumpoint[idx]
            arrayC[idx, 1] = arrayCsum[idx, 1] / arrayCnumpoint[idx]


'''
    kmeans辅助代码
'''
def kmeans(arrayP, arrayPcluster,
           arrayC, arrayCsum, arrayCnumpoint,
           num_points, num_centroids):

    # 分配内存,并将数据从宿主传递至设备
    darrayP = cuda.to_device(arrayP)
    darrayPcluster = cuda.to_device(arrayPcluster)
    darrayC = cuda.to_device(arrayC)
    darrayCsum = cuda.to_device(arrayCsum)
    darrayCnumpoint = cuda.to_device(arrayCnumpoint)

    # 根据迭代次数运行算法
    for i in range(ITERATIONS):
        # 调用分配聚类函数
        groupByCluster[ceil(NUMBER_OF_POINTS/25), 25](
            darrayP, darrayPcluster,
            darrayC,
            num_points, num_centroids
        )
        cuda.synchronize()

        # 调用计算聚类总值函数
        calCentroidsSum[ceil(NUMBER_OF_POINTS/25), 25](
            darrayP, darrayPcluster,
            darrayCsum, darrayCnumpoint,
            NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS
        )
        cuda.synchronize()

        # 调用计算聚类均值函数
        updateCentroids[ceil(NUMBER_OF_CENTROIDS/10), 10](
            darrayC, darrayCsum, darrayCnumpoint,
            NUMBER_OF_CENTROIDS
        )
        cuda.synchronize()

    # 将数据从设备传递至宿主
    arrayC = darrayC.copy_to_host()
    arrayCsum = darrayCsum.copy_to_host()
    arrayCnumpoint = darrayCnumpoint.copy_to_host()

    return arrayC, arrayCsum, arrayCnumpoint
