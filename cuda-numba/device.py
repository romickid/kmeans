from math import sqrt
from numba import cuda


'''
    计算点对象与聚类均值之间的距离的设备函数
'''
@cuda.jit('float32(float32, float32, float32, float32)', device='gpu')
def distance(px, py, cx, cy):
    dx = px - cx
    dy = py - cy
    return sqrt(dx * dx + dy * dy)
