#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "kmeans.h"
#include "point.h"
#include "config.h"


/**
* 分配聚类的核函数
*/
__global__ void groupByCluster(
    Point* points,
    Centroid* centroids,
    int num_centroids, 
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 序号不能超过点对象的数量大小
	if (idx < num_points) {
        // 使用负数初始化当前聚类的最短距离
        float minor_distance = -1.0;

	    for (int i = 0; i < num_centroids; i++) {
            // 计算当前聚类均值点与点对象的距离
            float my_distance = distance(&points[idx], &centroids[i]);
            // 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
            if (minor_distance > my_distance || minor_distance == -1.0) {
	            minor_distance = my_distance;
	            points[idx].cluster = i;
	        }
	    }
	}
}


/**
* 计算聚类总值的核函数
*/
__global__ void calCentroidsSum(
    Point* points, 
    Centroid* centroids,
    int num_centroids, 
    int num_points
) {
    // 为线程块定义共享的聚类变量
    extern __shared__ Centroid s_centroids[];
    
    int tdx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化聚类的总值信息
    if (idx < num_centroids) {
        centroids[idx].x_sum = 0.0;
        centroids[idx].y_sum = 0.0;
        centroids[idx].num_points = 0.0;
    }

    // 初始化共享的聚类变量的总值信息
    if (tdx < num_centroids) {
        s_centroids[tdx].x_sum = 0.0;
        s_centroids[tdx].y_sum = 0.0;
        s_centroids[tdx].num_points = 0.0;
    }

    __syncthreads();
    
    // 线程块内的每个线程各自完成
    // 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    if (idx < num_points) {
        int i = points[idx].cluster;
        atomicAdd(&s_centroids[i].x_sum, points[idx].x);
        atomicAdd(&s_centroids[i].y_sum, points[idx].y);
        atomicAdd(&s_centroids[i].num_points, 1);
	}

    __syncthreads();
    
    // 线程块间数据的汇总
    // 将各个线程块计算的聚类总值信息累加至函数调用时参数的变量
    if (tdx < num_centroids) {
        atomicAdd(&centroids[tdx].x_sum, s_centroids[tdx].x_sum);
        atomicAdd(&centroids[tdx].y_sum, s_centroids[tdx].y_sum);
        atomicAdd(&centroids[tdx].num_points, s_centroids[tdx].num_points);
    }
}


/**
* 计算聚类均值的核函数
*/
__global__ void updateCentroids(
    Centroid* centroids, 
    int num_centroids
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_centroids) {
        // 对已经计算好总值信息的聚类，计算其均值信息
	    if (centroids[idx].num_points > 0) {
	        centroids[idx].x = centroids[idx].x_sum / centroids[idx].num_points;
	        centroids[idx].y = centroids[idx].y_sum / centroids[idx].num_points;
	    }
	}
}


/**
* kmeans辅助代码
*/
void kmeans(
    Point* h_points,
    Centroid* h_centroids, 
    int num_points,
    int num_centroids
) {
    Point* d_points;
    Centroid* d_centroids;

    // 分配内存
    cudaMalloc((void **) &d_points, sizeof(Point) * num_points);
    cudaMalloc((void **) &d_centroids, sizeof(Centroid) * num_centroids);

    // 将数据从宿主传递至设备
    cudaMemcpy(d_points, h_points, sizeof(Point) * num_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(Centroid) * num_centroids, cudaMemcpyHostToDevice);   

    // 根据迭代次数运行算法
    for(int i = 0; i < ITERATIONS; i++) {
        // 调用分配聚类函数
        groupByCluster<<<ceil(num_points/100), 100>>>(
            d_points, 
            d_centroids,
            num_centroids, 
            num_points
        );
        cudaDeviceSynchronize();
        
        // 调用计算聚类总值函数
        calCentroidsSum<<<ceil(num_points/100), 100, num_centroids*sizeof(Centroid)>>>(
            d_points,
            d_centroids,
            num_centroids, 
            num_points
        );
        cudaDeviceSynchronize();

        // 调用计算聚类均值函数
        updateCentroids<<<ceil(num_centroids/10), 10>>>(
            d_centroids, 
            num_centroids
        );
        cudaDeviceSynchronize();
    }

    // 将数据从设备传递至宿主
    cudaMemcpy(h_centroids, d_centroids, sizeof(Centroid) * num_centroids, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_points);
    cudaFree(d_centroids);
}
