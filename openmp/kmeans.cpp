#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

#include "kmeans.h"
#include "point.h"
#include "config.h"


/**
* 分配聚类
*/
void groupByCluster(
    Point* points,
    Centroid* centroids,
    int num_centroids, 
    int num_points
) {
#pragma omp parallel for
	for(int i0 = 0; i0 < num_points; i0++) {
        // 使用负数初始化当前聚类的最短距离
		float minor_distance = -1.0;

		for (int i1 = 0; i1 < num_centroids; i1++) {
            // 计算当前聚类均值点与点对象的距离
			float my_distance = distance(&points[i0], &centroids[i1]);
            // 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
			if (minor_distance > my_distance || minor_distance == -1.0) {
				minor_distance = my_distance;
				points[i0].cluster = i1;
			}
		}
	}
}


/**
* 计算聚类总值
*/
void calCentroidsSum(
    Point* points, 
    Centroid* centroids,
    int num_centroids, 
    int num_points
) {
    // 初始化聚类的总值信息
#pragma omp parallel for
    for(int i = 0; i < num_centroids; i++) {
        centroids[i].x_sum = 0.0;
        centroids[i].y_sum = 0.0;
        centroids[i].num_points = 0.0;
    }

    // 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    for(int i = 0; i < num_points; i++) {
        int ci = points[i].cluster;
        centroids[ci].x_sum += points[i].x;
        centroids[ci].y_sum += points[i].y;
        centroids[ci].num_points += 1;
    }
}


/**
* 计算聚类均值
*/
void updateCentroids(
    Centroid* centroids, 
    int num_centroids
) {
	for(int i = 0; i < num_centroids; i++) {
        // 对已经计算好总值信息的聚类，计算其均值信息
	    if (centroids[i].num_points > 0) {
	        centroids[i].x = centroids[i].x_sum / centroids[i].num_points;
	        centroids[i].y = centroids[i].y_sum / centroids[i].num_points;
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
    for(int i = 0; i < ITERATIONS; i++) {
        // 调用分配聚类函数
        groupByCluster(
            h_points, 
            h_centroids,
            num_centroids, 
            num_points
        );
        
        // 调用计算聚类总值函数
        calCentroidsSum(
            h_points, 
            h_centroids,
            num_centroids, 
            num_points
        );

        // 调用计算聚类均值函数
        updateCentroids(
            h_centroids, 
            num_centroids
        );
    }
}
