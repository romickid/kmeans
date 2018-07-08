#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

#include "point.h"

__global__ void groupByCluster(Point* points, Centroid* centroids, int num_centroids);

__global__ void sumPointsCluster(Point* points, Centroid* centroids, int num_centroids);

__global__ void updateCentroids(Centroid* centroids);

void kmeans(Point* h_points, Centroid* h_centroids, int num_points, int num_centroids);


#endif
