#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED


/**
* 点对象类
*/
typedef struct {
    float x;
    float y;
    int cluster;
} Point;


/**
* 聚类对象类
*/
typedef struct {
    float x;
    float y;
    float x_sum;
    float y_sum;
    int num_points;
} Centroid;


float distance(Point* p, Centroid* c);

#endif