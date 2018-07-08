#include <stdlib.h>
#include <stdio.h>

#include "point.h"


/**
* 计算点对象与聚类均值之间的距离的设备函数
*/
__device__ float distance(
    Point* p, 
    Centroid* c
) {
    float dx = p->x - c->x;
    float dy = p->y - c->y;
    return sqrtf(dx*dx + dy*dy);
}
