#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "point.h"


/**
* 计算点对象与聚类均值之间的距离
*/
float distance(
    Point* p, 
    Centroid* c
) {
    float dx = p->x - c->x;
    float dy = p->y - c->y;
    return sqrtf(dx*dx + dy*dy);
}
