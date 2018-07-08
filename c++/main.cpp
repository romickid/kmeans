#include <sys/types.h>
#include <math.h>
#include <errno.h>

#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <jansson.h>
#include <sys/time.h>

#include "point.h"
#include "kmeans.h"
#include "config.h"


/**
* 打印聚类信息
*/
void printCentroids(
    Centroid* centroids
) {
    for (int i = 0; i < NUMBER_OF_CENTROIDS; i++) {
        printf("[x=%lf, y=%lf, x_sum=%lf, y_sum=%lf, num_points=%i]\n", 
               centroids[i].x, centroids[i].y, centroids[i].x_sum,
               centroids[i].y_sum, centroids[i].num_points);
    }

    printf("--------------------------------------------------\n");
}


/**
* 实验代码，用于计时与重复实验的确定
*/
float runKmeans(
    Point* points, 
    Centroid* centroids
) {
    struct timeval time_before, time_after, time_result;

    // 开始计时
    gettimeofday(&time_before, NULL);

    for (int i = 0; i < REPEAT; i++) {
        // 使用点对象中的前k个点初始化聚类
        for (int ci = 0; ci < NUMBER_OF_CENTROIDS; ci++) {
            centroids[ci].x = points[ci].x;
            centroids[ci].y = points[ci].y;
        }

        kmeans(points, centroids, NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS);
        
        if (i + 1 == REPEAT) {
            printCentroids(centroids);
        }
    }

    // 结束计时
    gettimeofday(&time_after, NULL);
    timersub(&time_after, &time_before, &time_result);
    float repeat_time = (time_result.tv_sec*1000.0) + (time_result.tv_usec/1000.0);

    return repeat_time / REPEAT;
}


int main() {
    json_t *json;
    json_error_t error;
    json_t *value;
    size_t index;
    float total_time = 0;

    // 初始化内存，分配变量
    Point* points = (Point*) malloc(NUMBER_OF_POINTS * sizeof(Point));
    Centroid* centroids = (Centroid*) malloc(NUMBER_OF_CENTROIDS * sizeof(Centroid));

    // 从json文件中读取数据集
    json = json_load_file("../points.json", 0, &error);
    if (!json) {
        printf("Error parsing Json file");
        fflush(stdout);
        return -1;
    }
    else {
        json_array_foreach(json, index, value) {
            float x = json_number_value(json_array_get(value, 0));
            float y = json_number_value(json_array_get(value, 1));
            points[index].x = x;
            points[index].y = y;
        }
    }

    // 开始实验
    total_time = runKmeans(points, centroids);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Average Time: %f ms\n", total_time);
	    
	// 内存释放
    free(centroids);
    free(points);

	return 0;
}
