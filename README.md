# 简介 / Introduction

该项目实现了不同语言下的k-means算法。<br>
其特点在于实现了cuda及cuda-numba的k-means代码。

我们在以下环境中进行了实验：
* Intel Core i5-4200H
* NVIDIA Geforce GTX850M
* Ubuntu 16.04

对其代码的运行速度进行了比较，并得到了以下运行结果(仅供参考)：
* cuda (9.0): 5.6ms 
* cuda-numba (0.38.0): 13.2ms
* numba (0.38.0): 75ms
* openmp (2 threads): 119ms
* c++ (g++ 5.4.0): 200ms
* python (3.6.5): 10000ms

该项目代码使用了Jansson依赖包用于处理JSON文件，并使用了CMake作为编译工具。

该项目参考了 https://github.com/andreaferretti/kmeans ，由 Romic Huang 在2018年春季完成。
***
This project implements k-means algorithm in different languages.<br>
Specially, it realize k-means' codes of CUDA and cuda-numba.

We conducted experiment in the following environments:
* Intel Core i5-4200H
* NVIDIA Geforce GTX850M
* Ubuntu 16.04

We compares the speed of the codes and obtains the following results (for reference only):
* cuda (9.0): 5.6ms 
* cuda-numba (0.38.0): 13.2ms
* numba (0.38.0): 75ms
* openmp (2 threads): 119ms
* c++ (g++ 5.4.0): 200ms
* python (3.6.5): 10000ms

This project uses Jansson to handle JSON file and uses CMake as a compilation tool.

This project reference https://github.com/andreaferretti/kmeans, and was completed by Romic Huang in the spring of 2018.<br><br><br>