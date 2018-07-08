# CMakeLists.txt for G4CU project                                                                                                                                 

project(kmeans)

# required cmake version

cmake_minimum_required(VERSION 2.8)

# Execute
add_executable(kmeans.out point.cpp kmeans.cpp main.cpp)
target_link_libraries(kmeans.out jansson)