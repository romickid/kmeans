# CMakeLists.txt for G4CU project                                                                                                                                 

project(kmeans)

# required cmake version

cmake_minimum_required(VERSION 2.8)

# Execute
add_compile_options(-fopenmp)
add_executable(kmeans.out point.cpp kmeans.cpp main.cpp)
target_link_libraries(kmeans.out jansson)


FIND_PACKAGE(OpenMP REQUIRED)  
if(OPENMP_FOUND)  
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()