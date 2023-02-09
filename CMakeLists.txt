cmake_minimum_required(VERSION 3.2)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

project(WuKong CXX)

message(STATUS "${CMAKE_BUILD_TYPE} Build")

# set(CMAKE_BUILD_TYPE Release) 

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../libigl/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O0 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O0 -fopenmp -Wno-unused-variable -Wno-unused-but-set-variable -Wsign-compare -Wreorder")
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -Wpedantic -fsanitize=undefined,address")

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 ON)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         On)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
option(LIBIGL_WITH_XML               "Use XML"                      ON)

option(LIBIGL_WITH_MOSEK             "Use MOSEK"          ON)



find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (SuiteSparse REQUIRED)
find_package (OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package (LIBIGL REQUIRED)


find_package(CGAL REQUIRED COMPONENTS Core)

add_subdirectory(Deps)
if(CUDA_FOUND)
	add_definitions(-DENABLE_AMGCL_CUDA)
endif()

add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DUSE_CHOLMOD)
add_definitions(-DMKL_LP64)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${MKLROOT}/include)
# include_directories(/home/yueli/Documents/ETH/WuKong/build/amgcl-src/)
# add_library(${MKLROOT}/lib/intel64)
add_subdirectory(Libs)
add_subdirectory(Solver)
add_subdirectory(Projects)


message("**************************************************")
message("C++ Cmake Flags: ${CMAKE_CXX_FLAGS}")
message("**************************************************")
