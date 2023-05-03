#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"

using T = double;

class SpatialHash
{
public:
    using TV = Vector<double, 3>;
    using IV = Vector<int, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    struct VectorHash
    {  
        size_t operator()(const IV& a) const{
            std::size_t h = 0;
            for (int d = 0; d < 3; ++d) {
                h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
            }
            return h;
        }
    };

    T h = 0;

    int dim = 3;
    std::unordered_map<IV, std::vector<int>, VectorHash> hash_table;

public:
    void build(T _h, const VectorXT& points);
    void getOneRingNeighbors(const TV& point, std::vector<int>& neighbors);

    void test();
public:
    SpatialHash() {}
    ~SpatialHash() {}
};

#endif