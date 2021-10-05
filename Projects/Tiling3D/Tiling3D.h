#ifndef TILING3D_H
#define TILING3D_H


#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"

#include "tactile/tiling.hpp"

class Tiling3D
{
public:
    using T = double;
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using IV = Vector<int, 3>;

    using PointLoops = std::vector<TV2>;
    using IdList = std::vector<int>;
    using Face = Vector<int, 3>;
    

public:
    Tiling3D() {}
    ~Tiling3D() {}

    void fetchOneFamily(std::vector<PointLoops>& raw_points, T width, T height);
    void extrudeToMesh(const std::vector<PointLoops>& raw_points, 
        T width, T height, std::string filename);

    void test();
};


#endif