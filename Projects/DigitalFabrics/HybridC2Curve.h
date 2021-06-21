#ifndef HYBRID_C2_CURVE_H
#define HYBRID_C2_CURVE_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include "VecMatDef.h"


template<class T, int dim>
class HybridC2Curve
{
public:
    using TV = Vector<T, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    

    std::vector<TV> data_points;    

public:
    HybridC2Curve() {}
    ~HybridC2Curve() {}

    void getLinearSegments(std::vector<TV>& points)
    {
        points = data_points;
    }

    void circularInterpolation()
    {

    }
};



#endif