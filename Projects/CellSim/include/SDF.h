#ifndef SDF_H
#define SDF_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

// #include <openvdb/openvdb.h>

#include "VecMatDef.h"

using T = double;

class SDF
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    int n_points = 0;

    VectorXT data_points;
    VectorXT data_point_normals;
    T radius;
    T search_radius;

public:
    void initialize(const VectorXT& _data_points, 
        const VectorXT& _data_point_normals, T _radius, T _search_radius);
    void valueIMLS(const TV& test_point, T& value);
    void gradientIMLS(const TV& test_point, TV& dphidx);
    void hessianIMLS(const TV& test_point, TM& d2phidx2);

private:
    T weightFunction(T d, T r) { return std::exp(-d*d, r*r); }

public:
    SDF() 
    {
        // openvdb::initialize();
    }
    ~SDF() {}
};

#endif 