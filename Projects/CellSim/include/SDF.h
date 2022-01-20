#ifndef SDF_H
#define SDF_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridOperators.h>

#include "VecMatDef.h"

using T = double;

using namespace openvdb;
class SDF
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    
public:
    
    virtual T value(const TV& test_point) = 0;
    virtual void gradient(const TV& test_point, TV& dphidx) = 0;
    virtual void hessian(const TV& test_point, TM& d2phidx2) = 0;

    bool inside(const TV& test_point) { return value(test_point) <= 0.0; }

    virtual void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon) = 0;
public:
    SDF() 
    {
        
    }
    ~SDF() {}
};

class IMLS : public SDF
{

public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;

    int n_points = 0;

    VectorXT data_points;
    VectorXT data_point_normals;
    T radius;
    T search_radius;

private:
    T weightFunction(T d, T r) { return std::exp(-d*d / r*r); }

public:
    T value(const TV& test_point);
    void gradient(const TV& test_point, TV& dphidx);
    void hessian(const TV& test_point, TM& d2phidx2);


    void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon);
public:
    IMLS() {}
    ~IMLS() {}
};

class VdbLevelSetSDF : public SDF
{

public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;

    openvdb::DoubleGrid::Ptr grid;
    openvdb::Vec3dGrid::Ptr grid_grad;
    openvdb::Vec3dGrid::Ptr ddx;
    openvdb::Vec3dGrid::Ptr ddy;
    openvdb::Vec3dGrid::Ptr ddz;

public:
    T value(const TV& test_point);
    void gradient(const TV& test_point, TV& dphidx);
    void hessian(const TV& test_point, TM& d2phidx2);

    void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon);

private:
    void levelsetFromMesh(const std::vector<Vec3s>& points,
        const std::vector<Vec3I>& triangles,
        const std::vector<Vec4I>& quads);
public:
    VdbLevelSetSDF() 
    {
        openvdb::initialize();
    }
    ~VdbLevelSetSDF() {}
};

#endif 