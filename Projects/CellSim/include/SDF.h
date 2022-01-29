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

#include "SpatialHash.h"

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

    TV min_corner, max_corner;
    
    virtual T value(const TV& test_point) = 0;
    virtual void gradient(const TV& test_point, TV& dphidx) = 0;
    virtual void hessian(const TV& test_point, TM& d2phidx2) = 0;

    bool inside(const TV& test_point) { return value(test_point) <= 0.0; }
    virtual std::string getName() const = 0;
    virtual void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon) = 0;

    virtual void sampleZeroLevelset(VectorXT& points) const = 0; 

    
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
    VectorXT radii;
    T search_radius;

    SpatialHash hash;

private:
    T weightFunction(T d, T r) { return std::exp(-d*d / r/r); }

    void thetaValue(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, double& energy);
    void thetaValueGradient(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 1>& energygradient);
    void thetaValueHessian(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 3>& energyhessian);

    void computeBBox();
public:
    T value(const TV& test_point);
    void gradient(const TV& test_point, TV& dphidx);
    void hessian(const TV& test_point, TM& d2phidx2);

    std::string getName() const { return "IMLS"; }

    void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon);
    
    void sampleZeroLevelset(VectorXT& points) const { points = data_points; }
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
    
    std::string getName() const { return "VDBLevelset"; }

    void sampleZeroLevelset(VectorXT& points) const {  }

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