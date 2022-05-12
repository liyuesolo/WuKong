#ifndef IMLS_H
#define IMLS_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

#include "SpatialHash.h"
template <int dim>
class SpatialHash;

template <int dim>
class IMLS
{
public:
    using TV = Vector<double, dim>;
    using TM = Matrix<double, dim, dim>;
    using IV = Vector<int, dim>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;

    int n_points = 0;

    VectorXT data_points;
    VectorXT data_point_normals;
    VectorXT radii;
    T search_radius;

    SpatialHash<dim> hash;

private:
    T weightFunction(T d, T r) { return std::exp(-d*d / r/r); }

    void thetaValue3D(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, double& energy);
    void thetaValue3DGradient(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 1>& energygradient);
    void thetaValue3DHessian(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 3>& energyhessian);

    void thetaValue2D(const Eigen::Matrix<double,2,1> & x, const Eigen::Matrix<double,2,1> & pi, double ri, double& energy);
    void thetaValue2DGradient(const Eigen::Matrix<double,2,1> & x, const Eigen::Matrix<double,2,1> & pi, double ri, Eigen::Matrix<double, 2, 1>& energygradient);
    void thetaValue2DHessian(const Eigen::Matrix<double,2,1> & x, const Eigen::Matrix<double,2,1> & pi, double ri, Eigen::Matrix<double, 2, 2>& energyhessian);
public:
    T value(const TV& test_point);
    void gradient(const TV& test_point, TV& dphidx);
    void hessian(const TV& test_point, TM& d2phidx2);

    IMLS() {}
    ~IMLS() {}
};

#endif