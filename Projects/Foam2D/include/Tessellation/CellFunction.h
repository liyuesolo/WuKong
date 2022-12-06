#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;


class CellFunction {
public:
    virtual void addValue(const VectorXT &points, double &value) const = 0;

    virtual void addGradient(const VectorXT &points, VectorXT &gradient) const = 0;

    virtual void addHessian(const VectorXT &points, MatrixXT &hessian) const = 0;
};
