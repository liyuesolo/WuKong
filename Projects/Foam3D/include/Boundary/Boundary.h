#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include <iostream>

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;

struct BoundaryVertex {
    TV3 pos;
    MatrixXT grad;
    MatrixXT hess[3];
};
struct BoundaryFace {
    IV3 vertices;
};

class Boundary {

public:
    int np;
    int nfree;

    VectorXT p;
    VectorXi free_idx;
    VectorXi free_map;

    std::vector<BoundaryVertex> v;
    std::vector<BoundaryFace> f;

private:
    virtual void computeVertices() = 0;

public:
    virtual bool checkValid() { return true; };

    virtual double computeEnergy() { return 0; };

    virtual VectorXT computeEnergyGradient() { return VectorXT::Zero(nfree); };

    virtual MatrixXT computeEnergyHessian() { return MatrixXT::Zero(nfree, nfree); };

protected:
    void initialize(int nv, const MatrixXi &f_);

    inline void setGradientEntry(int iv, int iCoord, int ip, double value) {
        if (free_map(ip) >= 0) {
            v[iv].grad(iCoord, free_map(ip)) = value;
        }
    }

    inline void setHessianEntry(int iv, int iCoord, int ip0, int ip1, double value) {
        if (free_map(ip0) >= 0 && free_map(ip1) >= 0) {
            v[iv].hess[iCoord](free_map(ip0), free_map(ip1)) = value;
        }
    }

    inline void setEnergyGradientEntry(VectorXT &gradient, int ip, double value) {
        if (free_map(ip) >= 0) {
            gradient(ip) = value;
        }
    }

    inline void setEnergyHessianEntry(MatrixXT &hessian, int ip0, int ip1, double value) {
        if (free_map(ip0) >= 0 && free_map(ip1) >= 0) {
            hessian(ip0, ip1) = value;
        }
    }

public:
    Boundary(const VectorXT &p_, const VectorXi &free_);

    void compute(const VectorXT &p_free);

    VectorXT get_p_free();
};
