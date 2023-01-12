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

class Boundary {

public:
    int np;
    int nfree;

    VectorXT p;
    VectorXi free_idx;
    VectorXi free_map;

    VectorXT v;
    MatrixXT dvdp;
    std::vector<MatrixXT> d2vdp2;

private:

    virtual void computeVertices() = 0;

    virtual void computeGradient() = 0;

    virtual void computeHessian() = 0;

protected:
    inline void setGradientEntry(int iv, int ip, double value) {
        if (free_map(ip) > 0) {
            dvdp(iv, free_map(ip)) = value;
        }
    };

    inline void setHessianEntry(int iv, int ip0, int ip1, double value) {
        if (free_map(ip0) > 0 && free_map(ip1) > 0) {
            d2vdp2[iv](ip0, ip1) = value;
        }
    };

public:
    Boundary(const VectorXT &p_, const VectorXi &free_) {
        p = p_;
        free_idx = free_;

        np = p.rows();
        nfree = free_idx.rows();

        free_map = -1 * VectorXi::Ones(np);
        for (int i = 0; i < nfree; i++) {
            free_map(free_idx(i)) = i;
        }
    }

    void compute(const VectorXT &p_free) {
        for (int i = 0; i < nfree; i++) {
            p(free_idx(i)) = p_free(i);
        }

        computeVertices();
        computeGradient();
        computeHessian();
    };
};
