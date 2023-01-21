#include "../../include/Boundary/SimpleBoundary.h"

void SimpleBoundary::computeVertices() {
    v = p;

    int n_vtx = v.rows() / 2;
    next.resize(n_vtx);
    if (n_vtx > 0) {
        next << Eigen::VectorXi::LinSpaced(n_vtx - 1, 1, n_vtx - 1), 0;
    }
}

void SimpleBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    for (int i = 0; i < v.rows(); i++) {
        setGradientEntry(i, i, 1.0);
    }
}

void SimpleBoundary::computeHessian() {
    d2vdp2.resize(v.rows());

    MatrixXT zeroHessian = MatrixXT::Zero(nfree, nfree);
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = zeroHessian;
    }
}

