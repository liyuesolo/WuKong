#include "../../include/Boundary/SimpleBoundary.h"

void SimpleBoundary::computeVertices() {
    v = p;
}

void SimpleBoundary::computeGradient() {
    dvdp = VectorXT::Ones(nfree);
}

void SimpleBoundary::computeHessian() {
    d2vdp2.resize(v.rows());

    MatrixXT zeroHessian = MatrixXT::Zero(nfree, nfree);
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = zeroHessian;
    }
}

