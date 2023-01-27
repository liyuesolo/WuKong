#include "../../include/Boundary/CircleBoundary.h"
#include <cmath>

void CircleBoundary::computeVertices() {
    double r = p(0);

    v.resize(4 * 2);
    v << r, 0, 0, r, -r, 0, 0, -r;
    r_map = VectorXi::Zero(4);

    radii.resize(1);
    radii(0) = r;

    int n_vtx = v.rows() / 2;
    next.resize(n_vtx);
    next << Eigen::VectorXi::LinSpaced(n_vtx - 1, 1, n_vtx - 1), 0;
}

void CircleBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    setGradientEntry(0, 0, 1);
    setGradientEntry(3, 0, 1);
    setGradientEntry(4, 0, -1);
    setGradientEntry(7, 0, -1);

    drdp = MatrixXT::Zero(radii.rows(), nfree);
    setRGradientEntry(0, 0, 1);
}

void CircleBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2rdp2.resize(radii.rows());
    for (int i = 0; i < radii.rows(); i++) {
        d2rdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

