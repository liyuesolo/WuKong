#include "../../include/Boundary/BezierCircleBoundary.h"
#include <cmath>

void BezierCircleBoundary::computeVertices() {
    double r = p(0);

    v.resize(4 * 2);
    v << r, 0, 0, r, -r, 0, 0, -r;

    q.resize(4);
    q << M_PI_2 + 0.001, M_PI + 0.001, 3 * M_PI_2 + 0.001, 0.001;

    int n_vtx = v.rows() / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 2;
        edges[i].q_idx = i;
    }
}

void BezierCircleBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    addGradientEntry(0, 0, 1);
    addGradientEntry(3, 0, 1);
    addGradientEntry(4, 0, -1);
    addGradientEntry(7, 0, -1);

    dqdp = MatrixXT::Zero(q.rows(), nfree);
}

void BezierCircleBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

