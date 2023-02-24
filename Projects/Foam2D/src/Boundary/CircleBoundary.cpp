#include "../../include/Boundary/CircleBoundary.h"
#include <cmath>

void CircleBoundary::computeVertices() {
    double r = p(0);

    v.resize(4 * 2);
    v << r, 0, 0, r, -r, 0, 0, -r;

    q.resize(2);
    q(0) = r;
    q(1) = -r;

    int n_vtx = v.rows() / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 1;
        edges[i].q_idx = 0;
    }
    edges[0].q_idx = 0; // Change if you want to invert one edge.
}

void CircleBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    addGradientEntry(0, 0, 1);
    addGradientEntry(3, 0, 1);
    addGradientEntry(4, 0, -1);
    addGradientEntry(7, 0, -1);

    dqdp = MatrixXT::Zero(q.rows(), nfree);
    addQGradientEntry(0, 0, 1);
    addQGradientEntry(1, 0, -1);
}

void CircleBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

