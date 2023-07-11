#include "../../include/Boundary/SimpleBoundary.h"

void MeshBoundary::computeVertices() {
    v = p;

    int n_vtx = v.rows() / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 0;
        edges[i].q_idx = -1;
    }
}

void MeshBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    for (int i = 0; i < v.rows(); i++) {
        addGradientEntry(i, i, 1.0);
    }
}

void MeshBoundary::computeHessian() {
    d2vdp2.resize(v.rows());

    MatrixXT zeroHessian = MatrixXT::Zero(nfree, nfree);
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = zeroHessian;
    }
}

