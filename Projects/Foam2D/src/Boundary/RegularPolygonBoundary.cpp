#include "../../include/Boundary/RegularPolygonBoundary.h"
#include <cmath>

void RegularPolygonBoundary::computeVertices() {
    double r = p(0);
    double t0 = p(1);

    v.resize(nsides * 2);
    for (int i = 0; i < nsides; i++) {
        double t = t0 + i * M_PI * 2.0 / nsides;

        v(i * 2 + 0) = r * cos(t);
        v(i * 2 + 1) = r * sin(t);
    }

    int n_vtx = v.rows() / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 0;
        edges[i].q_idx = -1;
    }
}

void RegularPolygonBoundary::computeGradient() {
    double r = p(0);
    double t0 = p(1);

    dvdp = MatrixXT::Zero(v.rows(), nfree);
    for (int i = 0; i < nsides; i++) {
        double t = t0 + i * M_PI * 2.0 / nsides;

        addGradientEntry(i * 2 + 0, 0, cos(t));
        addGradientEntry(i * 2 + 0, 1, -r * sin(t));

        addGradientEntry(i * 2 + 1, 0, sin(t));
        addGradientEntry(i * 2 + 1, 1, r * cos(t));
    }
}

void RegularPolygonBoundary::computeHessian() {
    double r = p(0);
    double t0 = p(1);

    d2vdp2.resize(v.rows());
    for (int i = 0; i < nsides; i++) {
        d2vdp2[i * 2 + 0] = MatrixXT::Zero(nfree, nfree);
        d2vdp2[i * 2 + 1] = MatrixXT::Zero(nfree, nfree);

        double t = t0 + i * M_PI * 2.0 / nsides;

        addHessianEntry(i * 2 + 0, 0, 0, 0);
        addHessianEntry(i * 2 + 0, 0, 1, -sin(t));
        addHessianEntry(i * 2 + 0, 1, 0, -sin(t));
        addHessianEntry(i * 2 + 0, 1, 1, -r * cos(t));

        addHessianEntry(i * 2 + 1, 0, 0, 0);
        addHessianEntry(i * 2 + 1, 0, 1, cos(t));
        addHessianEntry(i * 2 + 1, 1, 0, cos(t));
        addHessianEntry(i * 2 + 1, 1, 1, -r * sin(t));
    }
}

