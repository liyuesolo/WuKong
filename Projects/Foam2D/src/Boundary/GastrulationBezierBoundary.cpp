#include "../../include/Boundary/GastrulationBezierBoundary.h"
#include <cmath>

void GastrulationBezierBoundary::computeVertices() {
    int ncp = p.rows() / 2;
    v.resize(ncp * 2);
    q.resize(ncp * 2);

    int n_vtx = v.rows() / 2;
    int n_vtx_2 = n_vtx / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx_2; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx_2;
        edges[i].btype = 2;
        edges[i].q_idx = i;
    }
    for (int i = 0; i < n_vtx_2; i++) {
        edges[i + n_vtx_2].nextEdge = n_vtx_2 + (i + 1) % n_vtx_2;
        edges[i + n_vtx_2].btype = 2;
        edges[i + n_vtx_2].q_idx = i + n_vtx_2;
    }

    for (int i = 0; i < ncp; i++) {
        int j = edges[i].nextEdge;
        TV p0 = p.segment<2>(i * 2);
        TV p1 = p.segment<2>(j * 2);
        v.segment<2>(i * 2) = (p0 + p1) / 2.0;
        q(i) = atan2((p1 - p0).y(), (p1 - p0).x());
    }
}

void GastrulationBezierBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    dqdp = MatrixXT::Zero(q.rows(), nfree);

    int ncp = p.rows() / 2;
    for (int i = 0; i < ncp; i++) {
        int j = edges[i].nextEdge;
        double x0 = p(i * 2 + 0);
        double y0 = p(i * 2 + 1);
        double x1 = p(j * 2 + 0);
        double y1 = p(j * 2 + 1);

        addGradientEntry(i * 2 + 0, i * 2 + 0, 0.5);
        addGradientEntry(i * 2 + 1, i * 2 + 1, 0.5);
        addGradientEntry(i * 2 + 0, j * 2 + 0, 0.5);
        addGradientEntry(i * 2 + 1, j * 2 + 1, 0.5);

        double t1, t2, t3, t4;
        t1 = y0 - y1;
        t2 = x1 - x0;
        t2 = 0.1e1 / t2;
        t3 = pow(t2, 0.2e1);
        t4 = pow(t1, 0.2e1) * t3 + 0.1e1;
        t4 = 0.1e1 / t4;
        t2 = t2 * t4;
        t1 = t1 * t3 * t4;
        addQGradientEntry(i, i * 2 + 0, -t1);
        addQGradientEntry(i, i * 2 + 1, -t2);
        addQGradientEntry(i, j * 2 + 0, t1);
        addQGradientEntry(i, j * 2 + 1, t2);
    }
}

void GastrulationBezierBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }

    int ncp = p.rows() / 2;
    for (int i = 0; i < ncp; i++) {
        int j = edges[i].nextEdge;
        double x0 = p(i * 2 + 0);
        double y0 = p(i * 2 + 1);
        double x1 = p(j * 2 + 0);
        double y1 = p(j * 2 + 1);

        double t1, t2, t3, t4, t5;
        t1 = y0 - y1;
        t2 = x1 - x0;
        t2 = 0.1e1 / t2;
        t3 = pow(t2, 0.2e1);
        t4 = pow(t1, 0.2e1) * t3;
        t5 = 0.1e1 + t4;
        t5 = 0.1e1 / t5;
        t4 = t4 * t5;
        t1 = t1 * t2 * t3;
        t2 = t3 * t5 * (-0.1e1 + 0.2e1 * t4);
        t3 = -t2;
        t4 = 0.2e1 * t1 * t5 * (0.1e1 - t4);
        t1 = 0.2e1 * t1 * pow(t5, 0.2e1);

        int i0 = i * 2 + 0;
        int i1 = i * 2 + 1;
        int i2 = j * 2 + 0;
        int i3 = j * 2 + 1;
        addQHessianEntry(i, i0, i0, -t4);
        addQHessianEntry(i, i0, i1, t2);
        addQHessianEntry(i, i0, i2, t4);
        addQHessianEntry(i, i0, i3, t3);
        addQHessianEntry(i, i1, i0, t2);
        addQHessianEntry(i, i1, i1, t1);
        addQHessianEntry(i, i1, i2, t3);
        addQHessianEntry(i, i1, i3, -t1);
        addQHessianEntry(i, i2, i0, t4);
        addQHessianEntry(i, i2, i1, t3);
        addQHessianEntry(i, i2, i2, -t4);
        addQHessianEntry(i, i2, i3, t2);
        addQHessianEntry(i, i3, i0, t3);
        addQHessianEntry(i, i3, i1, -t1);
        addQHessianEntry(i, i3, i2, t2);
        addQHessianEntry(i, i3, i3, t1);
    }
}

bool GastrulationBezierBoundary::checkValid() {
    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        TV vertex = v.segment<2>(i * 2);
        TV normal(cos(q(i) + M_PI_2), sin(q(i) + M_PI_2));
        TV point = vertex + 1e-3 * normal;

        if (!pointInBounds(point)) return false;
    }
    return true;
}

