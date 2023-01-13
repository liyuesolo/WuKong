#include "../../include/Boundary/CircleBoundary.h"
#include <cmath>

void CircleBoundary::computeVertices() {
    double r = p(0);
    double t0 = p(1);

    v.resize(nsides * 2);
    for (int i = 0; i < nsides; i++) {
        double t = t0 + i * M_PI * 2.0 / nsides;

        v(i * 2 + 0) = r * cos(t);
        v(i * 2 + 1) = r * sin(t);
    }
}

void CircleBoundary::computeGradient() {
    double r = p(0);
    double t0 = p(1);

    dvdp = MatrixXT::Zero(v.rows(), nfree);
    for (int i = 0; i < nsides; i++) {
        double t = t0 + i * M_PI * 2.0 / nsides;

        setGradientEntry(i * 2 + 0, 0, cos(t));
        setGradientEntry(i * 2 + 0, 1, -r * sin(t));

        setGradientEntry(i * 2 + 1, 0, sin(t));
        setGradientEntry(i * 2 + 1, 1, r * cos(t));
    }
}

void CircleBoundary::computeHessian() {
    double r = p(0);
    double t0 = p(1);

    d2vdp2.resize(v.rows());
    for (int i = 0; i < nsides; i++) {
        d2vdp2[i * 2 + 0] = MatrixXT::Zero(nfree, nfree);
        d2vdp2[i * 2 + 1] = MatrixXT::Zero(nfree, nfree);

        double t = t0 + i * M_PI * 2.0 / nsides;

        setHessianEntry(i * 2 + 0, 0, 0, 0);
        setHessianEntry(i * 2 + 0, 0, 1, -sin(t));
        setHessianEntry(i * 2 + 0, 1, 0, -sin(t));
        setHessianEntry(i * 2 + 0, 1, 1, -r * cos(t));

        setHessianEntry(i * 2 + 1, 0, 0, 0);
        setHessianEntry(i * 2 + 1, 0, 1, cos(t));
        setHessianEntry(i * 2 + 1, 1, 0, cos(t));
        setHessianEntry(i * 2 + 1, 1, 1, -r * sin(t));
    }
}

