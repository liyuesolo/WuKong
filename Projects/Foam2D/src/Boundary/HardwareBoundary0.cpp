#include "../../include/Boundary/HardwareBoundary0.h"
#include <cmath>

void HardwareBoundary0::computeVertices() {
    double dx = p(0);
    double dy = p(1);

    int n_vtx = 9;
    v.resize(n_vtx * 2);
    v << -channel_width, 0,
            -channel_width, -channel_width,
            0, -channel_width,
            dx + corner_radius, -channel_width,
            dx + corner_radius, 0,
            corner_radius, 0,
            0, corner_radius,
            0, dy + corner_radius,
            -channel_width, dy + corner_radius;

    q.resize(2);
    q(0) = -corner_radius;
    q(1) = channel_width;

    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 0;
        edges[i].q_idx = -1;
    }
    edges[5].btype = 1;
    edges[5].q_idx = 0;
}

void HardwareBoundary0::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);

    setGradientEntry(3 * 2 + 0, 0, 1);
    setGradientEntry(4 * 2 + 0, 0, 1);
    setGradientEntry(7 * 2 + 1, 1, 1);
    setGradientEntry(8 * 2 + 1, 1, 1);

    dqdp = MatrixXT::Zero(q.rows(), nfree);
}

void HardwareBoundary0::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

bool HardwareBoundary0::checkValid() {
    double dx = p(0);
    double dy = p(1);

    return (dx > 0 && dy > 0);
}

double HardwareBoundary0::computeEnergy() {
    double dx = p(0);
    double dy = p(1);

    double energy = w_barrier * (pow(dx, -2) + pow(dy, -2)) + w_piston * (pow(dx, 2) + pow(dy, 2));
    return energy;
}

VectorXT HardwareBoundary0::computeEnergyGradient() {
    double dx = p(0);
    double dy = p(1);

    VectorXT energyGradient = VectorXT::Zero(nfree);
//    return energyGradient;
    if (free_map(0) >= 0) {
        energyGradient(free_map(0)) = -2 * w_barrier * pow(dx, -3) + 2 * w_piston * dx;
    }
    if (free_map(1) >= 0) {
        energyGradient(free_map(1)) = -2 * w_barrier * pow(dy, -3) + 2 * w_piston * dy;
    }
    return energyGradient;
};

MatrixXT HardwareBoundary0::computeEnergyHessian() {
    double dx = p(0);
    double dy = p(1);

    MatrixXT energyHessian = MatrixXT::Zero(nfree, nfree);
//    return energyHessian;
    if (free_map(0) >= 0) {
        energyHessian(free_map(0), free_map(0)) =
                6 * w_barrier * pow(dx, -4) + 2 * w_piston;
    }
    if (free_map(1) >= 0) {
        energyHessian(free_map(1), free_map(1)) =
                6 * w_barrier * pow(dy, -4) + 2 * w_piston;
    }
    return energyHessian;
};

