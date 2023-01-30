#include "../../include/Boundary/HardwareBoundary0.h"
#include <cmath>

void HardwareBoundary0::computeVertices() {
    double dx = p(0);
    double dy = p(1);

    int n_vtx = 7;
    v.resize(n_vtx * 2);
    v << -channel_width, -channel_width,
            dx + corner_radius, -channel_width,
            dx + corner_radius, 0,
            corner_radius, 0,
            0, corner_radius,
            0, dy + corner_radius,
            -channel_width, dy + corner_radius;

    r_map = -1 * VectorXi::Ones(n_vtx);
    r_map(3) = 0;
    radii.resize(1);
    radii(0) = -corner_radius;

    next.resize(n_vtx);
    next << Eigen::VectorXi::LinSpaced(n_vtx - 1, 1, n_vtx - 1), 0;
}

void HardwareBoundary0::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);

    setGradientEntry(1 * 2 + 0, 0, 1);
    setGradientEntry(2 * 2 + 0, 0, 1);
    setGradientEntry(5 * 2 + 1, 1, 1);
    setGradientEntry(6 * 2 + 1, 1, 1);

    drdp = MatrixXT::Zero(radii.rows(), nfree);
}

void HardwareBoundary0::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2rdp2.resize(radii.rows());
    for (int i = 0; i < radii.rows(); i++) {
        d2rdp2[i] = MatrixXT::Zero(nfree, nfree);
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

