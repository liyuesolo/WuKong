#include "../../include/Boundary/RigidBodyAgentBoundary.h"
#include <cmath>

void RigidBodyAgentBoundary::computeVertices() {
    VectorXT box(4 * 2);
    box << -bx, -by, bx, -by, bx, by, -bx, by;

    VectorXi r_box = -1 * VectorXi::Ones(4);
    VectorXi r_agent = r_map;
    r_map.resize(r_agent.rows() + r_box.rows());
    r_map << r_agent, r_box;

    double dx = p(0);
    double dy = p(1);
    double t = tmul * p(2);

    VectorXT agent(agentShape.rows());
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        agent(i * 2 + 0) = (x0 * cos(t) - y0 * sin(t)) + dx;
        agent(i * 2 + 1) = (x0 * sin(t) + y0 * cos(t)) + dy;
    }

    v.resize(agent.rows() + box.rows());
    v << agent, box;

    int n_vtx = v.rows() / 2;
    next.resize(n_vtx);
    next << Eigen::VectorXi::LinSpaced(nsides - 1, 1, nsides - 1), 0, Eigen::VectorXi::LinSpaced(3, nsides + 1,
                                                                                                 nsides + 3), nsides;

    holes.resize(1, 2);
    holes.row(0) = TV(dx, dy);
}

void RigidBodyAgentBoundary::computeGradient() {
    double dx = p(0);
    double dy = p(1);
    double t = tmul * p(2);

    dvdp = MatrixXT::Zero(v.rows(), nfree);
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        setGradientEntry(i * 2 + 0, 0, 1);
        setGradientEntry(i * 2 + 0, 1, 0);
        setGradientEntry(i * 2 + 0, 2, tmul * (-x0 * sin(t) - y0 * cos(t)));

        setGradientEntry(i * 2 + 1, 0, 0);
        setGradientEntry(i * 2 + 1, 1, 1);
        setGradientEntry(i * 2 + 1, 2, tmul * (x0 * cos(t) - y0 * sin(t)));
    }

    drdp = MatrixXT::Zero(radii.rows(), nfree);
}

void RigidBodyAgentBoundary::computeHessian() {
    double dx = p(0);
    double dy = p(1);
    double t = tmul * p(2);

    d2vdp2.resize(v.rows());
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        d2vdp2[i * 2 + 0] = MatrixXT::Zero(nfree, nfree);
        d2vdp2[i * 2 + 1] = MatrixXT::Zero(nfree, nfree);

        setHessianEntry(i * 2 + 0, 2, 2, tmul * tmul * (-x0 * cos(t) + y0 * sin(t)));
        setHessianEntry(i * 2 + 1, 2, 2, tmul * tmul * (-x0 * sin(t) - y0 * cos(t)));
    }
    for (int i = 0; i < 8; i++) {
        d2vdp2[nsides * 2 + i] = MatrixXT::Zero(nfree, nfree);
    }

    d2rdp2.resize(radii.rows());
    for (int i = 0; i < radii.rows(); i++) {
        d2rdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

// Prevent floating rigid body from going through walls. TODO: Log barriers using energy function?
bool RigidBodyAgentBoundary::checkValid() {
    double dx = p(0);
    double dy = p(1);
    double t = p(2);

    int nsides = agentShape.rows() / 2;
    double dmax = 0;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        double d = x0 * x0 + y0 * y0;
        dmax = fmax(dmax, d);
    }
    dmax = sqrt(dmax);

    return (dmax < bx - fabs(dx) && dmax < by - fabs(dy));
}

double RigidBodyAgentBoundary::computeEnergy() {
    double dx = p(0);
    double dy = p(1);

    int nsides = agentShape.rows() / 2;
    double dmax = 0;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        double d = x0 * x0 + y0 * y0;
        dmax = fmax(dmax, d);
    }
    dmax = sqrt(dmax);

    double BX = bx - dmax;
    double BY = by - dmax;
    double DX = fabs(dx);
    double DY = fabs(dy);

    double energy = epsilon * DX * DX * pow(DX - BX, -0.2e1) + epsilon * DY * DY * pow(DY - BY, -0.2e1);
    return energy;
}

VectorXT RigidBodyAgentBoundary::computeEnergyGradient() {
    double dx = p(0);
    double dy = p(1);

    int nsides = agentShape.rows() / 2;
    double dmax = 0;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        double d = x0 * x0 + y0 * y0;
        dmax = fmax(dmax, d);
    }
    dmax = sqrt(dmax);

    double BX = bx - dmax;
    double BY = by - dmax;
    double DX = fabs(dx);
    double DY = fabs(dy);

    VectorXT energyGradient = VectorXT::Zero(nfree);
//    return energyGradient;
    double t1, t2, t3;
    t1 = DX - BX;
    t1 = 0.1e1 / t1;
    t2 = BY - DY;
    t2 = 0.1e1 / t2;
    if (free_map(0) >= 0) {
        energyGradient(free_map(0)) = 0.2e1 * epsilon * DX * pow(t1, 0.2e1) * (-DX * t1 + 0.1e1);
        if (dx < 0) energyGradient(free_map(0)) *= -1;
    }
    if (free_map(1) >= 0) {
        energyGradient(free_map(1)) = 0.2e1 * epsilon * DY * pow(t2, 0.2e1) * (DY * t2 + 0.1e1);
        if (dy < 0) energyGradient(free_map(1)) *= -1;
    }
    return energyGradient;
};

MatrixXT RigidBodyAgentBoundary::computeEnergyHessian() {
    double dx = p(0);
    double dy = p(1);

    int nsides = agentShape.rows() / 2;
    double dmax = 0;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        double d = x0 * x0 + y0 * y0;
        dmax = fmax(dmax, d);
    }
    dmax = sqrt(dmax);

    double BX = bx - dmax;
    double BY = by - dmax;
    double DX = fabs(dx);
    double DY = fabs(dy);

    MatrixXT energyHessian = MatrixXT::Zero(nfree, nfree);
//    return energyHessian;
    double t1, t2, t3;
    t1 = DX - BX;
    t1 = 0.1e1 / t1;
    t2 = BY - DY;
    t2 = 0.1e1 / t2;
    if (free_map(0) >= 0) {
        energyHessian(free_map(0), free_map(0)) =
                epsilon * pow(t1, 0.2e1) * (DX * t1 * (0.6e1 * DX * t1 - 0.8e1) + 0.2e1);
    }
    if (free_map(1) >= 0) {
        energyHessian(free_map(1), free_map(1)) =
                epsilon * pow(t2, 0.2e1) * (DY * t2 * (0.6e1 * DY * t2 + 0.8e1) + 0.2e1);
    }
    return energyHessian;
};

