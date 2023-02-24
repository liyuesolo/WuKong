#include "../../include/Boundary/RigidBodyAgentBoundary.h"
#include <cmath>

void RigidBodyAgentBoundary::computeVertices() {
    VectorXT box(4 * 2);
    box << -bx, -by, bx, -by, bx, by, -bx, by;

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

    int n_agent = agent.rows() / 2;
    int n_box = 4;
    edges.resize(v.rows() / 2);
    for (int i = 0; i < n_agent; i++) {
        edges[i].nextEdge = (i + 1) % n_agent;
        edges[i].btype = q_map(i) >= 0 ? 1 : 0;
        edges[i].q_idx = q_map(i);
    }
    for (int i = 0; i < n_box; i++) {
        edges[i + n_agent].nextEdge = n_agent + (i + 1) % n_box;
        edges[i + n_agent].btype = 0;
        edges[i + n_agent].q_idx = -1;
    }

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

        addGradientEntry(i * 2 + 0, 0, 1);
        addGradientEntry(i * 2 + 0, 1, 0);
        addGradientEntry(i * 2 + 0, 2, tmul * (-x0 * sin(t) - y0 * cos(t)));

        addGradientEntry(i * 2 + 1, 0, 0);
        addGradientEntry(i * 2 + 1, 1, 1);
        addGradientEntry(i * 2 + 1, 2, tmul * (x0 * cos(t) - y0 * sin(t)));
    }

    dqdp = MatrixXT::Zero(q.rows(), nfree);
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

        addHessianEntry(i * 2 + 0, 2, 2, tmul * tmul * (-x0 * cos(t) + y0 * sin(t)));
        addHessianEntry(i * 2 + 1, 2, 2, tmul * tmul * (-x0 * sin(t) - y0 * cos(t)));
    }
    for (int i = 0; i < 8; i++) {
        d2vdp2[nsides * 2 + i] = MatrixXT::Zero(nfree, nfree);
    }

    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

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

