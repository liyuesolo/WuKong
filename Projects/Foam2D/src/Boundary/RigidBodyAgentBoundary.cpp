#include "../../include/Boundary/RigidBodyAgentBoundary.h"
#include <cmath>

// Prevent floating rigid body from going through walls. TODO: Log barriers using energy function?
bool RigidBodyAgentBoundary::checkValid() {
    double bx = 0.75, by = 0.75;

    double dx = p(0);
    double dy = p(1);
    double t = p(2);

    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);
        double x = (x0 * cos(t) - y0 * sin(t)) + dx;
        double y = (x0 * sin(t) + y0 * cos(t)) + dy;

        if (fabs(x) > bx || fabs(y) > by) return false;
    }

    return true;
}

void RigidBodyAgentBoundary::computeVertices() {
    double bx = 0.75, by = 0.75;
    VectorXT box(4 * 2);
    box << -bx, -by, bx, -by, bx, by, -bx, by;

    double dx = p(0);
    double dy = p(1);
    double t = p(2);

    VectorXT agent(agentShape.rows());
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        agent(i * 2 + 0) = (x0 * cos(t) - y0 * sin(t)) + dx;
        agent(i * 2 + 1) = (x0 * sin(t) + y0 * cos(t)) + dy;
    }

    v.resize(box.rows() + agent.rows());
    v << box, agent;

    int n_vtx = v.rows() / 2;
    next.resize(n_vtx);
    next << Eigen::VectorXi::LinSpaced(3, 1, 3), 0, Eigen::VectorXi::LinSpaced(nsides - 1, 5, nsides - 1 + 4), 4;
}

void RigidBodyAgentBoundary::computeGradient() {
    double dx = p(0);
    double dy = p(1);
    double t = p(2);

    dvdp = MatrixXT::Zero(v.rows(), nfree);
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        setGradientEntry(8 + i * 2 + 0, 0, 1);
        setGradientEntry(8 + i * 2 + 0, 1, 0);
        setGradientEntry(8 + i * 2 + 0, 2, -x0 * sin(t) - y0 * cos(t));

        setGradientEntry(8 + i * 2 + 1, 0, 0);
        setGradientEntry(8 + i * 2 + 1, 1, 1);
        setGradientEntry(8 + i * 2 + 1, 2, x0 * cos(t) - y0 * sin(t));
    }
}

void RigidBodyAgentBoundary::computeHessian() {
    double dx = p(0);
    double dy = p(1);
    double t = p(2);

    d2vdp2.resize(v.rows());
    for (int i = 0; i < 8; i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    int nsides = agentShape.rows() / 2;
    for (int i = 0; i < nsides; i++) {
        double x0 = agentShape(i * 2 + 0);
        double y0 = agentShape(i * 2 + 1);

        d2vdp2[8 + i * 2 + 0] = MatrixXT::Zero(nfree, nfree);
        d2vdp2[8 + i * 2 + 1] = MatrixXT::Zero(nfree, nfree);

        setHessianEntry(8 + i * 2 + 0, 2, 2, -x0 * cos(t) + y0 * sin(t));
        setHessianEntry(8 + i * 2 + 1, 2, 2, -x0 * sin(t) - y0 * cos(t));
    }
}

