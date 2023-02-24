#include "../../include/Boundary/BiArcBoundary.h"
#include "../../include/Boundary/BiArc.h"

void BiArcBoundary::computeVertices() {
    int ncp = p.rows() / 3;
    v.resize(ncp * 4);
    q.resize(ncp * 2);

    int n_vtx = v.rows() / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx;
        edges[i].btype = 1;
        edges[i].q_idx = i;
    }

    for (int i = 0; i < ncp; i++) {
        int j = edges[edges[2 * i].nextEdge].nextEdge / 2;
        v.segment<2>(i * 4) = p.segment<2>(i * 3);

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        VectorXT outputs;
        BiArc::getBiArcValues(inputs, outputs);

        v.segment<2>(i * 4 + 2) = outputs.segment<2>(0);
        q.segment<2>(i * 2) = outputs.segment<2>(2);
    }
}

void BiArcBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    dqdp = MatrixXT::Zero(q.rows(), nfree);

    int ncp = p.rows() / 3;
    for (int i = 0; i < ncp; i++) {
        int j = edges[edges[2 * i].nextEdge].nextEdge / 2;

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        MatrixXT outputs;
        BiArc::getBiArcGradient(inputs, outputs);

        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                addGradientEntry(i * 4 + 2 + ii, i * 3 + jj, outputs(ii, jj));
                addGradientEntry(i * 4 + 2 + ii, j * 3 + jj, outputs(ii, jj + 3));
                addQGradientEntry(i * 2 + ii, i * 3 + jj, outputs(ii + 2, jj));
                addQGradientEntry(i * 2 + ii, j * 3 + jj, outputs(ii + 2, jj + 3));
            }
        }

        addGradientEntry(i * 4 + 0, i * 3 + 0, 1);
        addGradientEntry(i * 4 + 1, i * 3 + 1, 1);
    }
}

void BiArcBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }

    int ncp = p.rows() / 3;
    for (int i = 0; i < ncp; i++) {
        int j = edges[edges[2 * i].nextEdge].nextEdge / 2;

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        std::vector<MatrixXT> outputs;
        BiArc::getBiArcHessian(inputs, outputs);

        MatrixXT hess;
        int idx;
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                hess = outputs[0];
                idx = i * 4 + 2 + 0;
                addHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                addHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                addHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                addHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[1];
                idx = i * 4 + 2 + 1;
                addHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                addHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                addHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                addHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[2];
                idx = i * 2 + 0;
                addQHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                addQHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                addQHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                addQHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[3];
                idx = i * 2 + 1;
                addQHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                addQHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                addQHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                addQHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));
            }
        }
    }
}

bool BiArcBoundary::checkValid() {
    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        for (int j = i + 1; j < n_vtx; j++) {
            if (edges[i].nextEdge == j || edges[j].nextEdge == i) continue;

            double r0 = q(edges[i].q_idx);
            double r1 = q(edges[j].q_idx);

            double x00 = v(i * 2 + 0);
            double y00 = v(i * 2 + 1);
            double x01 = v(edges[i].nextEdge * 2 + 0);
            double y01 = v(edges[i].nextEdge * 2 + 1);
            double xc0, yc0;
            {
                double a = (TV(x01, y01) - TV(x00, y00)).norm();
                double d = r0 / 2 * sqrt(4 - pow(a / r0, 2));
                double theta = atan2(-(x01 - x00), (y01 - y00));
                xc0 = (x00 + x01) / 2 - d * cos(theta);
                yc0 = (y00 + y01) / 2 - d * sin(theta);
            }

            double x10 = v(j * 2 + 0);
            double y10 = v(j * 2 + 1);
            double x11 = v(edges[j].nextEdge * 2 + 0);
            double y11 = v(edges[j].nextEdge * 2 + 1);
            double xc1, yc1;
            {
                double a = (TV(x11, y11) - TV(x10, y10)).norm();
                double d = r1 / 2 * sqrt(4 - pow(a / r1, 2));
                double theta = atan2(-(x11 - x10), (y11 - y10));
                xc1 = (x10 + x11) / 2 - d * cos(theta);
                yc1 = (y10 + y11) / 2 - d * sin(theta);
            }

            double R = (TV(xc1, yc1) - TV(xc0, yc0)).norm();
            double xint0 = 0.5 * (xc0 + xc1) + (r0 * r0 - r1 * r1) / (2 * R * R) * (xc1 - xc0) +
                           0.5 * sqrt(2 / pow(R, 2) * (r0 * r0 + r1 * r1) - pow(r0 * r0 - r1 * r1, 2) / pow(R, 4) - 1) *
                           (yc1 - yc0);
            double yint0 = 0.5 * (yc0 + yc1) + (r0 * r0 - r1 * r1) / (2 * R * R) * (yc1 - yc0) +
                           0.5 * sqrt(2 / pow(R, 2) * (r0 * r0 + r1 * r1) - pow(r0 * r0 - r1 * r1, 2) / pow(R, 4) - 1) *
                           (xc0 - xc1);
            double xint1 = 0.5 * (xc0 + xc1) + (r0 * r0 - r1 * r1) / (2 * R * R) * (xc1 - xc0) -
                           0.5 * sqrt(2 / pow(R, 2) * (r0 * r0 + r1 * r1) - pow(r0 * r0 - r1 * r1, 2) / pow(R, 4) - 1) *
                           (yc1 - yc0);
            double yint1 = 0.5 * (yc0 + yc1) + (r0 * r0 - r1 * r1) / (2 * R * R) * (yc1 - yc0) -
                           0.5 * sqrt(2 / pow(R, 2) * (r0 * r0 + r1 * r1) - pow(r0 * r0 - r1 * r1, 2) / pow(R, 4) - 1) *
                           (xc0 - xc1);

            TV p0((x00 + x01) / 2, (y00 + y01) / 2);
            TV c0(xc0, yc0);
            TV int0(xint0, yint0);

            TV p1((x10 + x11) / 2, (y10 + y11) / 2);
            TV c1(xc1, yc1);
            TV int1(xint1, yint1);

            if ((int0 - c0).dot(p0 - c0) > (p0 - c0).dot((p0 - c0))
                && (int0 - c1).dot(p1 - c1) > (p1 - c1).dot((p1 - c1))) {
                return false;
            }
            if ((int1 - c0).dot(p0 - c0) > (p0 - c0).dot((p0 - c0))
                && (int1 - c1).dot(p1 - c1) > (p1 - c1).dot((p1 - c1))) {
                return false;
            }
        }
    }

    return true;
}

double BiArcBoundary::computeEnergy() {
    double energy = 0;
    for (int i = 0; i < q.rows(); i++) {
        energy += epsilon / pow(q(i), 2);
    }
    return energy;
}

VectorXT BiArcBoundary::computeEnergyGradient() {
    VectorXT dEdr = VectorXT::Zero(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        dEdr(i) = -2 * epsilon / pow(q(i), 3);
    }
    return dEdr.transpose() * dqdp;
}

MatrixXT BiArcBoundary::computeEnergyHessian() {
    VectorXT dEdr = VectorXT::Zero(q.rows());
    VectorXT d2Edr2 = VectorXT::Zero(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        dEdr(i) = -2 * epsilon / pow(q(i), 3);
        d2Edr2(i) = 6 * epsilon / pow(q(i), 4);
    }

    MatrixXT energyHessian = dqdp.transpose() * d2Edr2.asDiagonal() * dqdp;
    for (int i = 0; i < q.rows(); i++) {
        energyHessian += dEdr(i) * d2qdp2[i];
    }
    return energyHessian;
}
