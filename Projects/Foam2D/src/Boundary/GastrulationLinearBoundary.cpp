#include "../../include/Boundary/GastrulationLinearBoundary.h"
#include <cmath>

void GastrulationLinearBoundary::computeVertices() {
    v = p;
    q.resize(0);

    int n_vtx = v.rows() / 2;
    int n_vtx_2 = n_vtx / 2;
    edges.resize(n_vtx);
    for (int i = 0; i < n_vtx_2; i++) {
        edges[i].nextEdge = (i + 1) % n_vtx_2;
        edges[i].btype = 0;
        edges[i].q_idx = -1;
    }
    for (int i = 0; i < n_vtx_2; i++) {
        edges[i + n_vtx_2].nextEdge = n_vtx_2 + (i + 1) % n_vtx_2;
        edges[i + n_vtx_2].btype = 0;
        edges[i + n_vtx_2].q_idx = -1;
    }
}

void GastrulationLinearBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    dqdp = MatrixXT::Zero(q.rows(), nfree);
    for (int i = 0; i < v.rows(); i++) {
        addGradientEntry(i, i, 1.0);
    }
}

void GastrulationLinearBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2qdp2.resize(q.rows());
    for (int i = 0; i < q.rows(); i++) {
        d2qdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
}

bool GastrulationLinearBoundary::checkValid() {
    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        for (int j = i + 1; j < n_vtx; j++) {
            if (edges[i].nextEdge == j || edges[j].nextEdge == i) continue;

            TV p0 = v.segment<2>(i * 2);
            TV p1 = v.segment<2>(edges[i].nextEdge * 2);
            TV p2 = v.segment<2>(j * 2);
            TV p3 = v.segment<2>(edges[j].nextEdge * 2);

            double s1_x, s1_y, s2_x, s2_y;
            s1_x = p1.x() - p0.x();
            s1_y = p1.y() - p0.y();
            s2_x = p3.x() - p2.x();
            s2_y = p3.y() - p2.y();

            double s, t;
            s = (-s1_y * (p0.x() - p2.x()) + s1_x * (p0.y() - p2.y())) / (-s2_x * s1_y + s1_x * s2_y);
            t = (s2_x * (p0.y() - p2.y()) - s2_y * (p0.x() - p2.x())) / (-s2_x * s1_y + s1_x * s2_y);

            if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
                // Collision detected
                return false;
            }
        }
    }

    for (int i = 0; i < n_vtx; i++) {
        int j = edges[i].nextEdge;
        int k = edges[j].nextEdge;

        TV p0 = v.segment<2>(i * 2);
        TV p1 = v.segment<2>(j * 2);
        TV p2 = v.segment<2>(k * 2);
        if ((p1 - p0).dot(p2 - p1) < 0) return false; // Acute angle not permitted
    }

    return true;
}

double GastrulationLinearBoundary::computeEnergy() {
    double energy = 0;

    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        int j = edges[i].nextEdge;
        int k = edges[j].nextEdge;

        double x0 = v(i * 2 + 0);
        double y0 = v(i * 2 + 1);
        double x1 = v(j * 2 + 0);
        double y1 = v(j * 2 + 1);
        double x2 = v(k * 2 + 0);
        double y2 = v(k * 2 + 1);

        double t1 = x1 - x0;
        double t2 = x2 - x1;
        double t4 = y1 - y0;
        double t5 = y2 - y1;
        double t9 = t1 * t1;
        double t10 = t4 * t4;
        double t12 = sqrt(t9 + t10);
        double t14 = t2 * t2;
        double t15 = t5 * t5;
        double t17 = sqrt(t14 + t15);
        energy += t17 * t12 / (t2 * t1 + t5 * t4) - 0.1e1;
    }

    return w * energy;
}

VectorXT GastrulationLinearBoundary::computeEnergyGradient() {
    VectorXT dEdv = VectorXT::Zero(v.rows());

    int n = 6;

    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        int j = edges[i].nextEdge;
        int k = edges[j].nextEdge;

        VectorXi indices(n);
        indices << i * 2 + 0, i * 2 + 1, j * 2 + 0, j * 2 + 1, k * 2 + 0, k * 2 + 1;

        double x0 = v(i * 2 + 0);
        double y0 = v(i * 2 + 1);
        double x1 = v(j * 2 + 0);
        double y1 = v(j * 2 + 1);
        double x2 = v(k * 2 + 0);
        double y2 = v(k * 2 + 1);

        double unknown[n];

        double t1 = x1 - x0;
        double t2 = x2 - x1;
        double t4 = y1 - y0;
        double t5 = y2 - y1;
        double t7 = t2 * t1 + t5 * t4;
        double t8 = t7 * t7;
        double t10 = t1 * t1;
        double t11 = t4 * t4;
        double t13 = sqrt(t10 + t11);
        double t14 = t13 / t8;
        double t15 = t2 * t2;
        double t16 = t5 * t5;
        double t18 = sqrt(t15 + t16);
        double t21 = 0.1e1 / t7;
        double t23 = 0.1e1 / t13 * t21;
        double t41 = t13 * t21;
        double t42 = 0.1e1 / t18;
        unknown[0] = -t1 * t18 * t23 + t2 * t18 * t14;
        unknown[1] = t5 * t18 * t14 - t4 * t18 * t23;
        unknown[2] = -(x2 - 0.2e1 * x1 + x0) * t18 * t14 + t1 * t18 * t23 - t2 * t42 * t41;
        unknown[3] = -(y2 - 0.2e1 * y1 + y0) * t18 * t14 + t4 * t18 * t23 - t5 * t42 * t41;
        unknown[4] = -t1 * t18 * t14 + t2 * t42 * t41;
        unknown[5] = -t4 * t18 * t14 + t5 * t42 * t41;

        for (int ii = 0; ii < n; ii++) {
            dEdv(indices(ii)) += w * unknown[ii];
        }
    }

    return dEdv.transpose() * dvdp;
};

MatrixXT GastrulationLinearBoundary::computeEnergyHessian() {
    VectorXT dEdv = VectorXT::Zero(v.rows());
    MatrixXT d2Edv2 = MatrixXT::Zero(v.rows(), v.rows());

    int n = 6;

    int n_vtx = v.rows() / 2;
    for (int i = 0; i < n_vtx; i++) {
        int j = edges[i].nextEdge;
        int k = edges[j].nextEdge;

        VectorXi indices(n);
        indices << i * 2 + 0, i * 2 + 1, j * 2 + 0, j * 2 + 1, k * 2 + 0, k * 2 + 1;

        double x0 = v(i * 2 + 0);
        double y0 = v(i * 2 + 1);
        double x1 = v(j * 2 + 0);
        double y1 = v(j * 2 + 1);
        double x2 = v(k * 2 + 0);
        double y2 = v(k * 2 + 1);

        {
            double unknown[n];

            double t1 = x1 - x0;
            double t2 = x2 - x1;
            double t4 = y1 - y0;
            double t5 = y2 - y1;
            double t7 = t2 * t1 + t5 * t4;
            double t8 = t7 * t7;
            double t10 = t1 * t1;
            double t11 = t4 * t4;
            double t13 = sqrt(t10 + t11);
            double t14 = t13 / t8;
            double t15 = t2 * t2;
            double t16 = t5 * t5;
            double t18 = sqrt(t15 + t16);
            double t21 = 0.1e1 / t7;
            double t23 = 0.1e1 / t13 * t21;
            double t41 = t13 * t21;
            double t42 = 0.1e1 / t18;
            unknown[0] = -t1 * t18 * t23 + t2 * t18 * t14;
            unknown[1] = t5 * t18 * t14 - t4 * t18 * t23;
            unknown[2] = -(x2 - 0.2e1 * x1 + x0) * t18 * t14 + t1 * t18 * t23 - t2 * t42 * t41;
            unknown[3] = -(y2 - 0.2e1 * y1 + y0) * t18 * t14 + t4 * t18 * t23 - t5 * t42 * t41;
            unknown[4] = -t1 * t18 * t14 + t2 * t42 * t41;
            unknown[5] = -t4 * t18 * t14 + t5 * t42 * t41;

            for (int ii = 0; ii < n; ii++) {
                dEdv(indices(ii)) += w * unknown[ii];
            }
        }
        {
            double unknown[n][n];

            double t1 = x1 - x0;
            double t2 = x2 - x1;
            double t4 = y1 - y0;
            double t5 = y2 - y1;
            double t7 = t2 * t1 + t5 * t4;
            double t8 = t7 * t7;
            double t11 = t1 * t1;
            double t12 = t4 * t4;
            double t13 = t11 + t12;
            double t14 = sqrt(t13);
            double t15 = t14 / t7 / t8;
            double t16 = t2 * t2;
            double t17 = t5 * t5;
            double t18 = t16 + t17;
            double t19 = sqrt(t18);
            double t23 = 0.1e1 / t8;
            double t24 = 0.1e1 / t14;
            double t25 = t24 * t23;
            double t26 = -t2 * t19;
            double t29 = 0.1e1 / t7;
            double t32 = 0.1e1 / t14 / t13 * t29;
            double t36 = t1 * t1 * t19 * t32;
            double t37 = t24 * t29;
            double t38 = t19 * t37;
            double t46 = -0.2e1 * t1 * t19;
            double t53 = -0.2e1 * t5 * t26 * t15 + t4 * t26 * t25 + t5 * t46 * t25 / 0.2e1 + t4 * t46 * t32 / 0.2e1;
            double t55 = x2 - 0.2e1 * x1 + x0;
            double t62 = t14 * t23;
            double t63 = 0.1e1 / t19;
            double t64 = -t2 * t63;
            double t68 = t19 * t62;
            double t75 = -0.2e1 * t1 * t63;
            double t79 = 0.2e1 * t55 * t26 * t15 - t1 * t26 * t25 + t2 * t64 * t62 - t68 - t55 * t46 * t25 / 0.2e1 -
                         t1 * t46 * t32 / 0.2e1 - t2 * t75 * t37 / 0.2e1 - t38;
            double t81 = y2 - 0.2e1 * y1 + y0;
            double t100 = 0.2e1 * t81 * t26 * t15 - t4 * t26 * t25 + t5 * t64 * t62 - t81 * t46 * t25 / 0.2e1 -
                          t4 * t46 * t32 / 0.2e1 - t5 * t75 * t37 / 0.2e1;
            double t113 =
                    0.2e1 * t1 * t26 * t15 - t2 * t64 * t62 + t68 - t1 * t46 * t25 / 0.2e1 + t2 * t75 * t37 / 0.2e1;
            double t126 = 0.2e1 * t4 * t26 * t15 - t5 * t64 * t62 - t4 * t46 * t25 / 0.2e1 + t5 * t75 * t37 / 0.2e1;
            double t130 = -t5 * t19;
            double t136 = t4 * t4 * t19 * t32;
            double t144 = -t5 * t63;
            double t148 = -0.2e1 * t4 * t19;
            double t155 = -0.2e1 * t4 * t63;
            double t159 = 0.2e1 * t55 * t130 * t15 - t1 * t130 * t25 + t2 * t144 * t62 - t55 * t148 * t25 / 0.2e1 -
                          t1 * t148 * t32 / 0.2e1 - t2 * t155 * t37 / 0.2e1;
            double t178 =
                    0.2e1 * t81 * t130 * t15 - t4 * t130 * t25 + t5 * t144 * t62 - t68 - t81 * t148 * t25 / 0.2e1 -
                    t4 * t148 * t32 / 0.2e1 - t5 * t155 * t37 / 0.2e1 - t38;
            double t191 = 0.2e1 * t1 * t130 * t15 - t2 * t144 * t62 - t1 * t148 * t25 / 0.2e1 + t2 * t155 * t37 / 0.2e1;
            double t204 =
                    0.2e1 * t4 * t130 * t15 - t5 * t144 * t62 + t68 - t4 * t148 * t25 / 0.2e1 + t5 * t155 * t37 / 0.2e1;
            double t205 = t55 * t55;
            double t209 = t55 * t19;
            double t212 = t55 * t63;
            double t215 = 0.2e1 * t68;
            double t216 = 0.2e1 * t1 * t63;
            double t220 = t14 * t29;
            double t222 = 0.1e1 / t19 / t18;
            double t226 = t2 * t2 * t222 * t220;
            double t227 = t63 * t220;
            double t238 = 0.2e1 * t1 * t19;
            double t248 = -0.2e1 * t2 * t63;
            double t255 = -0.2e1 * t2 * t222;
            double t259 = 0.2e1 * t81 * t209 * t15 - t4 * t209 * t25 + t5 * t212 * t62 - t81 * t238 * t25 / 0.2e1 -
                          t4 * t238 * t32 / 0.2e1 - t5 * t216 * t37 / 0.2e1 - t81 * t248 * t62 / 0.2e1 +
                          t4 * t248 * t37 / 0.2e1 + t5 * t255 * t220 / 0.2e1;
            double t278 = 0.2e1 * t1 * t209 * t15 - t2 * t212 * t62 - t68 - t1 * t238 * t25 / 0.2e1 +
                          t2 * t216 * t37 / 0.2e1 - t1 * t248 * t62 / 0.2e1 - t2 * t255 * t220 / 0.2e1 - t227;
            double t297 =
                    0.2e1 * t4 * t209 * t15 - t5 * t212 * t62 - t4 * t238 * t25 / 0.2e1 + t5 * t216 * t37 / 0.2e1 -
                    t4 * t248 * t62 / 0.2e1 - t5 * t255 * t220 / 0.2e1;
            double t298 = t81 * t81;
            double t302 = t81 * t19;
            double t305 = t81 * t63;
            double t308 = 0.2e1 * t4 * t63;
            double t315 = t5 * t5 * t222 * t220;
            double t323 = 0.2e1 * t4 * t19;
            double t330 = -0.2e1 * t5 * t63;
            double t334 = -0.2e1 * t5 * t222;
            double t338 =
                    0.2e1 * t1 * t302 * t15 - t2 * t305 * t62 - t1 * t323 * t25 / 0.2e1 + t2 * t308 * t37 / 0.2e1 -
                    t1 * t330 * t62 / 0.2e1 - t2 * t334 * t220 / 0.2e1;
            double t357 = 0.2e1 * t4 * t302 * t15 - t5 * t305 * t62 - t68 - t4 * t323 * t25 / 0.2e1 +
                          t5 * t308 * t37 / 0.2e1 - t4 * t330 * t62 / 0.2e1 - t5 * t334 * t220 / 0.2e1 - t227;
            double t361 = t1 * t63;
            double t380 = 0.2e1 * t4 * t1 * t19 * t15 - t5 * t2 * t222 * t220 - t4 * t2 * t63 * t62 - t5 * t361 * t62;
            unknown[0][0] = 0.2e1 * t1 * t26 * t25 + 0.2e1 * t16 * t19 * t15 - t36 + t38;
            unknown[0][1] = t53;
            unknown[0][2] = t79;
            unknown[0][3] = t100;
            unknown[0][4] = t113;
            unknown[0][5] = t126;
            unknown[1][0] = t53;
            unknown[1][1] = 0.2e1 * t4 * t130 * t25 + 0.2e1 * t17 * t19 * t15 - t136 + t38;
            unknown[1][2] = t159;
            unknown[1][3] = t178;
            unknown[1][4] = t191;
            unknown[1][5] = t204;
            unknown[2][0] = t79;
            unknown[2][1] = t159;
            unknown[2][2] =
                    -0.2e1 * t1 * t209 * t25 + 0.2e1 * t205 * t19 * t15 + 0.2e1 * t2 * t212 * t62 - t2 * t216 * t37 +
                    t215 - t226 + t227 - t36 + t38;
            unknown[2][3] = t259;
            unknown[2][4] = t278;
            unknown[2][5] = t297;
            unknown[3][0] = t100;
            unknown[3][1] = t178;
            unknown[3][2] = t259;
            unknown[3][3] =
                    0.2e1 * t298 * t19 * t15 - 0.2e1 * t4 * t302 * t25 + 0.2e1 * t5 * t305 * t62 - t5 * t308 * t37 -
                    t136 + t215 + t227 - t315 + t38;
            unknown[3][4] = t338;
            unknown[3][5] = t357;
            unknown[4][0] = t113;
            unknown[4][1] = t191;
            unknown[4][2] = t278;
            unknown[4][3] = t338;
            unknown[4][4] = 0.2e1 * t11 * t19 * t15 - 0.2e1 * t2 * t361 * t62 - t226 + t227;
            unknown[4][5] = t380;
            unknown[5][0] = t126;
            unknown[5][1] = t204;
            unknown[5][2] = t297;
            unknown[5][3] = t357;
            unknown[5][4] = t380;
            unknown[5][5] = -0.2e1 * t5 * t4 * t63 * t62 + 0.2e1 * t12 * t19 * t15 + t227 - t315;

            for (int ii = 0; ii < n; ii++) {
                for (int jj = 0; jj < n; jj++) {
                    d2Edv2(indices(ii), indices(jj)) += w * unknown[ii][jj];
                }
            }
        }
    }

    MatrixXT energyHessian = dvdp.transpose() * d2Edv2 * dvdp;
    for (int i = 0; i < v.rows(); i++) {
        energyHessian += dEdv(i) * d2vdp2[i];
    }
    return energyHessian;
};

