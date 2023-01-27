#include "../../include/ImageMatch/CellFunctionImageMatch2.h"
#include <iostream>

static void
get_closest_idx(const double &xp, const double &yp, const VectorXT &nodes, const VectorXi &next, int &closest_idx,
                bool &is_edge) {
    int n_nodes = nodes.rows() / CellFunction::nx;

    double mindistsq = 1e10;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * CellFunction::nx + 0;
        y0i = i * CellFunction::nx + 1;
        x1i = next(i) * CellFunction::nx + 0;
        y1i = next(i) * CellFunction::nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        double dsq = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
        double r = (xp - x0) * (x1 - x0) + (yp - y0) * (y1 - y0);
        r /= dsq;

        double distsq;
        if (r < 0) {
            distsq = (xp - x0) * (xp - x0) + (yp - y0) * (yp - y0);
        } else if (r > 1) {
            distsq = (xp - x1) * (xp - x1) + (yp - y1) * (yp - y1);
        } else {
            distsq = pow((xp - x0) * (y1 - y0) - (x1 - x0) * (yp - y0), 2) / dsq;
        }

        if (distsq < mindistsq) {
            mindistsq = distsq;
            if (r < 0) {
                closest_idx = i;
                is_edge = false;
            } else if (r > 1) {
                closest_idx = next(i);
                is_edge = false;
            } else {
                closest_idx = i;
                is_edge = true;
            }
        }
    }
}

void CellFunctionImageMatch2::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                                       const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        int i;
        bool is_edge;
        get_closest_idx(xp, yp, nodes, next, i, is_edge);

        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        if (!is_edge) {
            // Squared distance to point
            value += (xp - x0) * (xp - x0) + (yp - y0) * (yp - y0);
        } else {
            // Squared distance to edge
            value += pow((xp - x0) * (y1 - y0) - (x1 - x0) * (yp - y0), 2) /
                     ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        }
    }
}

void CellFunctionImageMatch2::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                          VectorXT &gradient_c,
                                          VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    double t1, t2, t3, t4, t5;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        int i;
        bool is_edge;
        get_closest_idx(xp, yp, nodes, next, i, is_edge);

        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        if (!is_edge) {
            t1 = -0.2e1;
            gradient_x(x0i) += t1 * (xp - x0);
            gradient_x(y0i) += t1 * (yp - y0);

        } else {
            t1 = xp - x0;
            t2 = -y1 + y0;
            t3 = -x1 + x0;
            t4 = yp - y0;
            t5 = pow(t2, 0.2e1) + pow(t3, 0.2e1);
            t5 = 0.1e1 / t5;
            t5 = (t1 * t2 - t3 * t4) * t5;
            t3 = t5 * t3;
            t2 = t5 * t2;
            gradient_x(x0i) += -0.2e1 * t5 * (-y1 + yp + t3);
            gradient_x(y0i) += 0.2e1 * t5 * (xp - x1 - t2);
            gradient_x(x1i) += 0.2e1 * t5 * (t3 + t4);
            gradient_x(y1i) += 0.2e1 * t5 * (t2 - t1);
        }
    }
}

void CellFunctionImageMatch2::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                         MatrixXT &hessian,
                                         const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        int i;
        bool is_edge;
        get_closest_idx(xp, yp, nodes, next, i, is_edge);

        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        if (!is_edge) {
            t1 = -y1 + yp;
            t2 = -x1 + x0;
            t3 = -y1 + y0;
            t4 = pow(t2, 0.2e1) + pow(t3, 0.2e1);
            t5 = -t2 * (yp - y0) + t3 * (xp - x0);
            t4 = 0.1e1 / t4;
            t6 = pow(t5, 0.2e1);
            t7 = t6 * t4;
            t8 = pow(t4, 0.2e1);
            t9 = t5 * t4;
            t5 = t5 * t8;
            t10 = xp - x1;
            t6 = t4 * (0.8e1 * t2 * t3 * t6 * t8 - 0.2e1 * t1 * t10) + 0.4e1 * t5 * (t1 * t3 - t10 * t2);
            hess_xx(x0i, x0i) += 0.2e1 * t4 * (pow(t1, 0.2e1) - t7) + 0.8e1 * t5 * t2 * (t9 * t2 + t1);
            hess_xx(x0i, y0i) += t6;
            hess_xx(y0i, x0i) += t6;
            hess_xx(y0i, y0i) += -0.2e1 * t4 * (-pow(t10, 0.2e1) + t7) - 0.8e1 * t5 * t3 * (-t9 * t3 + t10);
        } else {
            t1 = -y1 + yp;
            t2 = -x1 + x0;
            t3 = -y1 + y0;
            t4 = pow(t3, 0.2e1);
            t5 = pow(t2, 0.2e1);
            t6 = t4 + t5;
            t7 = xp - x0;
            t8 = yp - y0;
            t9 = -t2 * t8 + t3 * t7;
            t6 = 0.1e1 / t6;
            t10 = pow(t6, 0.2e1);
            t11 = t6 * t10;
            t12 = t9 * t6;
            t13 = t12 * t2;
            t14 = t9 * t10;
            t15 = t14 * t2;
            t16 = pow(t9, 0.2e1);
            t17 = t16 * t6;
            t18 = xp - x1;
            t19 = t1 * t3;
            t20 = t2 * t18;
            t21 = 0.4e1;
            t22 = 0.8e1 * t2 * t3 * t16;
            t10 = t22 * t10;
            t23 = t6 * (-0.2e1 * t1 * t18 + t10) + t21 * t14 * (t19 - t20);
            t16 = 0.8e1 * t16 * t11;
            t5 = -t21 * t15 * (t1 + t8) - 0.2e1 * t6 * (t1 * t8 - t17) - t16 * t5;
            t2 = t2 * t7;
            t11 = t22 * t11;
            t19 = -t21 * t14 * (t19 - t2) + 0.2e1 * t6 * (t1 * t7 + t9) - t11;
            t12 = t12 * t3;
            t22 = t14 * t3;
            t3 = t3 * t8;
            t9 = t21 * t14 * (-t3 + t20) - t11 - 0.2e1 * t6 * (-t18 * t8 + t9);
            t4 = t21 * t22 * (t7 + t18) - 0.2e1 * t6 * (t18 * t7 - t17) - t16 * t4;
            t2 = -t21 * t14 * (-t3 + t2) + t6 * (-0.2e1 * t7 * t8 + t10);
            hess_xx(x0i, x0i) += 0.8e1 * t15 * (t13 + t1) + 0.2e1 * t6 * (pow(t1, 0.2e1) - t17);
            hess_xx(x0i, y0i) += t23;
            hess_xx(x0i, x1i) += t5;
            hess_xx(x0i, y1i) += t19;
            hess_xx(y0i, x0i) += t23;
            hess_xx(y0i, y0i) += 0.2e1 * t6 * (pow(t18, 0.2e1) - t17) + 0.8e1 * t22 * (t12 - t18);
            hess_xx(y0i, x1i) += t9;
            hess_xx(y0i, y1i) += t4;
            hess_xx(x1i, x0i) += t5;
            hess_xx(x1i, y0i) += t9;
            hess_xx(x1i, x1i) += 0.2e1 * t6 * (pow(t8, 0.2e1) - t17) + 0.8e1 * t15 * (t13 + t8);
            hess_xx(x1i, y1i) += t2;
            hess_xx(y1i, x0i) += t19;
            hess_xx(y1i, y0i) += t4;
            hess_xx(y1i, x1i) += t2;
            hess_xx(y1i, y1i) += 0.2e1 * t6 * (pow(t7, 0.2e1) - t17) + 0.8e1 * t22 * (t12 - t7);
        }
    }
}
