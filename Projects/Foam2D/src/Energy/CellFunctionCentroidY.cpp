#include "../../include/Energy/CellFunctionCentroidY.h"
#include <iostream>

void
CellFunctionCentroidY::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                                const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, x1, y1, r;
    for (int i = 0; i < n_nodes; i++) {
        x0 = nodes(i * nx + 0);
        y0 = nodes(i * nx + 1);
        x1 = nodes(next(i) * nx + 0);
        y1 = nodes(next(i) * nx + 1);
        r = nodes(i * nx + 2);

        value += (x0 * y1 - x1 * y0) * (y0 + y1) / 6.0;
        if (fabs(r) > 1e-10) {
            value += (y0 / 0.2e1 + y1 / 0.2e1 -
                      r * sqrt(0.4e1 - (pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) * pow(r, -0.2e1)) * (x0 - x1) *
                      pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.2e1) * r * r *
                     (0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / r / 0.2e1) -
                      sin(0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / r / 0.2e1))) / 0.2e1 +
                     pow(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1), 0.3e1 / 0.2e1) * (x0 - x1) *
                     pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.12e2;
        }
    }
}

void CellFunctionCentroidY::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                        VectorXT &gradient_c,
                                        VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        ri = i * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        r = nodes(ri);

        t1 = y0 + y1;
        t2 = 0.1e1 / 0.6e1;
        gradient_x(x0i) += t2 * y1 * t1;
        gradient_x(y0i) += -t2 * (-x0 * y1 + (y0 + t1) * x1);
        gradient_x(x1i) += -t2 * y0 * t1;
        gradient_x(y1i) += t2 * ((y1 + t1) * x0 - x1 * y0);

        if (fabs(r) > 1e-10) {
            t1 = x0 - x1;
            t2 = -y1 + y0;
            t3 = pow(t1, 0.2e1);
            t4 = pow(t2, 0.2e1) + t3;
            t5 = 0.1e1 / r;
            t6 = pow(t5, 0.2e1);
            t7 = t4 * t6;
            t8 = 0.4e1 - t7;
            t9 = pow(t8, -0.1e1 / 0.2e1);
            t10 = pow(t4, -0.3e1 / 0.2e1);
            t11 = t4 * t10;
            t8 = t8 * t9;
            t12 = -t10 * t3 + t11;
            t13 = t5 * t9;
            t14 = pow(t4, -0.1e1 / 0.2e1);
            t15 = t4 * t14;
            t16 = 0.1e1 / 0.2e1;
            t5 = 0.2e1 * asin(t16 * t15 * t5);
            t17 = t5 - sin(t5);
            t18 = t8 * r;
            t19 = -t18 * t1 * t11 + y0 + y1;
            t5 = cos(t5) - 0.1e1;
            t20 = t13 * t14;
            t4 = pow(t4, 0.2e1) * t14;
            t14 = r * r;
            t21 = 0.1e1 / 0.4e1;
            t22 = t16 * t19 * t14;
            t3 = t4 * t12 / 0.12e2 - t21 * (-t15 * t3 * t11 + (t12 * t8 * r - t13 * t3 * t11) * t14 * t17) -
                 t22 * t20 * t1 * t5;
            t12 = t1 * t2;
            t13 = t12 * (t18 * t10 + t13 * t11);
            t18 = t12 * t15 * t11;
            t4 = t4 * t12 * t10 / 0.12e2;
            t2 = t22 * t20 * t2 * t5;
            gradient_x(x0i) += t3;
            gradient_x(y0i) += t21 * ((0.1e1 + t13) * t14 * t17 + t18) - t4 - t2;
            gradient_x(x1i) += -t3;
            gradient_x(y1i) += t21 * (-t18 + (0.1e1 - t13) * t14 * t17) + t2 + t4;
            gradient_x(ri) +=
                    t16 * t19 * r * (r * t15 * t5 * t6 * t9 + t17) - t21 * t1 * t11 * (t7 * t9 + t8) * t14 * t17;
        }
    }
}

void
CellFunctionCentroidY::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
                                  const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        ri = i * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        r = nodes(ri);

        t1 = y0 / 0.6e1;
        t2 = y1 / 0.3e1 + t1;
        t3 = y1 / 0.6e1;
        t4 = -y0 / 0.3e1 - t3;
        t5 = -x1 / 0.6e1 + x0 / 0.6e1;
        hess_xx(x0i, x0i) += 0;
        hess_xx(x0i, y0i) += t3;
        hess_xx(x0i, x1i) += 0;
        hess_xx(x0i, y1i) += t2;
        hess_xx(y0i, x0i) += t3;
        hess_xx(y0i, y0i) += -x1 / 0.3e1;
        hess_xx(y0i, x1i) += t4;
        hess_xx(y0i, y1i) += t5;
        hess_xx(x1i, x0i) += 0;
        hess_xx(x1i, y0i) += t4;
        hess_xx(x1i, x1i) += 0;
        hess_xx(x1i, y1i) += -t1;
        hess_xx(y1i, x0i) += t2;
        hess_xx(y1i, y0i) += t5;
        hess_xx(y1i, x1i) += -t1;
        hess_xx(y1i, y1i) += x0 / 0.3e1;

        if (fabs(r) > 1e-10) {
            t1 = x0 - x1;
            t2 = -y1 + y0;
            t3 = pow(t2, 0.2e1);
            t4 = pow(t1, 0.2e1);
            t5 = t1 * t4;
            t6 = t4 + t3;
            t7 = 0.1e1 / r;
            t8 = pow(t7, 0.2e1);
            t9 = pow(t8, 0.2e1);
            t10 = t7 * t8;
            t11 = t6 * t8;
            t12 = -t11 + 0.4e1;
            t13 = pow(t12, -0.3e1 / 0.2e1);
            t14 = pow(t6, -0.5e1 / 0.2e1);
            t15 = pow(t6, 0.2e1) * t14;
            t16 = t12 * t13;
            t17 = t6 * t14;
            t18 = pow(t12, 0.2e1) * t13;
            t19 = t8 * t13;
            t20 = t19 * t4;
            t21 = r * t18;
            t22 = t21 * t17;
            t23 = t7 * t15;
            t24 = t23 * (t20 + t16) + t22;
            t25 = t4 * t17;
            t26 = 0.1e1 / 0.2e1;
            t27 = 0.3e1 / 0.2e1 * t21;
            t28 = t1 * t24 * t26 + t1 * ((-t25 + t15) * t7 * t16 + t22) - t27 * t5 * t14;
            t29 = pow(t6, -0.3e1 / 0.2e1);
            t30 = pow(t6, 0.2e1);
            t31 = t30 * t29;
            t32 = 0.2e1;
            t33 = t32 * asin(t26 * t31 * t7);
            t34 = sin(t33);
            t35 = t33 - t34;
            t36 = t25 - t15;
            t37 = t23 * t16;
            t38 = t21 * t36 + t37 * t4;
            t39 = t6 * t29;
            t33 = cos(t33);
            t40 = t33 - 0.1e1;
            t41 = t39 * t7;
            t42 = t41 * t1 * t16 * t40;
            t21 = t21 * t1 * t15 - y0 - y1;
            t43 = t33 - 0.1e1;
            t33 = -t33 + 0.1e1;
            t44 = t39 * t33;
            t12 = 0.1e1 / t12;
            t45 = 0.1e1 / t6;
            t45 = 0.4e1 * t45 * t8;
            t20 = t32 * t7 * (t16 * (t43 * t29 * t4 + t44) + t44 * t20) + t45 * t4 * t12 * t34;
            t30 = t6 * t30 * t29;
            t46 = (-t39 * t4 - t31) * t15;
            t47 = t14 * t30;
            t48 = t47 * t4;
            t49 = r * r;
            t50 = t1 * (-t48 + t46);
            t51 = t21 * t49;
            t52 = t31 * t1;
            t53 = t52 * t36;
            t54 = 0.1e1 / 0.4e1;
            t55 = t30 * t17;
            t56 = t55 * t1;
            t57 = t38 * t49 * t42;
            t14 = t16 * t17 * t7 + t27 * t14;
            t27 = t4 * t2;
            t24 = t2 * t24 * t26 - t27 * t14;
            t41 = t41 * t2 * t16 * t40;
            t37 = t1 * (t37 + t22);
            t58 = t37 * t2;
            t59 = 0.1e1 + t58;
            t60 = t39 * t43;
            t61 = t60 * t19;
            t62 = -t32 * t7 * t2 * t1 * (t29 * t16 * t33 + t61) + t45 * t1 * t12 * t2 * t34;
            t46 = t2 * (t48 - t46);
            t48 = t38 * t41;
            t63 = t42 * t59;
            t27 = t27 * t31 * t17;
            t55 = t55 * t2 / 0.12e2;
            t64 = -t26 * (t49 * (-t24 * t35 + t48 + t63) + t27) + t54 * (-t51 * t62 + t46) - t55;
            t5 = t26 *
                 (t57 * t32 + t31 * t5 * t17 + (t14 * t5 - t26 * (t10 * t13 * t5 * t15 + 0.3e1 * t37)) * t49 * t35) +
                 t54 * (-0.3e1 * t52 * t15 - t5 * (t15 * t39 + t47) + t51 * t20 + t56);
            t37 = 0.1e1 - t58;
            t58 = t42 * t37;
            t65 = t26 * (t49 * (-t24 * t35 + t48 - t58) + t27) - t54 * (-t51 * t62 + t46) + t55;
            t11 = t11 * t16;
            t66 = t1 * t15;
            t67 = 0.4e1 * t10;
            t44 = -t32 * t8 * t1 * (t19 * t33 * t31 + t44 * t16) - t67 * t1 * t12 * t34;
            t68 = r * t31 * t8 * t16 * t40;
            t69 = t68 + t35;
            t70 = t66 * (t11 + t18) * r;
            t4 = (-t18 * t36 + t16 * (t15 * (t4 + t6) - t25 * t6) * t8 + t9 * t13 * t4 * t15 * t6) * t35;
            t25 = t21 * r;
            t36 = t25 * t42;
            t38 = t26 * r * (t38 * t69 + t70 * t42);
            t42 = -t54 * t49 * (t21 * t44 + t4) + t36 + t38;
            t14 = t26 * t1 * (t23 * (t19 * t3 + t16) + t22) - t1 * t3 * t14;
            t22 = -t32 * t7 * (t16 * (t29 * t33 * t3 + t60) + t61 * t3) + t45 * t3 * t12 * t34;
            t23 = t1 * (t15 * (t3 * t39 + t31) + t47 * t3);
            t3 = t52 * t17 * t3;
            t29 = t56 / 0.12e2;
            t33 = t26 * (t49 * (-t24 * t35 + t48 + t63) + t27) - t54 * (-t51 * t62 + t46) + t55;
            t39 = t26 * (t49 * (-t14 * t35 + t41 * (t59 - t37)) + t3) - t54 * (-t51 * t22 + t23) + t29;
            t45 = t19 * t6;
            t19 = t32 * t8 * t2 * (t19 * t43 * t31 + t60 * t16) - t67 * t2 * t12 * t34;
            t47 = t70 * t41;
            t1 = t1 * t2 * (t17 * (t11 + t18) - t8 * (t45 + t16) * t15) * t35;
            t2 = t25 * t41;
            t8 = t26 * r * (t59 * t69 + t47) - t54 * t49 * (t19 * t21 - t1) + t2;
            t11 = -t26 * (t49 * (-t24 * t35 + t48 - t58) + t27) + t54 * (-t51 * t62 + t46) - t55;
            t4 = -t54 * t49 * (-t21 * t44 - t4) - t36 - t38;
            t1 = t26 * r * (t37 * t69 - t47) - t54 * t49 * (-t19 * t21 + t1) - t2;
            hess_xx(x0i, x0i) += t26 * (t28 * t49 * t35 - t53) - (t51 * t20 + t50 + t56) * t54 - t57;
            hess_xx(x0i, y0i) += t64;
            hess_xx(x0i, x1i) += t5;
            hess_xx(x0i, y1i) += t65;
            hess_xx(x0i, ri) += t42;
            hess_xx(y0i, x0i) += t64;
            hess_xx(y0i, y0i) += -t26 * (-t14 * t49 * t35 + t3) + t54 * (-t51 * t22 + t23) - t29 - t59 * t49 * t41;
            hess_xx(y0i, x1i) += t33;
            hess_xx(y0i, y1i) += t39;
            hess_xx(y0i, ri) += t8;
            hess_xx(x1i, x0i) += t5;
            hess_xx(x1i, y0i) += t33;
            hess_xx(x1i, x1i) += t26 * (t28 * t49 * t35 - t53) - (t51 * t20 + t50 + t56) * t54 - t57;
            hess_xx(x1i, y1i) += t11;
            hess_xx(x1i, ri) += t4;
            hess_xx(y1i, x0i) += t65;
            hess_xx(y1i, y0i) += t39;
            hess_xx(y1i, x1i) += t11;
            hess_xx(y1i, y1i) += -t26 * (-t14 * t49 * t35 + t3) + t54 * (-t51 * t22 + t23) - t29 + t37 * t49 * t41;
            hess_xx(y1i, ri) += t1;
            hess_xx(ri, x0i) += t42;
            hess_xx(ri, y0i) += t8;
            hess_xx(ri, x1i) += t4;
            hess_xx(ri, y1i) += t1;
            hess_xx(ri, ri) += -t21 * (t26 * t35 + t68 * t32) - t54 * t49 * (t21 * (-t32 * t30 * t7 * t9 * t13 * t40 -
                                                                                    0.4e1 * t10 *
                                                                                    (-t6 * t7 * t12 * t34 +
                                                                                     t31 * t16 * t43)) -
                                                                             t35 * t66 * t6 * t10 * (t45 + t16)) -
                               t70 * (t68 + t35);
        }
    }
}
