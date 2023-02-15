#include "../../include/Energy/CellFunctionWeightedMeanX.h"
#include <iostream>

void
CellFunctionWeightedMeanX::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                    const VectorXi &btype, double &value,
                                    const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        value += (x0 * y1 - x1 * y0) * (x0 + x1) / 6.0;
        if (btype(i) == 1) {
            value += (x0 / 0.2e1 + x1 / 0.2e1 -
                      q0 * sqrt(0.4e1 - (pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) * pow(q0, -0.2e1)) * (y1 - y0) *
                      pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.2e1) * q0 * q0 *
                     (0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / q0 / 0.2e1) -
                      sin(0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / q0 / 0.2e1))) / 0.2e1 +
                     pow(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1), 0.3e1 / 0.2e1) * (y1 - y0) *
                     pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.12e2;
        }
    }
}

void CellFunctionWeightedMeanX::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                            const VectorXi &btype,
                                            VectorXT &gradient_c,
                                            VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        t1 = x0 + x1;
        t2 = 0.1e1 / 0.6e1;
        gradient_x(x0i) += t2 * (-x1 * y0 + (x0 + t1) * y1);
        gradient_x(y0i) += -t2 * x1 * t1;
        gradient_x(x1i) += -t2 * (-x0 * y1 + (x1 + t1) * y0);
        gradient_x(y1i) += t2 * x0 * t1;

        if (btype(i) == 1) {
            t1 = x0 - x1;
            t2 = -y1 + y0;
            t3 = pow(t2, 0.2e1);
            t4 = pow(t1, 0.2e1) + t3;
            t5 = 0.1e1 / q0;
            t6 = pow(t5, 0.2e1);
            t7 = t4 * t6;
            t8 = 0.4e1 - t7;
            t9 = pow(t8, -0.1e1 / 0.2e1);
            t10 = pow(t4, -0.3e1 / 0.2e1);
            t11 = t4 * t10;
            t8 = t8 * t9;
            t12 = t5 * t9;
            t13 = t12 * t11;
            t14 = q0 * t8;
            t15 = t2 * t1;
            t16 = t15 * (t14 * t10 + t13);
            t17 = pow(t4, -0.1e1 / 0.2e1);
            t18 = t4 * t17;
            t5 = 0.2e1 * asin(t18 * t5 / 0.2e1);
            t19 = -sin(t5) + t5;
            t20 = t14 * t2 * t11 + x0 + x1;
            t5 = -cos(t5) + 0.1e1;
            t12 = t12 * t17;
            t4 = pow(t4, 0.2e1) * t17;
            t17 = q0 * q0;
            t21 = t15 * t18 * t11;
            t22 = 0.1e1 / 0.4e1;
            t15 = t15 * t4 * t10 / 0.12e2;
            t23 = t20 * t17 / 0.2e1;
            t1 = t23 * t12 * t1 * t5;
            t10 = -t10 * t3 + t11;
            t3 = t22 * (-t18 * t3 * t11 + (t14 * t10 - t13 * t3) * t17 * t19) - t4 * t10 / 0.12e2 + t23 * t12 * t2 * t5;
            gradient_x(x0i) += t22 * ((0.1e1 - t16) * t17 * t19 - t21) + t15 + t1;
            gradient_x(y0i) += t3;
            gradient_x(x1i) += t22 * (t21 + (0.1e1 + t16) * t17 * t19) - t1 - t15;
            gradient_x(y1i) += -t3;
            gradient_x(q0i) +=
                    t20 * q0 * (-q0 * t18 * t5 * t6 * t9 + t19) / 0.2e1 + t22 * t2 * t11 * (t7 * t9 + t8) * t17 * t19;
        }
    }
}

void
CellFunctionWeightedMeanX::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                      const VectorXi &btype,
                                      MatrixXT &hessian,
                                      const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());
//    MatrixXT hess_xx = MatrixXT::Zero(nodes.rows(), nodes.rows());

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        t1 = 0.1e1 / 0.6e1;
        t2 = t1 * x1;
        t3 = x0 / 0.3e1 + t2;
        t4 = t1 * (y0 - y1);
        t1 = t1 * x0;
        t5 = -x1 / 0.3e1 - t1;
        hess_xx(x0i, x0i) += y1 / 0.3e1;
        hess_xx(x0i, y0i) += -t2;
        hess_xx(x0i, x1i) += -t4;
        hess_xx(x0i, y1i) += t3;
        hess_xx(y0i, x0i) += -t2;
        hess_xx(y0i, y0i) += 0;
        hess_xx(y0i, x1i) += t5;
        hess_xx(y0i, y1i) += 0;
        hess_xx(x1i, x0i) += -t4;
        hess_xx(x1i, y0i) += t5;
        hess_xx(x1i, x1i) += -y0 / 0.3e1;
        hess_xx(x1i, y1i) += t1;
        hess_xx(y1i, x0i) += t3;
        hess_xx(y1i, y0i) += 0;
        hess_xx(y1i, x1i) += t1;
        hess_xx(y1i, y1i) += 0;

        if (btype(i) == 1) {
            t1 = x0 - x1;
            t2 = -y1 + y0;
            t3 = pow(t2, 0.2e1);
            t4 = t2 * t3;
            t5 = pow(t1, 0.2e1);
            t6 = t3 + t5;
            t7 = 0.1e1 / q0;
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
            t20 = t19 * t5;
            t21 = q0 * t18;
            t22 = t21 * t17;
            t23 = t7 * t15;
            t24 = 0.1e1 / 0.2e1;
            t25 = 0.3e1 / 0.2e1 * t21 * t14;
            t26 = t16 * t17 * t7 + t25;
            t27 = t2 * t5;
            t28 = -t24 * t2 * (t23 * (t20 + t16) + t22) + t27 * t26;
            t29 = pow(t6, -0.3e1 / 0.2e1);
            t30 = pow(t6, 0.2e1);
            t31 = t30 * t29;
            t32 = 0.2e1;
            t33 = t32 * asin(t24 * t31 * t7);
            t34 = sin(t33);
            t35 = -t34 + t33;
            t36 = t23 * t16;
            t37 = t2 * (t36 + t22);
            t38 = t37 * t1;
            t39 = 0.1e1 - t38;
            t40 = t6 * t29;
            t33 = cos(t33);
            t41 = t33 - 0.1e1;
            t42 = t40 * t7;
            t43 = t42 * t1 * t16 * t41;
            t44 = t21 * t2 * t15 + x0 + x1;
            t45 = -t33 + 0.1e1;
            t33 = t33 - 0.1e1;
            t46 = t40 * t45;
            t47 = t33 * t29;
            t48 = 0.1e1 / t6;
            t12 = 0.1e1 / t12;
            t48 = 0.4e1 * t48 * t8;
            t20 = t32 * t7 * (t16 * (t47 * t5 + t46) + t46 * t20) + t48 * t5 * t12 * t34;
            t30 = t6 * t30 * t29;
            t49 = q0 * q0;
            t14 = t14 * t30;
            t50 = t14 * t5;
            t5 = (-t40 * t5 - t31) * t15;
            t51 = t44 * t49;
            t27 = t27 * t31 * t17;
            t52 = 0.1e1 / 0.4e1;
            t53 = t30 / 0.12e2;
            t54 = t53 * t2 * t17;
            t55 = t19 * t3;
            t23 = t23 * (t55 + t16) + t22;
            t56 = t3 * t1;
            t57 = -t1 * t23 * t24 + t56 * t26;
            t42 = -t42 * t2 * t16 * t41;
            t58 = t3 * t17;
            t59 = -t58 + t15;
            t21 = t21 * t59 - t36 * t3;
            t36 = t19 * t33;
            t29 = -t32 * t7 * t2 * t1 * (t29 * t16 * t45 + t36 * t40) + t48 * t1 * t12 * t2 * t34;
            t60 = t39 * t42;
            t61 = t43 * t21;
            t56 = t56 * t31 * t17;
            t62 = t14 * t3;
            t63 = (-t3 * t40 - t31) * t15;
            t53 = t53 * t17 * t1;
            t64 = -t24 * (t49 * (-t35 * t57 - t60 + t61) - t56) + t52 * (t1 * (-t62 + t63) + t51 * t29) + t53;
            t38 = 0.1e1 + t38;
            t65 = t2 * (t50 - t5);
            t66 = t24 * (t49 * (-t28 * t35 + t43 * (t39 - t38)) - t27) + t52 * (-t51 * t20 + t65) - t54;
            t62 = t62 - t63;
            t63 = t1 * t62;
            t60 = t24 * (t49 * (-t35 * t57 - t60 + t61) - t56) + t52 * (-t51 * t29 + t63) - t53;
            t11 = t11 * t16;
            t19 = t19 * t6;
            t67 = t2 * t15;
            t33 = t40 * t16 * t33 + t36 * t31;
            t36 = 0.4e1 * t10;
            t68 = -t36 * t1 * t12 * t34 + t1 * t32 * t33 * t8;
            t1 = t2 * t1 * (t17 * (-t11 - t18) + t8 * (t19 + t16) * t15) * t35;
            t69 = -q0 * t31 * t8 * t16 * t41;
            t70 = -t69 + t35;
            t11 = t67 * (t11 + t18) * q0;
            t71 = t11 * t43;
            t72 = t44 * q0;
            t73 = t72 * t43;
            t74 = t24 * q0 * (t39 * t70 - t71) + t52 * t49 * (t44 * t68 + t1) - t73;
            t22 = -t24 * t2 * t23 - t2 * ((-t58 + t15) * t7 * t16 + t22) + t25 * t4;
            t23 = t32 * t7 * (t16 * (t47 * t3 + t46) + t55 * t46) + t48 * t3 * t12 * t34;
            t25 = t31 * t2 * t59;
            t46 = t2 * t62;
            t47 = t30 * t2 * t17;
            t48 = t21 * t49 * t42;
            t55 = t42 * t38;
            t62 = t24 * (t49 * (-t35 * t57 + t55 + t61) - t56) + t52 * (-t51 * t29 + t63) - t53;
            t4 = -t24 *
                 (t48 * t32 + t31 * t4 * t17 - (t24 * (t10 * t13 * t4 * t15 + 0.3e1 * t37) - t26 * t4) * t49 * t35) +
                 t52 * (-t47 + 0.3e1 * t67 * t31 + t4 * (t15 * t40 + t14) - t51 * t23);
            t14 = -t36 * t2 * t12 * t34 + t2 * t32 * t33 * t8;
            t3 = (t18 * t59 + t16 * (t15 * (t3 + t6) - t58 * t6) * t8 + t9 * t13 * t3 * t15 * t6) * t35;
            t8 = t24 * q0 * (t11 * t42 + t21 * t70);
            t15 = t72 * t42;
            t17 = t52 * t49 * (t14 * t44 + t3) + t8 + t15;
            t18 = -t24 * (t49 * (-t35 * t57 + t55 + t61) - t56) - t52 * (-t51 * t29 + t63) + t53;
            t1 = t24 * q0 * (t38 * t70 + t71) - t52 * t49 * (t44 * t68 + t1) + t73;
            t3 = t52 * t49 * (-t14 * t44 - t3) - t15 - t8;
            hess_xx(x0i, x0i) +=
                    t24 * (t28 * t49 * t35 + t27) + t52 * (t2 * (-t50 + t5) + t51 * t20) + t54 - t39 * t49 * t43;
            hess_xx(x0i, y0i) += t64;
            hess_xx(x0i, x1i) += t66;
            hess_xx(x0i, y1i) += t60;
            hess_xx(x0i, q0i) += t74;
            hess_xx(y0i, x0i) += t64;
            hess_xx(y0i, y0i) += -t24 * (-t22 * t49 * t35 + t25) + t52 * (t51 * t23 - t46 + t47) + t48;
            hess_xx(y0i, x1i) += t62;
            hess_xx(y0i, y1i) += t4;
            hess_xx(y0i, q0i) += t17;
            hess_xx(x1i, x0i) += t66;
            hess_xx(x1i, y0i) += t62;
            hess_xx(x1i, x1i) += t24 * (t28 * t49 * t35 + t27) - t52 * (-t51 * t20 + t65) + t54 + t38 * t49 * t43;
            hess_xx(x1i, y1i) += t18;
            hess_xx(x1i, q0i) += t1;
            hess_xx(y1i, x0i) += t60;
            hess_xx(y1i, y0i) += t4;
            hess_xx(y1i, x1i) += t18;
            hess_xx(y1i, y1i) += -t24 * (-t22 * t49 * t35 + t25) + t52 * (t51 * t23 - t46 + t47) + t48;
            hess_xx(y1i, q0i) += t3;
            hess_xx(q0i, x0i) += t74;
            hess_xx(q0i, y0i) += t17;
            hess_xx(q0i, x1i) += t1;
            hess_xx(q0i, y1i) += t3;
            hess_xx(q0i, q0i) +=
                    t44 * (t24 * t35 - t69 * t32) + t52 * t49 * (-t35 * t67 * t6 * t10 * (t19 + t16) + t44 *
                                                                                                       (-t32 *
                                                                                                        t30 *
                                                                                                        t7 *
                                                                                                        t9 *
                                                                                                        t13 *
                                                                                                        t41 +
                                                                                                        0.4e1 *
                                                                                                        t10 *
                                                                                                        (t6 *
                                                                                                         t7 *
                                                                                                         t12 *
                                                                                                         t34 +
                                                                                                         t31 *
                                                                                                         t16 *
                                                                                                         t45))) +
                    t11 * (-t69 + t35);

        }
    }

//    VectorXT temp;
//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, temp, grad);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, temp, gradp);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "centroidx hess_xx(" << j << "," << i << "] " << (gradp[j] - gradient_x[j]) / eps << " "
//                      << hess_xx(j, i) << std::endl;
//        }
//    }
}
