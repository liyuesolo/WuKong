#include "../../include/Energy/CellFunctionArea.h"
#include <iostream>

void
CellFunctionArea::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                           double &value,
                           const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, x1, y1, q0, q1;
    for (int i = 0; i < n_nodes; i++) {
        x0 = nodes(i * nx + 0);
        y0 = nodes(i * nx + 1);
        q0 = nodes(i * nx + 2);
        x1 = nodes(next(i) * nx + 0);
        y1 = nodes(next(i) * nx + 1);
        q1 = nodes(next(i) * nx + 2);

        bool bad = true;

        value += 0.5 * (x0 * y1 - x1 * y0);
        if (btype(i) == 1) {
            value += q0 * q0 * (0.2e1 * asin(sqrt(pow(y1 - y0, 0.2e1) + pow(x1 - x0, 0.2e1)) / q0 / 0.2e1) -
                                sin(0.2e1 * asin(sqrt(pow(y1 - y0, 0.2e1) + pow(x1 - x0, 0.2e1)) / q0 / 0.2e1))) /
                     0.2e1;
        } else if (btype(i) == 2 && !bad) {
            value += ((-y0 + y1) * cos(q0) + sin(q0) * (x0 - x1)) * ((-y0 + y1) * cos(q1) + sin(q1) * (x0 - x1)) /
                     (0.3e1 * sin(q1) * cos(q0) - 0.3e1 * cos(q1) * sin(q0));
        } else if (btype(i) == 2 && bad) {
            value -= ((-y0 + y1) * cos(q0) + sin(q0) * (x0 - x1)) * ((-y0 + y1) * cos(q1) + sin(q1) * (x0 - x1)) /
                     (0.3e1 * sin(q1) * cos(q0) - 0.3e1 * cos(q1) * sin(q0));

//            std::cout << "Area " << x0 << " " << y0 << " " << q0 << " " << x1 << " " << y1 << " " << q1 << " "
//                      << ((-y0 + y1) * cos(q0) + sin(q0) * (x0 - x1)) * ((-y0 + y1) * cos(q1) + sin(q1) * (x0 - x1)) /
//                         (0.3e1 * sin(q1) * cos(q0) - 0.3e1 * cos(q1) * sin(q0)) << std::endl;
        }
    }
}

void
CellFunctionArea::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                              VectorXT &gradient_c,
                              VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t26, t32, t34, t37;
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

        bool bad = true;

        gradient_x(x0i) += 0.5 * y1;
        gradient_x(y0i) += -0.5 * x1;
        gradient_x(x1i) += -0.5 * y0;
        gradient_x(y1i) += 0.5 * x0;

        if (btype(i) == 1) {
            t1 = -y1 + y0;
            t2 = x1 - x0;
            t3 = pow(t1, 0.2e1) + pow(t2, 0.2e1);
            t4 = pow(t3, -0.1e1 / 0.2e1);
            t5 = 0.1e1 / q0;
            t6 = pow(t5, 0.2e1);
            t7 = -t3 * t6 + 0.4e1;
            t7 = pow(t7, -0.1e1 / 0.2e1);
            t3 = t3 * t4;
            t8 = 0.2e1 * asin(t3 * t5 / 0.2e1);
            t9 = cos(t8) - 0.1e1;
            t4 = t4 * t5;
            t5 = q0 * q0;
            t2 = t5 * t4 * t2 * t7 * t9;
            t1 = t5 * t4 * t1 * t7 * t9;
            gradient_x(x0i) += t2;
            gradient_x(y0i) += -t1;
            gradient_x(x1i) += -t2;
            gradient_x(y1i) += t1;
            gradient_x(q0i) += q0 * (q0 * t3 * t6 * t7 * t9 + t8 - sin(t8));
        } else if (btype(i) == 2 && !bad) {
            t1 = sin(q0);
            t2 = -y0 + y1;
            t3 = cos(q1);
            t5 = sin(q1);
            t6 = x0 - x1;
            t8 = t3 * t2 + t6 * t5;
            t10 = cos(q0);
            t13 = -t1 * t3 + t10 * t5;
            t14 = 0.1e1 / t13 / 0.3e1;
            t18 = t1 * t6 + t10 * t2;
            t21 = t14 * t8 * t1 + t14 * t5 * t18;
            t26 = -t14 * t8 * t10 - t14 * t3 * t18;
            t32 = t8 * t18;
            t34 = pow(t13, -0.2e1) / 0.9e1;
            t37 = -t1 * t5 - t10 * t3;
            gradient_x(x0i) += t21;
            gradient_x(y0i) += t26;
            gradient_x(q0i) += t14 * t8 * (-t2 * t1 + t6 * t10) - 0.3e1 * t37 * t34 * t32;
            gradient_x(x1i) += -t21;
            gradient_x(y1i) += -t26;
            gradient_x(q1i) += t14 * (-t5 * t2 + t6 * t3) * t18 + 0.3e1 * t37 * t34 * t32;
        } else if (btype(i) == 2 && bad) {
            t1 = sin(q0);
            t2 = -y0 + y1;
            t3 = cos(q1);
            t5 = sin(q1);
            t6 = x0 - x1;
            t8 = t3 * t2 + t6 * t5;
            t10 = cos(q0);
            t13 = -t1 * t3 + t10 * t5;
            t14 = 0.1e1 / t13 / 0.3e1;
            t18 = t1 * t6 + t10 * t2;
            t21 = t14 * t8 * t1 + t14 * t5 * t18;
            t26 = -t14 * t8 * t10 - t14 * t3 * t18;
            t32 = t8 * t18;
            t34 = pow(t13, -0.2e1) / 0.9e1;
            t37 = -t1 * t5 - t10 * t3;
            gradient_x(x0i) -= t21;
            gradient_x(y0i) -= t26;
            gradient_x(q0i) -= t14 * t8 * (-t2 * t1 + t6 * t10) - 0.3e1 * t37 * t34 * t32;
            gradient_x(x1i) -= -t21;
            gradient_x(y1i) -= -t26;
            gradient_x(q1i) -= t14 * (-t5 * t2 + t6 * t3) * t18 + 0.3e1 * t37 * t34 * t32;
        }
    }
}

void
CellFunctionArea::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                             MatrixXT &hessian,
                             const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100;
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

        bool bad = true;

        hess_xx(x0i, y1i) += 0.5;
        hess_xx(y0i, x1i) += -0.5;
        hess_xx(x1i, y0i) += -0.5;
        hess_xx(y1i, x0i) += 0.5;

        // @formatter:off
        if (btype(i) == 1) {
            t1 = -y1 + y0;
            t2 = x1 - x0;
            t3 = pow(t1, 0.2e1);
            t4 = pow(t2, 0.2e1);
            t5 = t3 + t4;
            t6 = pow(t5, -0.3e1 / 0.2e1);
            t7 = 0.1e1 / q0;
            t8 = pow(t7, 0.2e1);
            t9 = t7 * t8;
            t10 = -t5 * t8 + 0.4e1;
            t11 = pow(t10, -0.3e1 / 0.2e1);
            t12 = t10 * t11;
            t13 = t5 * t6;
            t14 = pow(t5, 0.2e1);
            t15 = t14 * t6;
            t16 = 0.2e1;
            t17 = t16 * asin(t15 * t7 / 0.2e1);
            t18 = cos(t17);
            t19 = sin(t17);
            t20 = -t18 + 0.1e1;
            t21 = t18 - 0.1e1;
            t22 = t13 * t21;
            t23 = t20 * t6;
            t10 = 0.1e1 / t10;
            t24 = 0.1e1 / t5;
            t24 = 0.4e1 * t24 * t8;
            t4 = -t16 * t7 * (t12 * (t23 * t4 + t22) + t22 * t4 * t8 * t11) + t24 * t4 * t10 * t19;
            t25 = t22 * t8 * t11;
            t26 = t7 * t2;
            t27 = t16 * t26 * t1 * (t23 * t12 + t25) - t24 * t2 * t10 * t1 * t19;
            t18 = -t18 + 0.1e1;
            t28 = t13 * t12;
            t20 = t20 * t8 * t15 * t11 + t28 * t20;
            t29 = 0.4e1 * t9;
            t2 = t29 * t2 * t10 * t19 + t16 * t2 * t20 * t8;
            t13 = t16 * t26 * t13 * t12 * t18;
            t26 = q0 / 0.2e1;
            t30 = q0 * (t26 * t2 - t13);
            t31 = q0 * q0 / 0.2e1;
            t32 = t31 * t27;
            t33 = -t31 * t4;
            t34 = -t31 * t27;
            t3 = -t16 * t7 * (t12 * (t23 * t3 + t22) + t25 * t3) + t24 * t3 * t10 * t19;
            t20 = -t29 * t1 * t10 * t19 - t1 * t16 * t20 * t8;
            t1 = t16 * t28 * t7 * t1 * t18;
            t22 = q0 * (t26 * t20 + t1);
            t23 = -t31 * t27;
            t24 = -t31 * t3;
            t2 = q0 * (-t26 * t2 + t13);
            t13 = t31 * t27;
            t1 = q0 * (-t26 * t20 - t1);
            hess_xx(x0i, x0i) += t31 * t4;
            hess_xx(x0i, y0i) += t32;
            hess_xx(x0i, x1i) += t33;
            hess_xx(x0i, y1i) += t34;
            hess_xx(x0i, q0i) += t30;
            hess_xx(y0i, x0i) += t32;
            hess_xx(y0i, y0i) += t31 * t3;
            hess_xx(y0i, x1i) += t23;
            hess_xx(y0i, y1i) += t24;
            hess_xx(y0i, q0i) += t22;
            hess_xx(x1i, x0i) += t33;
            hess_xx(x1i, y0i) += t23;
            hess_xx(x1i, x1i) += t31 * t4;
            hess_xx(x1i, y1i) += t13;
            hess_xx(x1i, q0i) += t2;
            hess_xx(y1i, x0i) += t34;
            hess_xx(y1i, y0i) += t24;
            hess_xx(y1i, x1i) += t13;
            hess_xx(y1i, y1i) += t31 * t3;
            hess_xx(y1i, q0i) += t1;
            hess_xx(q0i, x0i) += t30;
            hess_xx(q0i, y0i) += t22;
            hess_xx(q0i, x1i) += t2;
            hess_xx(q0i, y1i) += t1;
            hess_xx(q0i, q0i) += (-0.4e1 * t8 * t15 * t12 * t18 + t26 *
                                                                  (t16 * t5 * t14 * t6 * t7 * pow(t8, 0.2e1) * t11 *
                                                                   t18 -
                                                                   0.4e1 * t9 *
                                                                   (-t5 * t7 * t10 * t19 + t15 * t12 * t21))) * q0 +
                                 t17 -
                                 t19;
        } else if (btype(i) == 2 && !bad) {
            t1 = sin(q0);
            t2 = sin(q1);
            t3 = t2 * t1;
            t4 = cos(q0);
            t5 = t4 * t2;
            t6 = cos(q1);
            t7 = t1 * t6;
            t8 = t5 - t7;
            t9 = 0.1e1 / t8 / 0.3e1;
            t11 = 0.2e1 * t9 * t3;
            t14 = -t9 * t5 - t9 * t7;
            t15 = -y0 + y1;
            t17 = x0 - x1;
            t19 = t6 * t15 + t17 * t2;
            t20 = t19 * t4;
            t22 = t19 * t1;
            t23 = 0.9e1 * t8 * t8;
            t24 = 0.1e1 / t23;
            t25 = t4 * t6;
            t26 = -t3 - t25;
            t27 = 0.3e1 * t26 * t24;
            t31 = -t1 * t15 + t17 * t4;
            t36 = t17 * t1 + t4 * t15;
            t37 = t2 * t36;
            t39 = t9 * t2 * t31 + t9 * t20 - t27 * t22 - t27 * t37;
            t42 = -t2 * t15 + t17 * t6;
            t45 = -0.3e1 * t26 * t24;
            t47 = t6 * t36;
            t50 = t9 * t42 * t1 - t45 * t22 - t45 * t37 + t9 * t47;
            t52 = 0.2e1 * t9 * t25;
            t58 = -t9 * t6 * t31 + t27 * t20 + t9 * t22 + t27 * t47;
            t64 = -t9 * t42 * t4 + t45 * t20 + t9 * t37 + t45 * t47;
            t66 = -t9 * t19 * t36;
            t67 = t19 * t31;
            t70 = t19 * t36;
            t73 = 0.1e1 / t8 / t23 / 0.3e1;
            t77 = 0.18e2 * t26 * t26 * t73 * t70;
            t79 = -0.3e1 * t8 * t24 * t70;
            t84 = t42 * t36;
            t91 = -0.18e2 * t26 * t26 * t73 * t70 + t9 * t42 * t31 - t27 * t84 - t45 * t67 - t9 * t70;
            hess_xx(x0i, x0i) += t11;
            hess_xx(x0i, y0i) += t14;
            hess_xx(x0i, q0i) += t39;
            hess_xx(x0i, x1i) += -t11;
            hess_xx(x0i, y1i) += -t14;
            hess_xx(x0i, q1i) += t50;
            hess_xx(y0i, x0i) += t14;
            hess_xx(y0i, y0i) += t52;
            hess_xx(y0i, q0i) += t58;
            hess_xx(y0i, x1i) += -t14;
            hess_xx(y0i, y1i) += -t52;
            hess_xx(y0i, q1i) += t64;
            hess_xx(q0i, x0i) += t39;
            hess_xx(q0i, y0i) += t58;
            hess_xx(q0i, q0i) += -0.2e1 * t27 * t67 + t66 + t77 - t79;
            hess_xx(q0i, x1i) += -t39;
            hess_xx(q0i, y1i) += -t58;
            hess_xx(q0i, q1i) += t91;
            hess_xx(x1i, x0i) += -t11;
            hess_xx(x1i, y0i) += -t14;
            hess_xx(x1i, q0i) += -t39;
            hess_xx(x1i, x1i) += t11;
            hess_xx(x1i, y1i) += t14;
            hess_xx(x1i, q1i) += -t50;
            hess_xx(y1i, x0i) += -t14;
            hess_xx(y1i, y0i) += -t52;
            hess_xx(y1i, q0i) += -t58;
            hess_xx(y1i, x1i) += t14;
            hess_xx(y1i, y1i) += t52;
            hess_xx(y1i, q1i) += -t64;
            hess_xx(q1i, x0i) += t50;
            hess_xx(q1i, y0i) += t64;
            hess_xx(q1i, q0i) += t91;
            hess_xx(q1i, x1i) += -t50;
            hess_xx(q1i, y1i) += -t64;
            hess_xx(q1i, q1i) += -0.2e1 * t45 * t84 + t66 + t77 - t79;
        } else if (btype(i) == 2 && bad) {
            t1 = sin(q0);
            t2 = sin(q1);
            t3 = t2 * t1;
            t4 = cos(q0);
            t5 = t4 * t2;
            t6 = cos(q1);
            t7 = t1 * t6;
            t8 = t5 - t7;
            t9 = 0.1e1 / t8 / 0.3e1;
            t11 = 0.2e1 * t9 * t3;
            t14 = -t9 * t5 - t9 * t7;
            t15 = -y0 + y1;
            t17 = x0 - x1;
            t19 = t6 * t15 + t17 * t2;
            t20 = t19 * t4;
            t22 = t19 * t1;
            t23 = 0.9e1 * t8 * t8;
            t24 = 0.1e1 / t23;
            t25 = t4 * t6;
            t26 = -t3 - t25;
            t27 = 0.3e1 * t26 * t24;
            t31 = -t1 * t15 + t17 * t4;
            t36 = t17 * t1 + t4 * t15;
            t37 = t2 * t36;
            t39 = t9 * t2 * t31 + t9 * t20 - t27 * t22 - t27 * t37;
            t42 = -t2 * t15 + t17 * t6;
            t45 = -0.3e1 * t26 * t24;
            t47 = t6 * t36;
            t50 = t9 * t42 * t1 - t45 * t22 - t45 * t37 + t9 * t47;
            t52 = 0.2e1 * t9 * t25;
            t58 = -t9 * t6 * t31 + t27 * t20 + t9 * t22 + t27 * t47;
            t64 = -t9 * t42 * t4 + t45 * t20 + t9 * t37 + t45 * t47;
            t66 = -t9 * t19 * t36;
            t67 = t19 * t31;
            t70 = t19 * t36;
            t73 = 0.1e1 / t8 / t23 / 0.3e1;
            t77 = 0.18e2 * t26 * t26 * t73 * t70;
            t79 = -0.3e1 * t8 * t24 * t70;
            t84 = t42 * t36;
            t91 = -0.18e2 * t26 * t26 * t73 * t70 + t9 * t42 * t31 - t27 * t84 - t45 * t67 - t9 * t70;
            hess_xx(x0i, x0i) -= t11;
            hess_xx(x0i, y0i) -= t14;
            hess_xx(x0i, q0i) -= t39;
            hess_xx(x0i, x1i) -= -t11;
            hess_xx(x0i, y1i) -= -t14;
            hess_xx(x0i, q1i) -= t50;
            hess_xx(y0i, x0i) -= t14;
            hess_xx(y0i, y0i) -= t52;
            hess_xx(y0i, q0i) -= t58;
            hess_xx(y0i, x1i) -= -t14;
            hess_xx(y0i, y1i) -= -t52;
            hess_xx(y0i, q1i) -= t64;
            hess_xx(q0i, x0i) -= t39;
            hess_xx(q0i, y0i) -= t58;
            hess_xx(q0i, q0i) -= -0.2e1 * t27 * t67 + t66 + t77 - t79;
            hess_xx(q0i, x1i) -= -t39;
            hess_xx(q0i, y1i) -= -t58;
            hess_xx(q0i, q1i) -= t91;
            hess_xx(x1i, x0i) -= -t11;
            hess_xx(x1i, y0i) -= -t14;
            hess_xx(x1i, q0i) -= -t39;
            hess_xx(x1i, x1i) -= t11;
            hess_xx(x1i, y1i) -= t14;
            hess_xx(x1i, q1i) -= -t50;
            hess_xx(y1i, x0i) -= -t14;
            hess_xx(y1i, y0i) -= -t52;
            hess_xx(y1i, q0i) -= -t58;
            hess_xx(y1i, x1i) -= t14;
            hess_xx(y1i, y1i) -= t52;
            hess_xx(y1i, q1i) -= -t64;
            hess_xx(q1i, x0i) -= t50;
            hess_xx(q1i, y0i) -= t64;
            hess_xx(q1i, q0i) -= t91;
            hess_xx(q1i, x1i) -= -t50;
            hess_xx(q1i, y1i) -= -t64;
            hess_xx(q1i, q1i) -= -0.2e1 * t45 * t84 + t66 + t77 - t79;
        }
        // @formatter:on
    }
}
