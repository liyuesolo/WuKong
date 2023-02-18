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

        bool bad = true;

        value += (x0 * y1 - x1 * y0) * (x0 + x1) / 6.0;

        // @formatter:off
        if (btype(i) == 1) {
            value += (x0 / 0.2e1 + x1 / 0.2e1 -
                      q0 * sqrt(0.4e1 - (pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) * pow(q0, -0.2e1)) * (y1 - y0) *
                      pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.2e1) * q0 * q0 *
                     (0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / q0 / 0.2e1) -
                      sin(0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / q0 / 0.2e1))) / 0.2e1 +
                     pow(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1), 0.3e1 / 0.2e1) * (y1 - y0) *
                     pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.12e2;
        } else if (btype(i) == 2 && !bad) {
            value += (((y0 - y1) * (x0 - x1 + y0 - y1) * (x0 - x1 - y0 + y1) * pow(cos(q1), 0.2e1) - 0.3e1 * (pow(x0, 0.3e1) - 0.4e1 / 0.3e1 * x0 * x0 * x1 - (x1 + y0 - y1) * (x1 - y0 + y1) * x0 / 0.3e1 + 0.2e1 / 0.3e1 * (x1 * x1 + 0.2e1 * pow(y0 - y1, 0.2e1)) * x1) * sin(q1) * cos(q1) + 0.2e1 * (y0 - y1) * (x0 - x1) * (x0 + 0.3e1 / 0.2e1 * x1)) * pow(cos(q0), 0.2e1) - 0.2e1 * sin(q0) * ((-pow(x0, 0.3e1) + x0 * x0 * x1 / 0.2e1 + 0.2e1 * (x1 + y0 - y1) * (x1 - y0 + y1) * x0 - 0.3e1 / 0.2e1 * (x1 * x1 + pow(y0 - y1, 0.2e1) / 0.3e1) * x1) * pow(cos(q1), 0.2e1) + sin(q1) * pow(x0 - x1, 0.2e1) * (y0 - y1) * cos(q1) + pow(x0 - x1, 0.2e1) * (x0 + 0.3e1 / 0.2e1 * x1)) * cos(q0) + 0.3e1 * (x0 - x1) * ((-y0 + y1) * cos(q1) + sin(q1) * (x0 - x1)) * (x0 + 0.2e1 / 0.3e1 * x1) * cos(q1)) / ((0.30e2 * pow(cos(q1), 0.2e1) - 0.15e2) * pow(cos(q0), 0.2e1) + 0.30e2 * cos(q1) * cos(q0) * sin(q0) * sin(q1) - 0.15e2 * pow(cos(q1), 0.2e1));
        } else if (btype(i) == 2 && bad) {
            value -= (((y0 - y1) * (x0 - x1 + y0 - y1) * (x0 - x1 - y0 + y1) * pow(cos(q1), 0.2e1) - 0.3e1 * (pow(x0, 0.3e1) - 0.4e1 / 0.3e1 * x0 * x0 * x1 - (x1 + y0 - y1) * (x1 - y0 + y1) * x0 / 0.3e1 + 0.2e1 / 0.3e1 * (x1 * x1 + 0.2e1 * pow(y0 - y1, 0.2e1)) * x1) * sin(q1) * cos(q1) + 0.2e1 * (y0 - y1) * (x0 - x1) * (x0 + 0.3e1 / 0.2e1 * x1)) * pow(cos(q0), 0.2e1) - 0.2e1 * sin(q0) * ((-pow(x0, 0.3e1) + x0 * x0 * x1 / 0.2e1 + 0.2e1 * (x1 + y0 - y1) * (x1 - y0 + y1) * x0 - 0.3e1 / 0.2e1 * (x1 * x1 + pow(y0 - y1, 0.2e1) / 0.3e1) * x1) * pow(cos(q1), 0.2e1) + sin(q1) * pow(x0 - x1, 0.2e1) * (y0 - y1) * cos(q1) + pow(x0 - x1, 0.2e1) * (x0 + 0.3e1 / 0.2e1 * x1)) * cos(q0) + 0.3e1 * (x0 - x1) * ((-y0 + y1) * cos(q1) + sin(q1) * (x0 - x1)) * (x0 + 0.2e1 / 0.3e1 * x1) * cos(q1)) / ((0.30e2 * pow(cos(q1), 0.2e1) - 0.15e2) * pow(cos(q0), 0.2e1) + 0.30e2 * cos(q1) * cos(q0) * sin(q0) * sin(q1) - 0.15e2 * pow(cos(q1), 0.2e1));
        }
        // @formatter:on
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

        bool bad = true;

        t1 = x0 + x1;
        t2 = 0.1e1 / 0.6e1;
        gradient_x(x0i) += t2 * (-x1 * y0 + (x0 + t1) * y1);
        gradient_x(y0i) += -t2 * x1 * t1;
        gradient_x(x1i) += -t2 * (-x0 * y1 + (x1 + t1) * y0);
        gradient_x(y1i) += t2 * x0 * t1;

        // @formatter:off
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
        } else if (btype(i) == 2 && !bad) {
            t1 = y0 - y1;
            t2 = x0 - x1 - y0 + y1;
            t4 = cos(q1);
            t5 = t4 * t4;
            t6 = t5 * t2 * t1;
            t7 = x0 - x1 + y0 - y1;
            t8 = t7 * t1;
            t9 = t5 * t8;
            t10 = x0 * x0;
            t11 = 0.3e1 * t10;
            t12 = x0 * x1;
            t14 = x1 + y0 - y1;
            t15 = x1 - y0 + y1;
            t16 = t15 * t14;
            t19 = sin(q1);
            double t24 = x0 + 0.3e1 / 0.2e1 * x1;
            double t26 = 0.2e1 * t24 * t1;
            double t27 = x0 - x1;
            double t28 = t27 * t1;
            double t31 = cos(q0);
            double t32 = t31 * t31;
            double t34 = sin(q0);
            double t38 = t27 * t19;
            double t39 = t4 * t1;
            double t41 = 0.2e1 * t39 * t38;
            double t43 = 0.2e1 * t24 * t27;
            double t44 = t27 * t27;
            double t50 = -t4 * t1 + t38;
            double t52 = x0 + 0.2e1 / 0.3e1 * x1;
            double t55 = 0.3e1 * t4 * t52 * t50;
            double t56 = t4 * t52;
            double t58 = 0.3e1 * t56 * t38;
            double t59 = t50 * t27;
            double t60 = t4 * t59;
            double t64 = 0.30e2 * t5 - 0.15e2;
            double t71 = 0.30e2 * t19 * t34 * t31 * t4 + t32 * t64 - 0.15e2 * t5;
            double t72 = 0.1e1 / t71;
            double t75 = t5 * t2 * t7;
            double t76 = x0 * t15;
            double t77 = t76 / 0.3e1;
            double t78 = x0 * t14;
            double t79 = t78 / 0.3e1;
            double t88 = 0.2e1 * t76;
            double t89 = 0.2e1 * t78;
            double t94 = t44 * t19;
            double t95 = t4 * t94;
            double t102 = 0.3e1 * t52 * t5 * t27;
            double t107 = x0 * t10;
            double t108 = x1 * t10;
            double t110 = x0 * t16;
            double t112 = x1 * x1;
            double t113 = t1 * t1;
            double t118 = t107 - 0.4e1 / 0.3e1 * t108 - t110 / 0.3e1 + 0.2e1 / 0.3e1 * x1 * (t112 + 0.2e1 * t113);
            double t124 = -0.3e1 * t4 * t19 * t118 + t5 * t2 * t8 + 0.2e1 * t24 * t28;
            double t133 = -t107 + t108 / 0.2e1 + 0.2e1 * t110 - 0.3e1 / 0.2e1 * x1 * (t112 + t113 / 0.3e1);
            double t137 = t5 * t133 + t24 * t44 + t39 * t94;
            double t139 = t34 * t34;
            double t150 = t71 * t71;
            double t152 = 0.1e1 / t150 * (-0.2e1 * t31 * t137 * t34 + t32 * t124 + 0.3e1 * t56 * t59);
            double t160 = t19 * t32 * t4;
            double t212 = t19 * t19;
            gradient_x(x0i) += t72 * (t32 * (t6 + t9 - 0.3e1 * t4 * t19 * (t11 - 0.8e1 / 0.3e1 * t12 - t16 / 0.3e1) + t26 + 0.2e1 * t28) - 0.2e1 * t31 * (t5 * (-t11 + t12 + 0.2e1 * t16) + t41 + t43 + t44) * t34 + t55 + t58 + 0.3e1 * t60);
            gradient_x(y0i) += t72 * (t32 * (t75 + t6 - t9 - 0.3e1 * t4 * t19 * (-t77 + t79 + 0.8e1 / 0.3e1 * x1 * t1) + t43) - 0.2e1 * t31 * (t5 * (-x1 * t1 + t88 - t89) + t95) * t34 - t102);
            gradient_x(q0i) += 0.2e1 * t72 * (-t34 * t31 * t124 + t137 * t139 - t137 * t32) - (-0.30e2 * t19 * t139 * t4 - 0.2e1 * t34 * t31 * t64 + 0.30e2 * t160) * t152;
            gradient_x(x1i) += t72 * (t32 * (-t6 - t9 - 0.3e1 * t4 * t19 * (-0.4e1 / 0.3e1 * t10 - t77 - t79 + 0.2e1 * t112 + 0.4e1 / 0.3e1 * t113) - t26 + 0.3e1 * t28) - 0.2e1 * t31 * (t5 * (t10 / 0.2e1 + t88 + t89 - 0.9e1 / 0.2e1 * t112 - t113 / 0.2e1) - t41 - t43 + 0.3e1 / 0.2e1 * t44) * t34 - t55 - t58 + 0.2e1 * t60);
            gradient_x(y1i) += t72 * (t32 * (-t75 - t6 + t9 - 0.3e1 * t4 * t19 * (t77 - t79 - 0.8e1 / 0.3e1 * x1 * t1) - t43) - 0.2e1 * t31 * (t5 * (x1 * t1 - t88 + t89) - t95) * t34 + t102);
            gradient_x(q1i) += t72 * (t32 * (-0.2e1 * t19 * t4 * t2 * t8 + 0.3e1 * t212 * t118 - 0.3e1 * t5 * t118) - 0.2e1 * t31 * (-t1 * t44 * t212 + t1 * t44 * t5 - 0.2e1 * t19 * t4 * t133) * t34 + 0.3e1 * t56 * (t19 * t1 + t27 * t4) * t27 - 0.3e1 * t19 * t52 * t59) - (-0.30e2 * t34 * t31 * t212 + 0.30e2 * t34 * t31 * t5 + 0.30e2 * t4 * t19 - 0.60e2 * t160) * t152;
        } else if (btype(i) == 2 && bad) {
            t1 = y0 - y1;
            t2 = x0 - x1 - y0 + y1;
            t4 = cos(q1);
            t5 = t4 * t4;
            t6 = t5 * t2 * t1;
            t7 = x0 - x1 + y0 - y1;
            t8 = t7 * t1;
            t9 = t5 * t8;
            t10 = x0 * x0;
            t11 = 0.3e1 * t10;
            t12 = x0 * x1;
            t14 = x1 + y0 - y1;
            t15 = x1 - y0 + y1;
            t16 = t15 * t14;
            t19 = sin(q1);
            double t24 = x0 + 0.3e1 / 0.2e1 * x1;
            double t26 = 0.2e1 * t24 * t1;
            double t27 = x0 - x1;
            double t28 = t27 * t1;
            double t31 = cos(q0);
            double t32 = t31 * t31;
            double t34 = sin(q0);
            double t38 = t27 * t19;
            double t39 = t4 * t1;
            double t41 = 0.2e1 * t39 * t38;
            double t43 = 0.2e1 * t24 * t27;
            double t44 = t27 * t27;
            double t50 = -t4 * t1 + t38;
            double t52 = x0 + 0.2e1 / 0.3e1 * x1;
            double t55 = 0.3e1 * t4 * t52 * t50;
            double t56 = t4 * t52;
            double t58 = 0.3e1 * t56 * t38;
            double t59 = t50 * t27;
            double t60 = t4 * t59;
            double t64 = 0.30e2 * t5 - 0.15e2;
            double t71 = 0.30e2 * t19 * t34 * t31 * t4 + t32 * t64 - 0.15e2 * t5;
            double t72 = 0.1e1 / t71;
            double t75 = t5 * t2 * t7;
            double t76 = x0 * t15;
            double t77 = t76 / 0.3e1;
            double t78 = x0 * t14;
            double t79 = t78 / 0.3e1;
            double t88 = 0.2e1 * t76;
            double t89 = 0.2e1 * t78;
            double t94 = t44 * t19;
            double t95 = t4 * t94;
            double t102 = 0.3e1 * t52 * t5 * t27;
            double t107 = x0 * t10;
            double t108 = x1 * t10;
            double t110 = x0 * t16;
            double t112 = x1 * x1;
            double t113 = t1 * t1;
            double t118 = t107 - 0.4e1 / 0.3e1 * t108 - t110 / 0.3e1 + 0.2e1 / 0.3e1 * x1 * (t112 + 0.2e1 * t113);
            double t124 = -0.3e1 * t4 * t19 * t118 + t5 * t2 * t8 + 0.2e1 * t24 * t28;
            double t133 = -t107 + t108 / 0.2e1 + 0.2e1 * t110 - 0.3e1 / 0.2e1 * x1 * (t112 + t113 / 0.3e1);
            double t137 = t5 * t133 + t24 * t44 + t39 * t94;
            double t139 = t34 * t34;
            double t150 = t71 * t71;
            double t152 = 0.1e1 / t150 * (-0.2e1 * t31 * t137 * t34 + t32 * t124 + 0.3e1 * t56 * t59);
            double t160 = t19 * t32 * t4;
            double t212 = t19 * t19;
            gradient_x(x0i) -= t72 * (t32 * (t6 + t9 - 0.3e1 * t4 * t19 * (t11 - 0.8e1 / 0.3e1 * t12 - t16 / 0.3e1) + t26 + 0.2e1 * t28) - 0.2e1 * t31 * (t5 * (-t11 + t12 + 0.2e1 * t16) + t41 + t43 + t44) * t34 + t55 + t58 + 0.3e1 * t60);
            gradient_x(y0i) -= t72 * (t32 * (t75 + t6 - t9 - 0.3e1 * t4 * t19 * (-t77 + t79 + 0.8e1 / 0.3e1 * x1 * t1) + t43) - 0.2e1 * t31 * (t5 * (-x1 * t1 + t88 - t89) + t95) * t34 - t102);
            gradient_x(q0i) -= 0.2e1 * t72 * (-t34 * t31 * t124 + t137 * t139 - t137 * t32) - (-0.30e2 * t19 * t139 * t4 - 0.2e1 * t34 * t31 * t64 + 0.30e2 * t160) * t152;
            gradient_x(x1i) -= t72 * (t32 * (-t6 - t9 - 0.3e1 * t4 * t19 * (-0.4e1 / 0.3e1 * t10 - t77 - t79 + 0.2e1 * t112 + 0.4e1 / 0.3e1 * t113) - t26 + 0.3e1 * t28) - 0.2e1 * t31 * (t5 * (t10 / 0.2e1 + t88 + t89 - 0.9e1 / 0.2e1 * t112 - t113 / 0.2e1) - t41 - t43 + 0.3e1 / 0.2e1 * t44) * t34 - t55 - t58 + 0.2e1 * t60);
            gradient_x(y1i) -= t72 * (t32 * (-t75 - t6 + t9 - 0.3e1 * t4 * t19 * (t77 - t79 - 0.8e1 / 0.3e1 * x1 * t1) - t43) - 0.2e1 * t31 * (t5 * (x1 * t1 - t88 + t89) - t95) * t34 + t102);
            gradient_x(q1i) -= t72 * (t32 * (-0.2e1 * t19 * t4 * t2 * t8 + 0.3e1 * t212 * t118 - 0.3e1 * t5 * t118) - 0.2e1 * t31 * (-t1 * t44 * t212 + t1 * t44 * t5 - 0.2e1 * t19 * t4 * t133) * t34 + 0.3e1 * t56 * (t19 * t1 + t27 * t4) * t27 - 0.3e1 * t19 * t52 * t59) - (-0.30e2 * t34 * t31 * t212 + 0.30e2 * t34 * t31 * t5 + 0.30e2 * t4 * t19 - 0.60e2 * t160) * t152;
        }
        // @formatter:on
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

        bool bad = true;

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

        // @formatter:off
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

        } else if (btype(i) == 2 && !bad) {
            t1 = y0 - y1;
            t2 = cos(q1);
            t3 = t2 * t2;
            t4 = t3 * t1;
            t5 = 0.2e1 * t4;
            t6 = 0.6e1 * x0;
            t7 = 0.8e1 / 0.3e1 * x1;
            t9 = sin(q1);
            t16 = cos(q0);
            t17 = t16 * t16;
            t19 = sin(q0);
            t24 = 0.2e1 * t2 * t1 * t9;
            t29 = 0.2e1 / 0.3e1 * x1;
            t30 = x0 + t29;
            t31 = t30 * t9;
            t33 = 0.6e1 * t2 * t31;
            t35 = x0 - x1;
            t36 = t35 * t9;
            t37 = -t2 * t1 + t36;
            t38 = t2 * t37;
            t40 = t2 * t36;
            t43 = 0.30e2 * t3;
            t44 = t43 - 0.15e2;
            t45 = t17 * t44;
            t48 = t9 * t19 * t16 * t2;
            t51 = t45 + 0.30e2 * t48 - 0.15e2 * t3;
            t52 = 0.1e1 / t51;
            t54 = x0 - x1 - y0 + y1;
            t55 = t3 * t54;
            t56 = x0 - x1 + y0 - y1;
            t57 = t3 * t56;
            t61 = 0.4e1 * x0;
            t65 = 0.2e1 * t40;
            t71 = 0.3e1 * t30 * t3;
            t72 = t3 * t35;
            t73 = 0.3e1 * t72;
            double t75 = t52 * (t17 * (-0.2e1 * t2 * t1 * t9 + t55 + t57 + t61 + x1) - 0.2e1 * t16 * (-0.4e1 * t3 * t1 + t65) * t19 - t71 - t73);
            double t76 = t54 * t1;
            double t77 = t3 * t76;
            double t78 = t56 * t1;
            double t79 = t3 * t78;
            double t80 = x0 * x0;
            double t81 = 0.3e1 * t80;
            double t82 = x0 * x1;
            double t84 = x1 + y0 - y1;
            double t85 = x1 - y0 + y1;
            double t86 = t85 * t84;
            double t88 = t81 - 0.8e1 / 0.3e1 * t82 - t86 / 0.3e1;
            double t93 = x0 + 0.3e1 / 0.2e1 * x1;
            double t95 = 0.2e1 * t93 * t1;
            double t96 = t35 * t1;
            double t98 = -0.3e1 * t2 * t9 * t88 + t77 + t79 + t95 + 0.2e1 * t96;
            double t102 = -t81 + t82 + 0.2e1 * t86;
            double t104 = t2 * t1;
            double t106 = 0.2e1 * t104 * t36;
            double t108 = 0.2e1 * t93 * t35;
            double t109 = t35 * t35;
            double t110 = t3 * t102 + t106 + t108 + t109;
            double t112 = t19 * t19;
            double t120 = t30 * t37;
            double t122 = 0.3e1 * t2 * t120;
            double t123 = t2 * t30;
            double t124 = t123 * t36;
            double t125 = 0.3e1 * t124;
            double t126 = t37 * t35;
            double t127 = t2 * t126;
            double t130 = t51 * t51;
            double t131 = 0.1e1 / t130;
            double t132 = t131 * (-0.2e1 * t16 * t110 * t19 + t17 * t98 + t122 + t125 + 0.3e1 * t127);
            double t140 = t9 * t17 * t2;
            double t142 = -0.30e2 * t9 * t112 * t2 - 0.2e1 * t19 * t16 * t44 + 0.30e2 * t140;
            double t144 = 0.2e1 * t52 * (-t19 * t16 * t98 + t110 * t112 - t110 * t17) - t142 * t132;
            double t152 = 0.4e1 * x1;
            double t160 = t52 * (t17 * (-t5 - 0.3e1 * t2 * t9 * (-0.8e1 / 0.3e1 * x0 - t29) + y0 - y1) - 0.2e1 * t16 * (t3 * (x0 + t152) - t24 - x0 - t152) * t19 - t33 - t38 - t40);
            double t172 = t52 * (t17 * (0.2e1 * t2 * t1 * t9 - t55 - t57 - t61 - x1) - 0.2e1 * t16 * (0.4e1 * t3 * t1 - t65) * t19 + t71 + t73);
            double t173 = t9 * t2;
            double t175 = 0.2e1 * t173 * t76;
            double t177 = 0.2e1 * t173 * t78;
            double t180 = t9 * t9;
            double t187 = t1 * t72;
            double t188 = t35 * t180;
            double t189 = t1 * t188;
            double t196 = t1 * t9 + t35 * t2;
            double t199 = 0.3e1 * t2 * t30 * t196;
            double t201 = 0.3e1 * t9 * t120;
            double t203 = 0.3e1 * t30 * t72;
            double t205 = 0.3e1 * t30 * t188;
            double t206 = t196 * t35;
            double t207 = t2 * t206;
            double t209 = t9 * t126;
            double t217 = t16 * t3;
            double t221 = -0.30e2 * t19 * t16 * t180 + 0.30e2 * t19 * t217 - 0.60e2 * t140 + 0.30e2 * t173;
            double t223 = t52 * (t17 * (0.3e1 * t180 * t88 - 0.3e1 * t3 * t88 - t175 - t177) - 0.4e1 * t16 * (-t9 * t2 * t102 + t187 - t189) * t19 + t199 - t201 + t203 - t205 + 0.3e1 * t207 - 0.3e1 * t209) - t221 * t132;
            double t224 = 0.2e1 * t55;
            double t225 = 0.2e1 * t57;
            double t226 = 0.2e1 / 0.3e1 * x0;
            double t227 = t226 + t7;
            double t233 = -t61 - x1;
            double t238 = t52 * (t17 * (-0.3e1 * t2 * t9 * t227 + t224 - t225 - t5) - 0.2e1 * t217 * t233 * t19);
            double t239 = t54 * t56;
            double t240 = t3 * t239;
            double t241 = x0 * t85;
            double t242 = t241 / 0.3e1;
            double t243 = x0 * t84;
            double t244 = t243 / 0.3e1;
            double t247 = -t242 + t244 + 0.8e1 / 0.3e1 * x1 * t1;
            double t251 = -0.3e1 * t2 * t9 * t247 + t108 + t240 + t77 - t79;
            double t254 = 0.2e1 * t241;
            double t255 = 0.2e1 * t243;
            double t258 = -x1 * t1 + t254 - t255;
            double t260 = t109 * t9;
            double t261 = t2 * t260;
            double t262 = t3 * t258 + t261;
            double t272 = t131 * (-0.2e1 * t16 * t262 * t19 + t17 * t251 - t203);
            double t274 = 0.2e1 * t52 * (-t19 * t16 * t251 + t262 * t112 - t262 * t17) - t142 * t272;
            double t278 = 0.6e1 * x1;
            double t286 = 0.2e1 * t72;
            double t288 = t52 * (t17 * (-0.8e1 * t2 * t1 * t9 - t278 - t55 - t57 + x0) - 0.2e1 * t16 * (-t3 * t1 - t65) * t19 + t71 - t286);
            double t298 = t52 * (t17 * (0.3e1 * t2 * t9 * t227 - t224 + t225 + t5) + 0.2e1 * t217 * t233 * t19);
            double t300 = 0.2e1 * t173 * t239;
            double t310 = t109 * t3;
            double t311 = t109 * t180;
            double t316 = 0.6e1 * t124;
            double t320 = t52 * (t17 * (0.3e1 * t180 * t247 - 0.3e1 * t3 * t247 - t175 + t177 - t300) - 0.2e1 * t16 * (-0.2e1 * t9 * t2 * t258 + t310 - t311) * t19 + t316) - t221 * t272;
            double t321 = t55 * t78;
            double t322 = x0 * t80;
            double t323 = x1 * t80;
            double t325 = x0 * t86;
            double t327 = x1 * x1;
            double t328 = t1 * t1;
            double t333 = t322 - 0.4e1 / 0.3e1 * t323 - t325 / 0.3e1 + 0.2e1 / 0.3e1 * x1 * (t327 + 0.2e1 * t328);
            double t335 = t2 * t9 * t333;
            double t339 = 0.2e1 * t93 * t96 + t321 - 0.3e1 * t335;
            double t342 = t17 * t339;
            double t350 = -t322 + t323 / 0.2e1 + 0.2e1 * t325 - 0.3e1 / 0.2e1 * x1 * (t327 + t328 / 0.3e1);
            double t351 = t3 * t350;
            double t352 = t104 * t260;
            double t354 = t93 * t109 + t351 + t352;
            double t356 = t16 * t354 * t19;
            double t365 = 0.2e1 * t131 * (-t19 * t16 * t339 + t354 * t112 - t354 * t17);
            double t370 = 0.3e1 * t123 * t126;
            double t371 = t342 - 0.2e1 * t356 + t370;
            double t374 = 0.1e1 / t51 / t130 * t371;
            double t375 = t142 * t142;
            double t378 = t131 * t371;
            double t382 = 0.120e3 * t48;
            double t389 = -0.4e1 / 0.3e1 * t80 - t242 - t244 + 0.2e1 * t327 + 0.4e1 / 0.3e1 * t328;
            double t394 = -0.3e1 * t2 * t9 * t389 - t77 - t79 - t95 + 0.3e1 * t96;
            double t400 = t80 / 0.2e1 + t254 + t255 - 0.9e1 / 0.2e1 * t327 - t328 / 0.2e1;
            double t403 = t3 * t400 - t106 - t108 + 0.3e1 / 0.2e1 * t109;
            double t414 = t131 * (-0.2e1 * t16 * t403 * t19 + t17 * t394 - t122 - t125 + 0.2e1 * t127);
            double t416 = 0.2e1 * t52 * (-t19 * t16 * t394 + t403 * t112 - t403 * t17) - t142 * t414;
            double t419 = t242 - t244 - 0.8e1 / 0.3e1 * x1 * t1;
            double t423 = -0.3e1 * t2 * t9 * t419 - t108 - t240 - t77 + t79;
            double t428 = x1 * t1 - t254 + t255;
            double t430 = t3 * t428 - t261;
            double t440 = t131 * (-0.2e1 * t16 * t430 * t19 + t17 * t423 + t203);
            double t442 = 0.2e1 * t52 * (-t19 * t16 * t423 + t430 * t112 - t430 * t17) - t142 * t440;
            double t451 = -0.2e1 * t9 * t2 * t54 * t78 + 0.3e1 * t180 * t333 - 0.3e1 * t3 * t333;
            double t459 = -0.2e1 * t9 * t2 * t350 + t1 * t310 - t1 * t311;
            double t474 = t131 * (-0.2e1 * t16 * t459 * t19 + 0.3e1 * t123 * t206 - 0.3e1 * t31 * t126 + t17 * t451);
            double t483 = t17 * t180;
            double t485 = t17 * t3;
            double t489 = 0.2e1 * t52 * (-t19 * t16 * t451 + t459 * t112 - t459 * t17) - t221 * t365 - t142 * t474 + 0.2e1 * t221 * t142 * t374 - (0.30e2 * t112 * t180 - 0.30e2 * t112 * t3 + t382 - 0.30e2 * t483 + 0.30e2 * t485) * t378;
            double t498 = 0.9e1 * x1;
            double t519 = t52 * (t17 * (0.8e1 * t2 * t1 * t9 + t278 + t55 + t57 - x0) - 0.2e1 * t16 * (t4 + t65) * t19 - t71 + t286);
            double t537 = t52 * (t17 * (0.3e1 * t180 * t389 - 0.3e1 * t3 * t389 + t175 + t177) - 0.4e1 * t16 * (-t9 * t2 * t400 - t187 + t189) * t19 - t199 + t201 - t203 + t205 + 0.2e1 * t207 - 0.2e1 * t209) - t221 * t414;
            double t554 = t52 * (t17 * (0.3e1 * t180 * t419 - 0.3e1 * t3 * t419 + t175 - t177 + t300) - 0.2e1 * t16 * (-0.2e1 * t9 * t2 * t428 - t310 + t311) * t19 - t316) - t221 * t440;
            double t579 = t221 * t221;
            hess_xx(x0i, x0i) += t52 * (t17 * (t5 - 0.3e1 * t2 * t9 * (t6 - t7) + 0.4e1 * y0 - 0.4e1 * y1) - 0.2e1 * t16 * (t3 * (-t6 + x1) + t24 + t6 - x1) * t19 + t33 + 0.6e1 * t38 + 0.6e1 * t40);
            hess_xx(x0i, y0i) += t75;
            hess_xx(x0i, q0i) += t144;
            hess_xx(x0i, x1i) += t160;
            hess_xx(x0i, y1i) += t172;
            hess_xx(x0i, q1i) += t223;
            hess_xx(y0i, x0i) += t75;
            hess_xx(y0i, y0i) += t238;
            hess_xx(y0i, q0i) += t274;
            hess_xx(y0i, x1i) += t288;
            hess_xx(y0i, y1i) += t298;
            hess_xx(y0i, q1i) += t320;
            hess_xx(q0i, x0i) += t144;
            hess_xx(q0i, y0i) += t274;
            hess_xx(q0i, q0i) += t52 * (0.2e1 * t112 * t339 - 0.2e1 * t342 + 0.8e1 * t356) - 0.2e1 * t142 * t365 + 0.2e1 * t375 * t374 - (0.2e1 * t112 * t44 - t382 - 0.2e1 * t45) * t378;
            hess_xx(q0i, x1i) += t416;
            hess_xx(q0i, y1i) += t442;
            hess_xx(q0i, q1i) += t489;
            hess_xx(x1i, x0i) += t160;
            hess_xx(x1i, y0i) += t288;
            hess_xx(x1i, q0i) += t416;
            hess_xx(x1i, x1i) += t52 * (t17 * (t5 - 0.3e1 * t2 * t9 * (-t226 + t152) - 0.6e1 * y0 + 0.6e1 * y1) - 0.2e1 * t16 * (t3 * (t61 - t498) + t24 - t61 + t498) * t19 + t33 - 0.4e1 * t38 - 0.4e1 * t40);
            hess_xx(x1i, y1i) += t519;
            hess_xx(x1i, q1i) += t537;
            hess_xx(y1i, x0i) += t172;
            hess_xx(y1i, y0i) += t298;
            hess_xx(y1i, q0i) += t442;
            hess_xx(y1i, x1i) += t519;
            hess_xx(y1i, y1i) += t238;
            hess_xx(y1i, q1i) += t554;
            hess_xx(q1i, x0i) += t223;
            hess_xx(q1i, y0i) += t320;
            hess_xx(q1i, q0i) += t489;
            hess_xx(q1i, x1i) += t537;
            hess_xx(q1i, y1i) += t554;
            hess_xx(q1i, q1i) += t52 * (t17 * (0.2e1 * t180 * t54 * t78 - 0.2e1 * t321 + 0.12e2 * t335) - 0.2e1 * t16 * (0.2e1 * t180 * t350 - 0.2e1 * t351 - 0.4e1 * t352) * t19 - 0.3e1 * t123 * t37 * t35 - 0.6e1 * t31 * t206 - t370) - 0.2e1 * t221 * t474 + 0.2e1 * t579 * t374 - (0.60e2 * t483 - 0.60e2 * t485 - t382 - 0.30e2 * t180 + t43) * t378;
        } else if (btype(i) == 2 && bad) {
            t1 = y0 - y1;
            t2 = cos(q1);
            t3 = t2 * t2;
            t4 = t3 * t1;
            t5 = 0.2e1 * t4;
            t6 = 0.6e1 * x0;
            t7 = 0.8e1 / 0.3e1 * x1;
            t9 = sin(q1);
            t16 = cos(q0);
            t17 = t16 * t16;
            t19 = sin(q0);
            t24 = 0.2e1 * t2 * t1 * t9;
            t29 = 0.2e1 / 0.3e1 * x1;
            t30 = x0 + t29;
            t31 = t30 * t9;
            t33 = 0.6e1 * t2 * t31;
            t35 = x0 - x1;
            t36 = t35 * t9;
            t37 = -t2 * t1 + t36;
            t38 = t2 * t37;
            t40 = t2 * t36;
            t43 = 0.30e2 * t3;
            t44 = t43 - 0.15e2;
            t45 = t17 * t44;
            t48 = t9 * t19 * t16 * t2;
            t51 = t45 + 0.30e2 * t48 - 0.15e2 * t3;
            t52 = 0.1e1 / t51;
            t54 = x0 - x1 - y0 + y1;
            t55 = t3 * t54;
            t56 = x0 - x1 + y0 - y1;
            t57 = t3 * t56;
            t61 = 0.4e1 * x0;
            t65 = 0.2e1 * t40;
            t71 = 0.3e1 * t30 * t3;
            t72 = t3 * t35;
            t73 = 0.3e1 * t72;
            double t75 = t52 * (t17 * (-0.2e1 * t2 * t1 * t9 + t55 + t57 + t61 + x1) - 0.2e1 * t16 * (-0.4e1 * t3 * t1 + t65) * t19 - t71 - t73);
            double t76 = t54 * t1;
            double t77 = t3 * t76;
            double t78 = t56 * t1;
            double t79 = t3 * t78;
            double t80 = x0 * x0;
            double t81 = 0.3e1 * t80;
            double t82 = x0 * x1;
            double t84 = x1 + y0 - y1;
            double t85 = x1 - y0 + y1;
            double t86 = t85 * t84;
            double t88 = t81 - 0.8e1 / 0.3e1 * t82 - t86 / 0.3e1;
            double t93 = x0 + 0.3e1 / 0.2e1 * x1;
            double t95 = 0.2e1 * t93 * t1;
            double t96 = t35 * t1;
            double t98 = -0.3e1 * t2 * t9 * t88 + t77 + t79 + t95 + 0.2e1 * t96;
            double t102 = -t81 + t82 + 0.2e1 * t86;
            double t104 = t2 * t1;
            double t106 = 0.2e1 * t104 * t36;
            double t108 = 0.2e1 * t93 * t35;
            double t109 = t35 * t35;
            double t110 = t3 * t102 + t106 + t108 + t109;
            double t112 = t19 * t19;
            double t120 = t30 * t37;
            double t122 = 0.3e1 * t2 * t120;
            double t123 = t2 * t30;
            double t124 = t123 * t36;
            double t125 = 0.3e1 * t124;
            double t126 = t37 * t35;
            double t127 = t2 * t126;
            double t130 = t51 * t51;
            double t131 = 0.1e1 / t130;
            double t132 = t131 * (-0.2e1 * t16 * t110 * t19 + t17 * t98 + t122 + t125 + 0.3e1 * t127);
            double t140 = t9 * t17 * t2;
            double t142 = -0.30e2 * t9 * t112 * t2 - 0.2e1 * t19 * t16 * t44 + 0.30e2 * t140;
            double t144 = 0.2e1 * t52 * (-t19 * t16 * t98 + t110 * t112 - t110 * t17) - t142 * t132;
            double t152 = 0.4e1 * x1;
            double t160 = t52 * (t17 * (-t5 - 0.3e1 * t2 * t9 * (-0.8e1 / 0.3e1 * x0 - t29) + y0 - y1) - 0.2e1 * t16 * (t3 * (x0 + t152) - t24 - x0 - t152) * t19 - t33 - t38 - t40);
            double t172 = t52 * (t17 * (0.2e1 * t2 * t1 * t9 - t55 - t57 - t61 - x1) - 0.2e1 * t16 * (0.4e1 * t3 * t1 - t65) * t19 + t71 + t73);
            double t173 = t9 * t2;
            double t175 = 0.2e1 * t173 * t76;
            double t177 = 0.2e1 * t173 * t78;
            double t180 = t9 * t9;
            double t187 = t1 * t72;
            double t188 = t35 * t180;
            double t189 = t1 * t188;
            double t196 = t1 * t9 + t35 * t2;
            double t199 = 0.3e1 * t2 * t30 * t196;
            double t201 = 0.3e1 * t9 * t120;
            double t203 = 0.3e1 * t30 * t72;
            double t205 = 0.3e1 * t30 * t188;
            double t206 = t196 * t35;
            double t207 = t2 * t206;
            double t209 = t9 * t126;
            double t217 = t16 * t3;
            double t221 = -0.30e2 * t19 * t16 * t180 + 0.30e2 * t19 * t217 - 0.60e2 * t140 + 0.30e2 * t173;
            double t223 = t52 * (t17 * (0.3e1 * t180 * t88 - 0.3e1 * t3 * t88 - t175 - t177) - 0.4e1 * t16 * (-t9 * t2 * t102 + t187 - t189) * t19 + t199 - t201 + t203 - t205 + 0.3e1 * t207 - 0.3e1 * t209) - t221 * t132;
            double t224 = 0.2e1 * t55;
            double t225 = 0.2e1 * t57;
            double t226 = 0.2e1 / 0.3e1 * x0;
            double t227 = t226 + t7;
            double t233 = -t61 - x1;
            double t238 = t52 * (t17 * (-0.3e1 * t2 * t9 * t227 + t224 - t225 - t5) - 0.2e1 * t217 * t233 * t19);
            double t239 = t54 * t56;
            double t240 = t3 * t239;
            double t241 = x0 * t85;
            double t242 = t241 / 0.3e1;
            double t243 = x0 * t84;
            double t244 = t243 / 0.3e1;
            double t247 = -t242 + t244 + 0.8e1 / 0.3e1 * x1 * t1;
            double t251 = -0.3e1 * t2 * t9 * t247 + t108 + t240 + t77 - t79;
            double t254 = 0.2e1 * t241;
            double t255 = 0.2e1 * t243;
            double t258 = -x1 * t1 + t254 - t255;
            double t260 = t109 * t9;
            double t261 = t2 * t260;
            double t262 = t3 * t258 + t261;
            double t272 = t131 * (-0.2e1 * t16 * t262 * t19 + t17 * t251 - t203);
            double t274 = 0.2e1 * t52 * (-t19 * t16 * t251 + t262 * t112 - t262 * t17) - t142 * t272;
            double t278 = 0.6e1 * x1;
            double t286 = 0.2e1 * t72;
            double t288 = t52 * (t17 * (-0.8e1 * t2 * t1 * t9 - t278 - t55 - t57 + x0) - 0.2e1 * t16 * (-t3 * t1 - t65) * t19 + t71 - t286);
            double t298 = t52 * (t17 * (0.3e1 * t2 * t9 * t227 - t224 + t225 + t5) + 0.2e1 * t217 * t233 * t19);
            double t300 = 0.2e1 * t173 * t239;
            double t310 = t109 * t3;
            double t311 = t109 * t180;
            double t316 = 0.6e1 * t124;
            double t320 = t52 * (t17 * (0.3e1 * t180 * t247 - 0.3e1 * t3 * t247 - t175 + t177 - t300) - 0.2e1 * t16 * (-0.2e1 * t9 * t2 * t258 + t310 - t311) * t19 + t316) - t221 * t272;
            double t321 = t55 * t78;
            double t322 = x0 * t80;
            double t323 = x1 * t80;
            double t325 = x0 * t86;
            double t327 = x1 * x1;
            double t328 = t1 * t1;
            double t333 = t322 - 0.4e1 / 0.3e1 * t323 - t325 / 0.3e1 + 0.2e1 / 0.3e1 * x1 * (t327 + 0.2e1 * t328);
            double t335 = t2 * t9 * t333;
            double t339 = 0.2e1 * t93 * t96 + t321 - 0.3e1 * t335;
            double t342 = t17 * t339;
            double t350 = -t322 + t323 / 0.2e1 + 0.2e1 * t325 - 0.3e1 / 0.2e1 * x1 * (t327 + t328 / 0.3e1);
            double t351 = t3 * t350;
            double t352 = t104 * t260;
            double t354 = t93 * t109 + t351 + t352;
            double t356 = t16 * t354 * t19;
            double t365 = 0.2e1 * t131 * (-t19 * t16 * t339 + t354 * t112 - t354 * t17);
            double t370 = 0.3e1 * t123 * t126;
            double t371 = t342 - 0.2e1 * t356 + t370;
            double t374 = 0.1e1 / t51 / t130 * t371;
            double t375 = t142 * t142;
            double t378 = t131 * t371;
            double t382 = 0.120e3 * t48;
            double t389 = -0.4e1 / 0.3e1 * t80 - t242 - t244 + 0.2e1 * t327 + 0.4e1 / 0.3e1 * t328;
            double t394 = -0.3e1 * t2 * t9 * t389 - t77 - t79 - t95 + 0.3e1 * t96;
            double t400 = t80 / 0.2e1 + t254 + t255 - 0.9e1 / 0.2e1 * t327 - t328 / 0.2e1;
            double t403 = t3 * t400 - t106 - t108 + 0.3e1 / 0.2e1 * t109;
            double t414 = t131 * (-0.2e1 * t16 * t403 * t19 + t17 * t394 - t122 - t125 + 0.2e1 * t127);
            double t416 = 0.2e1 * t52 * (-t19 * t16 * t394 + t403 * t112 - t403 * t17) - t142 * t414;
            double t419 = t242 - t244 - 0.8e1 / 0.3e1 * x1 * t1;
            double t423 = -0.3e1 * t2 * t9 * t419 - t108 - t240 - t77 + t79;
            double t428 = x1 * t1 - t254 + t255;
            double t430 = t3 * t428 - t261;
            double t440 = t131 * (-0.2e1 * t16 * t430 * t19 + t17 * t423 + t203);
            double t442 = 0.2e1 * t52 * (-t19 * t16 * t423 + t430 * t112 - t430 * t17) - t142 * t440;
            double t451 = -0.2e1 * t9 * t2 * t54 * t78 + 0.3e1 * t180 * t333 - 0.3e1 * t3 * t333;
            double t459 = -0.2e1 * t9 * t2 * t350 + t1 * t310 - t1 * t311;
            double t474 = t131 * (-0.2e1 * t16 * t459 * t19 + 0.3e1 * t123 * t206 - 0.3e1 * t31 * t126 + t17 * t451);
            double t483 = t17 * t180;
            double t485 = t17 * t3;
            double t489 = 0.2e1 * t52 * (-t19 * t16 * t451 + t459 * t112 - t459 * t17) - t221 * t365 - t142 * t474 + 0.2e1 * t221 * t142 * t374 - (0.30e2 * t112 * t180 - 0.30e2 * t112 * t3 + t382 - 0.30e2 * t483 + 0.30e2 * t485) * t378;
            double t498 = 0.9e1 * x1;
            double t519 = t52 * (t17 * (0.8e1 * t2 * t1 * t9 + t278 + t55 + t57 - x0) - 0.2e1 * t16 * (t4 + t65) * t19 - t71 + t286);
            double t537 = t52 * (t17 * (0.3e1 * t180 * t389 - 0.3e1 * t3 * t389 + t175 + t177) - 0.4e1 * t16 * (-t9 * t2 * t400 - t187 + t189) * t19 - t199 + t201 - t203 + t205 + 0.2e1 * t207 - 0.2e1 * t209) - t221 * t414;
            double t554 = t52 * (t17 * (0.3e1 * t180 * t419 - 0.3e1 * t3 * t419 + t175 - t177 + t300) - 0.2e1 * t16 * (-0.2e1 * t9 * t2 * t428 - t310 + t311) * t19 - t316) - t221 * t440;
            double t579 = t221 * t221;
            hess_xx(x0i, x0i) -= t52 * (t17 * (t5 - 0.3e1 * t2 * t9 * (t6 - t7) + 0.4e1 * y0 - 0.4e1 * y1) - 0.2e1 * t16 * (t3 * (-t6 + x1) + t24 + t6 - x1) * t19 + t33 + 0.6e1 * t38 + 0.6e1 * t40);
            hess_xx(x0i, y0i) -= t75;
            hess_xx(x0i, q0i) -= t144;
            hess_xx(x0i, x1i) -= t160;
            hess_xx(x0i, y1i) -= t172;
            hess_xx(x0i, q1i) -= t223;
            hess_xx(y0i, x0i) -= t75;
            hess_xx(y0i, y0i) -= t238;
            hess_xx(y0i, q0i) -= t274;
            hess_xx(y0i, x1i) -= t288;
            hess_xx(y0i, y1i) -= t298;
            hess_xx(y0i, q1i) -= t320;
            hess_xx(q0i, x0i) -= t144;
            hess_xx(q0i, y0i) -= t274;
            hess_xx(q0i, q0i) -= t52 * (0.2e1 * t112 * t339 - 0.2e1 * t342 + 0.8e1 * t356) - 0.2e1 * t142 * t365 + 0.2e1 * t375 * t374 - (0.2e1 * t112 * t44 - t382 - 0.2e1 * t45) * t378;
            hess_xx(q0i, x1i) -= t416;
            hess_xx(q0i, y1i) -= t442;
            hess_xx(q0i, q1i) -= t489;
            hess_xx(x1i, x0i) -= t160;
            hess_xx(x1i, y0i) -= t288;
            hess_xx(x1i, q0i) -= t416;
            hess_xx(x1i, x1i) -= t52 * (t17 * (t5 - 0.3e1 * t2 * t9 * (-t226 + t152) - 0.6e1 * y0 + 0.6e1 * y1) - 0.2e1 * t16 * (t3 * (t61 - t498) + t24 - t61 + t498) * t19 + t33 - 0.4e1 * t38 - 0.4e1 * t40);
            hess_xx(x1i, y1i) -= t519;
            hess_xx(x1i, q1i) -= t537;
            hess_xx(y1i, x0i) -= t172;
            hess_xx(y1i, y0i) -= t298;
            hess_xx(y1i, q0i) -= t442;
            hess_xx(y1i, x1i) -= t519;
            hess_xx(y1i, y1i) -= t238;
            hess_xx(y1i, q1i) -= t554;
            hess_xx(q1i, x0i) -= t223;
            hess_xx(q1i, y0i) -= t320;
            hess_xx(q1i, q0i) -= t489;
            hess_xx(q1i, x1i) -= t537;
            hess_xx(q1i, y1i) -= t554;
            hess_xx(q1i, q1i) -= t52 * (t17 * (0.2e1 * t180 * t54 * t78 - 0.2e1 * t321 + 0.12e2 * t335) - 0.2e1 * t16 * (0.2e1 * t180 * t350 - 0.2e1 * t351 - 0.4e1 * t352) * t19 - 0.3e1 * t123 * t37 * t35 - 0.6e1 * t31 * t206 - t370) - 0.2e1 * t221 * t474 + 0.2e1 * t579 * t374 - (0.60e2 * t483 - 0.60e2 * t485 - t382 - 0.30e2 * t180 + t43) * t378;
        }
        // @formatter:on
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
