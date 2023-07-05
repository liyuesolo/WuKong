#include "../../include/Energy/PerTriangleWeightedMeanY.h"

#define NINPUTS 9

// @formatter:off
void PerTriangleWeightedMeanY::getValue(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    value.value = ((-z1 * x0 + z0 * x1) * y2 + (z2 * x0 - x2 * z0) * y1 - y0 * (z2 * x1 - x2 * z1)) * (y0 + y1 + y2) / 0.24e2;
}

void PerTriangleWeightedMeanY::getGradient(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    double unknown[NINPUTS];

    double t4 = y0 + y1 + y2;
    double t9 = -z2 * x1 + x2 * z1;
    double t13 = -z1 * x0 + z0 * x1;
    double t14 = t13 * y2;
    double t17 = z2 * x0 - x2 * z0;
    double t18 = y1 * t17;
    double t19 = -t9 * y0;
    unknown[0] = t4 * (y1 * z2 - y2 * z1) / 0.24e2;
    unknown[1] = t4 * t9 / 0.24e2 + t14 / 0.24e2 + t18 / 0.24e2 - t19 / 0.24e2;
    unknown[2] = t4 * (y2 * x1 - x2 * y1) / 0.24e2;
    unknown[3] = t4 * (-y0 * z2 + y2 * z0) / 0.24e2;
    unknown[4] = t4 * t17 / 0.24e2 + t14 / 0.24e2 + t18 / 0.24e2 - t19 / 0.24e2;
    unknown[5] = t4 * (-y2 * x0 + x2 * y0) / 0.24e2;
    unknown[6] = t4 * (y0 * z1 - y1 * z0) / 0.24e2;
    unknown[7] = t4 * t13 / 0.24e2 + t14 / 0.24e2 + t18 / 0.24e2 - t19 / 0.24e2;
    unknown[8] = t4 * (y1 * x0 - y0 * x1) / 0.24e2;

    value.gradient = Eigen::Map<Eigen::VectorXd>(&unknown[0], NINPUTS);
}

void PerTriangleWeightedMeanY::getHessian(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    double unknown[NINPUTS][NINPUTS];

    double t1 = y1 * z2;
    double t2 = y2 * z1;
    double t3 = t1 - t2;
    double t4 = y0 + y1 + y2;
    double t5 = t4 * z2;
    double t6 = t5 + t1 - t2;
    double t8 = t4 * y2 / 0.24e2;
    double t9 = t4 * z1;
    double t10 = -t9 + t1 - t2;
    double t12 = t4 * y1 / 0.24e2;
    double t13 = x1 * z2;
    double t14 = x2 * z1;
    double t16 = x1 * y2;
    double t17 = x2 * y1;
    double t18 = t16 - t17;
    double t19 = z0 * y2;
    double t20 = y0 * z2;
    double t21 = -t5 + t19 - t20;
    double t22 = x0 * z2;
    double t23 = x2 * z0;
    double t24 = t22 - t13 - t23 + t14;
    double t25 = t4 * x2;
    double t26 = x0 * y2;
    double t27 = y0 * x2;
    double t28 = t25 - t26 + t27;
    double t29 = z0 * y1;
    double t30 = y0 * z1;
    double t31 = t9 - t29 + t30;
    double t32 = x0 * z1;
    double t33 = x1 * z0;
    double t34 = -t32 + t33 - t13 + t14;
    double t35 = t4 * x1;
    double t36 = x0 * y1;
    double t37 = y0 * x1;
    double t38 = -t35 + t36 - t37;
    double t39 = -t25 + t16 - t17;
    double t40 = t35 + t16 - t17;
    double t41 = -t20 + t19;
    double t42 = t4 * z0;
    double t43 = t42 - t20 + t19;
    double t45 = t4 * y0 / 0.24e2;
    double t47 = -t26 + t27;
    double t48 = -t42 - t29 + t30;
    double t49 = -t32 + t22 + t33 - t23;
    double t50 = t4 * x0;
    double t51 = t50 + t36 - t37;
    double t52 = -t50 - t26 + t27;
    double t53 = t30 - t29;
    double t55 = t36 - t37;
    unknown[0][0] = 0.0e0;
    unknown[0][1] = t3 / 0.24e2;
    unknown[0][2] = 0.0e0;
    unknown[0][3] = 0.0e0;
    unknown[0][4] = t6 / 0.24e2;
    unknown[0][5] = -t8;
    unknown[0][6] = 0.0e0;
    unknown[0][7] = t10 / 0.24e2;
    unknown[0][8] = t12;
    unknown[1][0] = t3 / 0.24e2;
    unknown[1][1] = -t13 / 0.12e2 + t14 / 0.12e2;
    unknown[1][2] = t18 / 0.24e2;
    unknown[1][3] = t21 / 0.24e2;
    unknown[1][4] = t24 / 0.24e2;
    unknown[1][5] = t28 / 0.24e2;
    unknown[1][6] = t31 / 0.24e2;
    unknown[1][7] = t34 / 0.24e2;
    unknown[1][8] = t38 / 0.24e2;
    unknown[2][0] = 0.0e0;
    unknown[2][1] = t18 / 0.24e2;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = t8;
    unknown[2][4] = t39 / 0.24e2;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = -t12;
    unknown[2][7] = t40 / 0.24e2;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = 0.0e0;
    unknown[3][1] = t21 / 0.24e2;
    unknown[3][2] = t8;
    unknown[3][3] = 0.0e0;
    unknown[3][4] = t41 / 0.24e2;
    unknown[3][5] = 0.0e0;
    unknown[3][6] = 0.0e0;
    unknown[3][7] = t43 / 0.24e2;
    unknown[3][8] = -t45;
    unknown[4][0] = t6 / 0.24e2;
    unknown[4][1] = t24 / 0.24e2;
    unknown[4][2] = t39 / 0.24e2;
    unknown[4][3] = t41 / 0.24e2;
    unknown[4][4] = t22 / 0.12e2 - t23 / 0.12e2;
    unknown[4][5] = t47 / 0.24e2;
    unknown[4][6] = t48 / 0.24e2;
    unknown[4][7] = t49 / 0.24e2;
    unknown[4][8] = t51 / 0.24e2;
    unknown[5][0] = -t8;
    unknown[5][1] = t28 / 0.24e2;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = 0.0e0;
    unknown[5][4] = t47 / 0.24e2;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = t45;
    unknown[5][7] = t52 / 0.24e2;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = 0.0e0;
    unknown[6][1] = t31 / 0.24e2;
    unknown[6][2] = -t12;
    unknown[6][3] = 0.0e0;
    unknown[6][4] = t48 / 0.24e2;
    unknown[6][5] = t45;
    unknown[6][6] = 0.0e0;
    unknown[6][7] = t53 / 0.24e2;
    unknown[6][8] = 0.0e0;
    unknown[7][0] = t10 / 0.24e2;
    unknown[7][1] = t34 / 0.24e2;
    unknown[7][2] = t40 / 0.24e2;
    unknown[7][3] = t43 / 0.24e2;
    unknown[7][4] = t49 / 0.24e2;
    unknown[7][5] = t52 / 0.24e2;
    unknown[7][6] = t53 / 0.24e2;
    unknown[7][7] = -t32 / 0.12e2 + t33 / 0.12e2;
    unknown[7][8] = t55 / 0.24e2;
    unknown[8][0] = t12;
    unknown[8][1] = t38 / 0.24e2;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = -t45;
    unknown[8][4] = t51 / 0.24e2;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = 0.0e0;
    unknown[8][7] = t55 / 0.24e2;
    unknown[8][8] = 0.0e0;

    value.hessian = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], NINPUTS, NINPUTS);
}
