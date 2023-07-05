#include "../../include/Energy/PerTriangleWeightedMeanX.h"

#define NINPUTS 9

// @formatter:off
void PerTriangleWeightedMeanX::getValue(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    value.value = ((y1 * z2 - z1 * y2) * x0 + (-y0 * z2 + z0 * y2) * x1 + x2 * (z1 * y0 - y1 * z0)) * (x0 + x1 + x2) / 0.24e2;
}

void PerTriangleWeightedMeanX::getGradient(TriangleValue &value) const {
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

    double t3 = y1 * z2 - z1 * y2;
    double t4 = x0 + x1 + x2;
    double t6 = x0 * t3;
    double t9 = -y0 * z2 + z0 * y2;
    double t10 = x1 * t9;
    double t13 = z1 * y0 - y1 * z0;
    double t14 = t13 * x2;
    unknown[0] = t4 * t3 / 0.24e2 + t10 / 0.24e2 + t14 / 0.24e2 + t6 / 0.24e2;
    unknown[1] = t4 * (-x1 * z2 + z1 * x2) / 0.24e2;
    unknown[2] = t4 * (x1 * y2 - y1 * x2) / 0.24e2;
    unknown[3] = t4 * t9 / 0.24e2 + t10 / 0.24e2 + t14 / 0.24e2 + t6 / 0.24e2;
    unknown[4] = t4 * (x0 * z2 - z0 * x2) / 0.24e2;
    unknown[5] = t4 * (-x0 * y2 + y0 * x2) / 0.24e2;
    unknown[6] = t4 * t13 / 0.24e2 + t10 / 0.24e2 + t14 / 0.24e2 + t6 / 0.24e2;
    unknown[7] = t4 * (-z1 * x0 + x1 * z0) / 0.24e2;
    unknown[8] = t4 * (y1 * x0 - x1 * y0) / 0.24e2;

    value.gradient = Eigen::Map<Eigen::VectorXd>(&unknown[0], NINPUTS);
}

void PerTriangleWeightedMeanX::getHessian(TriangleValue &value) const {
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
    double t4 = z2 * x1;
    double t5 = x2 * z1;
    double t6 = -t4 + t5;
    double t7 = y2 * x1;
    double t8 = x2 * y1;
    double t9 = t7 - t8;
    double t10 = y0 * z2;
    double t11 = y2 * z0;
    double t12 = -t10 + t1 + t11 - t2;
    double t13 = x0 + x1 + x2;
    double t14 = t13 * z2;
    double t15 = z2 * x0;
    double t16 = x2 * z0;
    double t17 = t14 + t15 - t16;
    double t18 = t13 * y2;
    double t19 = y2 * x0;
    double t20 = x2 * y0;
    double t21 = -t18 - t19 + t20;
    double t22 = y0 * z1;
    double t23 = y1 * z0;
    double t24 = t22 - t23 + t1 - t2;
    double t25 = t13 * z1;
    double t26 = z1 * x0;
    double t27 = z0 * x1;
    double t28 = -t25 - t26 + t27;
    double t29 = t13 * y1;
    double t30 = y1 * x0;
    double t31 = y0 * x1;
    double t32 = t29 + t30 - t31;
    double t33 = -t14 - t4 + t5;
    double t35 = t13 * x2 / 0.24e2;
    double t36 = t25 - t4 + t5;
    double t38 = t13 * x1 / 0.24e2;
    double t39 = t18 + t7 - t8;
    double t40 = -t29 + t7 - t8;
    double t42 = t15 - t16;
    double t43 = -t19 + t20;
    double t44 = t22 - t10 - t23 + t11;
    double t45 = t13 * z0;
    double t46 = t45 - t26 + t27;
    double t47 = t13 * y0;
    double t48 = -t47 + t30 - t31;
    double t49 = -t45 + t15 - t16;
    double t51 = t13 * x0 / 0.24e2;
    double t52 = t47 - t19 + t20;
    double t54 = -t26 + t27;
    double t55 = t30 - t31;
    unknown[0][0] = t1 / 0.12e2 - t2 / 0.12e2;
    unknown[0][1] = t6 / 0.24e2;
    unknown[0][2] = t9 / 0.24e2;
    unknown[0][3] = t12 / 0.24e2;
    unknown[0][4] = t17 / 0.24e2;
    unknown[0][5] = t21 / 0.24e2;
    unknown[0][6] = t24 / 0.24e2;
    unknown[0][7] = t28 / 0.24e2;
    unknown[0][8] = t32 / 0.24e2;
    unknown[1][0] = t6 / 0.24e2;
    unknown[1][1] = 0.0e0;
    unknown[1][2] = 0.0e0;
    unknown[1][3] = t33 / 0.24e2;
    unknown[1][4] = 0.0e0;
    unknown[1][5] = t35;
    unknown[1][6] = t36 / 0.24e2;
    unknown[1][7] = 0.0e0;
    unknown[1][8] = -t38;
    unknown[2][0] = t9 / 0.24e2;
    unknown[2][1] = 0.0e0;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = t39 / 0.24e2;
    unknown[2][4] = -t35;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = t40 / 0.24e2;
    unknown[2][7] = t38;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = t12 / 0.24e2;
    unknown[3][1] = t33 / 0.24e2;
    unknown[3][2] = t39 / 0.24e2;
    unknown[3][3] = -t10 / 0.12e2 + t11 / 0.12e2;
    unknown[3][4] = t42 / 0.24e2;
    unknown[3][5] = t43 / 0.24e2;
    unknown[3][6] = t44 / 0.24e2;
    unknown[3][7] = t46 / 0.24e2;
    unknown[3][8] = t48 / 0.24e2;
    unknown[4][0] = t17 / 0.24e2;
    unknown[4][1] = 0.0e0;
    unknown[4][2] = -t35;
    unknown[4][3] = t42 / 0.24e2;
    unknown[4][4] = 0.0e0;
    unknown[4][5] = 0.0e0;
    unknown[4][6] = t49 / 0.24e2;
    unknown[4][7] = 0.0e0;
    unknown[4][8] = t51;
    unknown[5][0] = t21 / 0.24e2;
    unknown[5][1] = t35;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = t43 / 0.24e2;
    unknown[5][4] = 0.0e0;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = t52 / 0.24e2;
    unknown[5][7] = -t51;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = t24 / 0.24e2;
    unknown[6][1] = t36 / 0.24e2;
    unknown[6][2] = t40 / 0.24e2;
    unknown[6][3] = t44 / 0.24e2;
    unknown[6][4] = t49 / 0.24e2;
    unknown[6][5] = t52 / 0.24e2;
    unknown[6][6] = t22 / 0.12e2 - t23 / 0.12e2;
    unknown[6][7] = t54 / 0.24e2;
    unknown[6][8] = t55 / 0.24e2;
    unknown[7][0] = t28 / 0.24e2;
    unknown[7][1] = 0.0e0;
    unknown[7][2] = t38;
    unknown[7][3] = t46 / 0.24e2;
    unknown[7][4] = 0.0e0;
    unknown[7][5] = -t51;
    unknown[7][6] = t54 / 0.24e2;
    unknown[7][7] = 0.0e0;
    unknown[7][8] = 0.0e0;
    unknown[8][0] = t32 / 0.24e2;
    unknown[8][1] = -t38;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = t48 / 0.24e2;
    unknown[8][4] = t51;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = t55 / 0.24e2;
    unknown[8][7] = 0.0e0;
    unknown[8][8] = 0.0e0;

    value.hessian = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], NINPUTS, NINPUTS);
}
