#include "../../include/Energy/PerTriangleWeightedMeanZ.h"

#define NINPUTS 9

// @formatter:off
void PerTriangleWeightedMeanZ::getValue(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    value.value = ((x1 * y2 - x2 * y1) * z0 + (-x0 * y2 + x2 * y0) * z1 + z2 * (x0 * y1 - x1 * y0)) * (z0 + z1 + z2) / 0.24e2;;
}

void PerTriangleWeightedMeanZ::getGradient(TriangleValue &value) const {
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

    double t4 = z0 + z1 + z2;
    double t14 = x1 * y2 - x2 * y1;
    double t16 = z0 * t14;
    double t19 = -x0 * y2 + x2 * y0;
    double t20 = z1 * t19;
    double t23 = x0 * y1 - x1 * y0;
    double t24 = t23 * z2;
    unknown[0] = t4 * (z2 * y1 - y2 * z1) / 0.24e2;
    unknown[1] = t4 * (-z2 * x1 + x2 * z1) / 0.24e2;
    unknown[2] = t4 * t14 / 0.24e2 + t16 / 0.24e2 + t20 / 0.24e2 + t24 / 0.24e2;
    unknown[3] = t4 * (-z2 * y0 + y2 * z0) / 0.24e2;
    unknown[4] = t4 * (z2 * x0 - x2 * z0) / 0.24e2;
    unknown[5] = t4 * t19 / 0.24e2 + t16 / 0.24e2 + t20 / 0.24e2 + t24 / 0.24e2;
    unknown[6] = t4 * (y0 * z1 - y1 * z0) / 0.24e2;
    unknown[7] = t4 * (-x0 * z1 + x1 * z0) / 0.24e2;
    unknown[8] = t4 * t23 / 0.24e2 + t16 / 0.24e2 + t20 / 0.24e2 + t24 / 0.24e2;

    value.gradient = Eigen::Map<Eigen::VectorXd>(&unknown[0], NINPUTS);
}

void PerTriangleWeightedMeanZ::getHessian(TriangleValue &value) const {
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

    double t1 = z2 * y1;
    double t2 = y2 * z1;
    double t3 = t1 - t2;
    double t4 = z0 + z1 + z2;
    double t6 = t4 * z2 / 0.24e2;
    double t7 = t4 * y2;
    double t8 = -t7 + t1 - t2;
    double t10 = t4 * z1 / 0.24e2;
    double t11 = t4 * y1;
    double t12 = t11 + t1 - t2;
    double t13 = z2 * x1;
    double t14 = x2 * z1;
    double t15 = -t13 + t14;
    double t16 = t4 * x2;
    double t17 = t16 - t13 + t14;
    double t18 = t4 * x1;
    double t19 = -t18 - t13 + t14;
    double t20 = x1 * y2;
    double t21 = x2 * y1;
    double t23 = y2 * z0;
    double t24 = z2 * y0;
    double t25 = t7 + t23 - t24;
    double t26 = x2 * z0;
    double t27 = z2 * x0;
    double t28 = -t16 - t26 + t27;
    double t29 = x0 * y2;
    double t30 = x2 * y0;
    double t31 = -t29 + t20 + t30 - t21;
    double t32 = y1 * z0;
    double t33 = y0 * z1;
    double t34 = -t11 - t32 + t33;
    double t35 = x1 * z0;
    double t36 = x0 * z1;
    double t37 = t18 + t35 - t36;
    double t38 = x0 * y1;
    double t39 = x1 * y0;
    double t40 = t38 - t39 + t20 - t21;
    double t41 = -t24 + t23;
    double t43 = t4 * z0 / 0.24e2;
    double t44 = t4 * y0;
    double t45 = -t44 - t24 + t23;
    double t46 = -t26 + t27;
    double t47 = t4 * x0;
    double t48 = t47 + t27 - t26;
    double t50 = t44 - t32 + t33;
    double t51 = -t47 + t35 - t36;
    double t52 = t38 - t29 - t39 + t30;
    double t53 = t33 - t32;
    double t54 = -t36 + t35;
    unknown[0][0] = 0.0e0;
    unknown[0][1] = 0.0e0;
    unknown[0][2] = t3 / 0.24e2;
    unknown[0][3] = 0.0e0;
    unknown[0][4] = t6;
    unknown[0][5] = t8 / 0.24e2;
    unknown[0][6] = 0.0e0;
    unknown[0][7] = -t10;
    unknown[0][8] = t12 / 0.24e2;
    unknown[1][0] = 0.0e0;
    unknown[1][1] = 0.0e0;
    unknown[1][2] = t15 / 0.24e2;
    unknown[1][3] = -t6;
    unknown[1][4] = 0.0e0;
    unknown[1][5] = t17 / 0.24e2;
    unknown[1][6] = t10;
    unknown[1][7] = 0.0e0;
    unknown[1][8] = t19 / 0.24e2;
    unknown[2][0] = t3 / 0.24e2;
    unknown[2][1] = t15 / 0.24e2;
    unknown[2][2] = t20 / 0.12e2 - t21 / 0.12e2;
    unknown[2][3] = t25 / 0.24e2;
    unknown[2][4] = t28 / 0.24e2;
    unknown[2][5] = t31 / 0.24e2;
    unknown[2][6] = t34 / 0.24e2;
    unknown[2][7] = t37 / 0.24e2;
    unknown[2][8] = t40 / 0.24e2;
    unknown[3][0] = 0.0e0;
    unknown[3][1] = -t6;
    unknown[3][2] = t25 / 0.24e2;
    unknown[3][3] = 0.0e0;
    unknown[3][4] = 0.0e0;
    unknown[3][5] = t41 / 0.24e2;
    unknown[3][6] = 0.0e0;
    unknown[3][7] = t43;
    unknown[3][8] = t45 / 0.24e2;
    unknown[4][0] = t6;
    unknown[4][1] = 0.0e0;
    unknown[4][2] = t28 / 0.24e2;
    unknown[4][3] = 0.0e0;
    unknown[4][4] = 0.0e0;
    unknown[4][5] = t46 / 0.24e2;
    unknown[4][6] = -t43;
    unknown[4][7] = 0.0e0;
    unknown[4][8] = t48 / 0.24e2;
    unknown[5][0] = t8 / 0.24e2;
    unknown[5][1] = t17 / 0.24e2;
    unknown[5][2] = t31 / 0.24e2;
    unknown[5][3] = t41 / 0.24e2;
    unknown[5][4] = t46 / 0.24e2;
    unknown[5][5] = -t29 / 0.12e2 + t30 / 0.12e2;
    unknown[5][6] = t50 / 0.24e2;
    unknown[5][7] = t51 / 0.24e2;
    unknown[5][8] = t52 / 0.24e2;
    unknown[6][0] = 0.0e0;
    unknown[6][1] = t10;
    unknown[6][2] = t34 / 0.24e2;
    unknown[6][3] = 0.0e0;
    unknown[6][4] = -t43;
    unknown[6][5] = t50 / 0.24e2;
    unknown[6][6] = 0.0e0;
    unknown[6][7] = 0.0e0;
    unknown[6][8] = t53 / 0.24e2;
    unknown[7][0] = -t10;
    unknown[7][1] = 0.0e0;
    unknown[7][2] = t37 / 0.24e2;
    unknown[7][3] = t43;
    unknown[7][4] = 0.0e0;
    unknown[7][5] = t51 / 0.24e2;
    unknown[7][6] = 0.0e0;
    unknown[7][7] = 0.0e0;
    unknown[7][8] = t54 / 0.24e2;
    unknown[8][0] = t12 / 0.24e2;
    unknown[8][1] = t19 / 0.24e2;
    unknown[8][2] = t40 / 0.24e2;
    unknown[8][3] = t45 / 0.24e2;
    unknown[8][4] = t48 / 0.24e2;
    unknown[8][5] = t52 / 0.24e2;
    unknown[8][6] = t53 / 0.24e2;
    unknown[8][7] = t54 / 0.24e2;
    unknown[8][8] = t38 / 0.12e2 - t39 / 0.12e2;

    value.hessian = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], NINPUTS, NINPUTS);
}
