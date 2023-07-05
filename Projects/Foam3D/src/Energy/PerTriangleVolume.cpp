#include "../../include/Energy/PerTriangleVolume.h"

#define NINPUTS 9

// @formatter:off
void PerTriangleVolume::getValue(TriangleValue &value) const {
    double x0 = value.v0(0);
    double y0 = value.v0(1);
    double z0 = value.v0(2);
    double x1 = value.v1(0);
    double y1 = value.v1(1);
    double z1 = value.v1(2);
    double x2 = value.v2(0);
    double y2 = value.v2(1);
    double z2 = value.v2(2);

    value.value = (y1 * z2 - y2 * z1) * x0 / 0.6e1 + (-y0 * z2 + y2 * z0) * x1 / 0.6e1 + x2 * (y0 * z1 - y1 * z0) / 0.6e1;
}

void PerTriangleVolume::getGradient(TriangleValue &value) const {
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

    unknown[0] = y1 * z2 / 0.6e1 - y2 * z1 / 0.6e1;
    unknown[1] = -z2 * x1 / 0.6e1 + x2 * z1 / 0.6e1;
    unknown[2] = y2 * x1 / 0.6e1 - x2 * y1 / 0.6e1;
    unknown[3] = -y0 * z2 / 0.6e1 + y2 * z0 / 0.6e1;
    unknown[4] = z2 * x0 / 0.6e1 - x2 * z0 / 0.6e1;
    unknown[5] = -y2 * x0 / 0.6e1 + x2 * y0 / 0.6e1;
    unknown[6] = y0 * z1 / 0.6e1 - y1 * z0 / 0.6e1;
    unknown[7] = -z1 * x0 / 0.6e1 + z0 * x1 / 0.6e1;
    unknown[8] = y1 * x0 / 0.6e1 - y0 * x1 / 0.6e1;

    value.gradient = Eigen::Map<Eigen::VectorXd>(&unknown[0], NINPUTS);
}

void PerTriangleVolume::getHessian(TriangleValue &value) const {
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

    double t1 = z2 / 0.6e1;
    double t2 = y2 / 0.6e1;
    double t3 = z1 / 0.6e1;
    double t4 = y1 / 0.6e1;
    double t5 = x2 / 0.6e1;
    double t6 = x1 / 0.6e1;
    double t7 = z0 / 0.6e1;
    double t8 = y0 / 0.6e1;
    double t9 = x0 / 0.6e1;
    unknown[0][0] = 0.0e0;
    unknown[0][1] = 0.0e0;
    unknown[0][2] = 0.0e0;
    unknown[0][3] = 0.0e0;
    unknown[0][4] = t1;
    unknown[0][5] = -t2;
    unknown[0][6] = 0.0e0;
    unknown[0][7] = -t3;
    unknown[0][8] = t4;
    unknown[1][0] = 0.0e0;
    unknown[1][1] = 0.0e0;
    unknown[1][2] = 0.0e0;
    unknown[1][3] = -t1;
    unknown[1][4] = 0.0e0;
    unknown[1][5] = t5;
    unknown[1][6] = t3;
    unknown[1][7] = 0.0e0;
    unknown[1][8] = -t6;
    unknown[2][0] = 0.0e0;
    unknown[2][1] = 0.0e0;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = t2;
    unknown[2][4] = -t5;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = -t4;
    unknown[2][7] = t6;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = 0.0e0;
    unknown[3][1] = -t1;
    unknown[3][2] = t2;
    unknown[3][3] = 0.0e0;
    unknown[3][4] = 0.0e0;
    unknown[3][5] = 0.0e0;
    unknown[3][6] = 0.0e0;
    unknown[3][7] = t7;
    unknown[3][8] = -t8;
    unknown[4][0] = t1;
    unknown[4][1] = 0.0e0;
    unknown[4][2] = -t5;
    unknown[4][3] = 0.0e0;
    unknown[4][4] = 0.0e0;
    unknown[4][5] = 0.0e0;
    unknown[4][6] = -t7;
    unknown[4][7] = 0.0e0;
    unknown[4][8] = t9;
    unknown[5][0] = -t2;
    unknown[5][1] = t5;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = 0.0e0;
    unknown[5][4] = 0.0e0;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = t8;
    unknown[5][7] = -t9;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = 0.0e0;
    unknown[6][1] = t3;
    unknown[6][2] = -t4;
    unknown[6][3] = 0.0e0;
    unknown[6][4] = -t7;
    unknown[6][5] = t8;
    unknown[6][6] = 0.0e0;
    unknown[6][7] = 0.0e0;
    unknown[6][8] = 0.0e0;
    unknown[7][0] = -t3;
    unknown[7][1] = 0.0e0;
    unknown[7][2] = t6;
    unknown[7][3] = t7;
    unknown[7][4] = 0.0e0;
    unknown[7][5] = -t9;
    unknown[7][6] = 0.0e0;
    unknown[7][7] = 0.0e0;
    unknown[7][8] = 0.0e0;
    unknown[8][0] = t4;
    unknown[8][1] = -t6;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = -t8;
    unknown[8][4] = t9;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = 0.0e0;
    unknown[8][7] = 0.0e0;
    unknown[8][8] = 0.0e0;

    value.hessian = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], NINPUTS, NINPUTS);
}
