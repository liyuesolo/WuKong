#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/CellFunction.h"
#include <iostream>

void Voronoi::getStandardNode(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, VectorXT &node) {
    double x1 = v1(0);
    double y1 = v1(1);
    double x2 = v2(0);
    double y2 = v2(1);
    double x3 = v3(0);
    double y3 = v3(1);

//    double m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1));
//    double xn = 0.5 * (x1 + x3) - m * (y3 - y1);
//    double yn = 0.5 * (y1 + y3) + m * (x3 - x1);
//    node = {xn, yn};

    double xn, yn;
    // @formatter:off
    xn = 0.5e0 * x1 + 0.5e0 * x3 - 0.5e0 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1)) * (y3 - y1);
    yn = 0.5e0 * y1 + 0.5e0 * y3 + 0.5e0 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1)) * (x3 - x1);
//    xn = 1.0 / 3.0 * (x1 + x2 + x3);
//    yn = 1.0 / 3.0 * (y1 + y2 + y3);
    // @formatter:on
    node = TV3(xn, yn, 0);
}

void
Voronoi::getStandardNodeGradient(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, MatrixXT &nodeGrad) {
    double x1 = v1(0);
    double y1 = v1(1);
    double x2 = v2(0);
    double y2 = v2(1);
    double x3 = v3(0);
    double y3 = v3(1);

    double gradX[6], gradY[6];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60;

    t1 = x2 - x3;
    t2 = y3 - y1;
    t3 = x2 - x1;
    t4 = y2 - y1;
    t5 = x3 - x1;
    t6 = t2 * t3 - t4 * t5;
    t7 = y3 - y2;
    t8 = -t1 * t3 + t4 * t7;
    t6 = 0.1e1 / t6;
    t9 = t8 * t6;
    t10 = t6 * t2;
    gradX[0] = 0.1e1 / 0.2e1 - t10 * (t9 * t7 + t1) / 0.2e1;
    gradX[1] = t6 * (t2 * t7 + t8 * (-t10 * t1 + 0.1e1)) / 0.2e1;
    gradX[2] = -t10 * (-t10 * t8 + x1 - 0.2e1 * x2 + x3) / 0.2e1;
    gradX[3] = -t10 * (t9 * t5 + y1 - 0.2e1 * y2 + y3) / 0.2e1;
    gradX[4] = 0.1e1 / 0.2e1 - t10 * (t9 * t4 + t3) / 0.2e1;
    gradX[5] = -t6 * (t2 * t4 + t8 * (-t10 * t3 + 0.1e1)) / 0.2e1;

    t1 = x2 - x3;
    t2 = y3 - y1;
    t3 = x2 - x1;
    t4 = y2 - y1;
    t5 = x3 - x1;
    t6 = t2 * t3 - t4 * t5;
    t7 = y3 - y2;
    t8 = -t1 * t3 + t4 * t7;
    t6 = 0.1e1 / t6;
    t9 = t8 * t6;
    t10 = t6 * t5;
    gradY[0] = -t6 * (-t1 * t5 + t8 * (-t7 * t6 * t5 + 0.1e1)) / 0.2e1;
    gradY[1] = 0.1e1 / 0.2e1 + t10 * (t9 * t1 - t7) / 0.2e1;
    gradY[2] = t10 * (-t9 * t2 + x1 - 0.2e1 * x2 + x3) / 0.2e1;
    gradY[3] = t10 * (t10 * t8 + y1 - 0.2e1 * y2 + y3) / 0.2e1;
    gradY[4] = t6 * (t3 * t5 + t8 * (t10 * t4 + 0.1e1)) / 0.2e1;
    gradY[5] = 0.1e1 / 0.2e1 + t10 * (-t9 * t3 + t4) / 0.2e1;
    // @formatter:on

    nodeGrad = MatrixXT::Zero(CellFunction::nx, 6);
    nodeGrad.row(0) = Eigen::Map<VectorXT>(gradX, 6);
    nodeGrad.row(1) = Eigen::Map<VectorXT>(gradY, 6);
}

void
Voronoi::getStandardNodeHessian(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                                std::vector<MatrixXT> &nodeHess) {
    double x1 = v1(0);
    double y1 = v1(1);
    double x2 = v2(0);
    double y2 = v2(1);
    double x3 = v3(0);
    double y3 = v3(1);

    int n = 6;
    double hessX_c[n][n];
    double hessY_c[n][n];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60;

    t1 = x2 - x3;
    t2 = y3 - y1;
    t3 = x2 - x1;
    t4 = y2 - y1;
    t5 = x3 - x1;
    t6 = t2 * t3;
    t7 = -t4 * t5 + t6;
    t8 = y3 - y2;
    t9 = -t1 * t3 + t4 * t8;
    t7 = 0.1e1 / t7;
    t10 = pow(t7, 0.2e1);
    t11 = t7 * t10;
    t12 = t9 * t7;
    t13 = t12 * t8;
    t14 = t2 * t8;
    t15 = pow(t1, 0.2e1);
    t16 = 0.1e1 / 0.2e1;
    t17 = t14 * t9 * t11;
    t18 = -t16 * t7 * (-pow(t8, 0.2e1) * t7 * t2 + t15 * t7 * t2 - t1 - t13) - t17 * t1;
    t19 = -0.2e1 * x2 + x1 + x3;
    t20 = t1 * t2;
    t21 = t7 * t2;
    t22 = pow(t2, 0.2e1);
    t11 = t9 * t11;
    t23 = t11 * t22;
    t24 = t16 * t21 * (t7 * (-t19 * t8 + t20) - 0.1e1) + t23 * t8;
    t25 = -0.2e1 * y2 + y1 + y3;
    t26 = t10 * t2;
    t27 = -t16 * t26 * (t1 * t5 + t25 * t8 - t9) - t17 * t5;
    t28 = -t7 * (t1 * t4 + t3 * t8) + 0.1e1;
    t29 = t16 * t21 * t28 - t17 * t4;
    t30 = t21 * t3;
    t31 = -t30 + 0.1e1;
    t32 = t2 * t4;
    t33 = t12 * t2;
    t34 = t7 * (t32 + t9);
    t17 = -t16 * t7 * (t1 * t31 + t34 * t8 + t33) + t17 * t3;
    t35 = t26 * t9;
    t36 = t21 * t1;
    t37 = t16 * t7 * (-t21 * t19 * t1 - t8 * t7 * t22 + t19) + t35 * (t36 - 0.1e1);
    t38 = t7 * (t14 + t9);
    t20 = t20 * t11;
    t39 = -t16 * t7 * (t25 * (t36 - 0.1e1) + t2 - t38 * t5) - t20 * t5;
    t36 = t16 * t7 * (t3 * (-t36 + 0.1e1) + t33 + t38 * t4) - t20 * t4;
    t20 = t16 * t7 * (t2 * t28 + t4 + t8 - t12 * (t1 + t3)) + t20 * t3;
    t28 = -t16 * t26 * (t19 * t5 - t2 * t25) + t23 * t5;
    t23 = -t16 * t21 * (t7 * (t19 * t4 - t6) + 0.1e1) + t23 * t4;
    t22 = -t16 * t7 * (-t4 * t7 * t22 - t30 * t19 + t19) - t35 * (-0.1e1 + t30);
    t30 = t32 * t11;
    t38 = -t16 * t26 * (t25 * t4 + t3 * t5 + t9) - t30 * t5;
    t2 = -t16 * t7 * (t25 * t31 + t34 * t5 + t2) + t6 * t11 * t5;
    t6 = t12 * t4;
    t11 = pow(t3, 0.2e1);
    t16 = -t16 * t7 * (t21 * pow(t4, 0.2e1) - t21 * t11 + t3 + t6) + t30 * t3;
    hessX_c[0][0] = -t14 * t10 * (t1 + t13);
    hessX_c[0][1] = t18;
    hessX_c[0][2] = t24;
    hessX_c[0][3] = t27;
    hessX_c[0][4] = t29;
    hessX_c[0][5] = t17;
    hessX_c[1][0] = t18;
    hessX_c[1][1] = -t7 * (t35 * t15 + t8 + (-t14 - t9) * t7 * t1);
    hessX_c[1][2] = t37;
    hessX_c[1][3] = t39;
    hessX_c[1][4] = t36;
    hessX_c[1][5] = t20;
    hessX_c[2][0] = t24;
    hessX_c[2][1] = t37;
    hessX_c[2][2] = t21 * (t21 * (-t33 + t19) + 0.1e1);
    hessX_c[2][3] = t28;
    hessX_c[2][4] = t23;
    hessX_c[2][5] = t22;
    hessX_c[3][0] = t27;
    hessX_c[3][1] = t39;
    hessX_c[3][2] = t28;
    hessX_c[3][3] = t21 * (-t7 * t5 * (t12 * t5 + t25) + 0.1e1);
    hessX_c[3][4] = t38;
    hessX_c[3][5] = t2;
    hessX_c[4][0] = t29;
    hessX_c[4][1] = t36;
    hessX_c[4][2] = t23;
    hessX_c[4][3] = t38;
    hessX_c[4][4] = -t26 * t4 * (t3 + t6);
    hessX_c[4][5] = t16;
    hessX_c[5][0] = t17;
    hessX_c[5][1] = t20;
    hessX_c[5][2] = t22;
    hessX_c[5][3] = t2;
    hessX_c[5][4] = t16;
    hessX_c[5][5] = -t7 * (t35 * t11 + t4 + (-t32 - t9) * t7 * t3);

    t1 = x2 - x3;
    t2 = y3 - y1;
    t3 = x2 - x1;
    t4 = y2 - y1;
    t5 = x3 - x1;
    t6 = t4 * t5;
    t7 = t2 * t3 - t6;
    t8 = y3 - y2;
    t9 = -t1 * t3 + t4 * t8;
    t7 = 0.1e1 / t7;
    t10 = t1 * t5;
    t11 = pow(t8, 0.2e1);
    t12 = pow(t7, 0.2e1);
    t13 = t7 * t12;
    t14 = t9 * t12 * t5;
    t15 = (-t10 + t9) * t7;
    t16 = t9 * t7;
    t17 = t16 * t1;
    t18 = 0.1e1 / 0.2e1;
    t19 = t5 * t8;
    t20 = t19 * t9 * t13;
    t21 = t18 * t7 * (pow(t1, 0.2e1) * t7 * t5 - t11 * t7 * t5 - t17 + t8) + t20 * t1;
    t22 = -0.2e1 * x2 + x1 + x3;
    t23 = t19 * t7;
    t24 = t23 - 0.1e1;
    t25 = t18 * t7 * (t15 * t2 + t22 * t24 + t5) - t20 * t2;
    t26 = -0.2e1 * y2 + y1 + y3;
    t27 = pow(t5, 0.2e1);
    t28 = t1 * t7;
    t29 = 0.2e1 * t14;
    t13 = t9 * t13;
    t30 = t13 * t27;
    t23 = t18 * (t7 * (t23 * t26 + t28 * t27 - t26) - t29) + t30 * t8;
    t31 = -t7 * (t1 * t4 + t3 * t8) + 0.1e1;
    t32 = -t18 * t7 * (t31 * t5 - t1 + t3 + t16 * (t4 - t8)) + t20 * t4;
    t33 = t16 * t5;
    t20 = t18 * t7 * (t15 * t3 + t24 * t4 + t33) - t20 * t3;
    t24 = t12 * t5;
    t34 = t10 * t13;
    t35 = t18 * t24 * (t1 * t22 + t2 * t8 + t9) - t34 * t2;
    t36 = t7 * t5;
    t19 = -t18 * t36 * (t7 * (-t1 * t26 + t19) - 0.1e1) + t30 * t1;
    t37 = t36 * t4;
    t38 = -t37 - 0.1e1;
    t39 = t3 * t5;
    t40 = t39 + t9;
    t28 = t18 * t7 * (t28 * t40 + t38 * t8 - t33) + t34 * t4;
    t31 = -t18 * t31 * t36 - t34 * t3;
    t34 = t7 * t2;
    t41 = t18 * t24 * (-t2 * t26 + t22 * t5) - t30 * t2;
    t42 = t6 * t13;
    t5 = -t18 * t7 * (t22 * t38 + t34 * t40 - t5) - t42 * t2;
    t9 = -t18 * t24 * (t2 * t4 + t22 * t3 + t9) + t39 * t13 * t2;
    t13 = t18 * (t7 * (t3 * t7 * t27 + t37 * t26 + t26) + t29) + t30 * t4;
    t6 = t18 * t36 * (t7 * (-t26 * t3 + t6) + 0.1e1) - t30 * t3;
    t27 = pow(t4, 0.2e1);
    t29 = t16 * t3;
    t18 = -t18 * t7 * (t36 * pow(t3, 0.2e1) - t36 * t27 + t29 - t4) - t42 * t3;
    hessY_c[0][0] = -t7 * (-t14 * t11 + t15 * t8 + t1);
    hessY_c[0][1] = t21;
    hessY_c[0][2] = t25;
    hessY_c[0][3] = t23;
    hessY_c[0][4] = t32;
    hessY_c[0][5] = t20;
    hessY_c[1][0] = t21;
    hessY_c[1][1] = t10 * t12 * (-t8 + t17);
    hessY_c[1][2] = t35;
    hessY_c[1][3] = t19;
    hessY_c[1][4] = t28;
    hessY_c[1][5] = t31;
    hessY_c[2][0] = t25;
    hessY_c[2][1] = t35;
    hessY_c[2][2] = -t36 * (t34 * (-t16 * t2 + t22) + 0.1e1);
    hessY_c[2][3] = t41;
    hessY_c[2][4] = t5;
    hessY_c[2][5] = t9;
    hessY_c[3][0] = t23;
    hessY_c[3][1] = t19;
    hessY_c[3][2] = t41;
    hessY_c[3][3] = t36 * (t36 * (t33 + t26) - 0.1e1);
    hessY_c[3][4] = t13;
    hessY_c[3][5] = t6;
    hessY_c[4][0] = t32;
    hessY_c[4][1] = t28;
    hessY_c[4][2] = t5;
    hessY_c[4][3] = t13;
    hessY_c[4][4] = t7 * (t40 * t7 * t4 + t14 * t27 + t3);
    hessY_c[4][5] = t18;
    hessY_c[5][0] = t20;
    hessY_c[5][1] = t31;
    hessY_c[5][2] = t9;
    hessY_c[5][3] = t6;
    hessY_c[5][4] = t18;
    hessY_c[5][5] = t24 * t3 * (-t4 + t29);
    // @formatter:on

    nodeHess.resize(CellFunction::nx);
    nodeHess[0] = Eigen::Map<Eigen::MatrixXd>(&hessX_c[0][0], n, n);
    nodeHess[1] = Eigen::Map<Eigen::MatrixXd>(&hessY_c[0][0], n, n);
    nodeHess[2] = Eigen::MatrixXd::Zero(0, 0);

//    std::cout << "check write hess " << hessX_c[0][0] << " " << hessX(0, 0) << " " << hessY_c[5][1] << " " << hessY(5, 1) << std::endl;
}

void Voronoi::getBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, VectorXT &node) {
//    double x1 = (v1(0) + v2(0)) / 2;
//    double y1 = (v1(1) + v2(1)) / 2;
//    double x2 = x1 + (v2(1) - v1(1));
//    double y2 = y1 - (v2(0) - v1(0));
//    double x3 = b0(0);
//    double y3 = b0(1);
//    double x4 = b1(0);
//    double y4 = b1(1);
//    double t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (-(x4 - x3) * (y2 - y1) + (x2 - x1) * (y4 - y3));
//    double xn = x1 + t * (x2 - x1);
//    double yn = y1 + t * (y2 - y1);
//    node = {xn, yn};

    double v1x = v1(0);
    double v1y = v1(1);
    double v2x = v2(0);
    double v2y = v2(1);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    double xn, yn;
    // @formatter:off
    xn = 0.5e0 * v1x + 0.5e0 * v2x + ((x4 - x3) * (-y3 + 0.5e0 * v1y + 0.5e0 * v2y) - (y4 - y3) * (-x3 + 0.5e0 * v1x + 0.5e0 * v2x)) / (-(x4 - x3) * (-v2x + v1x) + (v2y - v1y) * (y4 - y3)) * (v2y - v1y);
    yn = 0.5e0 * v1y + 0.5e0 * v2y + ((x4 - x3) * (-y3 + 0.5e0 * v1y + 0.5e0 * v2y) - (y4 - y3) * (-x3 + 0.5e0 * v1x + 0.5e0 * v2x)) / (-(x4 - x3) * (-v2x + v1x) + (v2y - v1y) * (y4 - y3)) * (-v2x + v1x);
    // @formatter:on
    node = TV3(xn, yn, 0);
}

void
Voronoi::getBoundaryNodeGradient(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1,
                                 MatrixXT &nodeGrad) {
    double v1x = v1(0);
    double v1y = v1(1);
    double v2x = v2(0);
    double v2y = v2(1);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    double gradX[8], gradY[8];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60;

    t1 = y4 - y3;
    t2 = x4 - x3;
    t3 = v2x - v1x;
    t4 = v2y - v1y;
    t5 = t4 * t1;
    t6 = t2 * t3 + t5;
    t7 = 0.1e1 / 0.2e1;
    t8 = t7 * (v1y + v2y);
    t9 = t8 - y3;
    t10 = t7 * (v1x + v2x);
    t11 = t10 - x3;
    t1 = -t1 * t11 + t2 * t9;
    t6 = 0.1e1 / t6;
    t5 = -t5 * t6 + 0.1e1;
    t12 = t7 * t5;
    t13 = t1 * pow(t6, 0.2e1) * t4 * t2;
    t1 = t1 * t6;
    t5 = t1 * t5;
    t2 = t7 * t2 * t6 * t4;
    t3 = t1 * t3;
    t6 = t6 * t4;
    t1 = t4 * t1;
    gradX[0] = t12 + t13;
    gradX[1] = t2 - t5;
    gradX[2] = t12 - t13;
    gradX[3] = t2 + t5;
    gradX[4] = t6 * (t3 - t8 + y4);
    gradX[5] = t6 * (t1 + t10 - x4);
    gradX[6] = t6 * (-t3 + t9);
    gradX[7] = -t6 * (t1 + t11);

    t1 = y4 - y3;
    t2 = x4 - x3;
    t3 = v2x - v1x;
    t4 = v2y - v1y;
    t5 = t2 * t3;
    t6 = t1 * t4 + t5;
    t7 = 0.1e1 / 0.2e1;
    t8 = t7 * (v1y + v2y);
    t9 = t8 - y3;
    t10 = t7 * (v1x + v2x);
    t11 = t10 - x3;
    t2 = -t1 * t11 + t2 * t9;
    t6 = 0.1e1 / t6;
    t5 = t5 * t6 - 0.1e1;
    t12 = t2 * t6;
    t13 = t12 * t5;
    t14 = t7 * t1 * t6 * t3;
    t1 = t2 * pow(t6, 0.2e1) * t3 * t1;
    t2 = -t7 * t5;
    t5 = t12 * t3;
    t3 = t6 * t3;
    t4 = t12 * t4;
    gradY[0] = t14 - t13;
    gradY[1] = t2 - t1;
    gradY[2] = t14 + t13;
    gradY[3] = t2 + t1;
    gradY[4] = -t3 * (t5 + y4 - t8);
    gradY[5] = -t3 * (t4 + t10 - x4);
    gradY[6] = t3 * (t5 - t9);
    gradY[7] = t3 * (t4 + t11);
    // @formatter:on

    nodeGrad = MatrixXT::Zero(CellFunction::nx, 8);
    nodeGrad.row(0) = Eigen::Map<VectorXT>(gradX, 8);
    nodeGrad.row(1) = Eigen::Map<VectorXT>(gradY, 8);
}

void
Voronoi::getBoundaryNodeHessian(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1,
                                std::vector<MatrixXT> &nodeHess) {
    double v1x = v1(0);
    double v1y = v1(1);
    double v2x = v2(0);
    double v2y = v2(1);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    int n = 8;
    double hessX_c[n][n];
    double hessY_c[n][n];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60;

    t1 = y4 - y3;
    t2 = x4 - x3;
    t3 = v2x - v1x;
    t4 = v2y - v1y;
    t5 = t4 * t1;
    t6 = t2 * t3 + t5;
    t7 = 0.1e1 / 0.2e1;
    t8 = t7 * (v1y + v2y);
    t9 = t8 - y3;
    t10 = t7 * (v1x + v2x);
    t11 = t10 - x3;
    t12 = t2 * t9;
    t13 = -t1 * t11 + t12;
    t6 = 0.1e1 / t6;
    t14 = pow(t6, 0.2e1);
    t15 = 0.2e1 * t13;
    t16 = t15 * t6;
    t17 = t16 * t2;
    t18 = t14 * t4;
    t19 = t18 * t2;
    t20 = pow(t1, 0.2e1);
    t21 = pow(t2, 0.2e1);
    t22 = t7 * t6 * ((t21 - t20) * t6 * t4 + t1);
    t23 = t13 * t14;
    t24 = t23 * t2 * (0.2e1 * t5 * t6 - 0.1e1);
    t25 = t24 + t22;
    t20 = t7 * t6 * ((-t21 - t20) * t6 * t4 + t1);
    t26 = -t24 - t20;
    t8 = y4 - t8;
    t27 = t18 * (t2 * t8 - t13);
    t28 = t7 * t1;
    t29 = t18 * t3;
    t30 = t29 * (-t28 + t17);
    t31 = t30 + t27;
    t10 = -x4 + t10;
    t32 = -t5 * t6 + 0.1e1;
    t33 = t6 * t4;
    t34 = t16 * t4;
    t35 = t7 * t33 * t32;
    t36 = t19 * (t34 + t10);
    t37 = t36 + t35;
    t12 = t18 * (t12 + t13);
    t30 = -t30 + t12;
    t34 = t19 * (-t34 - t11);
    t38 = t34 - t35;
    t39 = t15 * t6 * t14;
    t40 = t39 * t4;
    t21 = t40 * t21;
    t41 = t2 * t6;
    t42 = t41 * t32;
    t23 = -0.2e1 * t23 * t1 * t32;
    t20 = -t24 + t20;
    t43 = -t23;
    t44 = t3 * t13;
    t45 = t6 * (t6 * (-t5 * t8 + t44) + t8);
    t41 = t7 * t33 * (-t41 * t3 + 0.1e1);
    t46 = t39 * t5 * t3;
    t47 = t46 - t41 - t45;
    t15 = t10 * t6 * t32 + t15 * t18;
    t48 = pow(t4, 0.2e1);
    t16 = t16 * t1;
    t2 = t7 * t2;
    t7 = t14 * t48;
    t14 = t7 * (t16 + t2) - t15;
    t5 = t6 * (t6 * (-t5 * t9 - t44) + t9);
    t49 = -t46 + t41 - t5;
    t50 = -t11 * t6 * t32;
    t32 = -0.2e1 * t18 * t13 * t32;
    t51 = t7 * t2;
    t52 = -t51 - t32 - t50;
    t22 = t24 - t22;
    t24 = t29 * (t28 + t17);
    t27 = -t24 - t27;
    t28 = -t36 + t35;
    t12 = t24 - t12;
    t24 = -t34 - t35;
    t34 = -t46 - t41 + t45;
    t2 = t7 * (-t16 + t2) + t15;
    t5 = t46 + t41 + t5;
    t15 = -t51 + t32 + t50;
    t16 = t44 * t6;
    t32 = t10 * t3;
    t35 = t4 * t8;
    t36 = t39 * t48 * t3;
    t41 = t36 + t18 * (t32 + t35);
    t40 = -t40 * pow(t3, 0.2e1) - t29 * (-t9 + t8);
    t3 = -t11 * t3;
    t35 = -t36 + t33 * (t6 * (t3 - t35) + 0.1e1);
    t13 = t33 * t13;
    t44 = t4 * t9;
    t6 = -t36 - t33 * (t6 * (-t44 + t32) + 0.1e1);
    t4 = -t39 * t4 * t48 - t7 * (t10 + t11);
    t3 = t36 - t18 * (t3 + t44);
    hessX_c[0][0] = t19 * (t17 - t1);
    hessX_c[0][1] = t25;
    hessX_c[0][2] = -t21;
    hessX_c[0][3] = t26;
    hessX_c[0][4] = t31;
    hessX_c[0][5] = t37;
    hessX_c[0][6] = t30;
    hessX_c[0][7] = t38;
    hessX_c[1][0] = t25;
    hessX_c[1][1] = t23 - t42;
    hessX_c[1][2] = t20;
    hessX_c[1][3] = t43;
    hessX_c[1][4] = t47;
    hessX_c[1][5] = t14;
    hessX_c[1][6] = t49;
    hessX_c[1][7] = t52;
    hessX_c[2][0] = -t21;
    hessX_c[2][1] = t20;
    hessX_c[2][2] = t19 * (t17 + t1);
    hessX_c[2][3] = t22;
    hessX_c[2][4] = t27;
    hessX_c[2][5] = t28;
    hessX_c[2][6] = t12;
    hessX_c[2][7] = t24;
    hessX_c[3][0] = t26;
    hessX_c[3][1] = t43;
    hessX_c[3][2] = t22;
    hessX_c[3][3] = t23 + t42;
    hessX_c[3][4] = t34;
    hessX_c[3][5] = t2;
    hessX_c[3][6] = t5;
    hessX_c[3][7] = t15;
    hessX_c[4][0] = t31;
    hessX_c[4][1] = t47;
    hessX_c[4][2] = t27;
    hessX_c[4][3] = t34;
    hessX_c[4][4] = 0.2e1 * t29 * (t16 + t8);
    hessX_c[4][5] = t41;
    hessX_c[4][6] = t40;
    hessX_c[4][7] = t35;
    hessX_c[5][0] = t37;
    hessX_c[5][1] = t14;
    hessX_c[5][2] = t28;
    hessX_c[5][3] = t2;
    hessX_c[5][4] = t41;
    hessX_c[5][5] = 0.2e1 * t7 * (t13 + t10);
    hessX_c[5][6] = t6;
    hessX_c[5][7] = t4;
    hessX_c[6][0] = t30;
    hessX_c[6][1] = t49;
    hessX_c[6][2] = t12;
    hessX_c[6][3] = t5;
    hessX_c[6][4] = t40;
    hessX_c[6][5] = t6;
    hessX_c[6][6] = 0.2e1 * t29 * (t16 - t9);
    hessX_c[6][7] = t3;
    hessX_c[7][0] = t38;
    hessX_c[7][1] = t52;
    hessX_c[7][2] = t24;
    hessX_c[7][3] = t15;
    hessX_c[7][4] = t35;
    hessX_c[7][5] = t4;
    hessX_c[7][6] = t3;
    hessX_c[7][7] = 0.2e1 * t7 * (t13 + t11);

    t1 = y4 - y3;
    t2 = x4 - x3;
    t3 = v2x - v1x;
    t4 = v2y - v1y;
    t5 = t2 * t3;
    t6 = t1 * t4 + t5;
    t7 = 0.1e1 / 0.2e1;
    t8 = t7 * (v1y + v2y);
    t9 = -y3 + t8;
    t10 = t7 * (v1x + v2x);
    t11 = t10 - x3;
    t12 = -t1 * t11 + t2 * t9;
    t6 = 0.1e1 / t6;
    t13 = t5 * t6;
    t14 = -0.1e1 + t13;
    t15 = t1 * t6;
    t16 = t15 * t14;
    t17 = pow(t6, 0.2e1);
    t18 = t6 * t17;
    t19 = t12 * t17;
    t20 = 0.2e1 * t19 * t2 * t14;
    t21 = pow(t2, 0.2e1);
    t22 = pow(t1, 0.2e1);
    t23 = t7 * t6 * ((t22 - t21) * t6 * t3 + t2);
    t13 = t19 * t1 * (0.1e1 - 0.2e1 * t13);
    t24 = t23 + t13;
    t21 = t7 * t6 * ((-t22 - t21) * t6 * t3 + t2);
    t25 = t21 - t13;
    t8 = y4 - t8;
    t19 = t19 * t3;
    t26 = -t8 * t6 * t14;
    t27 = pow(t3, 0.2e1);
    t28 = 0.2e1 * t19 * t14;
    t29 = t7 * t1;
    t30 = t29 * t17 * t27;
    t31 = t26 - t28 + t30;
    t10 = -x4 + t10;
    t32 = t6 * t3;
    t33 = t4 * t12;
    t34 = t6 * (t6 * (-t5 * t10 + t33) + t10);
    t35 = t7 * t32 * (t15 * t4 - 0.1e1);
    t36 = 0.2e1 * t33 * t5 * t18;
    t37 = t34 + t35 - t36;
    t19 = -t9 * t6 * t14 - 0.2e1 * t19;
    t38 = 0.2e1 * t12;
    t39 = t38 * t6 * t2;
    t40 = t17 * t27;
    t41 = t19 + t40 * (t39 - t29);
    t5 = t6 * (t6 * (t5 * t11 - t33) - t11);
    t42 = t5 - t35 + t36;
    t15 = t38 * t15;
    t43 = t1 * t17 * t3;
    t21 = -t21 - t13;
    t44 = t38 * t32;
    t14 = -t7 * t32 * t14;
    t45 = t43 * (t8 + t44);
    t46 = t14 - t45;
    t17 = t17 * t3;
    t47 = t17 * (t1 * t10 - t12);
    t7 = t7 * t2;
    t48 = t17 * t4;
    t49 = t48 * (t15 + t7);
    t50 = -t47 - t49;
    t44 = t43 * (-t9 + t44);
    t51 = -t14 + t44;
    t1 = t17 * (-t1 * t11 + t12);
    t49 = -t1 + t49;
    t38 = t38 * t18;
    t52 = t38 * t3;
    t22 = t52 * t22;
    t13 = -t23 + t13;
    t23 = -t26 + t28 + t30;
    t26 = -t34 + t35 + t36;
    t19 = -t19 - t40 * (t39 + t29);
    t5 = -t5 - t35 - t36;
    t28 = t14 + t45;
    t7 = t48 * (t15 - t7);
    t29 = t47 + t7;
    t14 = -t14 - t44;
    t1 = t1 - t7;
    t7 = t32 * t12;
    t12 = t10 * t3;
    t30 = t8 * t4;
    t18 = 0.2e1 * t33 * t18 * t27;
    t34 = -t17 * (t12 + t30) - t18;
    t27 = t40 * (-t9 + t8) + t38 * t3 * t27;
    t3 = -t11 * t3;
    t30 = -t32 * (t6 * (-t30 + t3) + 0.1e1) + t18;
    t33 = t33 * t6;
    t35 = t9 * t4;
    t6 = t32 * (t6 * (t12 - t35) + 0.1e1) + t18;
    t4 = t48 * (t10 + t11) + t52 * pow(t4, 0.2e1);
    t3 = t17 * (t3 + t35) - t18;
    hessY_c[0][0] = t16 - t20;
    hessY_c[0][1] = t24;
    hessY_c[0][2] = t20;
    hessY_c[0][3] = t25;
    hessY_c[0][4] = t31;
    hessY_c[0][5] = t37;
    hessY_c[0][6] = t41;
    hessY_c[0][7] = t42;
    hessY_c[1][0] = t24;
    hessY_c[1][1] = -t43 * (t2 + t15);
    hessY_c[1][2] = t21;
    hessY_c[1][3] = t22;
    hessY_c[1][4] = t46;
    hessY_c[1][5] = t50;
    hessY_c[1][6] = t51;
    hessY_c[1][7] = t49;
    hessY_c[2][0] = t20;
    hessY_c[2][1] = t21;
    hessY_c[2][2] = -t16 - t20;
    hessY_c[2][3] = t13;
    hessY_c[2][4] = t23;
    hessY_c[2][5] = t26;
    hessY_c[2][6] = t19;
    hessY_c[2][7] = t5;
    hessY_c[3][0] = t25;
    hessY_c[3][1] = t22;
    hessY_c[3][2] = t13;
    hessY_c[3][3] = t43 * (t2 - t15);
    hessY_c[3][4] = t28;
    hessY_c[3][5] = t29;
    hessY_c[3][6] = t14;
    hessY_c[3][7] = t1;
    hessY_c[4][0] = t31;
    hessY_c[4][1] = t46;
    hessY_c[4][2] = t23;
    hessY_c[4][3] = t28;
    hessY_c[4][4] = -0.2e1 * t40 * (t8 + t7);
    hessY_c[4][5] = t34;
    hessY_c[4][6] = t27;
    hessY_c[4][7] = t30;
    hessY_c[5][0] = t37;
    hessY_c[5][1] = t50;
    hessY_c[5][2] = t26;
    hessY_c[5][3] = t29;
    hessY_c[5][4] = t34;
    hessY_c[5][5] = -0.2e1 * t48 * (t10 + t33);
    hessY_c[5][6] = t6;
    hessY_c[5][7] = t4;
    hessY_c[6][0] = t41;
    hessY_c[6][1] = t51;
    hessY_c[6][2] = t19;
    hessY_c[6][3] = t14;
    hessY_c[6][4] = t27;
    hessY_c[6][5] = t6;
    hessY_c[6][6] = -0.2e1 * t40 * (-t9 + t7);
    hessY_c[6][7] = t3;
    hessY_c[7][0] = t42;
    hessY_c[7][1] = t49;
    hessY_c[7][2] = t5;
    hessY_c[7][3] = t1;
    hessY_c[7][4] = t30;
    hessY_c[7][5] = t4;
    hessY_c[7][6] = t3;
    hessY_c[7][7] = -0.2e1 * t48 * (t11 + t33);
    // @formatter:on

    nodeHess.resize(CellFunction::nx);
    nodeHess[0] = Eigen::Map<Eigen::MatrixXd>(&hessX_c[0][0], n, n);
    nodeHess[1] = Eigen::Map<Eigen::MatrixXd>(&hessY_c[0][0], n, n);
    nodeHess[2] = Eigen::MatrixXd::Zero(0, 0);
}

//VectorXi Voronoi::delaunayNaive(const VectorXT &vertices) {
//    int n_vtx = vertices.rows() / 2;
//    std::vector<int> tri1;
//    std::vector<int> tri2;
//    std::vector<int> tri3;
//
//    for (int i = 0; i < n_vtx; i++) {
//        TV vi = vertices.segment<2>(i * 2);
//        std::vector<int> neighbors;
//
//        for (int j = 0; j < n_vtx; j++) {
//            if (j == i) continue;
//
//            TV vj = vertices.segment<2>(j * 2);
//            TV line = {-(vj(1) - vi(1)), vj(0) - vi(0)};
//
//            double dmin = INFINITY;
//            double dmax = -INFINITY;
//
//            for (int k = 0; k < n_vtx; k++) {
//                if (k == i || k == j) continue;
//
//                TV vk = vertices.segment<2>(k * 2);
//                TV vc;
//                getNode(vi, vj, vk, vc);
//                double d = vc.dot(line);
//
//                if ((vk - vi).dot(line) > 0) {
//                    dmin = fmin(dmin, d);
//                } else {
//                    dmax = fmax(dmax, d);
//                }
//                if (dmax > dmin) break;
//            }
//
//            if (dmax < dmin || (dmax == dmin)) {
//                neighbors.push_back(j);
//            }
//        }
//
//        double xc = vertices(i * 2 + 0);
//        double yc = vertices(i * 2 + 1);
//
//        std::sort(neighbors.begin(), neighbors.end(), [vertices, xc, yc](int a, int b) {
//            double xa = vertices(a * 2 + 0);
//            double ya = vertices(a * 2 + 1);
//            double angle_a = atan2(ya - yc, xa - xc);
//
//            double xb = vertices(b * 2 + 0);
//            double yb = vertices(b * 2 + 1);
//            double angle_b = atan2(yb - yc, xb - xc);
//
//            return angle_a < angle_b;
//        });
//
//        if (neighbors.size() > 0) {
//            assert(neighbors.size() > 1);
//            for (int j = 0; j < neighbors.size(); j++) {
//                int v1 = i;
//                int v2 = neighbors[j];
//                int v3 = neighbors[(j + 1) % neighbors.size()];
//
//                if (v1 < v2 && v1 < v3) {
//                    double x1 = vertices(v1 * 2 + 0);
//                    double y1 = vertices(v1 * 2 + 1);
//                    double x2 = vertices(v2 * 2 + 0);
//                    double y2 = vertices(v2 * 2 + 1);
//                    double x3 = vertices(v3 * 2 + 0);
//                    double y3 = vertices(v3 * 2 + 1);
//
//                    if (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2 > 0) {
//                        tri1.push_back(v1);
//                        tri2.push_back(v2);
//                        tri3.push_back(v3);
//                    }
//                }
//            }
//        }
//    }
//
//    VectorXi tri(tri1.size() * 3);
//    for (int i = 0; i < tri1.size(); i++) {
//        tri(i * 3 + 0) = tri1[i];
//        tri(i * 3 + 1) = tri2[i];
//        tri(i * 3 + 2) = tri3[i];
//    }
//
//    return tri;
//}

VectorXi Voronoi::delaunayJRS(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    MatrixXT P;
    P.resize(n_vtx, 2);
    for (int i = 0; i < n_vtx; i++) {
        P.row(i) = vertices.segment<2>(i * 2);
    }

    MatrixXT V;
    MatrixXi F;
    igl::triangle::triangulate(P,
                               MatrixXi(),
                               MatrixXT(),
                               "cQ", // Enclose convex hull with segments
                               V, F);

    VectorXi tri;
    tri.resize(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++)
        tri.segment<3>(i * 3) = F.row(i);
    return tri;
}

VectorXi Voronoi::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    return delaunayJRS(vertices);
}

VectorXT Voronoi::getDefaultVertexParams(const VectorXT &vertices) {
    return {};
}

