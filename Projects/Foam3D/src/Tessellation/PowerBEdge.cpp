#include "../../include/Tessellation/Power.h"
#include <iostream>

// @formatter:off
void Power::getNodeBEdge(const TV3 &b0, const TV3 &b1, const VectorXT &v0, const VectorXT &v1, NodePosition &nodePos) {
    double x0 = v0(0);
    double y0 = v0(1);
    double z0 = v0(2);
    double w0 = v0(3);
    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double w1 = v1(3);
    double xb0 = b0(0);
    double yb0 = b0(1);
    double zb0 = b0(2);
    double xb1 = b1(0);
    double yb1 = b1(1);
    double zb1 = b1(2);

    double unknown[3];

    double t1 = y0 * y0;
    double t3 = 0.2e1 * yb1 * y0;
    double t4 = y1 * y1;
    double t6 = 0.2e1 * yb1 * y1;
    double t7 = z0 * z0;
    double t9 = 0.2e1 * z0 * zb1;
    double t10 = z1 * z1;
    double t12 = 0.2e1 * z1 * zb1;
    double t14 = wmul * (-w1 + w0);
    double t15 = x0 * x0;
    double t16 = x1 * x1;
    double t17 = -t1 + t3 + t4 - t6 - t7 + t9 + t10 - t12 + t14 - t15 + t16;
    double t20 = 0.2e1 * yb0 * y0;
    double t22 = 0.2e1 * yb0 * y1;
    double t24 = 0.2e1 * z0 * zb0;
    double t26 = 0.2e1 * z1 * zb0;
    double t27 = -t1 + t20 + t4 - t22 - t7 + t24 + t10 - t26 + t14 - t15 + t16;
    double t30 = x1 - x0;
    double t33 = yb1 - yb0;
    double t36 = -z1 + z0;
    double t39 = 0.2e1 * (zb0 - zb1) * t36;
    double t44 = 0.2e1 * xb1 * x0;
    double t46 = 0.2e1 * xb1 * x1;
    double t47 = -t15 + t44 + t16 - t46 - t7 + t9 + t10 - t12 + t14 - t1 + t4;
    double t50 = 0.2e1 * xb0 * x0;
    double t52 = 0.2e1 * xb0 * x1;
    double t53 = -t15 + t50 + t16 - t52 - t7 + t24 + t10 - t26 + t14 - t1 + t4;
    double t56 = y1 - y0;
    double t59 = xb1 - xb0;
    double t60 = 0.2e1 * x0 * t59;
    double t61 = -0.2e1 * x1 * t59;
    double t65 = -t15 + t44 + t16 - t46 - t1 + t3 + t4 - t6 + t14 - t7 + t10;
    double t67 = -t15 + t50 + t16 - t52 - t1 + t20 + t4 - t22 + t14 - t7 + t10;
    unknown[0] = 0.1e1 / (0.2e1 * xb0 * t30 - 0.2e1 * xb1 * t30 + 0.2e1 * y0 * t33 - 0.2e1 * y1 * t33 - t39) * (xb0 * t17 - t27 * xb1);
    unknown[1] = 0.1e1 / (0.2e1 * yb0 * t56 - 0.2e1 * yb1 * t56 - t39 + t60 + t61) * (yb0 * t47 - yb1 * t53);
    unknown[2] = 0.1e1 / (-0.2e1 * t33 * t56 - 0.2e1 * zb0 * t36 + 0.2e1 * zb1 * t36 + t60 + t61) * (zb0 * t65 - zb1 * t67);
    
    nodePos.pos = Eigen::Map<TV3>(unknown);
}
