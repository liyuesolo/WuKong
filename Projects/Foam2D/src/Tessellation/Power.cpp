#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "Projects/Foam2D/include/Tessellation/CellFunction.h"
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_2<K> Regular_triangulation;

void Power::getStandardNode(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, VectorXT &node) {
    assert(v1.rows() == 3 && v2.rows() == 3 && v3.rows() == 3);

    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double x2 = v2(0);
    double y2 = v2(1);
    double z2 = v2(2);
    double x3 = v3(0);
    double y3 = v3(1);
    double z3 = v3(2);

//    double rsq2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
//    double d2 = 0.5 + 0.5 * (z2 - z1) / rsq2;
//    double xp2 = x1 + d2 * (x2 - x1);
//    double yp2 = y1 + d2 * (y2 - y1);
//    double xl2 = -(y2 - y1);
//    double yl2 = (x2 - x1);
//    double rsq3 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1);
//    double d3 = 0.5 + 0.5 * (z3 - z1) / rsq3;
//    double xp3 = x1 + d3 * (x3 - x1);
//    double yp3 = y1 + d3 * (y3 - y1);
//    double xl3 = -(y3 - y1);
//    double yl3 = (x3 - x1);
//
//    double a2 = (yl3 * (xp3 - xp2) - xl3 * (yp3 - yp2)) / (xl2 * yl3 - xl3 * yl2);
//    double xn = xp2 + a2 * xl2;
//    double yn = yp2 + a2 * yl2;
//
//    node = {xn, yn};

    double xn, yn;
    // @formatter:off
    xn = ((-y3 + y2) * y1 * y1 + (-y2 * y2 + y3 * y3 + (-z2 + z3) * zmul - x2 * x2 + x3 * x3) * y1 + y2 * y2 * y3 + (-y3 * y3 + zmul * (-z3 + z1) + x1 * x1 - x3 * x3) * y2 - y3 * (zmul * (-z2 + z1) + x1 * x1 - x2 * x2)) / ((-0.2e1 * x2 + 0.2e1 * x3) * y1 + (0.2e1 * x1 - 0.2e1 * x3) * y2 + (-0.2e1 * x1 + 0.2e1 * x2) * y3);
    yn = ((x3 - x2) * x1 * x1 + (x2 * x2 - x3 * x3 + (-z3 + z2) * zmul + y2 * y2 - y3 * y3) * x1 - x3 * x2 * x2 + (x3 * x3 + (z3 - z1) * zmul - y1 * y1 + y3 * y3) * x2 + x3 * (zmul * (-z2 + z1) + y1 * y1 - y2 * y2)) / ((0.2e1 * y2 - 0.2e1 * y3) * x1 + (-0.2e1 * y1 + 0.2e1 * y3) * x2 + (0.2e1 * y1 - 0.2e1 * y2) * x3);
    // @formatter:on
    node = TV3(xn, yn, 0);
}

void
Power::getStandardNodeGradient(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, MatrixXT &nodeGrad) {
    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double x2 = v2(0);
    double y2 = v2(1);
    double z2 = v2(2);
    double x3 = v3(0);
    double y3 = v3(1);
    double z3 = v3(2);

    int n = 9;
    double gradX[n], gradY[n];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100,
            t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115, t116, t117, t118,
            t119, t120, t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132, t133, t134, t135, t136,
            t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147, t148, t149, t150, t151, t152, t153, t154,
            t155, t156, t157, t158, t159, t160, t161, t162, t163, t164, t165, t166, t167, t168, t169, t170, t171, t172,
            t173, t174, t175, t176, t177, t178, t179, t180, t181, t182, t183, t184, t185, t186, t187, t188, t189, t190,
            t191, t192, t193, t194, t195, t196, t197, t198, t199, t200, t201, t202, t203, t204, t205, t206, t207, t208,
            t209, t210, t211, t212, t213, t214, t215, t216, t217, t218, t219, t220, t221, t222, t223, t224, t225, t226,
            t227, t228, t229, t230, t231, t232, t233, t234, t235, t236, t237, t238, t239, t240, t241, t242, t243, t244,
            t245, t246, t247, t248, t249, t250;

    t1 = -y3 + y2;
    t2 = x2 - x3;
    t3 = x3 - x1;
    t4 = x2 - x1;
    t5 = t2 * y1 + t3 * y2 - t4 * y3;
    t6 = -t1;
    t7 = x2 * x2;
    t8 = y3 * y3;
    t9 = x3 * x3;
    t10 = y2 * y2;
    t11 = (-z2 + z3) * zmul - t7 + t8 + t9 - t10;
    t12 = x1 * x1;
    t13 = (z3 - z1) * zmul;
    t14 = zmul * (-z2 + z1);
    t15 = t6 * y1;
    t16 = (t15 - t11) * y1 + (-t12 + t8 + t9 + t13) * y2 + (-t10 - t7 + t12 + t14) * y3;
    t5 = 0.1e1 / t5;
    t17 = t16 * t5 / 0.2e1;
    t18 = 0.2e1;
    t16 = t16 * t5;
    t19 = y3 - y1;
    t20 = y1 * y1;
    t21 = y2 - y1;
    gradX[0] = -t5 * (x1 * t1 + t17 * t6);
    gradX[1] = -t5 * (-t15 * t18 + t16 * t2 + t11) / 0.2e1;
    gradX[2] = -zmul * t1 * t5 / 0.2e1;
    gradX[3] = t5 * (t17 * t19 - x2 * t19);
    gradX[4] = -t5 * (t18 * t19 * y2 + t16 * t3 + t12 - t13 + t20 - t8 - t9) / 0.2e1;
    gradX[5] = -zmul * t19 * t5 / 0.2e1;
    gradX[6] = t5 * (-t17 * t21 + x3 * t21);
    gradX[7] = -t5 * (-t18 * t21 * y3 - t16 * t4 + t10 - t12 - t14 - t20 + t7) / 0.2e1;
    gradX[8] = zmul * t21 * t5 / 0.2e1;

    t1 = x2 - x3;
    t2 = x2 * x2;
    t3 = y3 * y3;
    t4 = x3 * x3;
    t5 = y2 * y2;
    t6 = (-z2 + z3) * zmul - t2 + t3 + t4 - t5;
    t7 = t1 * x1;
    t8 = y3 - y2;
    t9 = y3 - y1;
    t10 = y2 - y1;
    t11 = -t10 * x3 - t8 * x1 + t9 * x2;
    t12 = y1 * y1;
    t13 = (z3 - z1) * zmul;
    t14 = zmul * (-z2 + z1);
    t11 = 0.1e1 / t11;
    t15 = ((t7 + t6) * x1 - (t4 - t12 + t3 + t13) * x2 + (t2 + t5 - t12 - t14) * x3) * t11;
    t16 = t15 / 0.2e1;
    t17 = x1 * x1;
    t18 = x3 - x1;
    t19 = x1 - x2;
    gradY[0] = t11 * (-t15 * t8 - t6 - 0.2e1 * t7) / 0.2e1;
    gradY[1] = t11 * (-t16 * t1 - y1 * t1);
    gradY[2] = -zmul * t1 * t11 / 0.2e1;
    gradY[3] = t11 * (t15 * t9 - 0.2e1 * x2 * t18 - t12 + t13 - t17 + t3 + t4) / 0.2e1;
    gradY[4] = -t11 * (t16 * t18 + y2 * t18);
    gradY[5] = -zmul * t18 * t11 / 0.2e1;
    gradY[6] = t11 * (-t15 * t10 - 0.2e1 * x3 * t19 + t12 + t14 + t17 - t2 - t5) / 0.2e1;
    gradY[7] = t11 * (-t16 * t19 - y3 * t19);
    gradY[8] = -zmul * t19 * t11 / 0.2e1;
    // @formatter:on

    nodeGrad = MatrixXT::Zero(CellFunction::nx, n);
    nodeGrad.row(0) = Eigen::Map<VectorXT>(gradX, n);
    nodeGrad.row(1) = Eigen::Map<VectorXT>(gradY, n);
}

void
Power::getStandardNodeHessian(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3,
                              std::vector<MatrixXT> &nodeHess) {
    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double x2 = v2(0);
    double y2 = v2(1);
    double z2 = v2(2);
    double x3 = v3(0);
    double y3 = v3(1);
    double z3 = v3(2);

    int n = 9;
    double unknown[n][n];
    nodeHess.resize(CellFunction::nx);
    nodeHess[2] = Eigen::MatrixXd::Zero(0, 0);

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100,
            t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115, t116, t117, t118,
            t119, t120, t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132, t133, t134, t135, t136,
            t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147, t148, t149, t150, t151, t152, t153, t154,
            t155, t156, t157, t158, t159, t160, t161, t162, t163, t164, t165, t166, t167, t168, t169, t170, t171, t172,
            t173, t174, t175, t176, t177, t178, t179, t180, t181, t182, t183, t184, t185, t186, t187, t188, t189, t190,
            t191, t192, t193, t194, t195, t196, t197, t198, t199, t200, t201, t202, t203, t204, t205, t206, t207, t208,
            t209, t210, t211, t212, t213, t214, t215, t216, t217, t218, t219, t220, t221, t222, t223, t224, t225, t226,
            t227, t228, t229, t230, t231, t232, t233, t234, t235, t236, t237, t238, t239, t240, t241, t242, t243, t244,
            t245, t246, t247, t248, t249, t250;

    t1 = -y3 + y2;
    t2 = 0.2e1 * x2;
    t3 = 0.2e1 * x3;
    t4 = -t2 + t3;
    t6 = 0.2e1 * x1;
    t7 = t6 - t3;
    t9 = -t6 + t2;
    t11 = y1 * t4 + y2 * t7 + y3 * t9;
    t12 = 0.1e1 / t11;
    t13 = 0.2e1 * t12 * t1;
    t17 = t11 * t11;
    t18 = 0.1e1 / t17;
    t19 = 0.2e1 * t18 * (y2 * x1 - y3 * x1);
    t20 = 0.2e1 * y2;
    t21 = 0.2e1 * y3;
    t22 = t20 - t21;
    t25 = y1 * y1;
    t27 = y2 * y2;
    t28 = y3 * y3;
    t30 = (-z2 + z3) * zmul;
    t31 = x2 * x2;
    t32 = x3 * x3;
    t37 = zmul * (-z3 + z1);
    t38 = x1 * x1;
    t42 = zmul * (-z2 + z1);
    t45 = t25 * t1 + y1 * (-t27 + t28 + t30 - t31 + t32) + y3 * t27 + y2 * (-t28 + t37 + t38 - t32) - (t42 + t38 - t31) * y3;
    t48 = 0.1e1 / t11 / t17 * t45;
    t49 = t22 * t22;
    t57 = t18 * (0.2e1 * y1 * t1 - t27 + t28 + t30 - t31 + t32);
    t62 = 0.2e1 * t4 * t22 * t48 - t4 * t19 - t22 * t57;
    t63 = zmul * y2;
    t64 = y3 * zmul;
    t66 = t18 * (t63 - t64);
    t67 = t22 * t66;
    t68 = 0.2e1 * y1;
    t69 = -t68 + t21;
    t74 = 0.2e1 * t18 * (-y1 * x2 + y3 * x2);
    t79 = 0.2e1 * t69 * t22 * t48 - t69 * t19 - t22 * t74;
    t81 = 0.2e1 * t12 * x1;
    t86 = 0.2e1 * y3 * y2;
    t88 = t18 * (-0.2e1 * y2 * y1 + t25 - t28 - t32 + t37 + t38 + t86);
    t94 = 0.2e1 * t18 * t45;
    t95 = 0.2e1 * t7 * t22 * t48 - t7 * t19 - t22 * t88 + t81 - t94;
    t96 = zmul * y1;
    t98 = t18 * (-t96 + t64);
    t99 = t22 * t98;
    t100 = t68 - t20;
    t105 = 0.2e1 * t18 * (y1 * x3 - y2 * x3);
    t110 = 0.2e1 * t100 * t22 * t48 - t100 * t19 - t22 * t105;
    t115 = t18 * (0.2e1 * y3 * y1 - t25 + t27 + t31 - t38 - t42 - t86);
    t120 = 0.2e1 * t9 * t22 * t48 - t22 * t115 - t9 * t19 - t81 + t94;
    t122 = t18 * (t96 - t63);
    t123 = t22 * t122;
    t126 = t4 * t4;
    t130 = t4 * t66;
    t132 = 0.2e1 * t12 * x2;
    t138 = 0.2e1 * t69 * t4 * t48 - t4 * t74 - t69 * t57 - t132 + t94;
    t140 = 0.2e1 * t12 * (-y2 + y1);
    t146 = 0.2e1 * t7 * t4 * t48 - t4 * t88 - t7 * t57 + t140;
    t147 = t12 * zmul;
    t149 = -t4 * t98 - t147;
    t151 = 0.2e1 * t12 * x3;
    t157 = 0.2e1 * t100 * t4 * t48 - t100 * t57 - t4 * t105 + t151 - t94;
    t159 = 0.2e1 * t12 * (y3 - y1);
    t165 = 0.2e1 * t9 * t4 * t48 - t4 * t115 - t9 * t57 + t159;
    t167 = -t4 * t122 + t147;
    t168 = t69 * t66;
    t170 = -t7 * t66 + t147;
    t171 = t100 * t66;
    t173 = -t9 * t66 - t147;
    t176 = t69 * t69;
    t185 = 0.2e1 * t7 * t69 * t48 - t69 * t88 - t7 * t74;
    t186 = t69 * t98;
    t192 = 0.2e1 * t100 * t69 * t48 - t100 * t74 - t69 * t105;
    t198 = 0.2e1 * t9 * t69 * t48 - t69 * t115 - t9 * t74 + t132 - t94;
    t199 = t69 * t122;
    t202 = t7 * t7;
    t206 = t7 * t98;
    t212 = 0.2e1 * t100 * t7 * t48 - t100 * t88 - t7 * t105 - t151 + t94;
    t218 = 0.2e1 * t9 * t7 * t48 - t7 * t115 - t9 * t88 + t13;
    t220 = -t7 * t122 - t147;
    t221 = t100 * t98;
    t223 = -t9 * t98 + t147;
    t226 = t100 * t100;
    t235 = 0.2e1 * t9 * t100 * t48 - t100 * t115 - t9 * t105;
    t236 = t100 * t122;
    t239 = t9 * t9;
    t243 = t9 * t122;
    unknown[0][0] = -0.2e1 * t22 * t19 + 0.2e1 * t49 * t48 + t13;
    unknown[0][1] = t62;
    unknown[0][2] = -t67;
    unknown[0][3] = t79;
    unknown[0][4] = t95;
    unknown[0][5] = -t99;
    unknown[0][6] = t110;
    unknown[0][7] = t120;
    unknown[0][8] = -t123;
    unknown[1][0] = t62;
    unknown[1][1] = 0.2e1 * t126 * t48 - 0.2e1 * t4 * t57 + t13;
    unknown[1][2] = -t130;
    unknown[1][3] = t138;
    unknown[1][4] = t146;
    unknown[1][5] = t149;
    unknown[1][6] = t157;
    unknown[1][7] = t165;
    unknown[1][8] = t167;
    unknown[2][0] = -t67;
    unknown[2][1] = -t130;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = -t168;
    unknown[2][4] = t170;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = -t171;
    unknown[2][7] = t173;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = t79;
    unknown[3][1] = t138;
    unknown[3][2] = -t168;
    unknown[3][3] = 0.2e1 * t176 * t48 - 0.2e1 * t69 * t74 + t159;
    unknown[3][4] = t185;
    unknown[3][5] = -t186;
    unknown[3][6] = t192;
    unknown[3][7] = t198;
    unknown[3][8] = -t199;
    unknown[4][0] = t95;
    unknown[4][1] = t146;
    unknown[4][2] = t170;
    unknown[4][3] = t185;
    unknown[4][4] = 0.2e1 * t202 * t48 - 0.2e1 * t7 * t88 + t159;
    unknown[4][5] = -t206;
    unknown[4][6] = t212;
    unknown[4][7] = t218;
    unknown[4][8] = t220;
    unknown[5][0] = -t99;
    unknown[5][1] = t149;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = -t186;
    unknown[5][4] = -t206;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = -t221;
    unknown[5][7] = t223;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = t110;
    unknown[6][1] = t157;
    unknown[6][2] = -t171;
    unknown[6][3] = t192;
    unknown[6][4] = t212;
    unknown[6][5] = -t221;
    unknown[6][6] = -0.2e1 * t100 * t105 + 0.2e1 * t226 * t48 + t140;
    unknown[6][7] = t235;
    unknown[6][8] = -t236;
    unknown[7][0] = t120;
    unknown[7][1] = t165;
    unknown[7][2] = t173;
    unknown[7][3] = t198;
    unknown[7][4] = t218;
    unknown[7][5] = t223;
    unknown[7][6] = t235;
    unknown[7][7] = -0.2e1 * t9 * t115 + 0.2e1 * t239 * t48 + t140;
    unknown[7][8] = -t243;
    unknown[8][0] = -t123;
    unknown[8][1] = t167;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = -t199;
    unknown[8][4] = t220;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = -t236;
    unknown[8][7] = -t243;
    unknown[8][8] = 0.0e0;

    nodeHess[0] = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], n, n);

    t1 = x3 - x2;
    t2 = 0.2e1 * y2;
    t3 = 0.2e1 * y3;
    t4 = t2 - t3;
    t6 = 0.2e1 * y1;
    t7 = -t6 + t3;
    t9 = t6 - t2;
    t11 = x1 * t4 + x2 * t7 + x3 * t9;
    t12 = 0.1e1 / t11;
    t13 = 0.2e1 * t12 * t1;
    t16 = x2 * x2;
    t17 = x3 * x3;
    t19 = (-z3 + z2) * zmul;
    t20 = y2 * y2;
    t21 = y3 * y3;
    t23 = t11 * t11;
    t24 = 0.1e1 / t23;
    t25 = t24 * (0.2e1 * x1 * t1 + t16 - t17 + t19 + t20 - t21);
    t28 = x1 * x1;
    t34 = (z3 - z1) * zmul;
    t35 = y1 * y1;
    t39 = zmul * (-z2 + z1);
    t42 = t28 * t1 + x1 * (t16 - t17 + t19 + t20 - t21) - t16 * x3 + x2 * (t17 + t34 - t35 + t21) + (t39 + t35 - t20) * x3;
    t45 = 0.1e1 / t11 / t23 * t42;
    t46 = t4 * t4;
    t50 = 0.2e1 * x2;
    t51 = 0.2e1 * x3;
    t52 = -t50 + t51;
    t57 = 0.2e1 * t24 * (-y1 * x2 + y1 * x3);
    t62 = 0.2e1 * t52 * t4 * t45 - t52 * t25 - t4 * t57;
    t63 = zmul * x2;
    t64 = x3 * zmul;
    t66 = t24 * (-t63 + t64);
    t67 = t4 * t66;
    t69 = 0.2e1 * t12 * (x2 - x1);
    t74 = 0.2e1 * x3 * x2;
    t76 = t24 * (0.2e1 * x2 * x1 + t17 + t21 - t28 + t34 - t35 - t74);
    t81 = 0.2e1 * t7 * t4 * t45 - t7 * t25 - t4 * t76 + t69;
    t83 = 0.2e1 * t12 * y2;
    t84 = 0.2e1 * x1;
    t85 = t84 - t51;
    t90 = 0.2e1 * t24 * (y2 * x1 - y2 * x3);
    t96 = 0.2e1 * t24 * t42;
    t97 = 0.2e1 * t85 * t4 * t45 - t85 * t25 - t4 * t90 + t83 - t96;
    t98 = t12 * zmul;
    t99 = zmul * x1;
    t101 = t24 * (t99 - t64);
    t103 = -t4 * t101 + t98;
    t105 = 0.2e1 * t12 * (-x3 + x1);
    t110 = t24 * (-0.2e1 * x3 * x1 - t16 - t20 + t28 + t35 + t39 + t74);
    t115 = 0.2e1 * t9 * t4 * t45 - t4 * t110 - t9 * t25 + t105;
    t117 = 0.2e1 * t12 * y3;
    t118 = -t84 + t50;
    t123 = 0.2e1 * t24 * (-y3 * x1 + y3 * x2);
    t128 = 0.2e1 * t118 * t4 * t45 - t118 * t25 - t4 * t123 - t117 + t96;
    t130 = t24 * (-t99 + t63);
    t132 = -t4 * t130 - t98;
    t135 = t52 * t52;
    t139 = t52 * t66;
    t141 = 0.2e1 * t12 * y1;
    t147 = 0.2e1 * t7 * t52 * t45 - t52 * t76 - t7 * t57 - t141 + t96;
    t153 = 0.2e1 * t85 * t52 * t45 - t52 * t90 - t85 * t57;
    t154 = t52 * t101;
    t160 = 0.2e1 * t9 * t52 * t45 - t52 * t110 - t9 * t57 + t141 - t96;
    t166 = 0.2e1 * t118 * t52 * t45 - t118 * t57 - t52 * t123;
    t167 = t52 * t130;
    t169 = -t7 * t66 - t98;
    t170 = t85 * t66;
    t172 = -t9 * t66 + t98;
    t173 = t118 * t66;
    t176 = t7 * t7;
    t185 = 0.2e1 * t85 * t7 * t45 - t7 * t90 - t85 * t76;
    t186 = t7 * t101;
    t192 = 0.2e1 * t9 * t7 * t45 - t7 * t110 - t9 * t76 + t13;
    t198 = 0.2e1 * t118 * t7 * t45 - t118 * t76 - t7 * t123 + t117 - t96;
    t200 = -t7 * t130 + t98;
    t203 = t85 * t85;
    t207 = t85 * t101;
    t213 = 0.2e1 * t9 * t85 * t45 - t85 * t110 - t9 * t90 - t83 + t96;
    t219 = 0.2e1 * t118 * t85 * t45 - t118 * t90 - t85 * t123;
    t220 = t85 * t130;
    t222 = -t9 * t101 - t98;
    t223 = t118 * t101;
    t226 = t9 * t9;
    t235 = 0.2e1 * t118 * t9 * t45 - t118 * t110 - t9 * t123;
    t236 = t9 * t130;
    t239 = t118 * t118;
    t243 = t118 * t130;
    unknown[0][0] = -0.2e1 * t4 * t25 + 0.2e1 * t46 * t45 + t13;
    unknown[0][1] = t62;
    unknown[0][2] = -t67;
    unknown[0][3] = t81;
    unknown[0][4] = t97;
    unknown[0][5] = t103;
    unknown[0][6] = t115;
    unknown[0][7] = t128;
    unknown[0][8] = t132;
    unknown[1][0] = t62;
    unknown[1][1] = 0.2e1 * t135 * t45 - 0.2e1 * t52 * t57 + t13;
    unknown[1][2] = -t139;
    unknown[1][3] = t147;
    unknown[1][4] = t153;
    unknown[1][5] = -t154;
    unknown[1][6] = t160;
    unknown[1][7] = t166;
    unknown[1][8] = -t167;
    unknown[2][0] = -t67;
    unknown[2][1] = -t139;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = t169;
    unknown[2][4] = -t170;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = t172;
    unknown[2][7] = -t173;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = t81;
    unknown[3][1] = t147;
    unknown[3][2] = t169;
    unknown[3][3] = 0.2e1 * t176 * t45 - 0.2e1 * t7 * t76 + t105;
    unknown[3][4] = t185;
    unknown[3][5] = -t186;
    unknown[3][6] = t192;
    unknown[3][7] = t198;
    unknown[3][8] = t200;
    unknown[4][0] = t97;
    unknown[4][1] = t153;
    unknown[4][2] = -t170;
    unknown[4][3] = t185;
    unknown[4][4] = 0.2e1 * t203 * t45 - 0.2e1 * t85 * t90 + t105;
    unknown[4][5] = -t207;
    unknown[4][6] = t213;
    unknown[4][7] = t219;
    unknown[4][8] = -t220;
    unknown[5][0] = t103;
    unknown[5][1] = -t154;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = -t186;
    unknown[5][4] = -t207;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = t222;
    unknown[5][7] = -t223;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = t115;
    unknown[6][1] = t160;
    unknown[6][2] = t172;
    unknown[6][3] = t192;
    unknown[6][4] = t213;
    unknown[6][5] = t222;
    unknown[6][6] = -0.2e1 * t9 * t110 + 0.2e1 * t226 * t45 + t69;
    unknown[6][7] = t235;
    unknown[6][8] = -t236;
    unknown[7][0] = t128;
    unknown[7][1] = t166;
    unknown[7][2] = -t173;
    unknown[7][3] = t198;
    unknown[7][4] = t219;
    unknown[7][5] = -t223;
    unknown[7][6] = t235;
    unknown[7][7] = -0.2e1 * t118 * t123 + 0.2e1 * t239 * t45 + t69;
    unknown[7][8] = -t243;
    unknown[8][0] = t132;
    unknown[8][1] = -t167;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = t200;
    unknown[8][4] = -t220;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = -t236;
    unknown[8][7] = -t243;
    unknown[8][8] = 0.0e0;

    nodeHess[1] = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], n, n);
    // @formatter:on
}

void Power::getBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, VectorXT &node) {
    assert(v1.rows() == 3 && v2.rows() == 3);

//    double rsq = (v2(0) - v1(0)) * (v2(0) - v1(0)) + (v2(1) - v1(1)) * (v2(1) - v1(1));
//    double d = 0.5 + 0.5 * (v2(2) - v1(2)) / rsq;
//
//    double x1 = d * v2(0) + (1 - d) * v1(0);
//    double y1 = d * v2(1) + (1 - d) * v1(1);
//    double x2 = x1 + (v2(1) - v1(1));
//    double y2 = y1 - (v2(0) - v1(0));
//    double x3 = b0(0);
//    double y3 = b0(1);
//    double x4 = b1(0);
//    double y4 = b1(1);
//
//    double t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (-(x4 - x3) * (y2 - y1) + (x2 - x1) * (y4 - y3));
//    double xn = x1 + t * (x2 - x1);
//    double yn = y1 + t * (y2 - y1);
//    node = {xn, yn};

    double v1x = v1(0);
    double v1y = v1(1);
    double v1z = v1(2);
    double v2x = v2(0);
    double v2y = v2(1);
    double v2z = v2(2);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    double xn, yn;
    // @formatter:off
    xn = ((0.5e0 * v1y * v1y - 0.1e1 * v1y * y4 - 0.5e0 * v2y * v2y + v2y * y4 + (-0.5e0 * v2z + 0.5e0 * v1z) * zmul + 0.5e0 * v1x * v1x - 0.5e0 * v2x * v2x) * x3 + (-0.5e0 * v1y * v1y + v1y * y3 + 0.5e0 * v2y * v2y - 0.1e1 * v2y * y3 + (0.5e0 * v2z - 0.5e0 * v1z) * zmul - 0.5e0 * v1x * v1x + 0.5e0 * v2x * v2x) * x4) / ((v1x - v2x) * x3 + (v2x - v1x) * x4 + (-v2y + v1y) * (-y4 + y3));
    yn = ((0.5e0 * v1x * v1x - 0.1e1 * x4 * v1x - 0.5e0 * v2x * v2x + x4 * v2x + (-0.5e0 * v2z + 0.5e0 * v1z) * zmul - 0.5e0 * v2y * v2y + 0.5e0 * v1y * v1y) * y3 + (-0.5e0 * v1x * v1x + x3 * v1x + 0.5e0 * v2x * v2x - 0.1e1 * x3 * v2x + (0.5e0 * v2z - 0.5e0 * v1z) * zmul + 0.5e0 * v2y * v2y - 0.5e0 * v1y * v1y) * y4) / ((-v2y + v1y) * y3 + (v2y - v1y) * y4 + (v1x - v2x) * (-x4 + x3));
    // @formatter:on
    node = TV3(xn, yn, 0);
}

void
Power::getBoundaryNodeGradient(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, MatrixXT &nodeGrad) {
    assert(v1.rows() == 3 && v2.rows() == 3);

    double v1x = v1(0);
    double v1y = v1(1);
    double v1z = v1(2);
    double v2x = v2(0);
    double v2y = v2(1);
    double v2z = v2(2);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    int n = 10;
    double gradX[n], gradY[n];

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100,
            t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115, t116, t117, t118,
            t119, t120, t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132, t133, t134, t135, t136,
            t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147, t148, t149, t150, t151, t152, t153, t154,
            t155, t156, t157, t158, t159, t160, t161, t162, t163, t164, t165, t166, t167, t168, t169, t170, t171, t172,
            t173, t174, t175, t176, t177, t178, t179, t180, t181, t182, t183, t184, t185, t186, t187, t188, t189, t190,
            t191, t192, t193, t194, t195, t196, t197, t198, t199, t200, t201, t202, t203, t204, t205, t206, t207, t208,
            t209, t210, t211, t212, t213, t214, t215, t216, t217, t218, t219, t220, t221, t222, t223, t224, t225, t226,
            t227, t228, t229, t230, t231, t232, t233, t234, t235, t236, t237, t238, t239, t240, t241, t242, t243, t244,
            t245, t246, t247, t248, t249, t250;

    t1 = x4 - x3;
    t2 = v2x - v1x;
    t3 = v2y - v1y;
    t4 = -y4 + y3;
    t5 = -t1 * t2 + t3 * t4;
    t6 = 0.1e1 / 0.2e1;
    t7 = t6 * ((v2z - v1z) * zmul - v1x * v1x - v1y * v1y + v2x * v2x + v2y * v2y);
    t8 = y4 * t3 - t7;
    t7 = -y3 * t3 + t7;
    t5 = 0.1e1 / t5;
    t9 = (t7 * x4 + t8 * x3) * t5;
    t10 = -t9 * t1;
    t4 = t9 * t4;
    t2 = t9 * t2;
    t3 = t3 * t5;
    t6 = -t6 * zmul * t1 * t5;
    gradX[0] = t5 * (v1x * t1 - t10);
    gradX[1] = t5 * (-t4 + (y4 - v1y) * x3 - (-v1y + y3) * x4);
    gradX[2] = -t6;
    gradX[3] = -t5 * (v2x * t1 - t10);
    gradX[4] = -t5 * (-t4 + (-v2y + y4) * x3 - (y3 - v2y) * x4);
    gradX[5] = t6;
    gradX[6] = -t5 * (-t2 + t8);
    gradX[7] = t3 * (t9 + x4);
    gradX[8] = -t5 * (t2 + t7);
    gradX[9] = -t3 * (t9 + x3);

    t1 = v2y - v1y;
    t2 = v2x - v1x;
    t3 = -x4 + x3;
    t4 = -y4 + y3;
    t5 = t1 * t4 + t2 * t3;
    t6 = 0.1e1 / 0.2e1;
    t7 = t6 * ((v2z - v1z) * zmul - v1x * v1x - v1y * v1y + v2x * v2x + v2y * v2y);
    t8 = x4 * t2 - t7;
    t7 = -x3 * t2 + t7;
    t5 = 0.1e1 / t5;
    t9 = (t7 * y4 + t8 * y3) * t5;
    t3 = t9 * t3;
    t10 = t9 * t4;
    t2 = t2 * t5;
    t1 = t9 * t1;
    t6 = t6 * zmul * t4 * t5;
    gradY[0] = t5 * ((x4 - v1x) * y3 - (-v1x + x3) * y4 - t3);
    gradY[1] = t5 * (-v1y * t4 - t10);
    gradY[2] = -t6;
    gradY[3] = -t5 * ((-v2x + x4) * y3 - (x3 - v2x) * y4 - t3);
    gradY[4] = -t5 * (-v2y * t4 - t10);
    gradY[5] = t6;
    gradY[6] = t2 * (t9 + y4);
    gradY[7] = -t5 * (t8 - t1);
    gradY[8] = -t2 * (t9 + y3);
    gradY[9] = -t5 * (t7 + t1);
    // @formatter:on

    nodeGrad = MatrixXT::Zero(CellFunction::nx, n);
    nodeGrad.row(0) = Eigen::Map<VectorXT>(gradX, n);
    nodeGrad.row(1) = Eigen::Map<VectorXT>(gradY, n);
}

void Power::getBoundaryNodeHessian(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1,
                                   std::vector<MatrixXT> &nodeHess) {
    assert(v1.rows() == 3 && v2.rows() == 3);

    double v1x = v1(0);
    double v1y = v1(1);
    double v1z = v1(2);
    double v2x = v2(0);
    double v2y = v2(1);
    double v2z = v2(2);
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    int n = 10;
    double hessX_c[n][n];
    double hessY_c[n][n];

    nodeHess.resize(CellFunction::nx);
    nodeHess[2] = Eigen::MatrixXd::Zero(0, 0);

    // @formatter:off
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100,
            t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115, t116, t117, t118,
            t119, t120, t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132, t133, t134, t135, t136,
            t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147, t148, t149, t150, t151, t152, t153, t154,
            t155, t156, t157, t158, t159, t160, t161, t162, t163, t164, t165, t166, t167, t168, t169, t170, t171, t172,
            t173, t174, t175, t176, t177, t178, t179, t180, t181, t182, t183, t184, t185, t186, t187, t188, t189, t190,
            t191, t192, t193, t194, t195, t196, t197, t198, t199, t200, t201, t202, t203, t204, t205, t206, t207, t208,
            t209, t210, t211, t212, t213, t214, t215, t216, t217, t218, t219, t220, t221, t222, t223, t224, t225, t226,
            t227, t228, t229, t230, t231, t232, t233, t234, t235, t236, t237, t238, t239, t240, t241, t242, t243, t244,
            t245, t246, t247, t248, t249, t250;

    t1 = -x4 + x3;
    t2 = v2x - v1x;
    t3 = v2y - v1y;
    t4 = -y4 + y3;
    t5 = t1 * t2 + t3 * t4;
    t6 = -v1x * t1;
    t7 = 0.1e1 / 0.2e1;
    t8 = t7 * ((v2z - v1z) * zmul - v1x * v1x - v1y * v1y + v2x * v2x + v2y * v2y);
    t9 = y4 * t3 - t8;
    t8 = -y3 * t3 + t8;
    t10 = t8 * x4 + t9 * x3;
    t5 = 0.1e1 / t5;
    t11 = pow(t5, 0.2e1);
    t12 = t10 * t5;
    t13 = t12 * t1;
    t14 = t11 * t1;
    t15 = t1 * t5;
    t16 = y4 - v1y;
    t17 = -v1y + y3;
    t18 = t16 * x3 - t17 * x4;
    t19 = t18 * t1;
    t20 = t6 * t4;
    t21 = 0.2e1 * t10 * t5 * t11;
    t22 = t21 * t1;
    t23 = t22 * t4;
    t24 = t11 * (t19 + t20) - t23;
    t25 = zmul * t1;
    t26 = -v2x * t1;
    t27 = -t14 * (t6 + t26) + t21 * pow(t1, 0.2e1);
    t28 = -v2y + y4;
    t29 = y3 - v2y;
    t30 = t28 * x3 - t29 * x4;
    t31 = t30 * t1;
    t20 = -t11 * (t20 + t31) + t23;
    t32 = t1 * t9;
    t33 = t2 * t6;
    t34 = t22 * t2;
    t35 = -(t5 * (t10 + t32 + t33) + v1x) * t5 + t34;
    t36 = x4 * t1;
    t37 = t11 * t3;
    t22 = t22 * t3;
    t38 = -t37 * (-t36 + t6) + t22;
    t39 = t1 * t8;
    t33 = -(t5 * (-t10 - t33 + t39) - v1x) * t5 - t34;
    t1 = x3 * t1;
    t40 = t37 * (-t1 + t6) - t22;
    t41 = t14 * t7 * t25;
    t42 = t12 * t4;
    t43 = t11 * t4;
    t44 = t26 * t4;
    t19 = -t11 * (t44 + t19) + t23;
    t45 = t21 * pow(t4, 0.2e1) - t43 * (t18 + t30);
    t46 = t2 * t18;
    t47 = t4 * t9;
    t48 = t21 * t4;
    t49 = t48 * t2;
    t16 = -(t5 * (t46 + t47) - t16) * t5 + t49;
    t50 = x4 * t4;
    t48 = t48 * t3;
    t51 = -t5 * (t5 * (t3 * (-t50 + t18) + t10) + x4) + t48;
    t52 = t4 * t8;
    t17 = -(t5 * (-t46 + t52) + t17) * t5 - t49;
    t4 = x3 * t4;
    t46 = t5 * (t5 * (t3 * (-t4 + t18) + t10) + x3) - t48;
    t53 = t43 * t7 * t25;
    t54 = t7 * t5 * (t25 * t5 * t2 - zmul);
    t7 = t37 * t7 * t25;
    t23 = -t23 + t11 * (t44 + t31);
    t25 = t2 * t26;
    t31 = -t34 + (t5 * (t10 + t32 + t25) + v2x) * t5;
    t32 = -t22 + t37 * (-t36 + t26);
    t25 = t34 - (t5 * (t10 - t39 + t25) + v2x) * t5;
    t1 = t22 - t37 * (-t1 + t26);
    t22 = t2 * t30;
    t28 = -t49 + (t5 * (t47 + t22) - t28) * t5;
    t34 = -t48 + t5 * (t5 * (t3 * (-t50 + t30) + t10) + x4);
    t22 = (t5 * (-t22 + t52) + t29) * t5 + t49;
    t4 = -t5 * (t5 * (t3 * (-t4 + t30) + t10) + x3) + t48;
    t10 = t12 * t2;
    t29 = t11 * t2;
    t36 = t2 * x4;
    t39 = t21 * t2 * t3;
    t44 = -t37 * (t36 - t9) - t39;
    t47 = -t29 * (t9 - t8) + t21 * pow(t2, 0.2e1);
    t2 = t2 * x3;
    t48 = t3 * t5;
    t49 = -t48 * (t5 * (-t2 + t9) + 0.1e1) + t39;
    t3 = pow(t3, 0.2e1);
    t11 = t11 * t3;
    t5 = t48 * (t5 * (t36 + t8) + 0.1e1) + t39;
    t3 = t11 * (x3 + x4) + t21 * t3;
    t2 = -t37 * (t2 + t8) - t39;
    hessX_c[0][0] = -0.2e1 * t14 * (-t6 + t13) - t15;
    hessX_c[0][1] = t24;
    hessX_c[0][2] = -t41;
    hessX_c[0][3] = t27;
    hessX_c[0][4] = t20;
    hessX_c[0][5] = t41;
    hessX_c[0][6] = t35;
    hessX_c[0][7] = t38;
    hessX_c[0][8] = t33;
    hessX_c[0][9] = t40;
    hessX_c[1][0] = t24;
    hessX_c[1][1] = 0.2e1 * t43 * (t18 - t42) - t15;
    hessX_c[1][2] = -t53;
    hessX_c[1][3] = t19;
    hessX_c[1][4] = t45;
    hessX_c[1][5] = t53;
    hessX_c[1][6] = t16;
    hessX_c[1][7] = t51;
    hessX_c[1][8] = t17;
    hessX_c[1][9] = t46;
    hessX_c[2][0] = -t41;
    hessX_c[2][1] = -t53;
    hessX_c[2][2] = 0;
    hessX_c[2][3] = t41;
    hessX_c[2][4] = t53;
    hessX_c[2][5] = 0;
    hessX_c[2][6] = t54;
    hessX_c[2][7] = t7;
    hessX_c[2][8] = -t54;
    hessX_c[2][9] = -t7;
    hessX_c[3][0] = t27;
    hessX_c[3][1] = t19;
    hessX_c[3][2] = t41;
    hessX_c[3][3] = 0.2e1 * t14 * (t26 - t13) + t15;
    hessX_c[3][4] = t23;
    hessX_c[3][5] = -t41;
    hessX_c[3][6] = t31;
    hessX_c[3][7] = t32;
    hessX_c[3][8] = t25;
    hessX_c[3][9] = t1;
    hessX_c[4][0] = t20;
    hessX_c[4][1] = t45;
    hessX_c[4][2] = t53;
    hessX_c[4][3] = t23;
    hessX_c[4][4] = 0.2e1 * t43 * (t30 - t42) + t15;
    hessX_c[4][5] = -t53;
    hessX_c[4][6] = t28;
    hessX_c[4][7] = t34;
    hessX_c[4][8] = t22;
    hessX_c[4][9] = t4;
    hessX_c[5][0] = t41;
    hessX_c[5][1] = t53;
    hessX_c[5][2] = 0;
    hessX_c[5][3] = -t41;
    hessX_c[5][4] = -t53;
    hessX_c[5][5] = 0;
    hessX_c[5][6] = -t54;
    hessX_c[5][7] = -t7;
    hessX_c[5][8] = t54;
    hessX_c[5][9] = t7;
    hessX_c[6][0] = t35;
    hessX_c[6][1] = t16;
    hessX_c[6][2] = t54;
    hessX_c[6][3] = t31;
    hessX_c[6][4] = t28;
    hessX_c[6][5] = -t54;
    hessX_c[6][6] = 0.2e1 * t29 * (t9 - t10);
    hessX_c[6][7] = t44;
    hessX_c[6][8] = t47;
    hessX_c[6][9] = t49;
    hessX_c[7][0] = t38;
    hessX_c[7][1] = t51;
    hessX_c[7][2] = t7;
    hessX_c[7][3] = t32;
    hessX_c[7][4] = t34;
    hessX_c[7][5] = -t7;
    hessX_c[7][6] = t44;
    hessX_c[7][7] = -0.2e1 * t11 * (t12 + x4);
    hessX_c[7][8] = t5;
    hessX_c[7][9] = t3;
    hessX_c[8][0] = t33;
    hessX_c[8][1] = t17;
    hessX_c[8][2] = -t54;
    hessX_c[8][3] = t25;
    hessX_c[8][4] = t22;
    hessX_c[8][5] = t54;
    hessX_c[8][6] = t47;
    hessX_c[8][7] = t5;
    hessX_c[8][8] = -0.2e1 * t29 * (t8 + t10);
    hessX_c[8][9] = t2;
    hessX_c[9][0] = t40;
    hessX_c[9][1] = t46;
    hessX_c[9][2] = -t7;
    hessX_c[9][3] = t1;
    hessX_c[9][4] = t4;
    hessX_c[9][5] = t7;
    hessX_c[9][6] = t49;
    hessX_c[9][7] = t3;
    hessX_c[9][8] = t2;
    hessX_c[9][9] = -0.2e1 * t11 * (t12 + x3);

    t1 = -y4 + y3;
    t2 = v2y - v1y;
    t3 = v2x - v1x;
    t4 = -x4 + x3;
    t5 = t1 * t2 + t3 * t4;
    t6 = x4 - v1x;
    t7 = -v1x + x3;
    t8 = t6 * y3 - t7 * y4;
    t9 = 0.1e1 / 0.2e1;
    t10 = t9 * ((v2z - v1z) * zmul - v1x * v1x - v1y * v1y + v2x * v2x + v2y * v2y);
    t11 = x4 * t3 - t10;
    t10 = -x3 * t3 + t10;
    t12 = t10 * y4 + t11 * y3;
    t5 = 0.1e1 / t5;
    t13 = pow(t5, 0.2e1);
    t14 = t12 * t5;
    t15 = t14 * t4;
    t16 = t13 * t4;
    t17 = t1 * t5;
    t18 = -v1y * t1;
    t19 = t8 * t1;
    t20 = t18 * t4;
    t21 = 0.2e1 * t12 * t5 * t13;
    t22 = t21 * t4;
    t23 = t22 * t1;
    t24 = -t23 + t13 * (t19 + t20);
    t25 = zmul * t1;
    t26 = -v2x + x4;
    t27 = x3 - v2x;
    t28 = t26 * y3 - t27 * y4;
    t29 = t21 * pow(t4, 0.2e1) - t16 * (t8 + t28);
    t30 = -v2y * t1;
    t31 = t30 * t4;
    t19 = t23 - t13 * (t19 + t31);
    t32 = y4 * t4;
    t33 = t22 * t3;
    t34 = t33 + t5 * (t5 * (t3 * (t32 - t8) - t12) - y4);
    t35 = t2 * t8;
    t36 = t4 * t11;
    t22 = t22 * t2;
    t6 = t22 + (t5 * (-t35 - t36) + t6) * t5;
    t37 = y3 * t4;
    t38 = -t33 - t5 * (t5 * (t3 * (t37 - t8) - t12) - y3);
    t4 = t4 * t10;
    t7 = -t22 + (t5 * (-t4 + t35) - t7) * t5;
    t35 = t16 * t9 * t25;
    t39 = t14 * t1;
    t40 = t13 * t1;
    t41 = t28 * t1;
    t20 = t23 - t13 * (t20 + t41);
    t42 = t21 * pow(t1, 0.2e1) - t40 * (t18 + t30);
    t43 = y4 * t1;
    t44 = t13 * t3;
    t45 = t21 * t1;
    t46 = t45 * t3;
    t47 = t46 + t44 * (t43 - t18);
    t48 = t2 * t18;
    t49 = t1 * t11;
    t45 = t45 * t2;
    t50 = t45 - (t5 * (t12 + t48 + t49) + v1y) * t5;
    t51 = y3 * t1;
    t52 = -t46 - t44 * (t51 - t18);
    t1 = t1 * t10;
    t48 = -t45 + (t5 * (t12 + t48 - t1) + v1y) * t5;
    t53 = t40 * t9 * t25;
    t54 = t9 * t5 * (t25 * t5 * t2 - zmul);
    t9 = t44 * t9 * t25;
    t23 = -t23 + t13 * (t31 + t41);
    t25 = -t33 - t5 * (t5 * (t3 * (t32 - t28) - t12) - y4);
    t31 = t2 * t28;
    t26 = -t22 - (t5 * (-t31 - t36) + t26) * t5;
    t32 = t33 + t5 * (t5 * (t3 * (t37 - t28) - t12) - y3);
    t4 = t22 + (t5 * (t4 - t31) + t27) * t5;
    t22 = -t46 - t44 * (t43 - t30);
    t27 = t2 * t30;
    t31 = -t45 + (t5 * (t12 + t49 + t27) + v2y) * t5;
    t33 = t46 + t44 * (t51 - t30);
    t1 = t45 - (t5 * (t12 + t27 - t1) + v2y) * t5;
    t12 = pow(t3, 0.2e1);
    t27 = t12 * t13;
    t36 = t2 * y4;
    t37 = t21 * t3 * t2;
    t41 = -t37 + t44 * (-t36 + t11);
    t12 = t27 * (y3 + y4) + t21 * t12;
    t3 = t3 * t5;
    t36 = t37 + t3 * (t5 * (t36 + t10) + 0.1e1);
    t43 = t14 * t2;
    t13 = t13 * t2;
    t45 = t2 * y3;
    t3 = t37 - t3 * (t5 * (-t45 + t11) + 0.1e1);
    t2 = -t13 * (t11 - t10) + t21 * pow(t2, 0.2e1);
    t5 = -t37 - t44 * (t45 + t10);
    hessY_c[0][0] = 0.2e1 * t16 * (-t15 + t8) - t17;
    hessY_c[0][1] = t24;
    hessY_c[0][2] = -t35;
    hessY_c[0][3] = t29;
    hessY_c[0][4] = t19;
    hessY_c[0][5] = t35;
    hessY_c[0][6] = t34;
    hessY_c[0][7] = t6;
    hessY_c[0][8] = t38;
    hessY_c[0][9] = t7;
    hessY_c[1][0] = t24;
    hessY_c[1][1] = 0.2e1 * t40 * (-t39 + t18) - t17;
    hessY_c[1][2] = -t53;
    hessY_c[1][3] = t20;
    hessY_c[1][4] = t42;
    hessY_c[1][5] = t53;
    hessY_c[1][6] = t47;
    hessY_c[1][7] = t50;
    hessY_c[1][8] = t52;
    hessY_c[1][9] = t48;
    hessY_c[2][0] = -t35;
    hessY_c[2][1] = -t53;
    hessY_c[2][2] = 0;
    hessY_c[2][3] = t35;
    hessY_c[2][4] = t53;
    hessY_c[2][5] = 0;
    hessY_c[2][6] = t9;
    hessY_c[2][7] = t54;
    hessY_c[2][8] = -t9;
    hessY_c[2][9] = -t54;
    hessY_c[3][0] = t29;
    hessY_c[3][1] = t20;
    hessY_c[3][2] = t35;
    hessY_c[3][3] = -0.2e1 * t16 * (t15 - t28) + t17;
    hessY_c[3][4] = t23;
    hessY_c[3][5] = -t35;
    hessY_c[3][6] = t25;
    hessY_c[3][7] = t26;
    hessY_c[3][8] = t32;
    hessY_c[3][9] = t4;
    hessY_c[4][0] = t19;
    hessY_c[4][1] = t42;
    hessY_c[4][2] = t53;
    hessY_c[4][3] = t23;
    hessY_c[4][4] = -0.2e1 * t40 * (t39 - t30) + t17;
    hessY_c[4][5] = -t53;
    hessY_c[4][6] = t22;
    hessY_c[4][7] = t31;
    hessY_c[4][8] = t33;
    hessY_c[4][9] = t1;
    hessY_c[5][0] = t35;
    hessY_c[5][1] = t53;
    hessY_c[5][2] = 0;
    hessY_c[5][3] = -t35;
    hessY_c[5][4] = -t53;
    hessY_c[5][5] = 0;
    hessY_c[5][6] = -t9;
    hessY_c[5][7] = -t54;
    hessY_c[5][8] = t9;
    hessY_c[5][9] = t54;
    hessY_c[6][0] = t34;
    hessY_c[6][1] = t47;
    hessY_c[6][2] = t9;
    hessY_c[6][3] = t25;
    hessY_c[6][4] = t22;
    hessY_c[6][5] = -t9;
    hessY_c[6][6] = -0.2e1 * t27 * (t14 + y4);
    hessY_c[6][7] = t41;
    hessY_c[6][8] = t12;
    hessY_c[6][9] = t36;
    hessY_c[7][0] = t6;
    hessY_c[7][1] = t50;
    hessY_c[7][2] = t54;
    hessY_c[7][3] = t26;
    hessY_c[7][4] = t31;
    hessY_c[7][5] = -t54;
    hessY_c[7][6] = t41;
    hessY_c[7][7] = 0.2e1 * t13 * (-t43 + t11);
    hessY_c[7][8] = t3;
    hessY_c[7][9] = t2;
    hessY_c[8][0] = t38;
    hessY_c[8][1] = t52;
    hessY_c[8][2] = -t9;
    hessY_c[8][3] = t32;
    hessY_c[8][4] = t33;
    hessY_c[8][5] = t9;
    hessY_c[8][6] = t12;
    hessY_c[8][7] = t3;
    hessY_c[8][8] = -0.2e1 * t27 * (t14 + y3);
    hessY_c[8][9] = t5;
    hessY_c[9][0] = t7;
    hessY_c[9][1] = t48;
    hessY_c[9][2] = -t54;
    hessY_c[9][3] = t4;
    hessY_c[9][4] = t1;
    hessY_c[9][5] = t54;
    hessY_c[9][6] = t36;
    hessY_c[9][7] = t2;
    hessY_c[9][8] = t5;
    hessY_c[9][9] = -0.2e1 * t13 * (t43 + t10);
    // @formatter:on

    nodeHess[0] = Eigen::Map<Eigen::MatrixXd>(&hessX_c[0][0], n, n);
    nodeHess[1] = Eigen::Map<Eigen::MatrixXd>(&hessY_c[0][0], n, n);
}

int idxClosest(const TV &p, const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices3d.segment<2>(i * 3);
        double d = (p2 - p).norm();
        if (d < dmin) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

VectorXi Power::powerDualCGAL(const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;

    std::vector<Regular_triangulation::Weighted_point> wpoints;
    for (int i = 0; i < n_vtx; i++) {
        TV3 v = vertices3d.segment<3>(i * 3);
        Regular_triangulation::Weighted_point wp({v(0), v(1)}, -zmul * v(2));
        wpoints.push_back(wp);
    }
    Regular_triangulation rt(wpoints.begin(), wpoints.end());

    VectorXi tri(rt.number_of_faces() * 3);
    int f = 0;
    for (auto it = rt.faces_begin(); it != rt.faces_end(); it++) {

        auto v0 = it->vertex(0)->point();
        TV V0 = {v0.x(), v0.y()};
        auto v1 = it->vertex(1)->point();
        TV V1 = {v1.x(), v1.y()};
        auto v2 = it->vertex(2)->point();
        TV V2 = {v2.x(), v2.y()};

        int i0 = idxClosest(V0, vertices3d);
        int i1 = idxClosest(V1, vertices3d);
        int i2 = idxClosest(V2, vertices3d);

        tri.segment<3>(f * 3) = IV3(i0, i1, i2);
        f++;
    }

    return tri;
}

//VectorXi Power::powerDualNaive(const VectorXT &vertices3d) {
//    int n_vtx = vertices3d.rows() / 3;
//    std::vector<int> tri1;
//    std::vector<int> tri2;
//    std::vector<int> tri3;
//
//    for (int i = 0; i < n_vtx; i++) {
//        TV3 vi = vertices3d.segment<3>(i * 3);
//        std::vector<int> neighbors;
//
//        for (int j = 0; j < n_vtx; j++) {
//            if (j == i) continue;
//
//            TV3 vj = vertices3d.segment<3>(j * 3);
//            TV3 line = (vj - vi).cross(TV3(0, 0, 1));
//
//            double dmin = INFINITY;
//            double dmax = -INFINITY;
//
//            for (int k = 0; k < n_vtx; k++) {
//                if (k == i || k == j) continue;
//
//                TV3 vk = vertices3d.segment<3>(k * 3);
//                TV vc2d;
//                getNode(vi, vj, vk, vc2d);
//                TV3 vc = {vc2d(0), vc2d(1), 0};
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
//        double xc = vertices3d(i * 3 + 0);
//        double yc = vertices3d(i * 3 + 1);
//
//        std::sort(neighbors.begin(), neighbors.end(), [vertices3d, xc, yc](int a, int b) {
//            double xa = vertices3d(a * 3 + 0);
//            double ya = vertices3d(a * 3 + 1);
//            double angle_a = atan2(ya - yc, xa - xc);
//
//            double xb = vertices3d(b * 3 + 0);
//            double yb = vertices3d(b * 3 + 1);
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
//                    double x1 = vertices3d(v1 * 3 + 0);
//                    double y1 = vertices3d(v1 * 3 + 1);
//                    double x2 = vertices3d(v2 * 3 + 0);
//                    double y2 = vertices3d(v2 * 3 + 1);
//                    double x3 = vertices3d(v3 * 3 + 0);
//                    double y3 = vertices3d(v3 * 3 + 1);
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

VectorXi Power::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    VectorXT vertices3d = combineVerticesParams(vertices, params);
    return powerDualCGAL(vertices3d);
}

VectorXT Power::getDefaultVertexParams(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    VectorXT z = VectorXT::Zero(n_vtx);

    return z;
}

