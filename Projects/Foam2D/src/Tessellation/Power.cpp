#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "Projects/Foam2D/include/Energy/CodeGen.h"
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_2<K> Regular_triangulation;

namespace PowerTVars {
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
}
using namespace PowerTVars;

void Power::getNode(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, TV &node) {
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
    xn = ((-y3 + y2) * y1 * y1 + (-x2 * x2 + x3 * x3 - y2 * y2 + y3 * y3 - z2 + z3) * y1 + y2 * y2 * y3 + (x1 * x1 - x3 * x3 - y3 * y3 + z1 - z3) * y2 - y3 * (x1 * x1 - x2 * x2 + z1 - z2)) / ((-0.2e1 * x2 + 0.2e1 * x3) * y1 + (0.2e1 * x1 - 0.2e1 * x3) * y2 + (-0.2e1 * x1 + 0.2e1 * x2) * y3);
    yn = ((-x2 + x3) * x1 * x1 + (x2 * x2 - x3 * x3 + y2 * y2 - y3 * y3 + z2 - z3) * x1 - x3 * x2 * x2 + (x3 * x3 - y1 * y1 + y3 * y3 - z1 + z3) * x2 + x3 * (y1 * y1 - y2 * y2 + z1 - z2)) / ((0.2e1 * y2 - 0.2e1 * y3) * x1 + (-0.2e1 * y1 + 0.2e1 * y3) * x2 + (0.2e1 * y1 - 0.2e1 * y2) * x3);
    // @formatter:on
    node = {xn, yn};
}

void
Power::getNodeGradient(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, VectorXT &gradX, VectorXT &gradY) {
    assert(v1.rows() == 3 && v2.rows() == 3 && v3.rows() == 3 && gradX.rows() == 9 && gradY.rows() == 9);

    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double x2 = v2(0);
    double y2 = v2(1);
    double z2 = v2(2);
    double x3 = v3(0);
    double y3 = v3(1);
    double z3 = v3(2);

    // @formatter:off
    t1 = -y3 + y2;
    t2 = -x2 + x3;
    t3 = x3 - x1;
    t4 = x2 - x1;
    t5 = t2 * y1 - t3 * y2 + t4 * y3;
    t6 = -t1;
    t7 = y2 * y2;
    t8 = y3 * y3;
    t9 = x2 * x2;
    t10 = x3 * x3;
    t11 = t8 - t9 + t10 - t7 + z3 - z2;
    t12 = x1 * x1;
    t13 = t6 * y1;
    t14 = (t11 - t13) * y1 - (t8 + t10 + z3 - t12 - z1) * y2 + (t9 - t12 - z1 + z2 + t7) * y3;
    t5 = 0.1e1 / t5;
    t15 = t14 * t5 / 0.2e1;
    t16 = 0.2e1;
    t14 = t14 * t5;
    t17 = y3 - y1;
    t18 = y2 - y1;
    gradX[0] = t5 * (x1 * t1 + t15 * t6);
    gradX[1] = -t5 * (t13 * t16 + t14 * t2 - t11) / 0.2e1;
    gradX[2] = -t6 * t5 / 0.2e1;
    gradX[3] = t5 * (-t15 * t17 + x2 * t17);
    gradX[4] = t5 * (t16 * t17 * y2 + t14 * t3 + y1 * y1 - t10 + t12 - t8 + z1 - z3) / 0.2e1;
    gradX[5] = t17 * t5 / 0.2e1;
    gradX[6] = t5 * (t15 * t18 - x3 * t18);
    gradX[7] = -t5 * (t16 * t18 * y3 + t14 * t4 + y1 * y1 + t12 - t7 - t9 + z1 - z2) / 0.2e1;
    gradX[8] = -t18 * t5 / 0.2e1;

    t1 = -x2 + x3;
    t2 = y2 * y2;
    t3 = y3 * y3;
    t4 = x2 * x2;
    t5 = x3 * x3;
    t6 = -t2 + t3 - t4 + t5 + z3 - z2;
    t7 = 0.2e1;
    t8 = t1 * x1;
    t9 = y3 - y2;
    t10 = y3 - y1;
    t11 = y2 - y1;
    t12 = -t10 * x2 + t11 * x3 + t9 * x1;
    t13 = y1 * y1;
    t12 = 0.1e1 / t12;
    t14 = ((t8 - t6) * x1 + (t3 + t5 - t13 + z3 - z1) * x2 + (-t4 - t2 + t13 + z1 - z2) * x3) * t12;
    t15 = t14 / 0.2e1;
    t16 = x3 - x1;
    t17 = x2 - x1;
    gradY[0] = -t12 * (-t14 * t9 + t8 * t7 - t6) / 0.2e1;
    gradY[1] = -t12 * (t15 * t1 + y1 * t1);
    gradY[2] = -t1 * t12 / 0.2e1;
    gradY[3] = -t12 * (-t16 * t7 * x2 + t14 * t10 - x1 * x1 - t13 + t3 + t5 - z1 + z3) / 0.2e1;
    gradY[4] = t12 * (t15 * t16 + y2 * t16);
    gradY[5] = t16 * t12 / 0.2e1;
    gradY[6] = t12 * (-t17 * t7 * x3 + t14 * t11 - x1 * x1 - t13 + t2 + t4 - z1 + z2) / 0.2e1;
    gradY[7] = -t12 * (t15 * t17 + y3 * t17);
    gradY[8] = -t17 * t12 / 0.2e1;
    // @formatter:on
}

void
Power::getNodeHessian(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3, MatrixXT &hessX, MatrixXT &hessY) {
    assert(v1.rows() == 3 && v2.rows() == 3 && v3.rows() == 3 && hessX.rows() == 9 && hessX.cols() == 9 &&
           hessY.rows() == 9 && hessY.cols() == 9);

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

    // @formatter:off
    Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], n, n) = hessX;

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
    t27 = x2 * x2;
    t28 = x3 * x3;
    t29 = y2 * y2;
    t30 = y3 * y3;
    t34 = x1 * x1;
    t39 = t25 * t1 + y1 * (-t27 + t28 - t29 + t30 - z2 + z3) + y3 * t29 + y2 * (t34 - t28 - t30 + z1 - z3) - (t34 - t27 + z1 - z2) * y3;
    t42 = 0.1e1 / t11 / t17 * t39;
    t43 = t22 * t22;
    t51 = t18 * (0.2e1 * y1 * t1 - t27 + t28 - t29 + t30 - z2 + z3);
    t56 = 0.2e1 * t4 * t22 * t42 - t4 * t19 - t22 * t51;
    t57 = t18 * t1;
    t58 = t22 * t57;
    t59 = 0.2e1 * y1;
    t60 = -t59 + t21;
    t65 = 0.2e1 * t18 * (-y1 * x2 + y3 * x2);
    t70 = 0.2e1 * t60 * t22 * t42 - t60 * t19 - t22 * t65;
    t72 = 0.2e1 * t12 * x1;
    t77 = 0.2e1 * y3 * y2;
    t79 = t18 * (-0.2e1 * y2 * y1 + t25 - t28 - t30 + t34 + t77 + z1 - z3);
    t85 = 0.2e1 * t18 * t39;
    t86 = 0.2e1 * t7 * t22 * t42 - t7 * t19 - t22 * t79 + t72 - t85;
    t87 = y3 - y1;
    t88 = t18 * t87;
    t89 = t22 * t88;
    t90 = t59 - t20;
    t95 = 0.2e1 * t18 * (y1 * x3 - y2 * x3);
    t100 = 0.2e1 * t90 * t22 * t42 - t90 * t19 - t22 * t95;
    t105 = t18 * (0.2e1 * y3 * y1 - t25 + t27 + t29 - t34 - t77 - z1 + z2);
    t110 = 0.2e1 * t9 * t22 * t42 - t22 * t105 - t9 * t19 - t72 + t85;
    t111 = -y2 + y1;
    t112 = t18 * t111;
    t113 = t22 * t112;
    t116 = t4 * t4;
    t120 = t4 * t57;
    t122 = 0.2e1 * t12 * x2;
    t128 = 0.2e1 * t60 * t4 * t42 - t4 * t65 - t60 * t51 - t122 + t85;
    t129 = 0.2e1 * t12 * t111;
    t135 = 0.2e1 * t7 * t4 * t42 - t4 * t79 - t7 * t51 + t129;
    t137 = -t4 * t88 - t12;
    t139 = 0.2e1 * t12 * x3;
    t145 = 0.2e1 * t90 * t4 * t42 - t4 * t95 - t90 * t51 + t139 - t85;
    t146 = 0.2e1 * t12 * t87;
    t152 = 0.2e1 * t4 * t9 * t42 - t4 * t105 - t9 * t51 + t146;
    t154 = -t4 * t112 + t12;
    t155 = t60 * t57;
    t157 = -t7 * t57 + t12;
    t158 = t90 * t57;
    t160 = -t9 * t57 - t12;
    t163 = t60 * t60;
    t172 = 0.2e1 * t7 * t60 * t42 - t60 * t79 - t7 * t65;
    t173 = t60 * t88;
    t179 = 0.2e1 * t90 * t60 * t42 - t60 * t95 - t90 * t65;
    t185 = 0.2e1 * t9 * t60 * t42 - t60 * t105 - t9 * t65 + t122 - t85;
    t186 = t60 * t112;
    t189 = t7 * t7;
    t193 = t7 * t88;
    t199 = 0.2e1 * t90 * t7 * t42 - t7 * t95 - t90 * t79 - t139 + t85;
    t205 = 0.2e1 * t9 * t7 * t42 - t7 * t105 - t9 * t79 + t13;
    t207 = -t7 * t112 - t12;
    t208 = t90 * t88;
    t210 = -t9 * t88 + t12;
    t213 = t90 * t90;
    t222 = 0.2e1 * t9 * t90 * t42 - t90 * t105 - t9 * t95;
    t223 = t90 * t112;
    t226 = t9 * t9;
    t230 = t9 * t112;
    unknown[0][0] = -0.2e1 * t22 * t19 + 0.2e1 * t43 * t42 + t13;
    unknown[0][1] = t56;
    unknown[0][2] = -t58;
    unknown[0][3] = t70;
    unknown[0][4] = t86;
    unknown[0][5] = -t89;
    unknown[0][6] = t100;
    unknown[0][7] = t110;
    unknown[0][8] = -t113;
    unknown[1][0] = t56;
    unknown[1][1] = 0.2e1 * t116 * t42 - 0.2e1 * t4 * t51 + t13;
    unknown[1][2] = -t120;
    unknown[1][3] = t128;
    unknown[1][4] = t135;
    unknown[1][5] = t137;
    unknown[1][6] = t145;
    unknown[1][7] = t152;
    unknown[1][8] = t154;
    unknown[2][0] = -t58;
    unknown[2][1] = -t120;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = -t155;
    unknown[2][4] = t157;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = -t158;
    unknown[2][7] = t160;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = t70;
    unknown[3][1] = t128;
    unknown[3][2] = -t155;
    unknown[3][3] = 0.2e1 * t163 * t42 - 0.2e1 * t60 * t65 + t146;
    unknown[3][4] = t172;
    unknown[3][5] = -t173;
    unknown[3][6] = t179;
    unknown[3][7] = t185;
    unknown[3][8] = -t186;
    unknown[4][0] = t86;
    unknown[4][1] = t135;
    unknown[4][2] = t157;
    unknown[4][3] = t172;
    unknown[4][4] = 0.2e1 * t189 * t42 - 0.2e1 * t7 * t79 + t146;
    unknown[4][5] = -t193;
    unknown[4][6] = t199;
    unknown[4][7] = t205;
    unknown[4][8] = t207;
    unknown[5][0] = -t89;
    unknown[5][1] = t137;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = -t173;
    unknown[5][4] = -t193;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = -t208;
    unknown[5][7] = t210;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = t100;
    unknown[6][1] = t145;
    unknown[6][2] = -t158;
    unknown[6][3] = t179;
    unknown[6][4] = t199;
    unknown[6][5] = -t208;
    unknown[6][6] = 0.2e1 * t213 * t42 - 0.2e1 * t90 * t95 + t129;
    unknown[6][7] = t222;
    unknown[6][8] = -t223;
    unknown[7][0] = t110;
    unknown[7][1] = t152;
    unknown[7][2] = t160;
    unknown[7][3] = t185;
    unknown[7][4] = t205;
    unknown[7][5] = t210;
    unknown[7][6] = t222;
    unknown[7][7] = -0.2e1 * t9 * t105 + 0.2e1 * t226 * t42 + t129;
    unknown[7][8] = -t230;
    unknown[8][0] = -t113;
    unknown[8][1] = t154;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = -t186;
    unknown[8][4] = t207;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = -t223;
    unknown[8][7] = -t230;
    unknown[8][8] = 0.0e0;

    Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], n, n) = hessY;

    t1 = -x2 + x3;
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
    t18 = y2 * y2;
    t19 = y3 * y3;
    t21 = t11 * t11;
    t22 = 0.1e1 / t21;
    t23 = t22 * (0.2e1 * x1 * t1 + t16 - t17 + t18 - t19 + z2 - z3);
    t26 = x1 * x1;
    t31 = y1 * y1;
    t36 = t26 * t1 + x1 * (t16 - t17 + t18 - t19 + z2 - z3) - t16 * x3 + x2 * (t17 - t31 + t19 - z1 + z3) + (t31 - t18 + z1 - z2) * x3;
    t39 = 0.1e1 / t11 / t21 * t36;
    t40 = t4 * t4;
    t44 = 0.2e1 * x2;
    t45 = 0.2e1 * x3;
    t46 = -t44 + t45;
    t51 = 0.2e1 * t22 * (-y1 * x2 + y1 * x3);
    t56 = 0.2e1 * t46 * t4 * t39 - t46 * t23 - t4 * t51;
    t57 = t22 * t1;
    t58 = t4 * t57;
    t59 = x2 - x1;
    t60 = 0.2e1 * t12 * t59;
    t65 = 0.2e1 * x3 * x2;
    t67 = t22 * (0.2e1 * x2 * x1 + t17 + t19 - t26 - t31 - t65 - z1 + z3);
    t72 = 0.2e1 * t7 * t4 * t39 - t7 * t23 - t4 * t67 + t60;
    t74 = 0.2e1 * t12 * y2;
    t75 = 0.2e1 * x1;
    t76 = t75 - t45;
    t81 = 0.2e1 * t22 * (y2 * x1 - y2 * x3);
    t87 = 0.2e1 * t22 * t36;
    t88 = 0.2e1 * t76 * t4 * t39 - t76 * t23 - t4 * t81 + t74 - t87;
    t89 = -x3 + x1;
    t90 = t22 * t89;
    t92 = -t90 * t4 + t12;
    t93 = 0.2e1 * t12 * t89;
    t98 = t22 * (-0.2e1 * x3 * x1 - t16 - t18 + t26 + t31 + t65 + z1 - z2);
    t103 = 0.2e1 * t4 * t9 * t39 - t9 * t23 - t4 * t98 + t93;
    t105 = 0.2e1 * t12 * y3;
    t106 = -t75 + t44;
    t111 = 0.2e1 * t22 * (-y3 * x1 + y3 * x2);
    t116 = 0.2e1 * t106 * t4 * t39 - t106 * t23 - t4 * t111 - t105 + t87;
    t117 = t22 * t59;
    t119 = -t4 * t117 - t12;
    t122 = t46 * t46;
    t126 = t46 * t57;
    t128 = 0.2e1 * t12 * y1;
    t134 = 0.2e1 * t7 * t46 * t39 - t46 * t67 - t7 * t51 - t128 + t87;
    t140 = 0.2e1 * t76 * t46 * t39 - t46 * t81 - t76 * t51;
    t141 = t46 * t90;
    t147 = 0.2e1 * t9 * t46 * t39 - t46 * t98 - t9 * t51 + t128 - t87;
    t153 = 0.2e1 * t106 * t46 * t39 - t106 * t51 - t46 * t111;
    t154 = t46 * t117;
    t156 = -t7 * t57 - t12;
    t157 = t76 * t57;
    t159 = -t9 * t57 + t12;
    t160 = t106 * t57;
    t163 = t7 * t7;
    t172 = 0.2e1 * t76 * t7 * t39 - t76 * t67 - t7 * t81;
    t173 = t7 * t90;
    t179 = 0.2e1 * t9 * t7 * t39 - t9 * t67 - t7 * t98 + t13;
    t185 = 0.2e1 * t106 * t7 * t39 - t106 * t67 - t7 * t111 + t105 - t87;
    t187 = -t7 * t117 + t12;
    t190 = t76 * t76;
    t194 = t76 * t90;
    t200 = 0.2e1 * t9 * t76 * t39 - t76 * t98 - t9 * t81 - t74 + t87;
    t206 = 0.2e1 * t106 * t76 * t39 - t106 * t81 - t76 * t111;
    t207 = t76 * t117;
    t209 = -t9 * t90 - t12;
    t210 = t106 * t90;
    t213 = t9 * t9;
    t222 = 0.2e1 * t106 * t9 * t39 - t106 * t98 - t9 * t111;
    t223 = t9 * t117;
    t226 = t106 * t106;
    t230 = t106 * t117;
    unknown[0][0] = -0.2e1 * t23 * t4 + 0.2e1 * t40 * t39 + t13;
    unknown[0][1] = t56;
    unknown[0][2] = -t58;
    unknown[0][3] = t72;
    unknown[0][4] = t88;
    unknown[0][5] = t92;
    unknown[0][6] = t103;
    unknown[0][7] = t116;
    unknown[0][8] = t119;
    unknown[1][0] = t56;
    unknown[1][1] = 0.2e1 * t122 * t39 - 0.2e1 * t46 * t51 + t13;
    unknown[1][2] = -t126;
    unknown[1][3] = t134;
    unknown[1][4] = t140;
    unknown[1][5] = -t141;
    unknown[1][6] = t147;
    unknown[1][7] = t153;
    unknown[1][8] = -t154;
    unknown[2][0] = -t58;
    unknown[2][1] = -t126;
    unknown[2][2] = 0.0e0;
    unknown[2][3] = t156;
    unknown[2][4] = -t157;
    unknown[2][5] = 0.0e0;
    unknown[2][6] = t159;
    unknown[2][7] = -t160;
    unknown[2][8] = 0.0e0;
    unknown[3][0] = t72;
    unknown[3][1] = t134;
    unknown[3][2] = t156;
    unknown[3][3] = 0.2e1 * t163 * t39 - 0.2e1 * t7 * t67 + t93;
    unknown[3][4] = t172;
    unknown[3][5] = -t173;
    unknown[3][6] = t179;
    unknown[3][7] = t185;
    unknown[3][8] = t187;
    unknown[4][0] = t88;
    unknown[4][1] = t140;
    unknown[4][2] = -t157;
    unknown[4][3] = t172;
    unknown[4][4] = 0.2e1 * t190 * t39 - 0.2e1 * t76 * t81 + t93;
    unknown[4][5] = -t194;
    unknown[4][6] = t200;
    unknown[4][7] = t206;
    unknown[4][8] = -t207;
    unknown[5][0] = t92;
    unknown[5][1] = -t141;
    unknown[5][2] = 0.0e0;
    unknown[5][3] = -t173;
    unknown[5][4] = -t194;
    unknown[5][5] = 0.0e0;
    unknown[5][6] = t209;
    unknown[5][7] = -t210;
    unknown[5][8] = 0.0e0;
    unknown[6][0] = t103;
    unknown[6][1] = t147;
    unknown[6][2] = t159;
    unknown[6][3] = t179;
    unknown[6][4] = t200;
    unknown[6][5] = t209;
    unknown[6][6] = 0.2e1 * t213 * t39 - 0.2e1 * t9 * t98 + t60;
    unknown[6][7] = t222;
    unknown[6][8] = -t223;
    unknown[7][0] = t116;
    unknown[7][1] = t153;
    unknown[7][2] = -t160;
    unknown[7][3] = t185;
    unknown[7][4] = t206;
    unknown[7][5] = -t210;
    unknown[7][6] = t222;
    unknown[7][7] = -0.2e1 * t106 * t111 + 0.2e1 * t226 * t39 + t60;
    unknown[7][8] = -t230;
    unknown[8][0] = t119;
    unknown[8][1] = -t154;
    unknown[8][2] = 0.0e0;
    unknown[8][3] = t187;
    unknown[8][4] = -t207;
    unknown[8][5] = 0.0e0;
    unknown[8][6] = -t223;
    unknown[8][7] = -t230;
    unknown[8][8] = 0.0e0;
    // @formatter:on
}

void Power::getBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, TV &node) {
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
    xn = ((0.5e0 * v1x * v1x + 0.5e0 * v1y * v1y - 0.1e1 * v1y * y4 - 0.5e0 * v2x * v2x - 0.5e0 * v2y * v2y + v2y * y4 + 0.5e0 * v1z - 0.5e0 * v2z) * x3 + (-0.5e0 * v1x * v1x - 0.5e0 * v1y * v1y + v1y * y3 + 0.5e0 * v2x * v2x + 0.5e0 * v2y * v2y - 0.1e1 * v2y * y3 - 0.5e0 * v1z + 0.5e0 * v2z) * x4) / ((v1x - v2x) * x3 + (v2x - v1x) * x4 + (-v2y + v1y) * (-y4 + y3));
    yn = ((0.5e0 * v1x * v1x - 0.1e1 * x4 * v1x + 0.5e0 * v1y * v1y - 0.5e0 * v2x * v2x + x4 * v2x - 0.5e0 * v2y * v2y + 0.5e0 * v1z - 0.5e0 * v2z) * y3 + (-0.5e0 * v1x * v1x + x3 * v1x - 0.5e0 * v1y * v1y + 0.5e0 * v2x * v2x - 0.1e1 * x3 * v2x + 0.5e0 * v2y * v2y - 0.5e0 * v1z + 0.5e0 * v2z) * y4) / ((-v2y + v1y) * y3 + (v2y - v1y) * y4 + (v1x - v2x) * (-x4 + x3));
    // @formatter:on
    node = {xn, yn};
}

void Power::getBoundaryNodeGradient(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, VectorXT &gradX,
                                    VectorXT &gradY) {
    assert(v1.rows() == 3 && v2.rows() == 3 && gradX.rows() == 6 && gradY.rows() == 6);

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

    // @formatter:off
    t1 = x4 - x3;
    t2 = -v2y + v1y;
    t3 = y4 - y3;
    t4 = -t1 * (v1x - v2x) - t2 * t3;
    t5 = 0.1e1 / 0.2e1;
    t6 = t5 * (v1x * v1x + v1y * v1y - v2x * v2x - v2y * v2y + v1z - v2z);
    t4 = 0.1e1 / t4;
    t2 = ((-y4 * t2 + t6) * x3 + (y3 * t2 - t6) * x4) * t4;
    t6 = t2 * t1;
    t2 = t2 * t3;
    t3 = t5 * t1 * t4;
    gradX[0] = t4 * (-v1x * t1 + t6);
    gradX[1] = -t4 * ((y4 - v1y) * x3 - (-v1y + y3) * x4 - t2);
    gradX[2] = -t3;
    gradX[3] = t4 * (v2x * t1 - t6);
    gradX[4] = t4 * ((-v2y + y4) * x3 - (y3 - v2y) * x4 - t2);
    gradX[5] = t3;

    t1 = v1x - v2x;
    t2 = x4 - x3;
    t3 = -y4 + y3;
    t4 = t3 * (-v2y + v1y) - t1 * t2;
    t5 = 0.1e1 / 0.2e1;
    t6 = t5 * (v1x * v1x + v1y * v1y - v2x * v2x - v2y * v2y + v1z - v2z);
    t4 = 0.1e1 / t4;
    t1 = ((-x4 * t1 + t6) * y3 + (x3 * t1 - t6) * y4) * t4;
    t2 = t1 * t2;
    t6 = -t3;
    t1 = t1 * t6;
    t5 = t5 * t6 * t4;
    gradY[0] = -t4 * ((x4 - v1x) * y3 - (-v1x + x3) * y4 - t2);
    gradY[1] = -t4 * (-v1y * t3 - t1);
    gradY[2] = -t5;
    gradY[3] = t4 * ((-v2x + x4) * y3 - (x3 - v2x) * y4 - t2);
    gradY[4] = t4 * (-v2y * t3 - t1);
    gradY[5] = t5;
    // @formatter:on
}

void Power::getBoundaryNodeHessian(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, MatrixXT &hessX,
                                   MatrixXT &hessY) {
    assert(v1.rows() == 3 && v2.rows() == 3 && hessX.rows() == 6 && hessX.cols() == 6 &&
           hessY.rows() == 6 && hessY.cols() == 6);

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

    int n = 6;
    double hessX_c[n][n];
    Eigen::Map<Eigen::MatrixXd>(&hessX_c[0][0], n, n) = hessX;
    double hessY_c[n][n];
    Eigen::Map<Eigen::MatrixXd>(&hessY_c[0][0], n, n) = hessY;

    // @formatter:off
    t1 = x4 - x3;
    t2 = -v2y + v1y;
    t3 = y4 - y3;
    t4 = -t1 * (v1x - v2x) - t2 * t3;
    t5 = v1x * t1;
    t6 = 0.1e1 / 0.2e1;
    t7 = t6 * (v1x * v1x + v1y * v1y - v2x * v2x - v2y * v2y + v1z - v2z);
    t2 = (-y4 * t2 + t7) * x3 + (y3 * t2 - t7) * x4;
    t4 = 0.1e1 / t4;
    t7 = pow(t4, 0.2e1);
    t8 = t1 * t4;
    t9 = t8 * t2;
    t10 = t7 * t1;
    t11 = (y4 - v1y) * x3 - (-v1y + y3) * x4;
    t12 = t11 * t1;
    t13 = t5 * t3;
    t14 = 0.2e1 * t2 * t4 * t7;
    t15 = t14 * t1 * t3;
    t16 = t15 - t7 * (t12 + t13);
    t17 = v2x * t1;
    t18 = pow(t1, 0.2e1);
    t19 = -t14 * t18 + t10 * (t5 + t17);
    t20 = (-v2y + y4) * x3 - (y3 - v2y) * x4;
    t1 = t20 * t1;
    t13 = -t15 + t7 * (t13 + t1);
    t18 = t6 * t18 * t7;
    t2 = t2 * t4 * t3;
    t4 = t7 * t3;
    t21 = t17 * t3;
    t12 = t7 * (t12 + t21) - t15;
    t14 = t4 * (t11 + t20) - t14 * pow(t3, 0.2e1);
    t3 = t10 * t6 * t3;
    t1 = -t7 * (t1 + t21) + t15;
    hessX_c[0][0] = -0.2e1 * t10 * (-t9 + t5) - t8;
    hessX_c[0][1] = t16;
    hessX_c[0][2] = -t18;
    hessX_c[0][3] = t19;
    hessX_c[0][4] = t13;
    hessX_c[0][5] = t18;
    hessX_c[1][0] = t16;
    hessX_c[1][1] = -0.2e1 * t4 * (-t2 + t11) - t8;
    hessX_c[1][2] = -t3;
    hessX_c[1][3] = t12;
    hessX_c[1][4] = t14;
    hessX_c[1][5] = t3;
    hessX_c[2][0] = -t18;
    hessX_c[2][1] = -t3;
    hessX_c[2][2] = 0;
    hessX_c[2][3] = t18;
    hessX_c[2][4] = t3;
    hessX_c[2][5] = 0;
    hessX_c[3][0] = t19;
    hessX_c[3][1] = t12;
    hessX_c[3][2] = t18;
    hessX_c[3][3] = 0.2e1 * t10 * (t9 - t17) + t8;
    hessX_c[3][4] = t1;
    hessX_c[3][5] = -t18;
    hessX_c[4][0] = t13;
    hessX_c[4][1] = t14;
    hessX_c[4][2] = t3;
    hessX_c[4][3] = t1;
    hessX_c[4][4] = 0.2e1 * t4 * (t2 - t20) + t8;
    hessX_c[4][5] = -t3;
    hessX_c[5][0] = t18;
    hessX_c[5][1] = t3;
    hessX_c[5][2] = 0;
    hessX_c[5][3] = -t18;
    hessX_c[5][4] = -t3;
    hessX_c[5][5] = 0;

    t1 = y4 - y3;
    t2 = v1x - v2x;
    t3 = x4 - x3;
    t4 = -t1 * (-v2y + v1y) - t2 * t3;
    t5 = (x4 - v1x) * y3 - (-v1x + x3) * y4;
    t6 = 0.1e1 / 0.2e1;
    t7 = t6 * (v1x * v1x + v1y * v1y - v2x * v2x - v2y * v2y + v1z - v2z);
    t2 = (-x4 * t2 + t7) * y3 + (x3 * t2 - t7) * y4;
    t4 = 0.1e1 / t4;
    t7 = pow(t4, 0.2e1);
    t8 = t2 * t4;
    t9 = t8 * t3;
    t10 = t7 * t3;
    t11 = t1 * t4;
    t12 = v1y * t1;
    t13 = t5 * t1;
    t14 = t12 * t3;
    t2 = 0.2e1 * t2 * t4 * t7;
    t4 = t2 * t3 * t1;
    t15 = t4 - t7 * (t13 + t14);
    t16 = (-v2x + x4) * y3 - (x3 - v2x) * y4;
    t17 = -t2 * pow(t3, 0.2e1) + t10 * (t5 + t16);
    t18 = v2y * t1;
    t3 = t18 * t3;
    t13 = -t4 + t7 * (t13 + t3);
    t19 = t10 * t6 * t1;
    t8 = t8 * t1;
    t20 = t7 * t1;
    t21 = t16 * t1;
    t14 = -t4 + t7 * (t14 + t21);
    t1 = pow(t1, 0.2e1);
    t2 = t20 * (t12 + t18) - t2 * t1;
    t1 = t6 * t1 * t7;
    t3 = t4 - t7 * (t3 + t21);
    hessY_c[0][0] = -0.2e1 * t10 * (-t9 + t5) - t11;
    hessY_c[0][1] = t15;
    hessY_c[0][2] = -t19;
    hessY_c[0][3] = t17;
    hessY_c[0][4] = t13;
    hessY_c[0][5] = t19;
    hessY_c[1][0] = t15;
    hessY_c[1][1] = -0.2e1 * t20 * (-t8 + t12) - t11;
    hessY_c[1][2] = -t1;
    hessY_c[1][3] = t14;
    hessY_c[1][4] = t2;
    hessY_c[1][5] = t1;
    hessY_c[2][0] = -t19;
    hessY_c[2][1] = -t1;
    hessY_c[2][2] = 0;
    hessY_c[2][3] = t19;
    hessY_c[2][4] = t1;
    hessY_c[2][5] = 0;
    hessY_c[3][0] = t17;
    hessY_c[3][1] = t14;
    hessY_c[3][2] = t19;
    hessY_c[3][3] = -0.2e1 * t10 * (-t9 + t16) + t11;
    hessY_c[3][4] = t3;
    hessY_c[3][5] = -t19;
    hessY_c[4][0] = t13;
    hessY_c[4][1] = t2;
    hessY_c[4][2] = t1;
    hessY_c[4][3] = t3;
    hessY_c[4][4] = -0.2e1 * t20 * (-t8 + t18) + t11;
    hessY_c[4][5] = -t1;
    hessY_c[5][0] = t19;
    hessY_c[5][1] = t1;
    hessY_c[5][2] = 0;
    hessY_c[5][3] = -t19;
    hessY_c[5][4] = -t1;
    hessY_c[5][5] = 0;
    // @formatter:on
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
        Regular_triangulation::Weighted_point wp({v(0), v(1)}, -v(2));
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

VectorXi Power::powerDualNaive(const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;
    std::vector<int> tri1;
    std::vector<int> tri2;
    std::vector<int> tri3;

    for (int i = 0; i < n_vtx; i++) {
        TV3 vi = vertices3d.segment<3>(i * 3);
        std::vector<int> neighbors;

        for (int j = 0; j < n_vtx; j++) {
            if (j == i) continue;

            TV3 vj = vertices3d.segment<3>(j * 3);
            TV3 line = (vj - vi).cross(TV3(0, 0, 1));

            double dmin = INFINITY;
            double dmax = -INFINITY;

            for (int k = 0; k < n_vtx; k++) {
                if (k == i || k == j) continue;

                TV3 vk = vertices3d.segment<3>(k * 3);
                TV vc2d;
                getNode(vi, vj, vk, vc2d);
                TV3 vc = {vc2d(0), vc2d(1), 0};
                double d = vc.dot(line);

                if ((vk - vi).dot(line) > 0) {
                    dmin = fmin(dmin, d);
                } else {
                    dmax = fmax(dmax, d);
                }
                if (dmax > dmin) break;
            }

            if (dmax < dmin || (dmax == dmin)) {
                neighbors.push_back(j);
            }
        }

        double xc = vertices3d(i * 3 + 0);
        double yc = vertices3d(i * 3 + 1);

        std::sort(neighbors.begin(), neighbors.end(), [vertices3d, xc, yc](int a, int b) {
            double xa = vertices3d(a * 3 + 0);
            double ya = vertices3d(a * 3 + 1);
            double angle_a = atan2(ya - yc, xa - xc);

            double xb = vertices3d(b * 3 + 0);
            double yb = vertices3d(b * 3 + 1);
            double angle_b = atan2(yb - yc, xb - xc);

            return angle_a < angle_b;
        });

        if (neighbors.size() > 0) {
            assert(neighbors.size() > 1);
            for (int j = 0; j < neighbors.size(); j++) {
                int v1 = i;
                int v2 = neighbors[j];
                int v3 = neighbors[(j + 1) % neighbors.size()];

                if (v1 < v2 && v1 < v3) {
                    double x1 = vertices3d(v1 * 3 + 0);
                    double y1 = vertices3d(v1 * 3 + 1);
                    double x2 = vertices3d(v2 * 3 + 0);
                    double y2 = vertices3d(v2 * 3 + 1);
                    double x3 = vertices3d(v3 * 3 + 0);
                    double y3 = vertices3d(v3 * 3 + 1);

                    if (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2 > 0) {
                        tri1.push_back(v1);
                        tri2.push_back(v2);
                        tri3.push_back(v3);
                    }
                }
            }
        }
    }

    VectorXi tri(tri1.size() * 3);
    for (int i = 0; i < tri1.size(); i++) {
        tri(i * 3 + 0) = tri1[i];
        tri(i * 3 + 1) = tri2[i];
        tri(i * 3 + 2) = tri3[i];
    }

    return tri;
}

VectorXi Power::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    VectorXT vertices3d = combineVerticesParams(vertices, params);
    return powerDualCGAL(vertices3d);
}

VectorXT Power::getDefaultVertexParams(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    VectorXT z = VectorXT::Zero(n_vtx);

    return z;
}

