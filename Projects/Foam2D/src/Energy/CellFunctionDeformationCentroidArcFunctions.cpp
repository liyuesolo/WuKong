#include "../../include/Energy/CellFunctionDeformationCentroid.h"
#include <iostream>

void
CellFunctionDeformationCentroid::getArcParams(int order, const VectorXT &x, VectorXT &p, MatrixXT &dpdx,
                                              std::vector<MatrixXT> &d2pdx2) const {
    double x0 = x(0);
    double y0 = x(1);
    double x1 = x(2);
    double y1 = x(3);
    double r = x(4);

    p.resize(5);
    dpdx.resize(5, 5);
    d2pdx2.resize(5);
    std::fill(d2pdx2.begin(), d2pdx2.end(), MatrixXT::Zero(5, 5));

    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80,
            t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99, t100,
            t101, t102, t103, t104, t105, t106, t107, t108, t109, t110, t111, t112, t113, t114, t115, t116, t117, t118, t119, t120,
            t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132, t133, t134, t135, t136, t137, t138, t139, t140,
            t141, t142, t143, t144, t145, t146, t147, t148, t149, t150, t151, t152, t153, t154, t155, t156, t157, t158, t159, t160,
            t161, t162, t163, t164, t165, t166, t167, t168, t169, t170, t171, t172, t173, t174, t175, t176, t177, t178, t179, t180,
            t181, t182, t183, t184, t185, t186, t187, t188, t189, t190, t191, t192, t193, t194, t195, t196, t197, t198, t199, t200,
            t201, t202, t203, t204, t205, t206, t207, t208, t209, t210, t211, t212, t213, t214, t215, t216, t217, t218, t219, t220,
            t221, t222, t223, t224, t225, t226, t227, t228, t229, t230, t231, t232, t233, t234, t235, t236, t237, t238, t239, t240,
            t241, t242, t243, t244, t245, t246, t247, t248, t249, t250, t251, t252, t253, t254, t255, t256, t257, t258, t259, t260,
            t261, t262, t263, t264, t265, t266, t267, t268, t269, t270, t271, t272, t273, t274, t275, t276, t277, t278, t279, t280,
            t281, t282, t283, t284, t285, t286, t287, t288, t289, t290, t291, t292, t293, t294, t295, t296, t297, t298, t299, t300,
            t301, t302, t303, t304, t305, t306, t307, t308, t309, t310, t311, t312, t313, t314, t315, t316, t317, t318, t319, t320,
            t321, t322, t323, t324, t325, t326, t327, t328, t329, t330, t331, t332, t333, t334, t335, t336, t337, t338, t339, t340,
            t341, t342, t343, t344, t345, t346, t347, t348, t349, t350, t351, t352, t353, t354, t355, t356, t357, t358, t359, t360,
            t361, t362, t363, t364, t365, t366, t367, t368, t369, t370, t371, t372, t373, t374, t375, t376, t377, t378, t379, t380,
            t381, t382, t383, t384, t385, t386, t387, t388, t389, t390, t391, t392, t393, t394, t395, t396, t397, t398, t399, t400,
            t401, t402, t403, t404, t405, t406, t407, t408, t409, t410, t411, t412, t413, t414, t415, t416, t417, t418, t419, t420,
            t421, t422, t423, t424, t425, t426, t427, t428, t429, t430, t431, t432, t433, t434, t435, t436, t437, t438, t439, t440;
    if (order >= 0) {
        p[0] = r;
        p[1] = x0 / 0.2e1 + x1 / 0.2e1 -
               r * sqrt(0.4e1 - (pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) * pow(r, -0.2e1)) * (y1 - y0) *
               pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.2e1;
        p[2] = y0 / 0.2e1 + y1 / 0.2e1 -
               r * sqrt(0.4e1 - (pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) * pow(r, -0.2e1)) * (x0 - x1) *
               pow(pow(x0 - x1, 0.2e1) + pow(y1 - y0, 0.2e1), -0.1e1 / 0.2e1) / 0.2e1;
        p[3] = atan2(x0 - x1, y1 - y0);
        p[4] = 0.2e1 * asin(sqrt(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1)) / r / 0.2e1);
    }
    if (order >= 1) {
        t1 = x0 - x1;
        t2 = -y1 + y0;
        t3 = pow(t1, 0.2e1);
        t4 = pow(t2, 0.2e1);
        t5 = t4 + t3;
        t6 = 0.1e1 / r;
        t7 = pow(t6, 0.2e1);
        t8 = t5 * t7;
        t9 = -t8 + 0.4e1;
        t10 = pow(t9, -0.1e1 / 0.2e1);
        t11 = pow(t5, -0.3e1 / 0.2e1);
        t12 = t5 * t11;
        t9 = t9 * t10;
        t6 = t6 * t10;
        t13 = t6 * t12;
        t14 = r * t9;
        t15 = t2 * t1 * (t14 * t11 + t13);
        t8 = t8 * t10 + t9;
        t9 = 0.1e1 / 0.2e1;
        t16 = t9 * (0.1e1 - t15);
        t4 = t9 * (-t13 * t4 + t14 * (-t11 * t4 + t12));
        t15 = t9 * (0.1e1 + t15);
        t11 = t9 * (-t13 * t3 + t14 * (-t11 * t3 + t12));
        t13 = 0.1e1 / t2;
        t14 = pow(t13, 0.2e1);
        t3 = t14 * t3 + 0.1e1;
        t3 = 0.1e1 / t3;
        t14 = t1 * t14 * t3;
        t3 = t13 * t3;
        t13 = pow(t5, -0.1e1 / 0.2e1);
        t6 = 0.2e1 * t6 * t13;
        t17 = t6 * t2;
        t6 = t6 * t1;
        dpdx(0, 0) = 0;
        dpdx(0, 1) = 0;
        dpdx(0, 2) = 0;
        dpdx(0, 3) = 0;
        dpdx(0, 4) = 1;
        dpdx(1, 0) = t16;
        dpdx(1, 1) = t4;
        dpdx(1, 2) = t15;
        dpdx(1, 3) = -t4;
        dpdx(1, 4) = t9 * t2 * t12 * t8;
        dpdx(2, 0) = -t11;
        dpdx(2, 1) = t15;
        dpdx(2, 2) = t11;
        dpdx(2, 3) = t16;
        dpdx(2, 4) = -t9 * t1 * t12 * t8;
        dpdx(3, 0) = -t3;
        dpdx(3, 1) = t14;
        dpdx(3, 2) = t3;
        dpdx(3, 3) = -t14;
        dpdx(3, 4) = 0;
        dpdx(4, 0) = t6;
        dpdx(4, 1) = t17;
        dpdx(4, 2) = -t6;
        dpdx(4, 3) = -t17;
        dpdx(4, 4) = -0.2e1 * t5 * t13 * t7 * t10;
    }
    if (order >= 2) {
        t2 = x1 - x0;
        t3 = t2 * t2;
        t4 = y1 - y0;
        t5 = t4 * t4;
        t6 = t3 + t5;
        t7 = r * r;
        t8 = 0.1e1 / t7;
        t10 = -t8 * t6 + 0.4e1;
        t11 = sqrt(t10);
        t12 = 0.1e1 / t11;
        t14 = sqrt(t6);
        t15 = 0.1e1 / t14;
        t16 = t6 * t15;
        t18 = 0.1e1 / r / t7;
        t19 = t18 * t16;
        t21 = t7 * t7;
        t23 = 0.1e1 / r / t21;
        t25 = 0.1e1 / t11 / t10;
        t26 = t25 * t23;
        t27 = t15 * t4;
        t28 = t6 * t6;
        t32 = t12 * t8;
        t36 = 0.1e1 / t21;
        t37 = t25 * t36;
        t38 = t4 * t37;
        t39 = 0.2e1 * t15 * t4;
        t40 = t6 * t39;
        t44 = t15 * t11 / 0.2e1;
        t46 = t16 * t32 / 0.2e1;
        t47 = t4 * t11;
        t48 = t14 * t6;
        t49 = 0.1e1 / t48;
        t50 = 0.2e1 * t4 * t49;
        t53 = t4 * t32;
        t54 = t6 * t50;
        t57 = -t4 * t27 * t32 / 0.2e1 - t40 * t38 / 0.4e1 - t44 - t46 + t50 * t47 / 0.4e1 + t54 * t53 / 0.4e1;
        t60 = 0.2e1 * t2 * t15;
        t61 = t6 * t60;
        t63 = 0.2e1 * t2 * t49;
        t65 = t6 * t63;
        t67 = -0.2e1 * t2 * t27 * t32 - t61 * t38 + t63 * t47 + t65 * t53;
        t71 = -0.2e1 * t15 * t4;
        t72 = t6 * t71;
        t75 = -0.2e1 * t4 * t49;
        t78 = t6 * t75;
        t81 = t4 * t27 * t32 / 0.2e1 - t72 * t38 / 0.4e1 + t44 + t46 + t75 * t47 / 0.4e1 + t78 * t53 / 0.4e1;
        t84 = -0.2e1 * t2 * t15;
        t85 = t6 * t84;
        t87 = -0.2e1 * t2 * t49;
        t89 = t6 * t87;
        t91 = 0.2e1 * t2 * t27 * t32 - t85 * t38 + t87 * t47 + t89 * t53;
        t92 = t25 * t18;
        t93 = 0.4e1 * t4 * t4;
        t96 = t93 * t27 * t92 / 0.8e1;
        t97 = 0.1e1 / r;
        t98 = t12 * t97;
        t99 = t39 * t98;
        t101 = t4 * t49;
        t104 = t93 * t101 * t98 / 0.4e1;
        t106 = t27 * t98 / 0.2e1;
        t107 = t11 * r;
        t108 = t50 * t107;
        t111 = 0.1e1 / t14 / t28;
        t112 = t111 * t4;
        t115 = 0.3e1 / 0.8e1 * t93 * t112 * t107;
        t117 = t101 * t107 / 0.2e1;
        t119 = t4 * t92;
        t120 = 0.2e1 * t4 * t60;
        t123 = t60 * t98;
        t124 = t123 / 0.4e1;
        t125 = t4 * t98;
        t126 = 0.2e1 * t4 * t63;
        t129 = t63 * t107;
        t130 = t129 / 0.4e1;
        t131 = t4 * t107;
        t133 = 0.4e1 * t4 * t2 * t111;
        t136 = t120 * t119 / 0.8e1 + t124 - t126 * t125 / 0.4e1 + t130 - 0.3e1 / 0.8e1 * t133 * t131;
        t137 = 0.2e1 * t4 * t71;
        t140 = t71 * t98;
        t141 = t140 / 0.4e1;
        t142 = 0.2e1 * t4 * t75;
        t145 = t99 / 0.4e1;
        t146 = t108 / 0.4e1;
        t147 = t75 * t107;
        t148 = t147 / 0.4e1;
        t149 = -0.2e1 * t111 * t4;
        t150 = 0.2e1 * t4 * t149;
        t153 = t137 * t119 / 0.8e1 + t141 - t142 * t125 / 0.4e1 - t106 - t145 - t146 + t148 -
               0.3e1 / 0.8e1 * t150 * t131 - t117;
        t154 = 0.2e1 * t4 * t84;
        t157 = t84 * t98;
        t158 = t157 / 0.4e1;
        t159 = 0.2e1 * t4 * t87;
        t162 = t87 * t107;
        t163 = t162 / 0.4e1;
        t164 = -0.2e1 * t2 * t111;
        t165 = 0.2e1 * t4 * t164;
        t168 = t154 * t119 / 0.8e1 + t158 - t159 * t125 / 0.4e1 + t163 - 0.3e1 / 0.8e1 * t165 * t131;
        t169 = 0.4e1 * t2 * t2;
        t179 = t169 * t27 * t92 / 0.8e1 - t169 * t101 * t98 / 0.4e1 + t106 - 0.3e1 / 0.8e1 * t169 * t112 * t107 + t117;
        t180 = 0.2e1 * t2 * t71;
        t183 = 0.2e1 * t2 * t75;
        t186 = 0.2e1 * t2 * t149;
        t189 = t180 * t119 / 0.8e1 - t183 * t125 / 0.4e1 - t124 - t130 - 0.3e1 / 0.8e1 * t186 * t131;
        t190 = 0.2e1 * t2 * t84;
        t193 = 0.2e1 * t2 * t87;
        t196 = 0.2e1 * t2 * t164;
        t199 = t190 * t119 / 0.8e1 - t193 * t125 / 0.4e1 - t106 - 0.3e1 / 0.8e1 * t196 * t131 - t117;
        t203 = -0.2e1 * t4 * t84;
        t206 = -0.2e1 * t4 * t87;
        t209 = -0.2e1 * t4 * t164;
        t212 = t203 * t119 / 0.8e1 - t158 - t206 * t125 / 0.4e1 - t163 - 0.3e1 / 0.8e1 * t209 * t131;
        t216 = -t2 * t15;
        t222 = -t2 * t37;
        t224 = -t2 * t11;
        t226 = -t2 * t32;
        t228 = -0.2e1 * t4 * t216 * t32 - t40 * t222 + t50 * t224 + t54 * t226;
        t238 = -t2 * t216 * t32 / 0.2e1 - t61 * t222 / 0.4e1 + t44 + t46 + t63 * t224 / 0.4e1 + t65 * t226 / 0.4e1;
        t244 = 0.2e1 * t4 * t216 * t32 - t72 * t222 + t75 * t224 + t78 * t226;
        t254 = t2 * t216 * t32 / 0.2e1 - t85 * t222 / 0.4e1 - t44 - t46 + t87 * t224 / 0.4e1 + t89 * t226 / 0.4e1;
        t258 = -t2 * t49;
        t263 = t216 * t98 / 0.2e1;
        t264 = -t2 * t111;
        t269 = t258 * t107 / 0.2e1;
        t270 = t93 * t216 * t92 / 0.8e1 - t93 * t258 * t98 / 0.4e1 + t263 - 0.3e1 / 0.8e1 * t93 * t264 * t107 + t269;
        t271 = -t2 * t92;
        t274 = -t2 * t98;
        t277 = -t2 * t107;
        t280 = t120 * t271 / 0.8e1 - t126 * t274 / 0.4e1 - t145 - t146 - 0.3e1 / 0.8e1 * t133 * t277;
        t287 = t137 * t271 / 0.8e1 - t142 * t274 / 0.4e1 - t263 - 0.3e1 / 0.8e1 * t150 * t277 - t269;
        t294 = t154 * t271 / 0.8e1 - t159 * t274 / 0.4e1 + t145 + t146 - 0.3e1 / 0.8e1 * t165 * t277;
        t297 = t169 * t216 * t92 / 0.8e1;
        t301 = t169 * t258 * t98 / 0.4e1;
        t305 = 0.3e1 / 0.8e1 * t169 * t264 * t107;
        t313 = t180 * t271 / 0.8e1 - t141 - t183 * t274 / 0.4e1 - t148 - 0.3e1 / 0.8e1 * t186 * t277;
        t320 = t190 * t271 / 0.8e1 - t158 - t193 * t274 / 0.4e1 - t263 + t124 + t130 - t163 -
               0.3e1 / 0.8e1 * t196 * t277 - t269;
        t327 = t203 * t271 / 0.8e1 - t206 * t274 / 0.4e1 + t141 + t148 - 0.3e1 / 0.8e1 * t209 * t277;
        t333 = 0.1e1 / t4 / t5;
        t335 = 0.1e1 / t5;
        t337 = t335 * t3 + 0.1e1;
        t338 = 0.1e1 / t337;
        t341 = t5 * t5;
        t345 = t337 * t337;
        t346 = 0.1e1 / t345;
        t348 = -t338 * t333 * t2 + t346 / t4 / t341 * t2 * t3;
        t354 = t338 * t335 - 0.2e1 * t3 * t346 / t341;
        t357 = -0.2e1 * t2 * t346 * t333;
        t366 = t8 * t15;
        t369 = t36 * t14;
        t372 = -0.2e1 * t4 * t12 * t366 - 0.2e1 * t25 * t4 * t369;
        t373 = 0.2e1 * t2 * t12;
        t375 = 0.2e1 * t25 * t2;
        t377 = -t373 * t366 - t375 * t369;
        t378 = -0.2e1 * t4 * t12;
        t380 = -0.2e1 * t25 * t4;
        t382 = -t378 * t366 - t380 * t369;
        t383 = -0.2e1 * t2 * t12;
        t385 = -0.2e1 * t25 * t2;
        t387 = -t383 * t366 - t385 * t369;
        t388 = t97 * t49;
        t394 = 0.2e1 * t12 * t97 * t15;
        t395 = t18 * t15;
        t399 = -t12 * t93 * t388 / 0.2e1 + t394 + t25 * t93 * t395 / 0.2e1;
        t404 = -0.2e1 * t4 * t373 * t388 + 0.2e1 * t4 * t375 * t395;
        t411 = -t4 * t378 * t388 + t4 * t380 * t395 - t394;
        t416 = -0.2e1 * t4 * t383 * t388 + 0.2e1 * t4 * t385 * t395;
        t423 = -t12 * t169 * t388 / 0.2e1 + t394 + t25 * t169 * t395 / 0.2e1;
        t428 = -0.2e1 * t2 * t378 * t388 + 0.2e1 * t2 * t380 * t395;
        t435 = -t2 * t383 * t388 + t2 * t385 * t395 - t394;
        t440 = 0.2e1 * t4 * t383 * t388 - 0.2e1 * t4 * t385 * t395;
        d2pdx2[0](0, 0) = 0;
        d2pdx2[0](0, 1) = 0;
        d2pdx2[0](0, 2) = 0;
        d2pdx2[0](0, 3) = 0;
        d2pdx2[0](0, 4) = 0;
        d2pdx2[0](1, 0) = 0;
        d2pdx2[0](1, 1) = 0;
        d2pdx2[0](1, 2) = 0;
        d2pdx2[0](1, 3) = 0;
        d2pdx2[0](1, 4) = 0;
        d2pdx2[0](2, 0) = 0;
        d2pdx2[0](2, 1) = 0;
        d2pdx2[0](2, 2) = 0;
        d2pdx2[0](2, 3) = 0;
        d2pdx2[0](2, 4) = 0;
        d2pdx2[0](3, 0) = 0;
        d2pdx2[0](3, 1) = 0;
        d2pdx2[0](3, 2) = 0;
        d2pdx2[0](3, 3) = 0;
        d2pdx2[0](3, 4) = 0;
        d2pdx2[0](4, 0) = 0;
        d2pdx2[0](4, 1) = 0;
        d2pdx2[0](4, 2) = 0;
        d2pdx2[0](4, 3) = 0;
        d2pdx2[0](4, 4) = 0;
        d2pdx2[1](0, 0) = t179;
        d2pdx2[1](0, 1) = t212;
        d2pdx2[1](0, 2) = t199;
        d2pdx2[1](0, 3) = t168;
        d2pdx2[1](0, 4) = t91 / 0.4e1;
        d2pdx2[1](1, 0) = t212;
        d2pdx2[1](1, 1) = t202;
        d2pdx2[1](1, 2) = t189;
        d2pdx2[1](1, 3) = t153;
        d2pdx2[1](1, 4) = t81;
        d2pdx2[1](2, 0) = t199;
        d2pdx2[1](2, 1) = t189;
        d2pdx2[1](2, 2) = t179;
        d2pdx2[1](2, 3) = t136;
        d2pdx2[1](2, 4) = t67 / 0.4e1;
        d2pdx2[1](3, 0) = t168;
        d2pdx2[1](3, 1) = t153;
        d2pdx2[1](3, 2) = t136;
        d2pdx2[1](3, 3) = t118;
        d2pdx2[1](3, 4) = t57;
        d2pdx2[1](4, 0) = t91 / 0.4e1;
        d2pdx2[1](4, 1) = t81;
        d2pdx2[1](4, 2) = t67 / 0.4e1;
        d2pdx2[1](4, 3) = t57;
        d2pdx2[1](4, 4) = t31 / 0.2e1;
        d2pdx2[2](0, 0) = t330;
        d2pdx2[2](0, 1) = t327;
        d2pdx2[2](0, 2) = t320;
        d2pdx2[2](0, 3) = t294;
        d2pdx2[2](0, 4) = t254;
        d2pdx2[2](1, 0) = t327;
        d2pdx2[2](1, 1) = t270;
        d2pdx2[2](1, 2) = t313;
        d2pdx2[2](1, 3) = t287;
        d2pdx2[2](1, 4) = t244 / 0.4e1;
        d2pdx2[2](2, 0) = t320;
        d2pdx2[2](2, 1) = t313;
        d2pdx2[2](2, 2) = t306;
        d2pdx2[2](2, 3) = t280;
        d2pdx2[2](2, 4) = t238;
        d2pdx2[2](3, 0) = t294;
        d2pdx2[2](3, 1) = t287;
        d2pdx2[2](3, 2) = t280;
        d2pdx2[2](3, 3) = t270;
        d2pdx2[2](3, 4) = t228 / 0.4e1;
        d2pdx2[2](4, 0) = t254;
        d2pdx2[2](4, 1) = t244 / 0.4e1;
        d2pdx2[2](4, 2) = t238;
        d2pdx2[2](4, 3) = t228 / 0.4e1;
        d2pdx2[2](4, 4) = t219 / 0.2e1;
        d2pdx2[3](0, 0) = -t357;
        d2pdx2[3](0, 1) = t354;
        d2pdx2[3](0, 2) = t357;
        d2pdx2[3](0, 3) = -t354;
        d2pdx2[3](0, 4) = 0;
        d2pdx2[3](1, 0) = t354;
        d2pdx2[3](1, 1) = 0.2e1 * t348;
        d2pdx2[3](1, 2) = -t354;
        d2pdx2[3](1, 3) = -0.2e1 * t348;
        d2pdx2[3](1, 4) = 0;
        d2pdx2[3](2, 0) = t357;
        d2pdx2[3](2, 1) = -t354;
        d2pdx2[3](2, 2) = -t357;
        d2pdx2[3](2, 3) = t354;
        d2pdx2[3](2, 4) = 0;
        d2pdx2[3](3, 0) = -t354;
        d2pdx2[3](3, 1) = -0.2e1 * t348;
        d2pdx2[3](3, 2) = t354;
        d2pdx2[3](3, 3) = 0.2e1 * t348;
        d2pdx2[3](3, 4) = 0;
        d2pdx2[3](4, 0) = 0;
        d2pdx2[3](4, 1) = 0;
        d2pdx2[3](4, 2) = 0;
        d2pdx2[3](4, 3) = 0;
        d2pdx2[3](4, 4) = 0;
        d2pdx2[4](0, 0) = t423;
        d2pdx2[4](0, 1) = t440 / 0.2e1;
        d2pdx2[4](0, 2) = t435;
        d2pdx2[4](0, 3) = t416 / 0.2e1;
        d2pdx2[4](0, 4) = t387;
        d2pdx2[4](1, 0) = t440 / 0.2e1;
        d2pdx2[4](1, 1) = t399;
        d2pdx2[4](1, 2) = t428 / 0.2e1;
        d2pdx2[4](1, 3) = t411;
        d2pdx2[4](1, 4) = t382;
        d2pdx2[4](2, 0) = t435;
        d2pdx2[4](2, 1) = t428 / 0.2e1;
        d2pdx2[4](2, 2) = t423;
        d2pdx2[4](2, 3) = t404 / 0.2e1;
        d2pdx2[4](2, 4) = t377;
        d2pdx2[4](3, 0) = t416 / 0.2e1;
        d2pdx2[4](3, 1) = t411;
        d2pdx2[4](3, 2) = t404 / 0.2e1;
        d2pdx2[4](3, 3) = t399;
        d2pdx2[4](3, 4) = t372;
        d2pdx2[4](4, 0) = t387;
        d2pdx2[4](4, 1) = t382;
        d2pdx2[4](4, 2) = t377;
        d2pdx2[4](4, 3) = t372;
        d2pdx2[4](4, 4) = t365;

        t1 = r * r;
        t3 = 0.1e1 / r / t1;
        t4 = x1 - x0;
        t5 = t4 * t4;
        t6 = y1 - y0;
        t7 = t6 * t6;
        t8 = t5 + t7;
        t11 = 0.4e1 - 0.1e1 / t1 * t8;
        t12 = sqrt(t11);
        t14 = 0.1e1 / t12 / t11;
        t15 = t14 * t3;
        t16 = sqrt(t8);
        t17 = 0.1e1 / t16;
        t18 = t17 * t6;
        t19 = 0.4e1 * t4 * t4;
        t24 = 0.1e1 / t12;
        t25 = t24 / r;
        t27 = 0.1e1 / t16 / t8;
        t28 = t27 * t6;
        t33 = t18 * t25 / 0.2e1;
        t34 = t12 * r;
        t35 = t8 * t8;
        t38 = 0.1e1 / t16 / t35 * t6;
        t43 = t28 * t34 / 0.2e1;
        t44 = t19 * t18 * t15 / 0.8e1 - t19 * t28 * t25 / 0.4e1 + t33 - 0.3e1 / 0.8e1 * t19 * t38 * t34 + t43;
        t45 = 0.4e1 * t6 * t6;
        t48 = t45 * t18 * t15 / 0.8e1;
        t54 = t45 * t28 * t25 / 0.4e1;
        t60 = 0.3e1 / 0.8e1 * t45 * t38 * t34;
        t73 = t1 * t1;
        d2pdx2[1](0, 0) = t44;
        d2pdx2[1](1, 1) = t17 * t6 * t25 + t27 * t6 * t34 + t33 + t43 + t48 - t54 - t60;
        d2pdx2[1](2, 2) = t44;
        d2pdx2[1](3, 3) = t17 * t6 * t25 + t27 * t6 * t34 + t33 + t43 + t48 - t54 - t60;
        d2pdx2[1](4, 4) = t3 * t8 * t17 * t6 * t24 / 0.2e1 + t35 * t18 * t14 / r / t73 / 0.2e1;

        t1 = r * r;
        t3 = 0.1e1 / r / t1;
        t4 = x1 - x0;
        t5 = t4 * t4;
        t6 = y1 - y0;
        t7 = t6 * t6;
        t8 = t5 + t7;
        t11 = 0.4e1 - 0.1e1 / t1 * t8;
        t12 = sqrt(t11);
        t14 = 0.1e1 / t12 / t11;
        t15 = t14 * t3;
        t16 = sqrt(t8);
        t17 = 0.1e1 / t16;
        t18 = -t17 * t4;
        t19 = 0.4e1 * t4 * t4;
        t22 = t19 * t18 * t15 / 0.8e1;
        t24 = 0.1e1 / t12;
        t25 = t24 / r;
        t30 = 0.1e1 / t16 / t8;
        t31 = -t30 * t4;
        t34 = t19 * t31 * t25 / 0.4e1;
        t36 = t18 * t25 / 0.2e1;
        t37 = t12 * r;
        t41 = t8 * t8;
        t44 = -0.1e1 / t16 / t41 * t4;
        t47 = 0.3e1 / 0.8e1 * t19 * t44 * t37;
        t49 = t31 * t37 / 0.2e1;
        t51 = 0.4e1 * t6 * t6;
        t61 = t51 * t18 * t15 / 0.8e1 - t51 * t31 * t25 / 0.4e1 + t36 - 0.3e1 / 0.8e1 * t51 * t44 * t37 + t49;
        t73 = t1 * t1;
        d2pdx2[2](0, 0) = -t17 * t4 * t25 - t30 * t4 * t37 + t22 - t34 + t36 - t47 + t49;
        d2pdx2[2](1, 1) = t61;
        d2pdx2[2](2, 2) = -t17 * t4 * t25 - t30 * t4 * t37 + t22 - t34 + t36 - t47 + t49;
        d2pdx2[2](3, 3) = t61;
        d2pdx2[2](4, 4) = -t3 * t8 * t17 * t4 * t24 / 0.2e1 + t41 * t18 * t14 / r / t73 / 0.2e1;

        t1 = y1 - y0;
        t2 = t1 * t1;
        t4 = 0.1e1 / t1 / t2;
        t5 = -x1 + x0;
        t6 = t5 * t5;
        t9 = 0.1e1 + 0.1e1 / t2 * t6;
        t10 = t9 * t9;
        t11 = 0.1e1 / t10;
        t14 = 0.2e1 * t5 * t11 * t4;
        t19 = t2 * t2;
        t24 = 0.1e1 / t9 * t4 * t5 - t11 / t1 / t19 * t5 * t6;
        d2pdx2[3](0, 0) = -t14;
        d2pdx2[3](1, 1) = 0.2e1 * t24;
        d2pdx2[3](2, 2) = -t14;
        d2pdx2[3](3, 3) = 0.2e1 * t24;
        d2pdx2[3](4, 4) = 0.0e0;

        t1 = x1 - x0;
        t2 = t1 * t1;
        t3 = y1 - y0;
        t4 = t3 * t3;
        t5 = t2 + t4;
        t6 = sqrt(t5);
        t7 = t5 * t6;
        t9 = 0.1e1 / r;
        t10 = t9 / t7;
        t11 = 0.4e1 * t1 * t1;
        t12 = r * r;
        t15 = 0.4e1 - 0.1e1 / t12 * t5;
        t16 = sqrt(t15);
        t17 = 0.1e1 / t16;
        t21 = 0.1e1 / t6;
        t24 = 0.2e1 * t17 * t9 * t21;
        t26 = 0.1e1 / r / t12;
        t27 = t26 * t21;
        t29 = 0.1e1 / t16 / t15;
        t33 = -t17 * t11 * t10 / 0.2e1 + t24 + t29 * t11 * t27 / 0.2e1;
        t34 = 0.4e1 * t3 * t3;
        t41 = -t17 * t34 * t10 / 0.2e1 + t24 + t29 * t34 * t27 / 0.2e1;
        t45 = t12 * t12;
        d2pdx2[4](0, 0) = t33;
        d2pdx2[4](1, 1) = t41;
        d2pdx2[4](2, 2) = t33;
        d2pdx2[4](3, 3) = t41;
        d2pdx2[4](4, 4) = 0.4e1 * t17 * t26 * t6 + 0.2e1 * t29 / r / t45 * t7;
    }
}

void CellFunctionDeformationCentroid::addValueArc(int i, const VectorXT &nodes, const VectorXi &next, double &value,
                                                  const CellInfo *cellInfo, double xc, double yc) const {

    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;

    x0i = i * nx + 0;
    y0i = i * nx + 1;
    x1i = next(i) * nx + 0;
    y1i = next(i) * nx + 1;
    ri = i * nx + 2;

    x0 = nodes(x0i);
    y0 = nodes(y0i);
    x1 = nodes(x1i);
    y1 = nodes(y1i);
    r = nodes(ri);

    VectorXT x(5);
    x << x0, y0, x1, y1, r;

    VectorXT p;
    MatrixXT dpdx;
    std::vector<MatrixXT> d2pdx2;
    getArcParams(0, x, p, dpdx, d2pdx2);

    // r = p(0)
    double xrc = p(1), yrc = p(2), theta = p(3), phi = p(4);
    double a = sqrt(cellInfo->target_area / M_PI);

    // @formatter:off
    double t1 = r * r;
    double t2 = yc - yrc;
    double t3 = t2 * t1;
    double t4 = xc - xrc;
    double t5 = phi / 0.2e1;
    double t6 = -theta + t5;
    double t7 = cos(t6);
    double t8 = t7 * t7;
    double t13 = r * (xc + yc - xrc - yrc);
    double t14 = xc - yc - xrc + yrc;
    double t15 = sin(t6);
    double t19 = a * a;
    double t20 = xc * xc;
    double t21 = xc * xrc;
    double t22 = 0.2e1 * t21;
    double t23 = xrc * xrc;
    double t24 = yc * yc;
    double t25 = yc * yrc;
    double t26 = 0.2e1 * t25;
    double t27 = yrc * yrc;
    double t28 = t19 - t1 - t20 + t22 - t23 - t24 + t26 - t27;
    double t29 = t28 * t2;
    double t34 = t4 * r;
    double t38 = theta + t5;
    double t39 = cos(t38);
    double t40 = t39 * t39;
    double t44 = sin(t38);
    double t55 = t1 * t1;
    double t66 = pow(t19 - t20 + t22 - t23 - t24 + t26 - t27, 0.2e1);
    value += (0.4e1 * t8 * t4 * t3 + 0.4e1 * t7 * r * (t15 * t14 * t13 / 0.2e1 + t29) + 0.4e1 * t15 * t28 * t34 - 0.4e1 * t40 * t4 * t3 - 0.4e1 * t39 * r * (-t44 * t14 * t13 / 0.2e1 + t29) + 0.4e1 * t44 * t28 * t34 + phi * (t55 + t1 * (-0.2e1 * t19 + 0.4e1 * t20 - 0.8e1 * t21 + 0.4e1 * t23 + 0.4e1 * t24 - 0.8e1 * t25 + 0.4e1 * t27) + t66)) * r;
    // @formatter:on
}

void
CellFunctionDeformationCentroid::addGradientArc(int i, const VectorXT &nodes, const VectorXi &next,
                                                VectorXT &gradient_x,
                                                VectorXT &gradient_centroid,
                                                const CellInfo *cellInfo, double xc,
                                                double yc) const {
    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;

    x0i = i * nx + 0;
    y0i = i * nx + 1;
    x1i = next(i) * nx + 0;
    y1i = next(i) * nx + 1;
    ri = i * nx + 2;

    x0 = nodes(x0i);
    y0 = nodes(y0i);
    x1 = nodes(x1i);
    y1 = nodes(y1i);
    r = nodes(ri);

    VectorXT x(5);
    x << x0, y0, x1, y1, r;

    VectorXT p;
    MatrixXT dpdx;
    std::vector<MatrixXT> d2pdx2;
    getArcParams(1, x, p, dpdx, d2pdx2);
    VectorXT gradient_p = VectorXT::Zero(5);

    // r = p(0)
    double xrc = p(1), yrc = p(2), theta = p(3), phi = p(4);
    double a = sqrt(cellInfo->target_area / M_PI);

    // @formatter:off
    double t1 = r * r;
    double t2 = yc - yrc;
    double t3 = t2 * t1;
    double t4 = xc - xrc;
    double t5 = phi / 0.2e1;
    double t6 = -theta + t5;
    double t7 = cos(t6);
    double t8 = t7 * t7;
    double t9 = t8 * t4;
    double t12 = xc + yc - xrc - yrc;
    double t13 = t12 * r;
    double t14 = xc - yc - xrc + yrc;
    double t15 = sin(t6);
    double t19 = a * a;
    double t20 = xc * xc;
    double t21 = xc * xrc;
    double t22 = 0.2e1 * t21;
    double t23 = xrc * xrc;
    double t24 = yc * yc;
    double t25 = yc * yrc;
    double t26 = 0.2e1 * t25;
    double t27 = yrc * yrc;
    double t28 = t19 - t1 - t20 + t22 - t23 - t24 + t26 - t27;
    double t29 = t28 * t2;
    double t30 = t15 * t14 * t13 / 0.2e1 + t29;
    double t31 = r * t30;
    double t34 = t4 * r;
    double t38 = theta + t5;
    double t39 = cos(t38);
    double t40 = t39 * t39;
    double t41 = t40 * t4;
    double t44 = sin(t38);
    double t48 = -t14 * t44 * t13 / 0.2e1 + t29;
    double t49 = r * t48;
    double t55 = t1 * t1;
    double t63 = -0.2e1 * t19 + 0.4e1 * t20 - 0.8e1 * t21 + 0.4e1 * t23 + 0.4e1 * t24 - 0.8e1 * t25 + 0.4e1 * t27;
    double t64 = t1 * t63;
    double t65 = t19 - t20 + t22 - t23 - t24 + t26 - t27;
    double t66 = t65 * t65;
    double t69 = r * t2;
    double t72 = t14 * t12;
    double t75 = 0.2e1 * t69;
    double t82 = t28 * t4;
    double t85 = t4 * t1;
    double t108 = 0.8e1 * t9 * t69 + 0.4e1 * t7 * r * (t15 * t72 / 0.2e1 - t75) + 0.4e1 * t7 * t30 + 0.4e1 * t15 * t82 - 0.8e1 * t15 * t85 - 0.8e1 * t41 * t69 - 0.4e1 * t39 * r * (-t44 * t72 / 0.2e1 - t75) - 0.4e1 * t39 * t48 + 0.4e1 * t44 * t82 - 0.8e1 * t44 * t85 + phi * (0.4e1 * r * t1 + 0.2e1 * r * t63);
    double t112 = 0.4e1 * t8 * t3;
    double t113 = t14 * r;
    double t115 = t15 * t113 / 0.2e1;
    double t117 = t15 * t13 / 0.2e1;
    double t118 = 0.2e1 * t4 * t2;
    double t123 = t28 * r;
    double t125 = 0.4e1 * t15 * t123;
    double t130 = 0.4e1 * t40 * t3;
    double t132 = t44 * t113 / 0.2e1;
    double t134 = t44 * t13 / 0.2e1;
    double t140 = 0.4e1 * t44 * t123;
    double t152 = 0.4e1 * t8 * t85;
    double t153 = 0.2e1 * t2 * t2;
    double t154 = -t115 + t117 - t19 + t1 + t20 - t22 + t23 + t24 - t26 + t27 + t153;
    double t162 = 0.4e1 * t40 * t85;
    double t163 = t132 - t134 - t19 + t1 + t20 - t22 + t23 + t24 - t26 + t27 + t153;
    double t179 = t15 * t7 * t4 * t3;
    double t181 = t12 * t1;
    double t183 = t8 * t14 * t181;
    double t185 = t15 * t31;
    double t188 = t7 * t28 * t34;
    double t192 = t44 * t39 * t4 * t3;
    double t195 = t40 * t14 * t181;
    double t197 = t49 * t44;
    double t200 = t39 * t28 * t34;
    double t210 = -0.4e1 * t179 + t183 - 0.2e1 * t185 + 0.2e1 * t188 + 0.4e1 * t192 + t195 + 0.2e1 * t197 + 0.2e1 * t200 + t55 + t64 + t66;
    double t212 = -0.2e1 * t4 * t2;
    double t234 = -0.2e1 * t2 * t2;
    double t235 = t115 - t117 + t19 - t1 - t20 + t22 - t23 - t24 + t26 - t27 + t234;
    double t242 = -t132 + t134 + t19 - t1 - t20 + t22 - t23 - t24 + t26 - t27 + t234;
    gradient_p[0] = 0.4e1 * t9 * t3 + 0.4e1 * t7 * t31 + 0.4e1 * t15 * t28 * t34 - 0.4e1 * t41 * t3 - 0.4e1 * t39 * t49 + 0.4e1 * t44 * t28 * t34 + phi * (t55 + t64 + t66) + t108 * r;
    gradient_p[1] = (-t112 + 0.4e1 * t7 * r * (-t115 - t117 + t118) - t125 + 0.8e1 * t15 * t4 * t34 + t130 - 0.4e1 * t39 * r * (t132 + t134 + t118) - t140 + 0.8e1 * t44 * t4 * t34 + phi * (-0.8e1 * t4 * t1 + 0.4e1 * t4 * t65)) * r;
    gradient_p[2] = (-t152 + 0.4e1 * t7 * r * t154 + 0.8e1 * t15 * t2 * t34 + t162 - 0.4e1 * t39 * r * t163 + 0.8e1 * t44 * t2 * t34 + phi * (-0.8e1 * t2 * t1 + 0.4e1 * t2 * t65)) * r;
    gradient_p[3] = (0.8e1 * t179 - 0.2e1 * t183 + 0.4e1 * t185 - 0.4e1 * t188 + 0.8e1 * t192 + 0.2e1 * t195 + 0.4e1 * t197 + 0.4e1 * t200) * r;
    gradient_p[4] = t210 * r;
    gradient_centroid[0] += (t112 + 0.4e1 * t7 * r * (t115 + t117 + t212) + t125 - 0.8e1 * t15 * t4 * t34 - t130 - 0.4e1 * t39 * r * (-t132 - t134 + t212) + t140 - 0.8e1 * t44 * t4 * t34 + phi * (0.8e1 * t4 * t1 - 0.4e1 * t4 * t65)) * r;
    gradient_centroid[1] += (t152 + 0.4e1 * t7 * r * t235 - 0.8e1 * t15 * t2 * t34 - t162 - 0.4e1 * t39 * r * t242 - 0.8e1 * t44 * t2 * t34 + phi * (0.8e1 * t2 * t1 - 0.4e1 * t2 * t65)) * r;
    // @formatter:on

    VectorXT gradient_x_undistributed = gradient_p.transpose() * dpdx;
    gradient_x(x0i) += gradient_x_undistributed(0);
    gradient_x(y0i) += gradient_x_undistributed(1);
    gradient_x(x1i) += gradient_x_undistributed(2);
    gradient_x(y1i) += gradient_x_undistributed(3);
    gradient_x(ri) += gradient_x_undistributed(4);

    if (std::isnan(gradient_x_undistributed.norm())) {
        std::cout << "nan gradient " << theta << " " << phi << std::endl;
    }
}
