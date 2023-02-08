#include "../../include/Energy/CellFunctionDeformationCentroid.h"
#include <iostream>

void
CellFunctionDeformationCentroid::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                          double &value,
                                          const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double xc = 0;
    xc_function.addValue(site, nodes, next, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, yc, cellInfo);

    double a = sqrt(cellInfo->target_area / M_PI);
    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    for (int i = 0; i < n_nodes; i++) {
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

        if (fabs(r) > 1e-10) {
            addValueArc(i, nodes, next, value, cellInfo, xc, yc);
        } else {
            // @formatter:off
            value += sqrt(x0 * x0 - 0.2e1 * x1 * x0 + x1 * x1 + pow(y0 - y1, 0.2e1)) * (pow(yc, 0.4e1) + (-0.2e1 * y0 - 0.2e1 * y1) * pow(yc, 0.3e1) + (0.2e1 * y1 * y1 + 0.2e1 * y1 * y0 + 0.2e1 * y0 * y0 + 0.2e1 * xc * xc + (-0.2e1 * x0 - 0.2e1 * x1) * xc + 0.2e1 / 0.3e1 * x1 * x0 + 0.2e1 / 0.3e1 * x0 * x0 + 0.2e1 / 0.3e1 * x1 * x1 - 0.2e1 * a * a) * yc * yc + (-pow(y1, 0.3e1) - y0 * y1 * y1 + (-y0 * y0 - 0.2e1 * xc * xc + (0.4e1 / 0.3e1 * x0 + 0.8e1 / 0.3e1 * x1) * xc - x0 * x0 / 0.3e1 - x1 * x1 - 0.2e1 / 0.3e1 * x1 * x0 + 0.2e1 * a * a) * y1 + 0.2e1 * (-y0 * y0 / 0.2e1 - xc * xc + (0.4e1 / 0.3e1 * x0 + 0.2e1 / 0.3e1 * x1) * xc + a * a - x0 * x0 / 0.2e1 - x1 * x0 / 0.3e1 - x1 * x1 / 0.6e1) * y0) * yc + pow(y1, 0.4e1) / 0.5e1 + y0 * pow(y1, 0.3e1) / 0.5e1 + (y0 * y0 / 0.5e1 + 0.2e1 / 0.3e1 * xc * xc + (-x0 / 0.3e1 - x1) * xc + x1 * x0 / 0.5e1 - 0.2e1 / 0.3e1 * a * a + 0.2e1 / 0.5e1 * x1 * x1 + x0 * x0 / 0.15e2) * y1 * y1 - 0.2e1 / 0.3e1 * y0 * (-0.3e1 / 0.10e2 * y0 * y0 - xc * xc + (x0 + x1) * xc + a * a - 0.3e1 / 0.10e2 * x0 * x0 - 0.2e1 / 0.5e1 * x1 * x0 - 0.3e1 / 0.10e2 * x1 * x1) * y1 + pow(y0, 0.4e1) / 0.5e1 + (0.2e1 / 0.3e1 * xc * xc + (-x1 / 0.3e1 - x0) * xc + 0.2e1 / 0.5e1 * x0 * x0 + x1 * x0 / 0.5e1 - 0.2e1 / 0.3e1 * a * a + x1 * x1 / 0.15e2) * y0 * y0 + pow(xc, 0.4e1) + (-0.2e1 * x0 - 0.2e1 * x1) * pow(xc, 0.3e1) + (-0.2e1 * a * a + 0.2e1 * x0 * x0 + 0.2e1 * x1 * x0 + 0.2e1 * x1 * x1) * xc * xc + 0.2e1 * (a * a - x0 * x0 / 0.2e1 - x1 * x1 / 0.2e1) * (x0 + x1) * xc + pow(a, 0.4e1) + (-0.2e1 / 0.3e1 * x1 * x0 - 0.2e1 / 0.3e1 * x0 * x0 - 0.2e1 / 0.3e1 * x1 * x1) * a * a + pow(x1, 0.4e1) / 0.5e1 + x0 * pow(x1, 0.3e1) / 0.5e1 + pow(x0, 0.3e1) * x1 / 0.5e1 + x0 * x0 * x1 * x1 / 0.5e1 + pow(x0, 0.4e1) / 0.5e1);
            // @formatter:on
        }
    }
}

void CellFunctionDeformationCentroid::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                  VectorXT &gradient_c,
                                                  VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;
    VectorXT gradient_centroid = VectorXT::Zero(2);

    double xc = 0;
    xc_function.addValue(site, nodes, next, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, yc, cellInfo);

    VectorXT temp;
    VectorXT xc_grad = VectorXT::Zero(nodes.rows());
    xc_function.addGradient(site, nodes, next, temp, xc_grad, cellInfo);
    VectorXT yc_grad = VectorXT::Zero(nodes.rows());
    yc_function.addGradient(site, nodes, next, temp, yc_grad, cellInfo);

    double a = sqrt(cellInfo->target_area / M_PI);
    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
            t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60,
            t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74;
    for (int i = 0; i < n_nodes; i++) {
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

        if (fabs(r) > 1e-10) {
            addGradientArc(i, nodes, next, gradient_x, gradient_centroid, cellInfo, xc, yc);
        } else {
            // @formatter:off
            t1 = y0 - y1;
            t2 = x0 * x0;
            t3 = x0 * t2;
            t4 = x1 * x1;
            t5 = x1 * x0;
            t6 = pow(t1, 0.2e1) + t2 + t4 - 0.2e1 * t5;
            t7 = pow(t6, -0.1e1 / 0.2e1);
            t8 = y0 + y1;
            t9 = x0 + x1;
            t10 = y0 * y0;
            t11 = a * a;
            t12 = (t9 - xc) * xc;
            t13 = t8 * y1;
            t14 = x1 * t9;
            t15 = t2 + t14;
            t16 = 0.2e1 * t10 - 0.2e1 * t11 - 0.2e1 * t12 + 0.2e1 * t13 + 0.2e1 / 0.3e1 * t15;
            t17 = 0.4e1 / 0.3e1;
            t18 = t17 * x0;
            t19 = 0.8e1 / 0.3e1 * x1 + t18;
            t20 = t19 * xc - t10 - t4;
            t21 = xc * xc;
            t22 = -t21 + t11;
            t23 = 0.2e1 / 0.3e1 * x1;
            t24 = -x0 / 0.3e1;
            t25 = (-t23 + t24) * x0;
            t26 = t18 + t23;
            t27 = (t26 - xc) * xc;
            t28 = -0.1e1 / 0.2e1;
            t29 = (t24 - x1 / 0.6e1) * x1 + t28 * (t2 + t10) + t11 + t27;
            t13 = (-t13 + t20 + 0.2e1 * t22 + t25) * y1;
            t30 = 0.2e1 * y0;
            t31 = 0.2e1 / 0.5e1;
            t32 = 0.1e1 / 0.5e1;
            t22 = 0.2e1 / 0.3e1 * t22;
            t33 = t2 / 0.15e2 + t4 * t31 + (t24 - x1) * xc + t32 * (t10 + t5) - t22;
            t34 = -0.3e1 / 0.10e2;
            t5 = t34 * (t2 + t10 + t4) + t11 + t12 - t5 * t31;
            t12 = -x1 / 0.3e1;
            t34 = t32 * x0;
            t35 = t31 * t2;
            t22 = (x1 / 0.15e2 + t34) * x1 + (t12 - x0) * xc - t22 + t35;
            t36 = -t2 + t11 - t14;
            t37 = t28 * (t2 + t4) + t11;
            t38 = y1 * y1;
            t39 = yc * yc;
            t40 = yc * t39;
            t41 = y1 * y0;
            t6 = t6 * t7;
            t42 = 0.2e1 * xc;
            t18 = t18 - t42 + t23;
            t23 = -t42 + t9;
            t43 = t17 * xc;
            t44 = t43 + t12 - x0;
            t45 = 0.3e1 / 0.5e1;
            t46 = t31 * x0;
            t47 = -t45 * x1 - t46 + xc;
            t48 = t45 * x0;
            t49 = -t31 * x1 - t48 + xc;
            t50 = 0.4e1 / 0.5e1;
            t51 = t32 * x1;
            t52 = 0.2e1 / 0.3e1 * t41;
            t4 = t7 * (t32 * (t38 * y0 * y1 + t4 * x0 * x1 + pow(t10, 0.2e1) + pow(t2, 0.2e1) + t2 * t4 + t3 * x1 + pow(t38, 0.2e1) + pow(t4, 0.2e1)) - 0.2e1 * t40 * t8 - 0.2e1 * xc * (t36 * xc + t9 * (t21 - t37)) + t10 * t22 + pow(t11, 0.2e1) + pow(t21, 0.2e1) + t33 * t38 + (t30 * t29 + (t39 + t16) * yc + t13) * yc - 0.2e1 / 0.3e1 * t11 * t15 - 0.2e1 / 0.3e1 * t41 * t5);
            t7 = t4 * (x0 - x1);
            t15 = 0.2e1 / 0.3e1 * x0;
            t5 = 0.2e1 / 0.3e1 * t5;
            t1 = t4 * t1;
            t4 = t17 * x1 + t15 - t42;
            t17 = t43 + t24 - x1;
            t24 = t50 * x1;
            t20 = t6 * (t38 * (t45 * y0 + t50 * y1) + yc * ((t30 + 0.4e1 * y1) * yc + t20 + t25 - 0.3e1 * t38 - 0.2e1 * t21 + 0.2e1 * t11 - 0.2e1 * t41) + 0.2e1 * t33 * y1 - 0.2e1 * t40 - t5 * y0) - t1;
            t25 = t9 * (-0.6e1 * t21 + 0.2e1 * t37) + t10 * t44 + t17 * t38 + (t30 * t18 - 0.2e1 * t23 * yc + 0.2e1 * t4 * y1) * yc - 0.4e1 * xc * (-t21 + t36) - t52 * t23;
            gradient_x(x0i) += t6 * (t3 * t50 + (t2 * t45 + (t51 + t46) * x1) * x1 + t10 * (t50 * x0 + t51 - xc) - t11 * t26 + 0.3e1 * t21 * t26 - t38 * t47 / 0.3e1 + (t18 * yc - 0.2e1 / 0.3e1 * t23 * y1 + t30 * t44) * yc + 0.2e1 * (-t9 * x0 - t21 + t37) * xc - t52 * t49) + t7;
            gradient_x(y0i) += t6 * (t10 * t50 * y0 + t31 * t41 * t8 + (t32 * t38 - t5) * y1 + yc * ((0.2e1 * y1 + 0.4e1 * y0) * yc + (-t15 + t12) * x1 - 0.3e1 * t10 + 0.2e1 * t11 - t2 + 0.2e1 * t27 - t38 - 0.2e1 * t41) + 0.2e1 * t22 * y0 - 0.2e1 * t40) + t1;
            gradient_x(x1i) += t6 * (t3 * t32 + ((t24 + t48) * x1 + t35) * x1 + 0.2e1 * (-t21 - t14 + t37) * xc - t10 * t49 / 0.3e1 + t11 * t28 * t19 + 0.3e1 / 0.2e1 * t21 * t19 + t38 * (t24 + t34 - xc) + (t4 * yc + 0.2e1 * t17 * y1 - t30 * t23 / 0.3e1) * yc - t52 * t47) - t7;
            gradient_x(y1i) += t20;
            gradient_centroid(0) += t6 * t25;
            gradient_centroid(1) += t6 * (0.2e1 * t16 * yc + 0.2e1 * t29 * y0 + t39 * (-0.6e1 * t8 + 0.4e1 * yc) + t13);
            // @formatter:on
        }
    }

    MatrixXT d_centroid_d_x(2, nodes.rows());
    d_centroid_d_x.row(0) = xc_grad;
    d_centroid_d_x.row(1) = yc_grad;
    gradient_x += gradient_centroid.transpose() * d_centroid_d_x;
}

void CellFunctionDeformationCentroid::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                 MatrixXT &hessian,
                                                 const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;
    VectorXT gradient_centroid = VectorXT::Zero(2);

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    MatrixXT hess_Cx = MatrixXT::Zero(2, nodes.rows());
    MatrixXT hess_CC = MatrixXT::Zero(2, 2);

    double xc = 0;
    xc_function.addValue(site, nodes, next, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, yc, cellInfo);

    VectorXT temp;
    VectorXT xc_grad = VectorXT::Zero(nodes.rows());
    xc_function.addGradient(site, nodes, next, temp, xc_grad, cellInfo);
    VectorXT yc_grad = VectorXT::Zero(nodes.rows());
    yc_function.addGradient(site, nodes, next, temp, yc_grad, cellInfo);

    MatrixXT xc_hess = MatrixXT::Zero(hessian.rows(), hessian.cols());
    xc_function.addHessian(site, nodes, next, xc_hess, cellInfo);
    MatrixXT yc_hess = MatrixXT::Zero(hessian.rows(), hessian.cols());
    yc_function.addHessian(site, nodes, next, yc_hess, cellInfo);

    double a = sqrt(cellInfo->target_area / M_PI);
    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
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
            t421, t422, t423, t424, t425, t426, t427, t428, t429, t430, t431, t432, t433, t434, t435, t436, t437, t438, t439, t440,
            t441, t442, t443, t444, t445, t446, t447, t448, t449, t450, t451, t452, t453, t454, t455, t456, t457, t458, t459, t460,
            t461, t462, t463, t464, t465, t466, t467, t468, t469, t470, t471, t472, t473, t474, t475, t476, t477, t478, t479, t480,
            t481, t482, t483, t484, t485, t486, t487, t488, t489, t490, t491, t492, t493, t494, t495, t496, t497, t498, t499, t500,
            t501, t502, t503, t504, t505, t506, t507, t508, t509, t510, t511, t512, t513, t514, t515, t516, t517, t518, t519, t520,
            t521, t522, t523, t524, t525, t526, t527, t528, t529, t530, t531, t532, t533, t534, t535, t536, t537, t538, t539, t540,
            t541, t542, t543, t544, t545, t546, t547, t548, t549, t550, t551, t552, t553, t554, t555, t556, t557, t558, t559, t560,
            t561, t562, t563, t564, t565, t566, t567, t568, t569, t570, t571, t572, t573, t574, t575, t576, t577, t578, t579, t580,
            t581, t582, t583, t584, t585, t586, t587, t588, t589, t590, t591, t592, t593, t594, t595, t596, t597, t598, t599, t600,
            t601, t602, t603, t604, t605, t606, t607, t608, t609, t610, t611, t612, t613, t614, t615, t616, t617, t618, t619, t620,
            t621, t622, t623, t624, t625, t626, t627, t628, t629, t630, t631, t632, t633, t634, t635, t636, t637, t638, t639, t640,
            t641, t642, t643, t644, t645, t646, t647, t648, t649, t650, t651, t652, t653, t654, t655, t656, t657, t658, t659, t660,
            t661, t662, t663, t664, t665, t666, t667, t668, t669, t670, t671, t672, t673, t674, t675, t676, t677, t678, t679, t680,
            t681, t682, t683, t684, t685, t686, t687, t688, t689, t690, t691, t692, t693, t694, t695, t696, t697, t698, t699, t700,
            t701, t702, t703, t704, t705, t706, t707, t708, t709, t710, t711, t712, t713, t714, t715, t716, t717, t718, t719, t720,
            t721, t722, t723, t724, t725, t726, t727, t728, t729, t730, t731, t732, t733, t734, t735, t736, t737, t738, t739, t740,
            t741, t742, t743, t744, t745, t746, t747, t748, t749, t750, t751, t752, t753, t754, t755, t756, t757, t758, t759, t760,
            t761, t762, t763, t764, t765, t766, t767, t768, t769, t770, t771, t772, t773, t774, t775, t776, t777, t778, t779, t780,
            t781, t782, t783, t784, t785, t786, t787, t788, t789, t790, t791, t792, t793, t794, t795, t796, t797, t798, t799, t800,
            t801, t802, t803, t804, t805, t806, t807, t808, t809, t810, t811, t812, t813, t814, t815, t816, t817, t818, t819, t820,
            t821, t822, t823, t824, t825, t826, t827, t828, t829, t830, t831, t832, t833, t834, t835, t836, t837, t838, t839, t840,
            t841, t842, t843, t844, t845, t846, t847, t848, t849, t850, t851, t852, t853, t854, t855, t856, t857, t858, t859, t860,
            t861, t862, t863, t864, t865, t866, t867, t868, t869, t870, t871, t872, t873, t874, t875, t876, t877, t878, t879, t880,
            t881, t882, t883, t884, t885, t886, t887, t888, t889, t890, t891, t892, t893, t894, t895, t896, t897, t898, t899, t900,
            t901, t902, t903, t904, t905, t906, t907, t908, t909, t910, t911, t912, t913, t914, t915, t916, t917, t918, t919, t920,
            t921, t922, t923, t924, t925, t926, t927, t928, t929, t930, t931, t932, t933, t934, t935, t936, t937, t938, t939, t940,
            t941, t942, t943, t944, t945, t946, t947, t948, t949, t950, t951, t952, t953, t954, t955, t956, t957, t958, t959, t960,
            t961, t962, t963, t964, t965, t966, t967, t968, t969, t970, t971, t972, t973, t974, t975, t976, t977, t978, t979, t980,
            t981, t982, t983, t984, t985, t986, t987, t988, t989, t990, t991, t992, t993, t994, t995, t996, t997, t998, t999, t1000,
            t1001, t1002, t1003, t1004, t1005, t1006, t1007, t1008, t1009, t1010, t1011, t1012, t1013, t1014, t1015, t1016, t1017, t1018, t1019, t1020,
            t1021, t1022, t1023, t1024, t1025, t1026, t1027, t1028, t1029, t1030, t1031, t1032, t1033, t1034, t1035, t1036, t1037, t1038, t1039, t1040,
            t1041, t1042, t1043, t1044, t1045, t1046, t1047, t1048, t1049, t1050, t1051, t1052, t1053, t1054, t1055, t1056, t1057, t1058, t1059, t1060,
            t1061, t1062, t1063, t1064, t1065, t1066, t1067, t1068, t1069, t1070, t1071, t1072, t1073, t1074, t1075, t1076, t1077, t1078, t1079, t1080,
            t1081, t1082, t1083, t1084, t1085, t1086, t1087, t1088, t1089, t1090, t1091, t1092, t1093, t1094, t1095, t1096, t1097, t1098, t1099, t1100,
            t1101, t1102, t1103, t1104, t1105, t1106, t1107, t1108, t1109, t1110, t1111, t1112, t1113, t1114, t1115, t1116, t1117, t1118, t1119, t1120,
            t1121, t1122, t1123, t1124, t1125, t1126, t1127, t1128, t1129, t1130, t1131, t1132, t1133, t1134, t1135, t1136, t1137, t1138, t1139, t1140,
            t1141, t1142, t1143, t1144, t1145, t1146, t1147, t1148, t1149, t1150, t1151, t1152, t1153, t1154, t1155, t1156, t1157, t1158, t1159, t1160,
            t1161, t1162, t1163, t1164, t1165, t1166, t1167, t1168, t1169, t1170, t1171, t1172, t1173, t1174, t1175, t1176, t1177, t1178, t1179, t1180,
            t1181, t1182, t1183, t1184, t1185, t1186, t1187, t1188, t1189, t1190, t1191, t1192, t1193, t1194, t1195, t1196, t1197, t1198, t1199, t1200,
            t1201, t1202, t1203, t1204, t1205, t1206, t1207, t1208, t1209, t1210, t1211, t1212, t1213, t1214, t1215, t1216, t1217, t1218, t1219, t1220,
            t1221, t1222, t1223, t1224, t1225, t1226, t1227, t1228, t1229, t1230, t1231, t1232, t1233, t1234, t1235, t1236, t1237, t1238, t1239, t1240,
            t1241, t1242, t1243, t1244, t1245, t1246, t1247, t1248, t1249, t1250, t1251, t1252, t1253, t1254, t1255, t1256, t1257, t1258, t1259, t1260,
            t1261, t1262, t1263, t1264, t1265, t1266, t1267, t1268, t1269, t1270, t1271, t1272, t1273, t1274, t1275, t1276, t1277, t1278, t1279, t1280,
            t1281, t1282, t1283, t1284, t1285, t1286, t1287, t1288, t1289, t1290, t1291, t1292, t1293, t1294, t1295, t1296, t1297, t1298, t1299, t1300,
            t1301, t1302, t1303, t1304, t1305, t1306, t1307, t1308, t1309, t1310, t1311, t1312, t1313, t1314, t1315, t1316, t1317, t1318, t1319, t1320,
            t1321, t1322, t1323, t1324, t1325, t1326, t1327, t1328, t1329, t1330, t1331, t1332, t1333, t1334, t1335, t1336, t1337, t1338, t1339, t1340,
            t1341, t1342, t1343, t1344, t1345, t1346, t1347, t1348, t1349, t1350, t1351, t1352, t1353, t1354, t1355, t1356, t1357, t1358, t1359, t1360,
            t1361, t1362, t1363, t1364, t1365, t1366, t1367, t1368, t1369, t1370, t1371, t1372, t1373, t1374, t1375, t1376, t1377, t1378, t1379, t1380,
            t1381, t1382, t1383, t1384, t1385, t1386, t1387, t1388, t1389, t1390, t1391, t1392, t1393, t1394, t1395, t1396, t1397, t1398, t1399, t1400,
            t1401, t1402, t1403, t1404, t1405, t1406, t1407, t1408, t1409, t1410, t1411, t1412, t1413, t1414, t1415, t1416, t1417, t1418, t1419, t1420,
            t1421, t1422, t1423, t1424, t1425, t1426, t1427, t1428, t1429, t1430, t1431, t1432, t1433, t1434, t1435, t1436, t1437, t1438, t1439, t1440,
            t1441, t1442, t1443, t1444, t1445, t1446, t1447, t1448, t1449, t1450, t1451, t1452, t1453, t1454, t1455, t1456, t1457, t1458, t1459, t1460,
            t1461, t1462, t1463, t1464, t1465, t1466, t1467, t1468, t1469, t1470, t1471, t1472, t1473, t1474, t1475, t1476, t1477, t1478, t1479, t1480,
            t1481, t1482, t1483, t1484, t1485, t1486, t1487, t1488, t1489, t1490, t1491, t1492, t1493, t1494, t1495, t1496, t1497, t1498, t1499, t1500, t1533;
    for (int i = 0; i < n_nodes; i++) {
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

        if (fabs(r) > 1e-10) {
            addHessianArc(i, nodes, next, gradient_centroid, hess_xx, hess_Cx, hess_CC, cellInfo, xc, yc);
        } else {
            double unknown[6][6];

            // @formatter:off
            t1 = y0 - y1;
            t2 = x0 * x0;
            t3 = x0 * t2;
            t4 = x1 * x1;
            t5 = x1 * x0;
            t6 = pow(t1, 0.2e1) + t2 + t4 - 0.2e1 * t5;
            t7 = pow(t6, -0.1e1 / 0.2e1);
            t8 = y0 + y1;
            t9 = x0 + x1;
            t10 = y0 * y0;
            t11 = a * a;
            t12 = (t9 - xc) * xc;
            t13 = t8 * y1;
            t14 = x1 * t9;
            t15 = t2 + t14;
            t16 = 0.2e1 * t10 - 0.2e1 * t11 - 0.2e1 * t12 + 0.2e1 * t13 + 0.2e1 / 0.3e1 * t15;
            t17 = 0.4e1 / 0.3e1;
            t18 = t17 * x0;
            t19 = 0.8e1 / 0.3e1 * x1 + t18;
            t20 = t19 * xc - t10 - t4;
            t21 = xc * xc;
            t22 = -t21 + t11;
            t23 = 0.2e1 / 0.3e1 * x1;
            t24 = -x0 / 0.3e1;
            t25 = (-t23 + t24) * x0;
            t26 = t18 + t23;
            t27 = (t26 - xc) * xc;
            t28 = -0.1e1 / 0.2e1;
            t29 = (t24 - x1 / 0.6e1) * x1 + t28 * (t2 + t10) + t11 + t27;
            t13 = (-t13 + t20 + 0.2e1 * t22 + t25) * y1;
            t30 = 0.2e1 * y0;
            t31 = 0.2e1 / 0.5e1;
            t32 = 0.1e1 / 0.5e1;
            t22 = 0.2e1 / 0.3e1 * t22;
            t33 = t2 / 0.15e2 + t4 * t31 + (t24 - x1) * xc + t32 * (t10 + t5) - t22;
            t34 = -0.3e1 / 0.10e2;
            t5 = t34 * (t2 + t10 + t4) + t11 + t12 - t5 * t31;
            t12 = -x1 / 0.3e1;
            t34 = t32 * x0;
            t35 = t31 * t2;
            t22 = (x1 / 0.15e2 + t34) * x1 + (t12 - x0) * xc - t22 + t35;
            t36 = -t2 + t11 - t14;
            t37 = t28 * (t2 + t4) + t11;
            t38 = y1 * y1;
            t39 = yc * yc;
            t40 = yc * t39;
            t41 = y1 * y0;
            t6 = t6 * t7;
            t42 = 0.2e1 * xc;
            t18 = t18 - t42 + t23;
            t23 = -t42 + t9;
            t43 = t17 * xc;
            t44 = t43 + t12 - x0;
            t45 = 0.3e1 / 0.5e1;
            t46 = t31 * x0;
            t47 = -t45 * x1 - t46 + xc;
            t48 = t45 * x0;
            t49 = -t31 * x1 - t48 + xc;
            t50 = 0.4e1 / 0.5e1;
            t51 = t32 * x1;
            t52 = 0.2e1 / 0.3e1 * t41;
            t4 = t7 * (t32 * (t38 * y0 * y1 + t4 * x0 * x1 + pow(t10, 0.2e1) + pow(t2, 0.2e1) + t2 * t4 + t3 * x1 + pow(t38, 0.2e1) + pow(t4, 0.2e1)) - 0.2e1 * t40 * t8 - 0.2e1 * xc * (t36 * xc + t9 * (t21 - t37)) + t10 * t22 + pow(t11, 0.2e1) + pow(t21, 0.2e1) + t33 * t38 + (t30 * t29 + (t39 + t16) * yc + t13) * yc - 0.2e1 / 0.3e1 * t11 * t15 - 0.2e1 / 0.3e1 * t41 * t5);
            t7 = t4 * (x0 - x1);
            t15 = 0.2e1 / 0.3e1 * x0;
            t5 = 0.2e1 / 0.3e1 * t5;
            t1 = t4 * t1;
            t4 = t17 * x1 + t15 - t42;
            t17 = t43 + t24 - x1;
            t24 = t50 * x1;
            t20 = t6 * (t38 * (t45 * y0 + t50 * y1) + yc * ((t30 + 0.4e1 * y1) * yc + t20 + t25 - 0.3e1 * t38 - 0.2e1 * t21 + 0.2e1 * t11 - 0.2e1 * t41) + 0.2e1 * t33 * y1 - 0.2e1 * t40 - t5 * y0) - t1;
            t25 = t9 * (-0.6e1 * t21 + 0.2e1 * t37) + t10 * t44 + t17 * t38 + (t30 * t18 - 0.2e1 * t23 * yc + 0.2e1 * t4 * y1) * yc - 0.4e1 * xc * (-t21 + t36) - t52 * t23;
            gradient_centroid(0) += t6 * t25;
            gradient_centroid(1) += t6 * (0.2e1 * t16 * yc + 0.2e1 * t29 * y0 + t39 * (-0.6e1 * t8 + 0.4e1 * yc) + t13);

            t1 = x0 * x0;
            t2 = x1 * x0;
            t4 = x1 * x1;
            t5 = y0 - y1;
            t6 = t5 * t5;
            t7 = t1 - 0.2e1 * t2 + t4 + t6;
            t8 = sqrt(t7);
            t11 = yc * yc;
            t12 = t11 * t11;
            t13 = -y0 - y1;
            t14 = yc * t11;
            t16 = y1 * y1;
            t18 = y1 * y0;
            t19 = 0.2e1 * t18;
            t20 = y0 * y0;
            t22 = xc * xc;
            t23 = 0.2e1 * t22;
            t24 = -x0 - x1;
            t25 = 0.2e1 * xc * t24;
            t26 = 0.2e1 / 0.3e1 * t2;
            t29 = a * a;
            t30 = 0.2e1 * t29;
            t31 = 0.2e1 * t16 + t19 + 0.2e1 * t20 + t23 + t25 + t26 + 0.2e1 / 0.3e1 * t1 + 0.2e1 / 0.3e1 * t4 - t30;
            t33 = y1 * t16;
            t34 = t16 * y0;
            t35 = 0.4e1 / 0.3e1 * x0;
            t36 = 0.8e1 / 0.3e1 * x1;
            t38 = xc * (t35 + t36);
            t39 = t1 / 0.3e1;
            t41 = y1 * (-t20 - t23 + t38 - t39 - t4 - t26 + t30);
            t43 = 0.2e1 / 0.3e1 * x1;
            t44 = t35 + t43;
            t45 = xc * t44;
            t46 = t1 / 0.2e1;
            t51 = 0.2e1 * y0 * (-t20 / 0.2e1 - t22 + t45 + t29 - t46 - t2 / 0.3e1 - t4 / 0.6e1);
            t54 = t16 * t16;
            t58 = t20 / 0.5e1;
            t59 = 0.2e1 / 0.3e1 * t22;
            t60 = x0 / 0.3e1;
            t62 = xc * (-t60 - x1);
            t63 = t2 / 0.5e1;
            t64 = 0.2e1 / 0.3e1 * t29;
            t65 = 0.2e1 / 0.5e1 * t4;
            t67 = t58 + t59 + t62 + t63 - t64 + t65 + t1 / 0.15e2;
            t70 = -xc * t24;
            t72 = 0.2e1 / 0.5e1 * t2;
            t74 = -0.3e1 / 0.10e2 * t20 - t22 + t70 + t29 - 0.3e1 / 0.10e2 * t1 - t72 - 0.3e1 / 0.10e2 * t4;
            t75 = t74 * y0;
            t78 = t20 * t20;
            t80 = x1 / 0.3e1;
            t82 = xc * (-t80 - x0);
            t83 = 0.2e1 / 0.5e1 * t1;
            t85 = t59 + t82 + t83 + t63 - t64 + t4 / 0.15e2;
            t88 = t22 * t22;
            t89 = xc * t22;
            t91 = -t29 + t1 + t2 + t4;
            t94 = t29 - t46 - t4 / 0.2e1;
            t95 = -t24 * t94;
            t98 = t29 * t29;
            t101 = t4 * t4;
            t103 = x1 * t4;
            t106 = x0 * t1;
            t111 = t1 * t1;
            t113 = t88 + 0.2e1 * t89 * t24 + 0.2e1 * t22 * t91 + 0.2e1 * xc * t95 + t98 + 0.2e1 / 0.3e1 * t29 * (-t1 - t2 - t4) + t101 / 0.5e1 + t103 * x0 / 0.5e1 + x1 * t106 / 0.5e1 + t4 * t1 / 0.5e1 + t111 / 0.5e1;
            t114 = t12 + 0.2e1 * t14 * t13 + t11 * t31 + yc * (-t33 - t34 + t41 + t51) + t54 / 0.5e1 + t33 * y0 / 0.5e1 + t16 * t67 - 0.2e1 / 0.3e1 * y1 * t75 + t78 / 0.5e1 + t20 * t85 + t113;
            t115 = t114 / t8 / t7;
            t116 = x0 - x1;
            t119 = t116 * t116 * t115;
            t120 = 0.1e1 / t8;
            t121 = 0.2e1 * xc;
            t122 = -t121 + t43 + t35;
            t124 = 0.4e1 / 0.3e1 * xc;
            t125 = 0.2e1 / 0.3e1 * x0;
            t126 = t124 - t125 - t43;
            t127 = y1 * t126;
            t128 = t124 - x0 - t80;
            t130 = 0.2e1 * y0 * t128;
            t133 = xc / 0.3e1;
            t134 = x1 / 0.5e1;
            t136 = -t133 + t134 + 0.2e1 / 0.15e2 * x0;
            t140 = xc - 0.3e1 / 0.5e1 * x0 - 0.2e1 / 0.5e1 * x1;
            t141 = t140 * y0;
            t145 = -xc + 0.4e1 / 0.5e1 * x0 + t134;
            t147 = 0.2e1 * t89;
            t149 = 0.2e1 * x1;
            t150 = 0.4e1 * x0 + t149;
            t152 = -t24 * x0;
            t156 = 0.2e1 * xc * t94;
            t159 = x1 * t1;
            t161 = t4 * x0;
            t164 = t11 * t122 + yc * (t127 + t130) + t16 * t136 - 0.2e1 / 0.3e1 * y1 * t141 + t20 * t145 - t147 + t22 * t150 - 0.2e1 * xc * t152 + t156 - t29 * t44 + t103 / 0.5e1 + 0.3e1 / 0.5e1 * t159 + 0.2e1 / 0.5e1 * t161 + 0.4e1 / 0.5e1 * t106;
            t165 = t164 * t120;
            t167 = t114 * t120;
            t168 = 0.4e1 / 0.3e1 * t11;
            t170 = 0.2e1 * y0;
            t174 = 0.2e1 / 0.5e1 * t18;
            t176 = 0.4e1 * t22;
            t177 = 0.2e1 * t70;
            t178 = x0 * xc;
            t180 = 0.4e1 / 0.3e1 * t29;
            t181 = 0.6e1 / 0.5e1 * t2;
            t183 = t168 + yc * (-0.2e1 / 0.3e1 * y1 - t170) + 0.2e1 / 0.15e2 * t16 + t174 + 0.4e1 / 0.5e1 * t20 + t176 - t177 - 0.4e1 * t178 - t180 + t181 + t65 + 0.12e2 / 0.5e1 * t1;
            t188 = t5 * t116 * t115;
            t189 = 0.2e1 * t14;
            t190 = 0.2e1 * y1;
            t192 = t190 + 0.4e1 * y0;
            t194 = 0.3e1 * t20;
            t195 = 0.2e1 * t45;
            t196 = t4 / 0.3e1;
            t210 = (-t189 + t11 * t192 + yc * (-t16 - t19 - t194 - t23 + t195 + t30 - t1 - t26 - t196) + t33 / 0.5e1 + 0.2e1 / 0.5e1 * t34 - 0.2e1 / 0.3e1 * y1 * t74 + 0.2e1 / 0.5e1 * y1 * t20 + 0.4e1 / 0.5e1 * y0 * t20 + 0.2e1 * y0 * t85) * t120;
            t215 = 0.8e1 / 0.3e1 * xc;
            t216 = 0.2e1 * x0;
            t225 = -t188 + t116 * t210 + t5 * t165 + (yc * (t215 - t216 - t43) - 0.2e1 / 0.3e1 * y1 * t140 + 0.2e1 * y0 * t145) * t8;
            t229 = 0.4e1 / 0.3e1 * x1;
            t230 = -t121 + t125 + t229;
            t232 = t215 - t149 - t125;
            t233 = y1 * t232;
            t237 = 0.2e1 * y0 * (0.2e1 / 0.3e1 * xc - t60 - t80);
            t240 = x0 / 0.5e1;
            t242 = -xc + t240 + 0.4e1 / 0.5e1 * x1;
            t246 = xc - 0.2e1 / 0.5e1 * x0 - 0.3e1 / 0.5e1 * x1;
            t247 = t246 * y0;
            t251 = -t133 + t240 + 0.2e1 / 0.15e2 * x1;
            t254 = t216 + 0.4e1 * x1;
            t256 = -t24 * x1;
            t265 = t11 * t230 + yc * (t233 + t237) + t16 * t242 - 0.2e1 / 0.3e1 * y1 * t247 + t20 * t251 - t147 + t22 * t254 - 0.2e1 * xc * t256 + t156 + t29 * (-t125 - t229) + 0.4e1 / 0.5e1 * t103 + 0.3e1 / 0.5e1 * t161 + t106 / 0.5e1 + 0.2e1 / 0.5e1 * t159;
            t266 = t265 * t120;
            t276 = x1 * xc;
            t281 = 0.2e1 / 0.3e1 * t11 + 0.2e1 / 0.3e1 * yc * t13 + t16 / 0.5e1 + 0.4e1 / 0.15e2 * t18 + t58 + t23 - 0.2e1 * t178 - 0.2e1 * t276 - t64 + 0.3e1 / 0.5e1 * t4 + 0.3e1 / 0.5e1 * t1 + 0.4e1 / 0.5e1 * t2;
            t283 = t116 * t116 * t115 - t116 * t165 + t116 * t266 + t281 * t8 - t167;
            t286 = -t5 * t116 * t115;
            t288 = 0.4e1 * y1 + t170;
            t290 = 0.3e1 * t16;
            t299 = (-t189 + t11 * t288 + yc * (-t290 - t19 - t20 - t23 + t38 - t39 - t4 - t26 + t30) + 0.4e1 / 0.5e1 * t33 + 0.3e1 / 0.5e1 * t34 + 0.2e1 * y1 * t67 - 0.2e1 / 0.3e1 * t75) * t120;
            t304 = yc * t126;
            t310 = -t286 + t116 * t299 - t5 * t165 + (t304 + 0.2e1 * y1 * t136 - 0.2e1 / 0.3e1 * t141) * t8;
            t311 = 0.4e1 * xc;
            t312 = t311 - t216 - t149;
            t314 = -t311 + t35 + t36;
            t315 = y1 * t314;
            t317 = 0.2e1 * y0 * t122;
            t320 = t124 - t60 - x1;
            t322 = -t121 + x0 + x1;
            t323 = t322 * y0;
            t334 = (t11 * t312 + yc * (t315 + t317) + t16 * t320 - 0.2e1 / 0.3e1 * y1 * t323 + t20 * t128 + 0.4e1 * t89 + 0.6e1 * t22 * t24 + 0.4e1 * xc * t91 + 0.2e1 * t95) * t120;
            t337 = 0.2e1 * t11;
            t343 = 0.2e1 / 0.3e1 * t18;
            t344 = 0.6e1 * t22;
            t348 = -t337 + (0.4e1 / 0.3e1 * y1 + 0.8e1 / 0.3e1 * y0) * yc - t16 / 0.3e1 - t343 - t20 - t344 + 0.2e1 * xc * t150 - 0.2e1 * t152 + t30 - t1 - t4;
            t350 = t116 * t334 + t348 * t8;
            t357 = (0.6e1 * t11 * t13 + 0.2e1 * yc * t31 + 0.4e1 * t14 - t33 - t34 + t41 + t51) * t120;
            t364 = t116 * t357 + (0.2e1 * yc * t122 + t127 + t130) * t8;
            t367 = t5 * t5 * t115;
            t369 = 0.4e1 * t11;
            t374 = 0.6e1 / 0.5e1 * t18;
            t376 = 0.4e1 / 0.3e1 * t22;
            t380 = t369 + yc * (-t190 - 0.6e1 * y0) + 0.2e1 / 0.5e1 * t16 + t374 + 0.12e2 / 0.5e1 * t20 + t376 + 0.2e1 * t82 + 0.4e1 / 0.5e1 * t1 + t72 - t180 + 0.2e1 / 0.15e2 * t4;
            t393 = -t286 + t5 * t266 - t116 * t210 + (t304 - 0.2e1 / 0.3e1 * y1 * t246 + 0.2e1 * y0 * t251) * t8;
            t401 = 0.2e1 * yc * t13;
            t409 = t337 + t401 + 0.3e1 / 0.5e1 * t16 + 0.4e1 / 0.5e1 * t18 + 0.3e1 / 0.5e1 * t20 + t59 - 0.2e1 / 0.3e1 * t70 - t64 + t1 / 0.5e1 + 0.4e1 / 0.15e2 * t2 + t4 / 0.5e1;
            t411 = t5 * t5 * t115 - t5 * t210 + t5 * t299 + t409 * t8 - t167;
            t421 = t5 * t334 + (yc * (-t311 + t229 + 0.8e1 / 0.3e1 * x0) - 0.2e1 / 0.3e1 * y1 * t322 + t130) * t8;
            t424 = 0.6e1 * t11;
            t427 = 0.2e1 * yc * t192 - t1 - t16 - t19 - t194 + t195 - t196 - t23 - t26 + t30 - t424;
            t429 = t5 * t357 + t427 * t8;
            t438 = t168 + yc * (-t190 - 0.2e1 / 0.3e1 * y0) + 0.4e1 / 0.5e1 * t16 + t174 + 0.2e1 / 0.15e2 * t20 + t176 - t177 - 0.4e1 * t276 - t180 + 0.12e2 / 0.5e1 * t4 + t181 + t83;
            t451 = -t188 - t116 * t299 - t5 * t266 + (yc * t232 + 0.2e1 * y1 * t242 - 0.2e1 / 0.3e1 * t247) * t8;
            t462 = -t337 + (0.8e1 / 0.3e1 * y1 + 0.4e1 / 0.3e1 * y0) * yc - t16 - t343 - t20 / 0.3e1 - t344 + 0.2e1 * xc * t254 - 0.2e1 * t256 + t30 - t1 - t4;
            t464 = -t116 * t334 + t462 * t8;
            t471 = -t116 * t357 + (0.2e1 * yc * t230 + t233 + t237) * t8;
            t481 = t369 + yc * (-0.6e1 * y1 - t170) + 0.12e2 / 0.5e1 * t16 + t374 + 0.2e1 / 0.5e1 * t20 + t376 + 0.2e1 * t62 + t72 - t180 + 0.4e1 / 0.5e1 * t4 + 0.2e1 / 0.15e2 * t1;
            t492 = -t5 * t334 + (yc * t314 + 0.2e1 * y1 * t320 - 0.2e1 / 0.3e1 * t323) * t8;
            t497 = 0.2e1 * yc * t288 - t19 - t20 - t23 - t26 - t290 + t30 + t38 - t39 - t4 - t424;
            t499 = -t5 * t357 + t497 * t8;
            t506 = 0.4e1 * t29;
            t510 = t369 + 0.4e1 * yc * t13 + 0.4e1 / 0.3e1 * t16 + 0.4e1 / 0.3e1 * t18 + 0.4e1 / 0.3e1 * t20 + 0.12e2 * t22 + 0.6e1 * t25 - t506 + 0.4e1 * t1 + 0.4e1 * t2 + 0.4e1 * t4;
            t515 = (0.2e1 * yc * t312 + t315 + t317) * t8;
            t525 = 0.12e2 * t11 + 0.6e1 * t401 + 0.4e1 * t16 + 0.4e1 * t18 + 0.4e1 * t20 + t176 + 0.2e1 * t25 + 0.4e1 / 0.3e1 * t2 + 0.4e1 / 0.3e1 * t1 + 0.4e1 / 0.3e1 * t4 - t506;
            unknown[0][0] = 0.2e1 * t116 * t165 + t183 * t8 - t119 + t167;
            unknown[0][1] = t225;
            unknown[0][2] = t283;
            unknown[0][3] = t310;
            unknown[0][4] = t350;
            unknown[0][5] = t364;
            unknown[1][0] = t225;
            unknown[1][1] = 0.2e1 * t5 * t210 + t380 * t8 + t167 - t367;
            unknown[1][2] = t393;
            unknown[1][3] = t411;
            unknown[1][4] = t421;
            unknown[1][5] = t429;
            unknown[2][0] = t283;
            unknown[2][1] = t393;
            unknown[2][2] = -0.2e1 * t116 * t266 + t438 * t8 - t119 + t167;
            unknown[2][3] = t451;
            unknown[2][4] = t464;
            unknown[2][5] = t471;
            unknown[3][0] = t310;
            unknown[3][1] = t411;
            unknown[3][2] = t451;
            unknown[3][3] = -0.2e1 * t5 * t299 + t481 * t8 + t167 - t367;
            unknown[3][4] = t492;
            unknown[3][5] = t499;
            unknown[4][0] = t350;
            unknown[4][1] = t421;
            unknown[4][2] = t464;
            unknown[4][3] = t492;
            unknown[4][4] = t510 * t8;
            unknown[4][5] = t515;
            unknown[5][0] = t364;
            unknown[5][1] = t429;
            unknown[5][2] = t471;
            unknown[5][3] = t499;
            unknown[5][4] = t515;
            unknown[5][5] = t525 * t8;
            // @formatter:on

            MatrixXT hessian_maple = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], 6, 6);
            hess_CC += hessian_maple.bottomRightCorner(2, 2);
            MatrixXT hess_Cx_undistributed = hessian_maple.bottomLeftCorner(2, 4);
            MatrixXT hess_xx_undistributed = hessian_maple.topLeftCorner(4, 4);

            VectorXi idx(4);
            idx << x0i, y0i, x1i, y1i;
            for (int j = 0; j < 4; j++) {
                hess_Cx(0, idx(j)) += hess_Cx_undistributed(0, j);
                hess_Cx(1, idx(j)) += hess_Cx_undistributed(1, j);
                for (int k = 0; k < 4; k++) {
                    hess_xx(idx(j), idx(k)) += hess_xx_undistributed(j, k);
                }
            }
        }
    }

    MatrixXT d_centroid_d_x(2, nodes.rows());
    d_centroid_d_x.row(0) = xc_grad;
    d_centroid_d_x.row(1) = yc_grad;
    hess_xx += d_centroid_d_x.transpose() * hess_CC * d_centroid_d_x + d_centroid_d_x.transpose() * hess_Cx +
               hess_Cx.transpose() * d_centroid_d_x +
               gradient_centroid(0) * xc_hess.bottomRightCorner(nodes.rows(), nodes.rows()) +
               gradient_centroid(1) * yc_hess.bottomRightCorner(nodes.rows(), nodes.rows());


//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, next, temp, grad, cellInfo);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, next, temp, gradp, cellInfo);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "deformation hess_xx(" << j << "," << i << ") " << (gradp[j] - grad[j]) / eps << " "
//                      << hess_xx(j, i) << std::endl;
//        }
//    }

//    VectorXT gradc = VectorXT::Zero(site.rows());
//    VectorXT gradx = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, next, gradc, gradx, cellInfo);
//    VectorXT grad(gradc.rows() + gradx.rows());
//    grad << gradc, gradx;
//    double eps = 1e-6;
//    VectorXT y(site.rows() + nodes.rows());
//    y << site, nodes;
//    for (int i = 0; i < hessian.rows(); i++) {
//        VectorXT yp = y;
//        yp(i) += eps;
//        VectorXT gradcp = VectorXT::Zero(site.rows());
//        VectorXT gradxp = VectorXT::Zero(nodes.rows());
//        addGradient(yp.segment(0, site.rows()), yp.segment(site.rows(), nodes.rows()), next, gradcp, gradxp, cellInfo);
//        VectorXT gradp(gradc.rows() + gradx.rows());
//        gradp << gradcp, gradxp;
//        for (int j = 0; j < hessian.rows(); j++) {
//            std::cout << "deformation hessian(" << j << "," << i << ") " << (gradp[j] - grad[j]) / eps << " "
//                      << hessian(j, i) << std::endl;
//        }
//    }
}
