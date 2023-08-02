#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"
#include "Projects/Foam2D/include/Tessellation/Tessellation.h"
#include "Projects/Foam2D/include/Boundary/Boundary.h"

class Foam2DInfo {
public:
    std::vector<Tessellation *> tessellations;
    int tessellation = 2;

    int n_free;
    int n_fixed;
    Boundary *boundary;
    VectorXT c_fixed;

    VectorXT energy_area_targets;
    double energy_area_weight = 0.005;
    double energy_length_weight = 0.003;
    double energy_centroid_weight = 0.05;
    double energy_drag_target_weight = 0.00;
    double energy_adhesion_weight = 0.00;
    double energy_deformation_weight = 0.00;
    double energy_deformation_moment_weight = 0.00;

//    double dynamics_m = 0.0001;
//    double dynamics_eta = 0.1;
//    double dynamics_dt = 0.7;
    double dynamics_m = 0.001;
    double dynamics_eta = 0.1;
    double dynamics_dt = 0.003;

    int trajOpt_N = 30; // Number of time steps (states from 0 to N, inclusive)
    double trajOpt_target_weight = 0 * 1e-1;
    double trajOpt_velocity_weight = 0 * 1e-3;
    double trajOpt_input_weight = 1e-2;
    double trajOpt_stepsize_weight = 0 * 3e-3;

    int imageMatch_N = 30;
    double imageMatch_h0 = 0.003;
    double imageMatch_hgrowth = 1.3;

    int selected = -1;
    TV selected_target_pos = {0, 0};

    Tessellation *getTessellation() { return tessellations[tessellation]; }
};
