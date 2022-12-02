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

class Foam2DInfo {
public:
    std::vector<Tessellation *> tessellations;
    int tessellation = 0;

    int n_free;
    int n_fixed;
    VectorXd boundary;
    VectorXd c_fixed;

    VectorXd energy_area_targets;
    double energy_area_weight = 0.1;
    double energy_length_weight = 0.003;
    double energy_centroid_weight = 0.05;
    double energy_drag_target_weight = 0.00;

    double dynamics_m = 0.002;
    double dynamics_eta = 0.01;
    double dynamics_dt = 0.01;

    int trajOpt_N = 50; // Number of time steps (states from 0 to N, inclusive)
    double trajOpt_target_weight = 1e-1;
    double trajOpt_velocity_weight = 0 * 1e-3;
    double trajOpt_input_weight = 0 * 1e-2;

    int selected = -1;
    TV selected_target_pos = {0, 0};

    Tessellation *getTessellation() { return tessellations[tessellation]; }
};
