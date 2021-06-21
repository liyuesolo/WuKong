#ifndef UI_H
#define UI_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_on_plane.h>
#include <igl/readOBJ.h>

#include "VecMatDef.h"

void appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        double scale = 1.0, Vector<double, 3> shift = Vector<double, 3>::Zero());

void removeSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

void appendCylinderMesh(igl::opengl::glfw::Viewer& viewer,
        Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        std::vector<Vector<double, 2>>& points_on_curve, bool backward = false, int n_div = 8);

#endif