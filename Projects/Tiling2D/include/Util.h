#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include "VecMatDef.h"
#include <fstream>
#include <iostream>

void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

#endif