#ifndef UTIL_H
#define UTIL_H

#include <fstream>
#include <iostream>

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F);

void loadPBCDataFromMSHFile(const std::string& filename, 
    std::vector<std::vector<Vector<int ,2>>>& pbc_pairs);
    // std::vector<Vector<int, 3>>& pbc_pairs);
#endif