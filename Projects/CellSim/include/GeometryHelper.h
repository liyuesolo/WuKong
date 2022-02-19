#ifndef GEOMETRY_HELPER_H
#define GEOMETRY_HELPER_H
#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>



#include "VecMatDef.h"

namespace GeometryHelper
{
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using TV = Vector<T, 3>;
    using TM = Matrix<T, 3, 3>;

    void registerPointCloudAToB(const VectorXT& point_cloud_A, 
        const VectorXT& point_cloud_B, 
        VectorXT& result);

    void normalizePointCloud(MatrixXT& point_cloud);
};

#endif