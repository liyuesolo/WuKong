#include <igl/rigid_alignment.h>
#include <igl/AABB.h>
#include <igl/slice.h>
#include <igl/writeOBJ.h>
#include "../include/GeometryHelper.h"
#include "../include/SpatialHash.h"
#include "../icp/simpleicp.h"


void GeometryHelper::normalizePointCloud(MatrixXT& point_cloud)
{
    TV min_corner, max_corner;
    
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < (int)point_cloud.rows(); i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], point_cloud(i, d));
            min_corner[d] = std::min(min_corner[d], point_cloud(i, d));
        }
    }

    TV delta = max_corner - min_corner;
    T scale = std::max(delta[0], std::max(delta[1], delta[2]));
    TV center = 0.5 * (max_corner + min_corner);
    for (int i = 0; i < (int)point_cloud.rows(); i++)
    {
        point_cloud.row(i) = (point_cloud.row(i) - center.transpose()) / scale;
    }
    std::cout << center.transpose() << " " << scale << std::endl;
}

void GeometryHelper::registerPointCloudAToB(const VectorXT& point_cloud_A, 
    const VectorXT& point_cloud_B, VectorXT& result)
{

    TM rotation; 
    Eigen::Matrix<T, 1, 3> translation;

    rotation.setIdentity(3,3);
    translation.setConstant(1,3,0);

    MatrixXT VA, VB, NB;
    
    VA.resize(point_cloud_A.rows()/3, 3);
    VB.resize(point_cloud_B.rows()/3, 3);
    // NB.resize(point_cloud_B_normal.rows()/3, 3);
    
    for (int i = 0; i < point_cloud_A.rows() / 3; i++)
    {
        VA.row(i) = point_cloud_A.segment<3>(i * 3);
    }
    
    for (int i = 0; i < point_cloud_B.rows() / 3; i++)
    {
        VB.row(i) = point_cloud_B.segment<3>(i * 3);
        // NB.row(i) = point_cloud_B_normal.segment<3>(i * 3);
    }

    normalizePointCloud(VA);
    normalizePointCloud(VB);

    TM rotation_along_x = Eigen::AngleAxis<T>(0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();

    for (int i = 0; i < point_cloud_B.rows() / 3; i++)
    {
        VB.row(i) = (rotation_along_x * VB.row(i).transpose()).transpose();
    }

    Matrix<T, 4, 4> trans = SimpleICP(VB, VA);
    std::cout << trans << std::endl;
    rotation = trans.block(0, 0, 3, 3);
    translation = trans.block(0, 3, 3, 1);
    
    Eigen::MatrixXi F;

    igl::writeOBJ("test_VA.obj", VA, F);
    igl::writeOBJ("test_VB.obj", VB, F);
    std::cout << translation << std::endl;
    for (int i = 0; i < VA.rows(); i++)
    {
        VA.row(i) = (rotation * VA.row(i).transpose() + translation.transpose()).transpose();
    }
    // normalizePointCloud(VA);
    
    igl::writeOBJ("test_VA_aligned.obj", VA, F);
}