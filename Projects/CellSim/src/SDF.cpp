#include "../include/SDF.h"

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

// void VdbLevelSetSDF::initialize(const VectorXT& _data_points, 
//     const VectorXT& _data_point_normals, T _radius, T _search_radius)
// {
//     data_points = _data_points;
//     data_point_normals = _data_point_normals;
//     radius = _radius;
//     search_radius = _search_radius;
//     n_points = data_points.rows();
 
// }  

T IMLS::value(const TV& test_point)
{
    T weights_sum = 0.0;
    for (int i = 0; i < n_points; i++)
    {
        
    }
}

void VdbLevelSetSDF::initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
    const VectorXT& normals, T epsilon)
{
    int nv = vertices.rows() / 3;
    int nf = indices.rows() / 3;
    std::vector<Vec3s> points(nv);
    std::vector<Vec3I> triangles(nf);
    std::vector<Vec4I> quads;
    tbb::parallel_for(0, nv, [&](int i){
        points[i] = Vec3s(vertices[i * 3 + 0] + normals[i * 3 + 0] * epsilon, 
                        vertices[i * 3 + 1] + normals[i * 3 + 1] * epsilon,  
                        vertices[i * 3 + 2] + normals[i * 3 + 2] * epsilon);
    });
    tbb::parallel_for(0, nf, [&](int i){
        triangles[i] = Vec3I(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]);
    });
    levelsetFromMesh(points, triangles, quads);
}

void VdbLevelSetSDF::levelsetFromMesh(
    const std::vector<Vec3s>& points,
    const std::vector<Vec3I>& triangles,
    const std::vector<Vec4I>& quads)
{
    // int pointsPerVoxel = 8;
    // openvdb::points::PointAttributeVector<openvdb::Vec3s> positionsWrapper(points);
    // float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);   
    // std::cout << "voxel size " << voxelSize << std::endl;
    float voxelSize = 5e-3;
    openvdb::math::Transform::Ptr xForm = openvdb::math::Transform::createLinearTransform(voxelSize);
    grid = openvdb::tools::meshToSignedDistanceField<openvdb::DoubleGrid>(
                *xForm, points, triangles, quads,
                10, 100);
    grid_grad = openvdb::tools::gradient(*grid);
    
}

void IMLS::gradient(const TV& test_point, TV& dphidx)
{

}

void IMLS::hessian(const TV& test_point, TM& d2phidx2)
{

}

void IMLS::initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon)
{

}        

T VdbLevelSetSDF::value(const TV& test_point) 
{
    openvdb::tools::GridSampler<DoubleGrid , openvdb::tools::BoxSampler> interpolator(*grid);
    openvdb::math::Vec3<T> P(test_point(0), test_point(1), test_point(2));
    float phi = interpolator.wsSample(test_point.data()); //ws denotes world space
    return (T)phi;
}

void VdbLevelSetSDF::gradient(const TV& test_point, TV& dphidx) 
{
    openvdb::tools::GridSampler<Vec3dGrid , openvdb::tools::BoxSampler> interpolator(*grid_grad);
    Vec3dGrid::ValueType grad = interpolator.wsSample(test_point.data());
    dphidx = TV(grad[0], grad[1], grad[2]);
}

void VdbLevelSetSDF::hessian(const TV& test_point, TM& d2phidx2) 
{
    TM fd = TM::Zero();
    const double h = 1e-2;
    for (int i = 0; i < 3; ++i) {
        TV dy = TV::Zero();
        dy[i] = h;

        TV gp;
        gradient(test_point + dy, gp);
        TV gm;
        gradient(test_point - dy, gm);
        fd.col(i) = (gp-gm) / (2*h);
    }
    d2phidx2 = 0.5*(fd + fd.transpose());

    // openvdb::tools::GridSampler<Vec3dGrid , openvdb::tools::BoxSampler> interpolator_ddx(*ddx);
    // openvdb::tools::GridSampler<Vec3dGrid , openvdb::tools::BoxSampler> interpolator_ddy(*ddy);
    // openvdb::tools::GridSampler<Vec3dGrid , openvdb::tools::BoxSampler> interpolator_ddz(*ddz);
    // Vec3dGrid::ValueType ddx = interpolator_ddx.wsSample(test_point.data());
    // Vec3dGrid::ValueType ddy = interpolator_ddy.wsSample(test_point.data());
    // Vec3dGrid::ValueType ddz = interpolator_ddz.wsSample(test_point.data());
    
    // d2phidx2 << ddx[0], ddy[0], ddz[0],
    //         ddx[1], ddy[1], ddz[1],
    //         ddx[2], ddy[2], ddz[2];
}