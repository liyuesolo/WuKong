#include <igl/readOBJ.h>

#include "../include/SDF.h"

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

bool use_spacial_hash = true;

void IMLS::computeBBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < n_points; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], data_points[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], data_points[i * 3 + d]);
        }
    }
}

void IMLS::initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon)
{
    n_points = vertices.rows() / 3;
    data_points = vertices;
    data_point_normals = normals;
    // T h = 0.1;
    T h = ref_dis;
    radii = VectorXT::Ones(n_points) * h;
    data_points += normals * epsilon;
    hash.build(h * 2.0, data_points); 
    computeBBox();
}   

void IMLS::initializeFromMeshFile(const std::string& filename)
{

}

T IMLS::value(const TV& test_point)
{
    T value = 0.0;
    T weights_sum = 0.0;
    std::vector<int> neighbors;
    if (use_spacial_hash)
        hash.getOneRingNeighbors(test_point, neighbors);
    else 
        for (int i = 0; i < n_points; i++) 
            neighbors.push_back(i);
    // std::cout << "#neighbors" << neighbors.size() << std::endl;
    // std::getchar();
    for (int i : neighbors)
    {
        TV pi = data_points.segment<3>(i * 3);
        TV ni = data_point_normals.segment<3>(i * 3);
        T wi = weightFunction((test_point - pi).norm(), radii[i]);
        
        weights_sum += wi;
        T dot_term = (test_point - pi).dot(ni);
        value += wi * dot_term;
    }
    if (weights_sum >= 1e-6)
        value /= weights_sum;
    else
        value = -10; // let's assume it's inside in this case, since those are outside will be penalized.
    return value;
}

void IMLS::gradient(const TV& test_point, TV& dphidx)
{
    dphidx = TV::Zero();
    T theta_sum = 0.0;
    TV sum_df = TV::Zero();
    T sum_dg = 0;
    TV sum_dtheta = TV::Zero();
    std::vector<int> neighbors;
    if (use_spacial_hash)
        hash.getOneRingNeighbors(test_point, neighbors);
    else 
        for (int i = 0; i < n_points; i++) 
            neighbors.push_back(i);

    for (int i : neighbors)
    {
        TV pi = data_points.segment<3>(i * 3);
        TV ni = data_point_normals.segment<3>(i * 3);
        TV dtheta_dx;
        thetaValueGradient(test_point, pi, radii[i], dtheta_dx);
        T wi = weightFunction((test_point - pi).norm(), radii[i]);
        theta_sum += wi;

        T dot_term = (test_point - pi).dot(ni);
        
        sum_df += (dtheta_dx * dot_term + wi * ni);
        sum_dg += wi * dot_term;
        sum_dtheta += dtheta_dx;
    }
    dphidx = (theta_sum * sum_df - sum_dg * sum_dtheta) / theta_sum / theta_sum; 
}

void IMLS::hessian(const TV& test_point, TM& d2phidx2)
{
    d2phidx2 = TM::Zero();
    TV sum_dtheta_dx = TV::Zero();
    TM sum_d2theta_dx2 = TM::Zero();
    T sum_theta = 0.0;

    TV sum_a = TV::Zero();
    TM sum_b = TM::Zero();
    TV sum_c = TV::Zero();
    T sum_d = 0.0;

    // for (int i = 0; i < n_points; i++)
    std::vector<int> neighbors;
    if (use_spacial_hash)
        hash.getOneRingNeighbors(test_point, neighbors);
    else 
        for (int i = 0; i < n_points; i++) 
            neighbors.push_back(i);

    for (int i : neighbors)
    {
        TV pi = data_points.segment<3>(i * 3);
        TV ni = data_point_normals.segment<3>(i * 3);
        TV dtheta_dx;
        thetaValueGradient(test_point, pi, radii[i], dtheta_dx);
        TM d2theta_dx2;
        thetaValueHessian(test_point, pi, radii[i], d2theta_dx2);
        sum_dtheta_dx += dtheta_dx;
        sum_d2theta_dx2 += d2theta_dx2;

        T wi = weightFunction((test_point - pi).norm(), radii[i]);
        sum_theta += wi;

        T dot_term = (test_point - pi).dot(ni);
        TV niwi = ni * wi;
        sum_a += dtheta_dx * dot_term + niwi;
        sum_b += d2theta_dx2 * dot_term + dtheta_dx * ni.transpose() + ni * dtheta_dx.transpose();
        sum_c += dtheta_dx * dot_term + niwi;
        sum_d += wi * dot_term;
    }

    d2phidx2 += (sum_theta * sum_b - sum_dtheta_dx * sum_a.transpose()) / sum_theta / sum_theta;
    d2phidx2 -= (sum_theta * sum_theta * (sum_c * sum_dtheta_dx.transpose() + sum_d * sum_d2theta_dx2) - 
        2.0 * sum_d * sum_theta * sum_dtheta_dx * sum_dtheta_dx.transpose()) / std::pow(sum_theta, 4);
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

void IMLS::thetaValue(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, double& energy){
	double _i_var[14];
	_i_var[0] = (x(1,0))-(pi(1,0));
	_i_var[1] = (x(0,0))-(pi(0,0));
	_i_var[2] = (x(2,0))-(pi(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = -(_i_var[8]);
	_i_var[10] = (_i_var[9])*(_i_var[8]);
	_i_var[11] = (_i_var[10])/(ri);
	_i_var[12] = (_i_var[11])/(ri);
	_i_var[13] = std::exp(_i_var[12]);
	energy = _i_var[13];
}

void IMLS::thetaValueGradient(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 1>& energygradient){
	double _i_var[33];
	_i_var[0] = (x(1,0))-(pi(1,0));
	_i_var[1] = (x(0,0))-(pi(0,0));
	_i_var[2] = (x(2,0))-(pi(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = -(_i_var[8]);
	_i_var[10] = (_i_var[9])*(_i_var[8]);
	_i_var[11] = (_i_var[10])/(ri);
	_i_var[12] = 1;
	_i_var[13] = (_i_var[11])/(ri);
	_i_var[14] = (_i_var[12])/(ri);
	_i_var[15] = std::exp(_i_var[13]);
	_i_var[16] = (_i_var[15])*(_i_var[14]);
	_i_var[17] = (_i_var[16])*(_i_var[14]);
	_i_var[18] = 2;
	_i_var[19] = -1;
	_i_var[20] = (_i_var[17])*(_i_var[8]);
	_i_var[21] = (_i_var[18])*(_i_var[8]);
	_i_var[22] = (_i_var[20])*(_i_var[19]);
	_i_var[23] = (_i_var[17])*(_i_var[9]);
	_i_var[24] = (_i_var[12])/(_i_var[21]);
	_i_var[25] = (_i_var[23])+(_i_var[22]);
	_i_var[26] = (_i_var[25])*(_i_var[24]);
	_i_var[27] = (_i_var[26])*(_i_var[1]);
	_i_var[28] = (_i_var[26])*(_i_var[0]);
	_i_var[29] = (_i_var[26])*(_i_var[2]);
	_i_var[30] = (_i_var[18])*(_i_var[27]);
	_i_var[31] = (_i_var[18])*(_i_var[28]);
	_i_var[32] = (_i_var[18])*(_i_var[29]);
	energygradient(0,0) = _i_var[30];
	energygradient(1,0) = _i_var[31];
	energygradient(2,0) = _i_var[32];
}

void IMLS::thetaValueHessian(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 3>& energyhessian){
	double _i_var[68];
	_i_var[0] = (x(1,0))-(pi(1,0));
	_i_var[1] = (x(0,0))-(pi(0,0));
	_i_var[2] = (x(2,0))-(pi(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = -(_i_var[8]);
	_i_var[10] = (_i_var[9])*(_i_var[8]);
	_i_var[11] = (_i_var[10])/(ri);
	_i_var[12] = 1;
	_i_var[13] = (_i_var[11])/(ri);
	_i_var[14] = (_i_var[12])/(ri);
	_i_var[15] = std::exp(_i_var[13]);
	_i_var[16] = (_i_var[14])*(_i_var[14]);
	_i_var[17] = 2;
	_i_var[18] = (_i_var[16])*(_i_var[15]);
	_i_var[19] = (_i_var[17])*(_i_var[8]);
	_i_var[20] = (_i_var[15])*(_i_var[14]);
	_i_var[21] = (_i_var[16])*(_i_var[18]);
	_i_var[22] = (_i_var[8])*(_i_var[9]);
	_i_var[23] = (_i_var[19])*(_i_var[19]);
	_i_var[24] = (_i_var[20])*(_i_var[14]);
	_i_var[25] = (_i_var[22])*(_i_var[21]);
	_i_var[26] = (_i_var[12])/(_i_var[23]);
	_i_var[27] = (_i_var[25])+(_i_var[24]);
	_i_var[28] = -2;
	_i_var[29] = (_i_var[9])*(_i_var[9]);
	_i_var[30] = -(_i_var[26]);
	_i_var[31] = -1;
	_i_var[32] = (_i_var[24])*(_i_var[8]);
	_i_var[33] = (_i_var[8])*(_i_var[8]);
	_i_var[34] = (_i_var[28])*(_i_var[27]);
	_i_var[35] = (_i_var[29])*(_i_var[21]);
	_i_var[36] = (_i_var[12])/(_i_var[19]);
	_i_var[37] = (_i_var[30])*(_i_var[17]);
	_i_var[38] = (_i_var[32])*(_i_var[31]);
	_i_var[39] = (_i_var[24])*(_i_var[9]);
	_i_var[40] = (_i_var[33])*(_i_var[21]);
	_i_var[41] = (_i_var[35])+(_i_var[34]);
	_i_var[42] = (_i_var[37])*(_i_var[36]);
	_i_var[43] = (_i_var[39])+(_i_var[38]);
	_i_var[44] = (_i_var[41])+(_i_var[40]);
	_i_var[45] = (_i_var[36])*(_i_var[36]);
	_i_var[46] = (_i_var[43])*(_i_var[42]);
	_i_var[47] = (_i_var[45])*(_i_var[44]);
	_i_var[48] = (_i_var[17])*(_i_var[1]);
	_i_var[49] = (_i_var[17])*(_i_var[0]);
	_i_var[50] = (_i_var[17])*(_i_var[2]);
	_i_var[51] = (_i_var[43])*(_i_var[36]);
	_i_var[52] = (_i_var[47])+(_i_var[46]);
	_i_var[53] = (_i_var[48])*(_i_var[48]);
	_i_var[54] = (_i_var[49])*(_i_var[49]);
	_i_var[55] = (_i_var[50])*(_i_var[50]);
	_i_var[56] = (_i_var[51])*(_i_var[17]);
	_i_var[57] = (_i_var[53])*(_i_var[52]);
	_i_var[58] = (_i_var[48])*(_i_var[52]);
	_i_var[59] = (_i_var[50])*(_i_var[52]);
	_i_var[60] = (_i_var[54])*(_i_var[52]);
	_i_var[61] = (_i_var[55])*(_i_var[52]);
	_i_var[62] = (_i_var[57])+(_i_var[56]);
	_i_var[63] = (_i_var[49])*(_i_var[58]);
	_i_var[64] = (_i_var[48])*(_i_var[59]);
	_i_var[65] = (_i_var[60])+(_i_var[56]);
	_i_var[66] = (_i_var[49])*(_i_var[59]);
	_i_var[67] = (_i_var[61])+(_i_var[56]);
	energyhessian(0,0) = _i_var[62];
	energyhessian(1,0) = _i_var[63];
	energyhessian(2,0) = _i_var[64];
	energyhessian(0,1) = _i_var[63];
	energyhessian(1,1) = _i_var[65];
	energyhessian(2,1) = _i_var[66];
	energyhessian(0,2) = _i_var[64];
	energyhessian(1,2) = _i_var[66];
	energyhessian(2,2) = _i_var[67];
}