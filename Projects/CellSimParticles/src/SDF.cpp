#include "../include/SDF.h"
#include<random>
#include<cmath>
#include<chrono>

template <int dim>
void IMLS<dim>::sampleSphere(const TV& center, T radius, 
    int n_samples, std::vector<TV>& samples)
{
    if constexpr (dim == 3)
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator (seed);
        std::uniform_real_distribution<T> uniform01(0.0, 1.0);
        samples.resize(n_samples);
        tbb::parallel_for(0, n_samples, [&](int i){
            T r = radius * uniform01(generator);
            T theta = 2 * M_PI * uniform01(generator);
            T phi = std::acos(1 - 2 * uniform01(generator));
            T x = r * std::sin(phi) * std::cos(theta);
            T y = r * std::sin(phi) * std::sin(theta);
            T z = r * std::cos(phi);
            samples[i] = center + TV(x, y, z);
        });
    }
}

template <int dim>
void IMLS<dim>::computeBBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < n_points; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            max_corner[d] = std::max(max_corner[d], data_points[i * dim + d]);
            min_corner[d] = std::min(min_corner[d], data_points[i * dim + d]);
        }
    }
}

template <int dim>
void IMLS<dim>::initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon)
{
    if constexpr (dim == 2)
        return;
    n_points = vertices.rows() / 3;
    data_points = vertices;
    data_point_normals = normals;
    // T h = 0.1;
    T h = ref_dis;
    radii = VectorXT::Ones(n_points) * h;
    data_points += normals * epsilon;
    hash.build(2.0 * h, data_points); 
    computeBBox();
}   

template <int dim>
void IMLS<dim>::sampleZeroLevelset(VectorXT& points)
{
    if constexpr (dim == 2)
    {

    }
    else
    {
        int n_pt = data_points.rows() / 3;
        int n_samples_per_cell = 2000;
        std::vector<std::vector<TV>> on_surface_points(n_pt);
        tbb::parallel_for(0, n_pt, [&](int i)
        {
            TV pt = data_points.segment<dim>(i * dim);
            std::vector<TV> samples;
            sampleSphere(pt, 2.0 * hash.h, n_samples_per_cell, samples);
            samples.push_back(pt);
            std::vector<TV> valid;
            for (TV& sample : samples)
                if (std::abs(value(sample)) < 1e-4)
                    valid.push_back(sample);
            on_surface_points[i] = valid;
        });
        int n_valid_point = 0;
        for (auto& vec : on_surface_points)
            n_valid_point += vec.size();
        points.resize(n_valid_point * 3);
        int cnt = 0;
        for (auto& vec : on_surface_points)
        {
            for (TV& pt : vec)
            {
                points.segment<3>(cnt * 3) = pt;
                cnt ++;
            }
        }

    }
}

template <int dim>
T IMLS<dim>::value(const TV& test_point)
{
    T value = 0.0;
    if constexpr (dim == 2)
    {

    }
    else
    {
        T weights_sum = 0.0;
        std::vector<int> neighbors;
        hash.getOneRingNeighbors(test_point, neighbors);
        for (int i : neighbors)
        {
            TV pi = data_points.segment<3>(i * 3);
            TV ni = data_point_normals.segment<3>(i * 3);
            T wi = weightFunction((test_point - pi).norm(), radii[i]);
            
            weights_sum += wi;
            T dot_term = (test_point - pi).dot(ni);
            value += wi * dot_term;
        }
        // std::cout << "#neighbors" << neighbors.size() << std::endl;
        // std::cout << weights_sum << std::endl;
        // std::getchar();
        if (weights_sum >= 1e-6)
            value /= weights_sum;
        else
            value = -10; // let's assume it's inside in this case, since those are outside will be penalized.
    }
    return value;
}

template <int dim>
void IMLS<dim>::gradient(const TV& test_point, TV& dphidx)
{
    if constexpr (dim == 2)
    {

    }
    else
    {
        dphidx = TV::Zero();
        T theta_sum = 0.0;
        TV sum_df = TV::Zero();
        T sum_dg = 0;
        TV sum_dtheta = TV::Zero();
        std::vector<int> neighbors;
        hash.getOneRingNeighbors(test_point, neighbors);

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
        if (theta_sum > 1e-8)
            dphidx = (theta_sum * sum_df - sum_dg * sum_dtheta) / theta_sum / theta_sum; 
    }
}

template <int dim>
void IMLS<dim>::hessian(const TV& test_point, TM& d2phidx2)
{
    if constexpr (dim == 2)
    {

    }
    else
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
        hash.getOneRingNeighbors(test_point, neighbors);

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
        if (sum_theta > 1e-8)
        {
            d2phidx2 += (sum_theta * sum_b - sum_dtheta_dx * sum_a.transpose()) / sum_theta / sum_theta;
            d2phidx2 -= (sum_theta * sum_theta * (sum_c * sum_dtheta_dx.transpose() + sum_d * sum_d2theta_dx2) - 
                2.0 * sum_d * sum_theta * sum_dtheta_dx * sum_dtheta_dx.transpose()) / std::pow(sum_theta, 4);
        }
    }
}

template <int dim>
void IMLS<dim>::thetaValue(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, double& energy){
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

template <int dim>
void IMLS<dim>::thetaValueGradient(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 1>& energygradient){
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

template <int dim>
void IMLS<dim>::thetaValueHessian(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 3>& energyhessian){
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


template class IMLS<2>;
template class IMLS<3>;