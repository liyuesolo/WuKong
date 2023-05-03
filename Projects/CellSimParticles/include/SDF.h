#ifndef SDF_H
#define SDF_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"
#include "SpatialHash.h"

template <int dim>
class SDF
{
public:
    using TV = Vector<double, dim>;
    using TM = Matrix<double, dim, dim>;
    using IV = Vector<int, dim>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;

    
public:
    virtual T value(const TV& test_point) = 0;
    virtual void gradient(const TV& test_point, TV& dphidx) = 0;
    virtual void hessian(const TV& test_point, TM& d2phidx2) = 0;

    bool inside(const TV& test_point) { return value(test_point) <= 0.0; }

    virtual void sampleZeroLevelset(VectorXT& points) = 0; 
public:
    SDF<dim>() 
    {
        
    }
    ~SDF<dim>() {}
};

template <int dim>
class SphereSDF : public SDF<dim>
{
public:
    using TV = Vector<double, dim>;
    using TM = Matrix<double, dim, dim>;
    using IV = Vector<int, dim>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
public:
    TV center; 
    T radius;
public:
    T value(const TV& test_point)
    {
        return (test_point - center).norm() - radius;
    }
    void gradient(const TV& test_point, TV& dphidx)
    {
        if constexpr (dim == 2)
        {
            double _i_var[14];
            _i_var[0] = (test_point(1,0))-(center(1,0));
            _i_var[1] = (test_point(0,0))-(center(0,0));
            _i_var[2] = (_i_var[0])*(_i_var[0]);
            _i_var[3] = (_i_var[1])*(_i_var[1]);
            _i_var[4] = (_i_var[3])+(_i_var[2]);
            _i_var[5] = std::sqrt(_i_var[4]);
            _i_var[6] = 2;
            _i_var[7] = (_i_var[6])*(_i_var[5]);
            _i_var[8] = 1;
            _i_var[9] = (_i_var[8])/(_i_var[7]);
            _i_var[10] = (_i_var[9])*(_i_var[1]);
            _i_var[11] = (_i_var[9])*(_i_var[0]);
            _i_var[12] = (_i_var[6])*(_i_var[10]);
            _i_var[13] = (_i_var[6])*(_i_var[11]);
            dphidx[0] = _i_var[12];
            dphidx[1] = _i_var[13];
        }
    }
    void hessian(const TV& test_point, TM& d2phidx2)
    {
        if constexpr (dim == 2)
        {
            double _i_var[26];
            _i_var[0] = (test_point(1,0))-(center(1,0));
            _i_var[1] = (test_point(0,0))-(center(0,0));
            _i_var[2] = (_i_var[0])*(_i_var[0]);
            _i_var[3] = (_i_var[1])*(_i_var[1]);
            _i_var[4] = (_i_var[3])+(_i_var[2]);
            _i_var[5] = std::sqrt(_i_var[4]);
            _i_var[6] = 2;
            _i_var[7] = (_i_var[6])*(_i_var[5]);
            _i_var[8] = (_i_var[7])*(_i_var[7]);
            _i_var[9] = 1;
            _i_var[10] = (_i_var[9])/(_i_var[8]);
            _i_var[11] = -(_i_var[10]);
            _i_var[12] = (_i_var[9])/(_i_var[7]);
            _i_var[13] = (_i_var[11])*(_i_var[6]);
            _i_var[14] = (_i_var[6])*(_i_var[1]);
            _i_var[15] = (_i_var[6])*(_i_var[0]);
            _i_var[16] = (_i_var[13])*(_i_var[12]);
            _i_var[17] = (_i_var[14])*(_i_var[14]);
            _i_var[18] = (_i_var[15])*(_i_var[15]);
            _i_var[19] = (_i_var[12])*(_i_var[6]);
            _i_var[20] = (_i_var[17])*(_i_var[16]);
            _i_var[21] = (_i_var[14])*(_i_var[16]);
            _i_var[22] = (_i_var[18])*(_i_var[16]);
            _i_var[23] = (_i_var[20])+(_i_var[19]);
            _i_var[24] = (_i_var[15])*(_i_var[21]);
            _i_var[25] = (_i_var[22])+(_i_var[19]);
            d2phidx2(0,0) = _i_var[23];
            d2phidx2(1,0) = _i_var[24];
            d2phidx2(0,1) = _i_var[24];
            d2phidx2(1,1) = _i_var[25];
        }
    }

    void sampleZeroLevelset(VectorXT& points)
    {
        int n_samples = 100;
        T dtheta = 2.0 * M_PI / T(n_samples);
        points.resize(n_samples * dim);
        if constexpr (dim == 2)
        {
            for (int i = 0; i < n_samples; i++)
                points.segment<dim>(i * dim) = center + radius * TV(std::cos(T(i) * dtheta), std::sin(T(i) * dtheta));
        }
    }
public:
    SphereSDF<dim>(const TV& _center, T _radius) : center(_center), radius(_radius) 
    {
        
    }
    SphereSDF<dim>() : center(TV::Zero()), radius(1.0) 
    {
        
    }
    ~SphereSDF<dim>() {}
};

template <int dim>
class IMLS : public SDF<dim>
{

public:
    using TV = Vector<double, dim>;
    using TM = Matrix<double, dim, dim>;
    using IV = Vector<int, dim>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;

    int n_points = 0;

    VectorXT data_points;
    VectorXT data_point_normals;
    VectorXT radii;
    T search_radius;
    T ref_dis = 0.1;
    SpatialHash<dim> hash;
    TV min_corner, max_corner;

private:
    T weightFunction(T d, T r) { return std::exp(-d*d / r/r); }

    void thetaValue(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, double& energy);
    void thetaValueGradient(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 1>& energygradient);
    void thetaValueHessian(const Eigen::Matrix<double,3,1> & x, const Eigen::Matrix<double,3,1> & pi, double ri, Eigen::Matrix<double, 3, 3>& energyhessian);

    void computeBBox();
public:
    void setRefDis(T _ref_dis) { ref_dis = _ref_dis; }
    T value(const TV& test_point);
    void gradient(const TV& test_point, TV& dphidx);
    void hessian(const TV& test_point, TM& d2phidx2);

    std::string getName() const { return "IMLS"; }

    void initializedMeshData(const VectorXT& vertices, const VectorXi& indices,
        const VectorXT& normals, T epsilon);


    void sampleZeroLevelset(VectorXT& points);
    void sampleSphere(const TV& center, T radius, 
        int n_samples, std::vector<TV>& samples);
public:
    IMLS() {}
    ~IMLS() {}
};

#endif