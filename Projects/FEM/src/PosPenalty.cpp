#include "../include/FEMSolver.h"

#include <igl/writeOBJ.h>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;
 
double centerPenalty = 1e7;
double PBCPenalty = 1e7;
double rotationalPenalty = 1e7;

template <typename DerivedLocalGrad, typename DerivedGrad>
void local_gradient_to_global_gradient(
    const Eigen::MatrixBase<DerivedLocalGrad>& local_grad,
    const std::vector<long>& ids,
    int dim,
    Eigen::PlainObjectBase<DerivedGrad>& grad)
{
    //std::cout<<ids.size()<<" "<<local_grad.size()<<std::endl;
    for (int i = 0; i < ids.size(); i++) {
        grad.segment(dim * ids[i], dim) += local_grad.segment(dim * i, dim);
    }
    
}

template <typename Derived>
void local_hessian_to_global_triplets(
    const Eigen::MatrixBase<Derived>& local_hessian,
    const std::vector<long>& ids,
    int dim,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    for (int i = 0; i < ids.size(); i++) {
        for (int j = 0; j < ids.size(); j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    triplets.emplace_back(
                        dim * ids[i] + k, dim * ids[j] + l,
                        local_hessian(dim * i + k, dim * j + l));
                }
            }
        }
    }
}

//Compute the Center of One Object
double computePosPenaltyImpl_(Eigen::MatrixXd& x, double CoM2)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    return 0.5*centerPenalty*(CoM1-CoM2)*(CoM1-CoM2);
}

double computePosPenaltyUpperImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, Eigen::MatrixXd& x, double initial)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    double CoM2 = 0.0;
    double x_min = 0.0;
    double x_max = 0.0;

    for(int j=0; j<left.rows(); ++j)
    {
        x_min += left(j,0);
    }
    x_min /= left.rows();

    for(int j=0; j<right.rows(); ++j)
    {
        x_max += right(j,0);
    }
    x_max /= right.rows();

    CoM2 = WIDTH_1/2.0+(x_max-x_min)/WIDTH_1*initial;

    return 0.5*centerPenalty*(CoM1-CoM2)*(CoM1-CoM2);
}

Eigen::VectorXd computePosPenaltyGradImpl_(Eigen::MatrixXd& x, double CoM2)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    //std::cout<<"Center of Mass Compare: "<<CoM1-CoM2<<std::endl;

    Eigen::VectorXd dCoM1dX(x.rows()*2);
    dCoM1dX.setZero();
    for(int i=0; i<x.rows(); ++i)
    {
        dCoM1dX(2*i) = 1/double(x.rows());
    }
    return centerPenalty*(CoM1-CoM2)*dCoM1dX;
}

Eigen::VectorXd computePosPenaltyGradUpperImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, Eigen::MatrixXd& x, double initial)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    double CoM2 = 0.0;
    double x_min = 0.0;
    double x_max = 0.0;

    for(int j=0; j<left.rows(); ++j)
    {
        x_min += left(j,0);
    }
    x_min /= left.rows();

    for(int j=0; j<right.rows(); ++j)
    {
        x_max += right(j,0);
    }
    x_max /= right.rows();

    CoM2 = WIDTH_1/2.0+(x_max-x_min)/WIDTH_1*initial;

    Eigen::VectorXd dCoM1dX((left.rows()+right.rows()+x.rows())*2);
    dCoM1dX.setZero();
    for(int i=0; i<x.rows(); ++i)
    {
        dCoM1dX(2*(i+left.rows()+right.rows())) = 1/double(x.rows());
    }

    Eigen::VectorXd dCoM2dX((left.rows()+right.rows()+x.rows())*2);
    dCoM2dX.setZero();
    for(int i=0; i<left.rows(); ++i)
    {
        dCoM2dX(2*i) = -1/double(left.rows())*1.0/WIDTH_1*initial;
    }
    for(int i=0; i<left.rows(); ++i)
    {
        dCoM2dX(2*(i+left.rows())) = +1/double(right.rows())*1.0/WIDTH_1*initial;
    }

    return centerPenalty*(CoM1-CoM2)*dCoM1dX - centerPenalty*(CoM1-CoM2)*dCoM2dX;
}

Eigen::MatrixXd computePosPenaltyHessImpl_(Eigen::MatrixXd& x, double CoM2)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    Eigen::VectorXd dCoM1dX(x.rows()*2);
    dCoM1dX.setZero();
    for(int i=0; i<x.rows(); ++i)
    {
        dCoM1dX(2*i) = 1/double(x.rows());
    }

    return centerPenalty*dCoM1dX*dCoM1dX.transpose();
}

Eigen::MatrixXd computePosPenaltyHessUpperImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, Eigen::MatrixXd& x, double initial)
{   
    double CoM1 = 0.0;
    for(int i=0; i<x.rows(); ++i)
    {
        CoM1 += x(i,0);
    }
    CoM1 /= double(x.rows());

    double CoM2 = 0.0;
    double x_min = 0.0;
    double x_max = 0.0;

    for(int j=0; j<left.rows(); ++j)
    {
        x_min += left(j,0);
    }
    x_min /= left.rows();

    for(int j=0; j<right.rows(); ++j)
    {
        x_max += right(j,0);
    }
    x_max /= right.rows();

    CoM2 = WIDTH_1/2.0+(x_max-x_min)/WIDTH_1*initial;

    Eigen::VectorXd dCoM1dX((left.rows()+right.rows()+x.rows())*2);
    dCoM1dX.setZero();
    for(int i=0; i<x.rows(); ++i)
    {
        dCoM1dX(2*(i+left.rows()+right.rows())) = 1/double(x.rows());
    }

    Eigen::VectorXd dCoM2dX((left.rows()+right.rows()+x.rows())*2);
    dCoM2dX.setZero();
    for(int i=0; i<left.rows(); ++i)
    {
        dCoM2dX(2*i) = -1/double(left.rows())*1.0/WIDTH_1*initial;
    }
    for(int i=0; i<left.rows(); ++i)
    {
        dCoM2dX(2*(i+left.rows())) = +1/double(right.rows())*1.0/WIDTH_1*initial;
    }

    return centerPenalty*dCoM1dX*dCoM1dX.transpose() + centerPenalty*dCoM2dX*dCoM2dX.transpose() - centerPenalty*dCoM1dX*dCoM2dX.transpose() - centerPenalty*dCoM2dX*dCoM1dX.transpose();
}

double computePBCPenaltyImpl_(double x, double X)
{   
    double energy = 0.0;
    energy += (x-X)*(x-X);
    return 0.5*centerPenalty*energy;
}

Eigen::VectorXd computePBCPenaltyGradImpl_(double x, double X)
{   
    Eigen::VectorXd dEdx(4);
    dEdx.setZero();
    dEdx(1) = 2*(x-X);
    dEdx(3) = -2*(x-X);

    return 0.5*centerPenalty*dEdx;
}

Eigen::MatrixXd computePBCPenaltyHessImpl_(double x, double X)
{   
    Eigen::MatrixXd Hess(4,4);
    Hess.setZero();
    Hess(1,1) = 2;
    Hess(1,3) = -2;
    Hess(3,1) = -2;
    Hess(3,3) = 2;
    
    return 0.5*centerPenalty*Hess;
}

double computeVirtualSpringImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, double k1, double k2)
{   
    double x_left = 0.0;
    for(int i=0; i<left.rows(); ++i)
    {
        x_left += left(i,0);
    }
    x_left /= left.rows();

    double x_right = 0.0;
    for(int i=0; i<right.rows(); ++i)
    {
        x_right += right(i,0);
    }
    x_right /= right.rows();
    x_right = WIDTH_1 - x_right;

    //std::cout<<"x_left: "<<x_left<<" x_right: "<<x_right<<std::endl;

    if(x_left >= 0 && x_right>=0)
        return 0.5*k1*x_left*x_left + 0.5*k2*x_right*x_right;
    else
        return 1e8;
}

double computeVirtualSpringImpl_2(Eigen::MatrixXd& all, double k1, double k2)
{   
    double x_left = 0.0, x_right = 0.0;
    double center = 0.0;
    for(int i=0; i<all.rows(); ++i)
    {
        center += all(i,0);
    }
    center /= all.rows();
    x_left = center;
    x_right = WIDTH_1-center;

    //std::cout<<"x_left: "<<x_left<<" x_right: "<<x_right<<std::endl;

    if(x_left >= 0 && x_right>=0)
        return 0.5*k1*x_left*x_left + 0.5*k2*x_right*x_right;
    else
        return 1e8;
}

Eigen::VectorXd computeVirtualSpringGradImpl_2(Eigen::MatrixXd& all, double k1, double k2)
{   
    double x_left = 0.0, x_right = 0.0;
    double center = 0.0;
    Eigen::VectorXd zeros((all.rows())*2);
    zeros.setZero();
    for(int i=0; i<all.rows(); ++i)
    {
        center += all(i,0);
    }
    center /= all.rows();
    x_left = center;
    x_right = WIDTH_1-center;

    Eigen::VectorXd dx_leftdx(all.rows()*2);
    Eigen::VectorXd dx_rightdx(all.rows()*2);
    dx_leftdx.setZero();
    dx_rightdx.setZero();
    for(int i=0; i<all.rows(); ++i)
    {
        dx_leftdx(2*i) = 1.0/double(all.rows());
        dx_rightdx(2*i) = -1.0/double(all.rows());
    }
    // std::cout<<"-------------------------------"<<std::endl;

    // std::cout<<dx_leftdx.transpose()<<std::endl;
    // std::cout<<dx_rightdx.transpose()<<std::endl;

    // std::cout<<"-------------------------------"<<std::endl;
    if(x_left >= 0 && x_right>=0)
        return k1*x_left*dx_leftdx + k2*x_right*dx_rightdx;
    else
        return zeros;
}


Eigen::VectorXd computeVirtualSpringGradImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, double k1, double k2)
{   
    double x_left = 0.0;
    Eigen::VectorXd zeros((left.rows()+ right.rows())*2);
    zeros.setZero();
    for(int i=0; i<left.rows(); ++i)
    {
        x_left += left(i,0);
    }
    x_left /= left.rows();

    double x_right = 0.0;
    for(int i=0; i<right.rows(); ++i)
    {
        x_right += right(i,0);
    }
    x_right /= right.rows();
    x_right = WIDTH_1 - x_right;

    Eigen::VectorXd dx_leftdx((left.rows()+ right.rows())*2);
    Eigen::VectorXd dx_rightdx((left.rows()+ right.rows())*2);
    dx_leftdx.setZero();
    dx_rightdx.setZero();
    for(int i=0; i<left.rows(); ++i)
    {
        dx_leftdx(2*i) = 1.0/double(left.rows());
    }
    for(int i=0; i<right.rows(); ++i)
    {
        dx_rightdx(2*(i+left.rows())) = -1.0/double(right.rows());
    }
    // std::cout<<"-------------------------------"<<std::endl;

    // std::cout<<dx_leftdx.transpose()<<std::endl;
    // std::cout<<dx_rightdx.transpose()<<std::endl;

    // std::cout<<"-------------------------------"<<std::endl;
    if(x_left >= 0 && x_right>=0)
        return k1*x_left*dx_leftdx + k2*x_right*dx_rightdx;
    else
        return zeros;
}

Eigen::MatrixXd computeVirtualSpringHessImpl_(Eigen::MatrixXd& left, Eigen::MatrixXd& right, double k1, double k2)
{   
    Eigen::MatrixXd zeros((left.rows()+ right.rows())*2,(left.rows()+ right.rows())*2);
    zeros.setZero();
    double x_left = 0.0;
    for(int i=0; i<left.rows(); ++i)
    {
        x_left += left(i,0);
    }
    x_left /= left.rows();

    double x_right = 0.0;
    for(int i=0; i<right.rows(); ++i)
    {
        x_right += right(i,0);
    }
    x_right /= right.rows();
    x_right = WIDTH_1 - x_right;

    Eigen::VectorXd dx_leftdx((left.rows()+ right.rows())*2);
    Eigen::VectorXd dx_rightdx((left.rows()+ right.rows())*2);
    dx_leftdx.setZero();
    dx_rightdx.setZero();
    for(int i=0; i<left.rows(); ++i)
    {
        dx_leftdx(2*i) = 1.0/double(left.rows());
    }
    for(int i=0; i<right.rows(); ++i)
    {
        dx_rightdx(2*(i+left.rows())) = 1.0/double(right.rows());
    }

    if(x_left >= 0 && x_right>=0)
        return k1*dx_leftdx*dx_leftdx.transpose() + k2*dx_rightdx*dx_rightdx.transpose();
    else
        return zeros;
}

Eigen::MatrixXd computeVirtualSpringHessImpl_2(Eigen::MatrixXd& all, double k1, double k2)
{   
    Eigen::MatrixXd zeros(all.rows()*2,all.rows()*2);
    zeros.setZero();
    double x_left = 0.0, x_right = 0.0;
    double center = 0.0;
    for(int i=0; i<all.rows(); ++i)
    {
        center += all(i,0);
    }
    center /= all.rows();
    x_left = center;
    x_right = WIDTH_1-center;

    Eigen::VectorXd dx_leftdx(all.rows()*2);
    Eigen::VectorXd dx_rightdx(all.rows()*2);
    dx_leftdx.setZero();
    dx_rightdx.setZero();
    for(int i=0; i<all.rows(); ++i)
    {
        dx_leftdx(2*i) = 1.0/double(all.rows());
        dx_rightdx(2*i) = -1.0/double(all.rows());
    }

    if(x_left >= 0 && x_right>=0)
        return k1*dx_leftdx*dx_leftdx.transpose() + k2*dx_rightdx*dx_rightdx.transpose();
    else
        return zeros;
}

var computeRotationalEnergy(const ArrayXvar& xs, const ArrayXvar& Xs)
{
    int num_pts = xs.size()/2;

    VectorXvar cx(2), cX(2);
    cx<<0,0;
    cx<<0,0;
    for(int i=0; i<num_pts; ++i)
    {
        cx(0) += xs(2*i);
        cx(1) += xs(2*i+1);
        cX(0) += Xs(2*i);
        cX(1) += Xs(2*i+1);
    }
    cx/=num_pts;
    cX/=num_pts;

    var result = 0;
    for(int i=0; i<num_pts; ++i)
    {
        var x1 = xs(2*i)-cx(0); var y1 = xs(2*i+1)-cx(1);
        var x2 = Xs(2*i)-cX(0); var y2 = Xs(2*i+1)-cX(1);
        result += (x1*y2-x2*y1);
    }
    return 0.5*result*result;
}

double computeRotationalEnergyImpl_(Eigen::MatrixXd& x,Eigen::MatrixXd& X)
{
    assert(x.rows() == X.rows());
    int num_pts = x.rows();

    VectorXvar xs(2*num_pts), Xs(2*num_pts);
    for(int i=0; i<num_pts; ++i)
    {
        xs(2*i) = x(i,0);
        xs(2*i+1) = x(i,1);
        Xs(2*i) = X(i,0);
        Xs(2*i+1) = X(i,1);
    }
    var u = computeRotationalEnergy(xs, Xs);
    return rotationalPenalty*double(u);
}

Eigen::VectorXd computeRotationalGradImpl_(Eigen::MatrixXd& x,Eigen::MatrixXd& X)
{
    assert(x.rows() == X.rows());
    int num_pts = x.rows();

    VectorXvar xs(2*num_pts), Xs(2*num_pts);
    for(int i=0; i<num_pts; ++i)
    {
        xs(2*i) = x(i,0);
        xs(2*i+1) = x(i,1);
        Xs(2*i) = X(i,0);
        Xs(2*i+1) = X(i,1);
    }
    Eigen::VectorXd grad;
    var u = computeRotationalEnergy(xs, Xs);
    hessian(u, xs, grad);
    return rotationalPenalty*grad;
}

Eigen::MatrixXd computeRotationalHessImpl_(Eigen::MatrixXd& x,Eigen::MatrixXd& X)
{
    assert(x.rows() == X.rows());
    int num_pts = x.rows();

    VectorXvar xs(2*num_pts), Xs(2*num_pts);
    for(int i=0; i<num_pts; ++i)
    {
        xs(2*i) = x(i,0);
        xs(2*i+1) = x(i,1);
        Xs(2*i) = X(i,0);
        Xs(2*i+1) = X(i,1);
    }
    Eigen::VectorXd grad;
    Eigen::MatrixXd hess;
    var u = computeRotationalEnergy(xs, Xs);
    hess = hessian(u, xs, grad);
    return rotationalPenalty*hess;
}

// Compute the Penalty Energy
template <int dim>
void FEMSolver<dim>::addPosPenaltyEnergy(T& energy)
{
    int size = 1;
    if(!use_virtual_spring) size = 2;

    // for(int i=0; i<Object_indices[0].size(); ++i)
    // {
    //     double x_cur = deformed(Object_indices[0](i) * 2);
    //     x_min = std::min(x_cur, x_min);
    //     x_max = std::max(x_cur, x_max);
    // }

    tbb::enumerable_thread_specific<double> storage(0);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), size),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                Eigen::MatrixXd x;
                if(i == 0)
                {
                    x.resize((left_boundaries_master.size()+right_boundaries_master.size()),2);
                    for(int j=0; j<left_boundaries_master.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                    }
                    for(int j=0; j<right_boundaries_master.size(); ++j)
                    {
                        x.row(j+left_boundaries_master.size()) = deformed.segment<2>(right_boundaries_master[j] * 2);
                    }
                }
                else
                {
                    x.resize((left_boundaries.size()+right_boundaries.size()),2);
                    for(int j=0; j<left_boundaries.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
                    }
                    for(int j=0; j<right_boundaries.size(); ++j)
                    {
                        x.row(j+left_boundaries.size()) = deformed.segment<2>(right_boundaries[j] * 2);
                    }

                    // x.resize(Object_indices[1].size(),2);
                    // for(int j=0; j<Object_indices[1].size(); ++j)
                    // {
                    //     x.row(j) = deformed.segment<2>(2*Object_indices[1](j));
                    // }
                
                }
                
                Eigen::MatrixXd left(left_boundaries_master.size(),2);
                Eigen::MatrixXd right(right_boundaries_master.size(),2);

                //std::cout<<deformed.size()<<" "<<x.rows()<<" "<<Object_indices[i].size()<<std::endl;
                if(i == 1)
                {
                    for(int j=0; j<left_boundaries_master.size(); ++j)
                    {
                        left.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                    }
                    for(int j=0; j<right_boundaries_master.size(); ++j)
                    {
                        right.row(j) = deformed.segment<2>(right_boundaries_master[j] * 2);
                    }
                }

                if(TEST)
                    local_potential += computePosPenaltyImpl_(x,6.0);
                else
                {
                    if(i == 0)
                    {
                        //local_potential += computePosPenaltyImpl_(x,WIDTH_1/2.0);
                    }
                    else
                    {
                        //local_potential += computePosPenaltyUpperImpl_(left,right,x,DISPLAYSMENT+WIDTH_2/2.0-WIDTH_1/2.0);
                        local_potential += computePosPenaltyImpl_(x,DISPLAYSMENT+WIDTH_2/2.0);
                    }
                }
                
            }
        });

    double potential = 0;
    for (const auto& local_potential : storage) {
        potential += local_potential;
    }
    energy += potential;
}

template <int dim>
void FEMSolver<dim>::addPosPenaltyForceEntries(VectorXT& residual)
{
    int size = 1;
    if(!use_virtual_spring) size = 2;
    // for(int i=0; i<Object_indices[0].size(); ++i)
    // {
    //     double x_cur = deformed(Object_indices[0](i) * 2);
    //     x_min = std::min(x_cur, x_min);
    //     x_max = std::max(x_cur, x_max);
    // }

    tbb::enumerable_thread_specific<Eigen::VectorXd> storage(
        Eigen::VectorXd::Zero(num_nodes*dim));
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), size),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_grad = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                Eigen::MatrixXd x;
                std::vector<long> vertices;
                if(i == 0)
                {
                    x.resize((left_boundaries_master.size()+right_boundaries_master.size()),2);
                    for(int j=0; j<left_boundaries_master.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                        vertices.push_back(left_boundaries_master[j]);
                    }
                    for(int j=0; j<right_boundaries_master.size(); ++j)
                    {
                        x.row(j+left_boundaries_master.size()) = deformed.segment<2>(right_boundaries_master[j] * 2);
                        vertices.push_back(right_boundaries_master[j]);
                    }
                }
                else
                {
                    x.resize((left_boundaries.size()+right_boundaries.size()),2);
                    for(int j=0; j<left_boundaries.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
                        vertices.push_back(left_boundaries[j]);
                    }
                    for(int j=0; j<right_boundaries.size(); ++j)
                    {
                        x.row(j+left_boundaries.size()) = deformed.segment<2>(right_boundaries[j] * 2);
                        vertices.push_back(right_boundaries[j]);
                    }
                    // x.resize(Object_indices[1].size(),2);
                    // for(int j=0; j<Object_indices[1].size(); ++j)
                    // {
                    //     x.row(j) = deformed.segment<2>(2*Object_indices[1](j));
                    //     vertices.push_back(Object_indices[1](j));
                    // }
                }

                Eigen::MatrixXd left(left_boundaries_master.size(),2);
                Eigen::MatrixXd right(left_boundaries_master.size(),2);

                //std::cout<<deformed.size()<<" "<<x.rows()<<" "<<Object_indices[i].size()<<std::endl;
                // if(!TEST && i == 1)
                // {
                //     for(int j=0; j<left_boundaries_master.size(); ++j)
                //     {
                //         left.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                //         vertices.push_back(left_boundaries_master[j]);
                //     }
                //     for(int j=0; j<right_boundaries_master.size(); ++j)
                //     {
                //         right.row(j) = deformed.segment<2>(right_boundaries_master[j] * 2);
                //         vertices.push_back(right_boundaries_master[j]);
                //     }
                // }
                if(TEST)
                    local_gradient_to_global_gradient(
                        computePosPenaltyGradImpl_(x, 6.0),
                        vertices, dim, local_grad);
                else
                {
                    if(i == 0)
                    {
                        // local_gradient_to_global_gradient(
                        // computePosPenaltyGradImpl_(x, WIDTH_1/2.0),
                        // vertices, dim, local_grad);
                    }
                        
                    else
                    {
                        // local_gradient_to_global_gradient(
                        // computePosPenaltyGradUpperImpl_(left,right,x,DISPLAYSMENT+WIDTH_2/2.0-WIDTH_1/2.0),
                        // vertices, dim, local_grad);
                        local_gradient_to_global_gradient(
                        computePosPenaltyGradImpl_(x, DISPLAYSMENT+WIDTH_2/2.0),
                        vertices, dim, local_grad);
                    }
                }
                
            }
        });

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes*dim);
    for (const auto& local_grad : storage) {
        grad += local_grad;
    }

    //std::cout<<"Pos Pen force norm: "<<grad.norm()<<std::endl;
    
    residual.segment(0, num_nodes * dim) += -grad.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addPosPenaltyHessianEntries(std::vector<Entry>& entries,bool project_PD)
{

    int size = 1;
    if(!use_virtual_spring) size = 2;
    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>>
        storage;

    // for(int i=0; i<Object_indices[0].size(); ++i)
    // {
    //     double x_cur = deformed(Object_indices[0](i) * 2);
    //     x_min = std::min(x_cur, x_min);
    //     x_max = std::max(x_cur, x_max);
    // }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), size),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_hess_triplets = storage.local();

            for (size_t i = r.begin(); i < r.end(); i++) {
                Eigen::MatrixXd x;
                std::vector<long> vertices;
                if(i == 0)
                {
                    x.resize((left_boundaries_master.size()+right_boundaries_master.size()),2);
                    for(int j=0; j<left_boundaries_master.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                        vertices.push_back(left_boundaries_master[j]);
                    }
                    for(int j=0; j<right_boundaries_master.size(); ++j)
                    {
                        x.row(j+left_boundaries_master.size()) = deformed.segment<2>(right_boundaries_master[j] * 2);
                        vertices.push_back(right_boundaries_master[j]);
                    }
                }
                else
                {
                    x.resize((left_boundaries.size()+right_boundaries.size()),2);
                    for(int j=0; j<left_boundaries.size(); ++j)
                    {
                        x.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
                        vertices.push_back(left_boundaries[j]);
                    }
                    for(int j=0; j<right_boundaries.size(); ++j)
                    {
                        x.row(j+left_boundaries.size()) = deformed.segment<2>(right_boundaries[j] * 2);
                        vertices.push_back(right_boundaries[j]);
                    }
                    // x.resize(Object_indices[1].size(),2);
                    // for(int j=0; j<Object_indices[1].size(); ++j)
                    // {
                    //     x.row(j) = deformed.segment<2>(2*Object_indices[1](j));
                    //     vertices.push_back(Object_indices[1](j));
                    // }
                }


                Eigen::MatrixXd left(left_boundaries_master.size(),2);
                Eigen::MatrixXd right(left_boundaries_master.size(),2);

                //std::cout<<deformed.size()<<" "<<x.rows()<<" "<<Object_indices[i].size()<<std::endl;
                // if(!TEST && i == 1)
                // {
                //     for(int j=0; j<left_boundaries_master.size(); ++j)
                //     {
                //         left.row(j) = deformed.segment<2>(left_boundaries_master[j] * 2);
                //         vertices.push_back(left_boundaries_master[j]);
                //     }
                //     for(int j=0; j<right_boundaries_master.size(); ++j)
                //     {
                //         right.row(j) = deformed.segment<2>(right_boundaries_master[j] * 2);
                //         vertices.push_back(right_boundaries_master[j]);
                //     }
                // }

                if(TEST)
                    local_hessian_to_global_triplets(
                    computePosPenaltyHessImpl_(x, 6.0),
                    vertices, dim, local_hess_triplets);
                else
                {
                    if(i == 0)
                    {
                        // local_hessian_to_global_triplets(
                        // computePosPenaltyHessImpl_(x, WIDTH_1/2.0),
                        // vertices, dim, local_hess_triplets);
                    }
                    else
                    {
                        // local_hessian_to_global_triplets(
                        // computePosPenaltyHessUpperImpl_(left,right,x,DISPLAYSMENT+WIDTH_2/2.0-WIDTH_1/2.0),
                        // vertices, dim, local_hess_triplets);
                        local_hessian_to_global_triplets(
                        computePosPenaltyHessImpl_(x, DISPLAYSMENT+WIDTH_2/2.0),
                        vertices, dim, local_hess_triplets);
                    }
                }
                

                
            }
        });
        
    Eigen::SparseMatrix<double> hess(num_nodes*dim, num_nodes*dim);
    for (const auto& local_hess_triplets : storage) {
        Eigen::SparseMatrix<double> local_hess(num_nodes*dim, num_nodes*dim);
        local_hess.setFromTriplets(
            local_hess_triplets.begin(), local_hess_triplets.end());
        hess += local_hess;
    }

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

// Compute the Penalty Energy
template <int dim>
void FEMSolver<dim>::addPBCPenaltyEnergy(T& energy)
{
    tbb::enumerable_thread_specific<double> storage(0);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), pbc_pairs.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                double x, X;
                x = deformed(pbc_pairs[i].first * 2+1);
                X = deformed(pbc_pairs[i].second * 2+1);
                local_potential += computePBCPenaltyImpl_(x,X);
            }
        });

    double potential = 0;
    for (const auto& local_potential : storage) {
        potential += local_potential;
    }
    potential /= num_nodes;
    energy += potential;
}

template <int dim>
void FEMSolver<dim>::addPBCPenaltyForceEntries(VectorXT& residual)
{
    tbb::enumerable_thread_specific<Eigen::VectorXd> storage(
        Eigen::VectorXd::Zero(num_nodes*dim));

    

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), pbc_pairs.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_grad = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                double x, X;
                x = deformed(pbc_pairs[i].first * 2+1);
                X = deformed(pbc_pairs[i].second * 2+1);
                std::vector<long> vertices;
                vertices.push_back(pbc_pairs[i].first);
                vertices.push_back(pbc_pairs[i].second);

                local_gradient_to_global_gradient(
                    computePBCPenaltyGradImpl_(x, X),
                    vertices, dim, local_grad);
            }
        });

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes*dim);
    for (const auto& local_grad : storage) {
        grad += local_grad;
    }
    grad /= num_nodes;
    residual.segment(0, num_nodes * dim) += -grad.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addPBCPenaltyHessianEntries(std::vector<Entry>& entries,bool project_PD)
{

    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>>
        storage;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), pbc_pairs.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_hess_triplets = storage.local();

            for (size_t i = r.begin(); i < r.end(); i++) {
                double x, X;
                x = deformed(pbc_pairs[i].first * 2+1);
                X = deformed(pbc_pairs[i].second * 2+1);
                std::vector<long> vertices;
                vertices.push_back(pbc_pairs[i].first);
                vertices.push_back(pbc_pairs[i].second);

                local_hessian_to_global_triplets(
                    computePBCPenaltyHessImpl_(x, X),
                    vertices, dim, local_hess_triplets);
            }
        });
        
    
    Eigen::SparseMatrix<double> hess(num_nodes*dim, num_nodes*dim);
    for (const auto& local_hess_triplets : storage) {
        Eigen::SparseMatrix<double> local_hess(num_nodes*dim, num_nodes*dim);
        local_hess.setFromTriplets(
            local_hess_triplets.begin(), local_hess_triplets.end());
        hess += local_hess;
    }
    hess /= num_nodes;
    std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringEnergy(T& energy)
{
    Eigen::MatrixXd left(left_boundaries.size(),2);
    Eigen::MatrixXd right(right_boundaries.size(),2);
    Eigen::MatrixXd all(Object_indices[1].rows(),2);
    //assert(left.rows() == right.rows());
    for(int j=0; j<left.rows(); ++j)
    {
        left.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
    }
    for(int j=0; j<right.rows(); ++j)
    {
        right.row(j) = deformed.segment<2>(right_boundaries[j] * 2);
    }
    for(int j=0; j<all.rows(); ++j)
    {
        all.row(j) = deformed.segment<2>(Object_indices[1](j) * 2);
    }
    double potential = computeVirtualSpringImpl_2(all, k1, k2);
    energy += potential;
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringForceEntries(VectorXT& residual)
{
    std::vector<long> vertices;
    Eigen::MatrixXd left(left_boundaries.size(),2);
    Eigen::MatrixXd right(right_boundaries.size(),2);
    Eigen::MatrixXd all(Object_indices[1].rows(),2);
    std::cout<<all.rows()<<std::endl;
    //assert(left.rows() == right.rows());
    for(int j=0; j<left.rows(); ++j)
    {
        left.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
        //vertices.push_back(left_boundaries[j]);
    }
    for(int j=0; j<right.rows(); ++j)
    {
        right.row(j) = deformed.segment<2>(right_boundaries[j] * 2);
        //vertices.push_back(right_boundaries[j]);
    }
    for(int j=0; j<all.rows(); ++j)
    {
        all.row(j) = deformed.segment<2>(Object_indices[1](j) * 2);
        vertices.push_back(Object_indices[1](j));
    }
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes*dim);
    // std::cout<<"fuck"<<std::endl;
    // std::cout<<computeVirtualSpringGradImpl_(left, right).transpose()<<std::endl;
    local_gradient_to_global_gradient(computeVirtualSpringGradImpl_2(all, k1, k2), vertices, dim, grad);

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     std::cout<<i<<" "<<grad(2*i)<<" "<<grad(2*i+1)<<std::endl;
    // }
    residual.segment(0, num_nodes * dim) += -grad.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringHessianEntries(std::vector<Entry>& entries,bool project_PD)
{        
    std::vector<long> vertices;
    Eigen::MatrixXd left(left_boundaries.size(),2);
    Eigen::MatrixXd right(right_boundaries.size(),2);
    Eigen::MatrixXd all(Object_indices[1].rows(),2);
    //assert(left.rows() == right.rows());
    for(int j=0; j<left.rows(); ++j)
    {
        left.row(j) = deformed.segment<2>(left_boundaries[j] * 2);
        //vertices.push_back(left_boundaries[j]);
    }
    for(int j=0; j<right.rows(); ++j)
    {
        right.row(j) = deformed.segment<2>(right_boundaries[j] * 2);
        //vertices.push_back(right_boundaries[j]);
    }
    for(int j=0; j<all.rows(); ++j)
    {
        all.row(j) = deformed.segment<2>(Object_indices[1](j) * 2);
         vertices.push_back(Object_indices[1](j));
    }

    Eigen::SparseMatrix<double> hess(num_nodes*dim, num_nodes*dim);
    std::vector<Eigen::Triplet<double>> local_hess_triplets;

    local_hessian_to_global_triplets(computeVirtualSpringHessImpl_2(all, k1, k2), vertices, dim, local_hess_triplets);
    hess.setFromTriplets(local_hess_triplets.begin(), local_hess_triplets.end());

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void FEMSolver<dim>::addRotationalPenaltyEnergy(T& energy)
{
    Eigen::MatrixXd x(Object_indices_rot[0].size(),2);
    Eigen::MatrixXd X(Object_indices_rot[0].size(),2);
    for(int j=0; j<x.rows(); ++j)
    {
        x.row(j) = deformed.segment<2>(Object_indices_rot[0](j) * 2);
        X.row(j) = undeformed.segment<2>(Object_indices_rot[0](j) * 2);
    }
    double potential = computeRotationalEnergyImpl_(x,X);
    energy += potential;
}

template <int dim>
void FEMSolver<dim>::addRotationalPenaltyForceEntries(VectorXT& residual)
{
    std::vector<long> vertices;
    Eigen::MatrixXd x(Object_indices_rot[0].size(),2);
    Eigen::MatrixXd X(Object_indices_rot[0].size(),2);
    for(int j=0; j<x.rows(); ++j)
    {
        x.row(j) = deformed.segment<2>(Object_indices_rot[0](j) * 2);
        X.row(j) = undeformed.segment<2>(Object_indices_rot[0](j) * 2);
        vertices.push_back(Object_indices_rot[0](j));
    }
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes*dim);
    // std::cout<<"fuck"<<std::endl;
    // std::cout<<computeVirtualSpringGradImpl_(left, right).transpose()<<std::endl;
    local_gradient_to_global_gradient(computeRotationalGradImpl_(x,X), vertices, dim, grad);

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     std::cout<<i<<" "<<grad(2*i)<<" "<<grad(2*i+1)<<std::endl;
    // }
    residual.segment(0, num_nodes * dim) += -grad.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addRotationalPenaltyHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    std::vector<long> vertices;
    Eigen::MatrixXd x(Object_indices_rot[0].size(),2);
    Eigen::MatrixXd X(Object_indices_rot[0].size(),2);
    for(int j=0; j<x.rows(); ++j)
    {
        x.row(j) = deformed.segment<2>(Object_indices_rot[0](j) * 2);
        X.row(j) = undeformed.segment<2>(Object_indices_rot[0](j) * 2);
        vertices.push_back(Object_indices_rot[0](j));
    }

    Eigen::SparseMatrix<double> hess(num_nodes*dim, num_nodes*dim);
    std::vector<Eigen::Triplet<double>> local_hess_triplets;

    local_hessian_to_global_triplets(computeRotationalHessImpl_(x,X), vertices, dim, local_hess_triplets);
    hess.setFromTriplets(local_hess_triplets.begin(), local_hess_triplets.end());

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void FEMSolver<dim>::showLeftRight()
{
    // double x_left = 0.0, x_right = 0.0;
    // double center = 0.0;
    // double y_c = 0.0;
    // for(int i=0; i<Object_indices[1].rows(); ++i)
    // {
    //     center += deformed(Object_indices[1](i) * 2);
    //     y_c += deformed(Object_indices[1](i) * 2+1);
    // }
    // center /= Object_indices[1].rows();
    // y_c /= Object_indices[1].rows();
    // x_left = center;
    // x_right = WIDTH_1-center;

    // // std::ofstream file( "Mortar_4_10000_1e8.csv", std::ios::app ) ;
    // // file<<DISPLAYSMENT<<","<<k2<<","<<x_left<<","<<x_right<<std::endl;
    // std::cout<<DISPLAYSMENT<<","<<k2<<","<<x_left<<","<<x_right<<std::endl;
    // std::cout<<"Left/Right Ratio: "<<x_left/x_right<<std::endl;

    // // Calculate lateral forces:

    // double contact_lateral = 0;
    // double spring_lateral = 0;

    // Eigen::VectorXd force_contact(2*num_nodes);
    // force_contact.setZero();
    // if(USE_MORTAR_METHOD)
    //     addMortarForceEntries(force_contact);
    // else
    //     addIPC2DForceEntries(force_contact);

    // Eigen::VectorXd force_spring(2*num_nodes);
    // force_spring.setZero();
    // addVirtualSpringForceEntries(force_spring);

    // for(int i=0; i<Object_indices[1].size(); ++i)
    // {
    //     std::cout<<"Index: "<<Object_indices[1][i]<<" contact force: "<<force_contact(2*Object_indices[1][i])<<" spring force: "<<force_spring(2*Object_indices[1][i])<<std::endl;
    //     contact_lateral+=force_contact(2*Object_indices[1][i]);
    //     spring_lateral+=force_spring(2*Object_indices[1][i]);
    // }

    // std::cout<<"Net contact force: "<<contact_lateral<<std::endl;
    // std::cout<<"Net spring force: "<<spring_lateral<<std::endl;
}

template <int dim>
bool FEMSolver<dim>::checkLeftRight(Eigen::MatrixXd& next_pos)
{
    // double x_left = 0.0, x_right = 0.0;
    // double center = 0.0;
    // for(int i=0; i<Object_indices[1].rows(); ++i)
    // {
    //     center += deformed(Object_indices[1](i) * 2);;
    // }
    // center /= Object_indices[1].rows();
    // x_left = center;
    // x_right = WIDTH_1-center;

    // std::cout<<x_left<<" "<<x_right<<std::endl;

    // if(x_left>=0 && x_right>= 0) return true;
    // else return false;
    return true;
}

template <int dim>
double FEMSolver<dim>::computeUpperCenter()
{
    double center = 0;
    // for(int i=0; i<Object_indices[1].size(); ++i)
    // {
    //     center += deformed(2*Object_indices[1](i));
    //     //std::cout<<Object_indices[1](i)<<" "<<center<<std::endl;
    // }
    // center/=Object_indices[1].size();
    // std::cout<<"center "<<center<<std::endl;


    // Eigen::VectorXd residual(dim*num_nodes);
    // residual.setZero();
    // if(USE_TRUE_IPC_2D)
    //     addIPC2DtrueForceEntries(residual);
    // else
    //     addIPC2DForceEntries(residual);
    
    // for(int i=0; i<master_nodes.size(); ++i)
    // {
    //     std::cout<<"Contact Force for: "<<master_nodes[i]<<" "<<residual.segment<dim>(2*master_nodes[i]).transpose()<<std::endl;
    // }

    return center;
}

template class FEMSolver<2>;
template class FEMSolver<3>;
