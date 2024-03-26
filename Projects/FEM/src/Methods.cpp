#include <igl/per_face_normals.h>
#include <igl/doublearea.h>
#include <igl/centroid.h>
#include <igl/massmatrix.h>
#include <fstream>

#include "../include/FEMSolver.h"
#include <ipc/ipc.hpp>

#include <igl/writeOBJ.h>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "pallas/differential_evolution.h"
#include <ipc/ipc.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

using namespace autodiff;

double stiffness_val = 1e8;
using StiffnessMatrix = Eigen::SparseMatrix<T>;

bool leftTurn(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& c){
     return ((b(0) - a(0))*(c(1) - a(1)) - (b(1) - a(1))*(c(0)- a(0))) > 0;
}

double computeVariance(Eigen::VectorXd& arr)
{
    double res = 0.0;
    double mean = arr.sum()/arr.size();
    for(int i=0; i<arr.size(); ++i)
    {
        res += (arr(i) - mean) * (arr(i) - mean);
    }
    return sqrt(res/arr.size());
}

var f_point2linedist(const ArrayXvar& x, bool slave = true)
{
    var A = x(0) - x(2);
    var B = x(1) - x(3);
    var C = x(4) - x(2);
    var D = x(5) - x(3);
    
    var dot = A * C + B * D;
    var len_sq = C * C + D * D;
    var param = -1;
    if (len_sq != 0) //in case of 0 length line
        param = dot / len_sq;

    var xx, yy;

    if (param < 0) {
        xx = x(2);
        yy = x(3);
    }
    else if (param > 1) {
        xx = x(4);
        yy = x(5);
    }
    else {
        xx = x(2) + param * C;
        yy = x(3) + param * D;
    }

    var dx = x(0) - xx;
    var dy = x(1) - yy;

    var sign = 1.;

    Eigen::VectorXd p(2);
    p<< double(x(0)),double(x(1));
    Eigen::VectorXd v1(2);
    v1<<double(x(2)),double(x(3));
    Eigen::VectorXd v2(2);
    v2<<double(x(4)),double(x(5));
    
    if((slave && leftTurn(v1,v2,p)) || (!slave && !leftTurn(v1,v2,p)))
    {
        sign = -1.;
    }
    return sign*sqrt(dx * dx + dy * dy);
}

Eigen::VectorXd EdgeNormal(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
    double dx = p2(0)-p1(0);
    double dy = p2(1)-p1(1);
    Eigen::VectorXd normal(2);
    normal<<-dy,dx;
    return normal.normalized();
}


double point2linedist(const Eigen::VectorXd& p, const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, bool slave = true)
{
    double A = p(0) - v1(0);
    double B = p(1) - v1(1);
    double C = v2(0) - v1(0);
    double D = v2(1) - v1(1);

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    double param = -1;
    if (len_sq != 0) //in case of 0 length line
        param = dot / len_sq;

    double xx, yy;

    if (param < 0) {
        xx = v1(0);
        yy = v1(1);
    }
    else if (param > 1) {
        xx = v2(0);
        yy = v2(1);
    }
    else {
        xx = v1(0) + param * C;
        yy = v1(1) + param * D;
    }

    double dx = p(0) - xx;
    double dy = p(1) - yy;
    double sign = 1.;
    if((slave && leftTurn(v1,v2,p)) || (!slave && !leftTurn(v1,v2,p)))
    {
        sign = -1.;
    }
    return sign*sqrt(dx * dx + dy * dy);
}

Eigen::MatrixXd point2linedistgrad(const Eigen::VectorXd& p, const Eigen::VectorXd& v1, const Eigen::VectorXd& v2)
{
    double A = p(0) - v1(0);
    double B = p(1) - v1(1);
    double C = v2(0) - v1(0);
    double D = v2(1) - v1(1);

    Eigen::MatrixXd dA(3,2), dB(3,2), dC(3,2), dD(3,2);
    dA.setZero();
    dB.setZero();
    dC.setZero();
    dD.setZero();

    dA(0,0) = 1; dA(1,0) = -1;
    dB(0,1) = 1; dB(1,1) = -1;
    dC(2,0) = 1; dC(1,0) = -1;
    dD(2,1) = 1; dD(1,1) = -1;

    double dot = A * C + B * D;
    double len_sq = C * C + D * D;
    double param = -1;
    if (len_sq != 0) //in case of 0 length line
        param = dot / len_sq;

    Eigen::MatrixXd ddot(3,2);
    ddot.setZero();
    ddot = dA*C+dC*A+dB*D+dD*B;

    Eigen::MatrixXd dlen_sq(3,2);
    dlen_sq.setZero();
    dlen_sq = 2*C*dC+2*D*dD;

    Eigen::MatrixXd dparam(3,2);
    dparam.setZero();
    if (len_sq != 0) //in case of 0 length line
        dparam = ddot/len_sq - dot*dlen_sq/(len_sq*len_sq);

    double xx, yy;

    Eigen::MatrixXd dxx(3,2), dyy(3,2);
    dxx.setZero(); dyy.setZero();


    if (param < 0) {
        xx = v1(0);
        yy = v1(1);
        dxx(1,0) = 1;
        dyy(1,1) = 1;
    }
    else if (param > 1) {
        xx = v2(0);
        yy = v2(1);
        dxx(2,0) = 1;
        dyy(2,1) = 1;
    }
    else {
        xx = v1(0) + param * C;
        yy = v1(1) + param * D;
        dxx = dparam*C + dC*param;
        dxx(1,0) += 1;
        dyy = dparam*D + dD*param;
        dyy(1,1) += 1;
    }

    double dx = p(0) - xx;
    double dy = p(1) - yy;

    Eigen::MatrixXd ddx(3,2), ddy(3,2);
    ddx.setZero(); ddy.setZero();
    ddx = -dxx;
    ddx(0,0) += 1;
    ddy = -dyy;
    ddy(0,1) +=1;

    double dist = sqrt(dx * dx + dy * dy);
    double sign = 1.;
    if(leftTurn(v1,v2,p))
    {
        sign = -1.;
    }

    Eigen::MatrixXd dist_grad(3,2);
    dist_grad.setZero();
    dist_grad = 0.5/sqrt(dx * dx + dy * dy) * (2*dx*ddx+2*dy*ddy);

    return sign*dist_grad;
}

void autodifftest()
{
    VectorXvar x(6);
    x << 1., 2., 3., 4., 5., 6.;
    var u = f_point2linedist(x);

    Eigen::VectorXd p(2);
    p << 1., 2.;
    Eigen::VectorXd x1(2);
    x1 << 3., 4.;
    Eigen::VectorXd x2(2);
    x2 << 5., 6.;

    Eigen::VectorXd g;  // the gradient vector to be computed in method `hessian`
    Eigen::MatrixXd H = hessian(u, x, g);  // evaluate the Hessian matrix H and the gradient vector g of u

    double u_p = point2linedist(p, x1, x2);
    Eigen::MatrixXd g_p = point2linedistgrad(p, x1, x2);

    std::cout << "u = " << u << std::endl;    // print the evaluated output variable u
    std::cout << "g = \n" << g << std::endl;  // print the evaluated gradient vector of u
    std::cout << "H = \n" << H << std::endl;  // print the evaluated Hessian matrix of u

    std::cout << "up = " << u_p << std::endl;    // print the evaluated output variable u
    std::cout << "gp = \n" << g_p << std::endl;  // print the evaluated gradient vector of u
}

void testDerivative()
{
    srand(time(NULL));
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,2);
    double dist = point2linedist(m.row(0), m.row(1), m.row(2));
    Eigen::MatrixXd ddist = point2linedistgrad(m.row(0), m.row(1), m.row(2));
    double eps = 1e-5;

    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<2; ++j)
        {
            m(i,j) += eps;
            double new_dist = point2linedist(m.row(0), m.row(1), m.row(2));
            m(i,j) -= eps;
            std::cout<<i<<" "<<j<<" "<<ddist(i,j)<<" "<<(new_dist-dist)/eps<<std::endl;
        }
    }
}

template <typename DerivedLocalGrad, typename DerivedGrad>
void local_gradient_to_global_gradient(
    const Eigen::MatrixBase<DerivedLocalGrad>& local_grad,
    const std::vector<long>& ids,
    int dim,
    Eigen::PlainObjectBase<DerivedGrad>& grad,
    bool IMLS_3D_VIRTUAL = false)
{
    for (int i = 0; i < ids.size(); i++) {
        double scale = 1.0;
        if(IMLS_3D_VIRTUAL && i<3) scale = 1.0/3.0;
        int index = i;
        if(IMLS_3D_VIRTUAL) index = 0;
        if(IMLS_3D_VIRTUAL && i >= 3) index = i-2;

        grad.segment(dim * ids[i], dim) += scale*local_grad.segment(dim * index, dim);
    }

}

template <typename Derived>
void local_hessian_to_global_triplets(
    const Eigen::MatrixBase<Derived>& local_hessian,
    const std::vector<long>& ids,
    int dim,
    std::vector<Eigen::Triplet<double>>& triplets,
    bool IMLS_3D_VIRTUAL = false)
{
    for (int i = 0; i < ids.size(); i++) {
        double scale1 = 1.0;
        if(IMLS_3D_VIRTUAL && i<3) scale1 = 1.0/3.0;
        int index1 = i;
        if(IMLS_3D_VIRTUAL) index1 = 0;
        if(IMLS_3D_VIRTUAL && i >= 3) index1 = i-2;
        for (int j = 0; j < ids.size(); j++) {
            double scale2 = 1.0;
            if(IMLS_3D_VIRTUAL && j<3) scale2 = 1.0/3.0;
            int index2 = j;
            if(IMLS_3D_VIRTUAL) index2 = 0;
            if(IMLS_3D_VIRTUAL && i >= 3) index2 = i-2;
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    triplets.emplace_back(
                        dim * ids[i] + k, dim * ids[j] + l,
                        scale1*scale2*local_hessian(dim * index1 + k, dim * index2 + l));
                }
            }
        }
    }
}

template <typename Derived>
void local_sparse_hessian_to_global_triplets(
    const Eigen::SparseMatrix<Derived>& local_hessian,
    const std::vector<long>& ids,
    int dim,
    std::vector<Eigen::Triplet<double>>& triplets,
    bool IMLS_3D_VIRTUAL = false)
{
    for (int k=0; k<local_hessian.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(local_hessian,k); it; ++it)
        {
            int row_index = it.row(); 
            int col_index = it.col();
            int i = row_index / dim;
            int j = col_index / dim;
            int k = row_index % dim;
            int l = col_index % dim;
            triplets.emplace_back(
            dim * ids[i] + k, dim * ids[j] + l,
            it.value());
        }
    }
}


double compute1PKStressImpl(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X)
{
    double lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	double mu = E / 2.0 / (1.0 + nu);

    Eigen::Matrix<double, 3, 2> dNdb;
    dNdb << -1.0, -1.0, 
        1.0, 0.0,
        0.0, 1.0;
    Eigen::MatrixXd dXdb = X.transpose() * dNdb;
    Eigen::MatrixXd dxdb = x.transpose() * dNdb;
    Eigen::MatrixXd F = dxdb * dXdb.inverse();

    double J = F.determinant();
    double lnJ = std::log(J);
    Eigen::Matrix<double, 2, 2> FNT = F.inverse().transpose();

    Eigen::MatrixXd P = mu*(F-FNT) + lambda*lnJ*FNT;
    return P.norm();
}

double compute1PKStressImplQuad(double E, double nu, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X)
{
    double lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	double mu = E / 2.0 / (1.0 + nu);

    Eigen::MatrixXd basis(4,2);
    basis<< -1.0/sqrt(3.0), -1.0/sqrt(3.0),
            1.0/sqrt(3.0), -1.0/sqrt(3.0),
            1.0/sqrt(3.0), 1.0/sqrt(3.0),
            -1.0/sqrt(3.0), 1.0/sqrt(3.0);

    double result = 0.0;

    for(int i=0; i<4; ++i)
    {
        Eigen::Matrix<double, 4, 2> dNdb;
        dNdb << -0.25*(1.0-basis(i,1)), -0.25*(1.0-basis(i,0)), 
                0.25*(1.0-basis(i,1)), -0.25*(1.0+basis(i,0)),
                0.25*(1.0+basis(i,1)), 0.25*(1.0+basis(i,0)),
                -0.25*(1.0+basis(i,1)), 0.25*(1.0-basis(i,0));
        
        Eigen::MatrixXd dXdb = X.transpose() * dNdb;
        Eigen::MatrixXd dxdb = x.transpose() * dNdb;
        Eigen::MatrixXd F = dxdb * dXdb.inverse();

        double J = F.determinant();
        double lnJ = std::log(J);
        if(std::isnan(lnJ)) std::cout<<F<<std::endl;
        Eigen::Matrix<double, 2, 2> FNT = F.inverse().transpose();

        Eigen::MatrixXd P = mu*(F-FNT) + lambda*lnJ*FNT;
        result += 0.25 * P.norm();
    }
    return result;
    
}

Eigen::Matrix<double,2,2> computeCauchyStressImplQuad(double E, double nu, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X)
{
    double lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
	double mu = E / 2.0 / (1.0 + nu);

    Eigen::MatrixXd basis(4,2);
    basis<< -1.0/sqrt(3.0), -1.0/sqrt(3.0),
            1.0/sqrt(3.0), -1.0/sqrt(3.0),
            1.0/sqrt(3.0), 1.0/sqrt(3.0),
            -1.0/sqrt(3.0), 1.0/sqrt(3.0);

    Eigen::Matrix<double,2,2> result;
    result.setZero();

    for(int i=0; i<4; ++i)
    {
        Eigen::Matrix<double, 4, 2> dNdb;
        dNdb << -0.25*(1.0-basis(i,1)), -0.25*(1.0-basis(i,0)), 
                0.25*(1.0-basis(i,1)), -0.25*(1.0+basis(i,0)),
                0.25*(1.0+basis(i,1)), 0.25*(1.0+basis(i,0)),
                -0.25*(1.0+basis(i,1)), 0.25*(1.0-basis(i,0));
        
        Eigen::MatrixXd dXdb = X.transpose() * dNdb;
        Eigen::MatrixXd dxdb = x.transpose() * dNdb;
        Eigen::MatrixXd F = dxdb * dXdb.inverse();

        double J = F.determinant();
        double lnJ = std::log(J);
        if(std::isnan(lnJ)) std::cout<<F<<std::endl;
        Eigen::Matrix<double, 2, 2> FNT = F.inverse().transpose();

        Eigen::MatrixXd P = mu*(F-FNT) + lambda*lnJ*FNT;
        Eigen::MatrixXd sigma = (P*F.transpose()/J).transpose();
        result += 0.25 * sigma;
    }
    return result;
    
}

template <int dim>
void FEMSolver<dim>::initializeBoundaryInfo()
{

    is_boundary.resize(num_nodes);
    is_boundary.setZero();
    for(int i=0; i<num_nodes; ++i)
    {
        if(fabs(undeformed(dim*i+1))- 0 < 1e-3){
            is_boundary[i] = 1;
        }
            
    };
    //autodifftest();

    //testDerivative();
    // Compute the point-segment pair
}

double compute_potential(double d_hat, double d, int i)
{
    // if(d <= 0){
    //     std::cout<<i<<" "<<"error! d = "<<d<<std::endl;
    //     return std::numeric_limits<double>::infinity();
    // } 
    // if(d >= d_hat) return 0;
    // else return -(d-d_hat)*(d-d_hat)*log(d/d_hat);
    //std::cout<<exp(stiffness_val*(d_hat-d))<<std::endl; 
    // if(d == 1e8 || d == -1e8)
    //     std::cerr<<d<<std::endl;
    if(d >= 0) return 0;
    else return -stiffness_val*d*d*d;

    // return -stiffness_val*d;

}

Eigen::VectorXd compute_potential_gradient(double d_hat, double d, const Eigen::VectorXd& dist_grad)
{
    // if(d <= 0){
    //     std::cout<<"error! d = "<<d<<std::endl;
    // }
    // double barrier_gradient;
    // if (d <= 0.0 || d >= d_hat) {
    //     barrier_gradient =  0.0;
    // }
    // else barrier_gradient = (d_hat - d) * (2*log(d/d_hat) - d_hat/d + 1);

    //std::cout<<"Barrier gradient: "<<barrier_gradient<<std::endl;
    double barrier_gradient;
    if(d >= 0) barrier_gradient = 0;
    else barrier_gradient = -3*stiffness_val*d*d;

    return dist_grad*barrier_gradient;

    //  return -stiffness_val*dist_grad;
}

Eigen::MatrixXd compute_potential_hessian(double d_hat, double d, const Eigen::VectorXd& dist_grad, const Eigen::MatrixXd& dist_hess)
{
    Eigen::MatrixXd hess;
    //double barrier_hess;

    // if (d <= 0.0 || d >= d_hat) {
    //     barrier_hess =  0.0;
    // }
    // else{
    //     double dhat_d = d_hat / d;
    //     barrier_hess =  (dhat_d + 2) * dhat_d - 2 * log(d / d_hat) - 3;
    // }

    // double barrier_gradient;
    // if (d <= 0.0 || d >= d_hat) {
    //     barrier_gradient =  0.0;
    // }
    // else barrier_gradient = (d_hat - d) * (2*log(d/d_hat) - d_hat/d + 1);
    double barrier_hess, barrier_gradient;
    if(d >= 0) barrier_gradient = 0;
    else barrier_gradient = -3*stiffness_val*d*d;

    if(d >= 0) barrier_hess = 0;
    else barrier_hess = -6*stiffness_val*d;
    hess = barrier_hess * dist_grad * dist_grad.transpose() + barrier_gradient*dist_hess;

    return hess;
}

StiffnessMatrix compute_potential_hessian_sparse(double d_hat, double d, const Eigen::VectorXd& dist_grad, const StiffnessMatrix& dist_hess)
{
    StiffnessMatrix hess;
    //double barrier_hess;

    // if (d <= 0.0 || d >= d_hat) {
    //     barrier_hess =  0.0;
    // }
    // else{
    //     double dhat_d = d_hat / d;
    //     barrier_hess =  (dhat_d + 2) * dhat_d - 2 * log(d / d_hat) - 3;
    // }

    // double barrier_gradient;
    // if (d <= 0.0 || d >= d_hat) {
    //     barrier_gradient =  0.0;
    // }
    // else barrier_gradient = (d_hat - d) * (2*log(d/d_hat) - d_hat/d + 1);
    double barrier_hess, barrier_gradient;
    if(d >= 0) barrier_gradient = 0;
    else barrier_gradient = -3*stiffness_val*d*d;

    if(d >= 0) barrier_hess = 0;
    else barrier_hess = -6*stiffness_val*d;
    hess = (barrier_hess * dist_grad * dist_grad.transpose()).sparseView()+ barrier_gradient*dist_hess;

    return hess;
}

double compute_potential_Sqaured_Norm(double d_hat, double d, int i)
{
    if(BARRIER_ENERGY) return -(d-d_hat)*(d-d_hat)*log(d/d_hat);
    else return 0.5*d*d;
}

Eigen::VectorXd compute_potential_gradient_Sqaured_Norm(double d_hat, double d, const Eigen::VectorXd& dist_grad)
{
    double barrier_gradient = d;
    if(BARRIER_ENERGY)
    {
        barrier_gradient = -(d-d_hat)*(2*log(d/d_hat)-d/d_hat+1);
    } 
    return dist_grad*barrier_gradient;
}

Eigen::MatrixXd compute_potential_hessian_Sqaured_Norm(double d_hat, double d, const Eigen::VectorXd& dist_grad, const Eigen::MatrixXd& dist_hess)
{
    Eigen::MatrixXd hess;
    double barrier_hess = 1;
    double barrier_gradient = d;
    if(BARRIER_ENERGY)
    {
        barrier_gradient = -(d-d_hat)*(2*log(d/d_hat)-d/d_hat+1);
        barrier_hess = (d_hat/d+2)*d_hat/d-2*log(d/d_hat)-3;
    }
    hess = barrier_hess * dist_grad * dist_grad.transpose() + barrier_gradient*dist_hess;

    return hess;
}

template <int dim>
void FEMSolver<dim>::computeIPC2DRestData(bool original)
{
    if(CALCULATE_IMLS_PROJECTION)
    {
        CalculateProjectionIMLS();
        // for(int i=0; i<slave_nodes.size(); ++i)
        // {
        //     std::cout<<"Original: "<<ipc_vertices.row(slave_nodes[i])<<std::endl;
        //     std::cout<<"Projection: "<<projectedPts.row(i)<<std::endl;
        // }
        SortedProjectionPoints2D();
        std::cout<<"Projected Points: "<<std::endl;
        std::cout<<projectedPts<<std::endl;
    }
    int num_nodes_all = num_nodes;
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();

    ipc_vertices.resize(num_nodes_all, dim);
    for (int i = 0; i < num_nodes_all; i++)
        ipc_vertices.row(i) = undeformed.segment<dim>(i * dim);
    num_ipc_vtx = ipc_vertices.rows();

    std::vector<Edge> edges;
    for (int i = 0; i < num_surface_faces; i++)
    {
        int vi,vj,vk;
        vi = surface_indices(3*i);
        vj = surface_indices(3*i+1);
        vk = surface_indices(3*i+2);

        if(is_boundary[vi] && is_boundary[vj]) edges.push_back(Edge(vi,vj));
        if(is_boundary[vj] && is_boundary[vk]) edges.push_back(Edge(vj,vk));
        if(is_boundary[vk] && is_boundary[vi]) edges.push_back(Edge(vk,vi));
    }

    ipc_edges.resize(edges.size(), 2);
    for (int i = 0; i < edges.size(); i++)
        ipc_edges.row(i) = edges[i];

    for (int i = 0; i < ipc_edges.rows(); i++)
    {
        Edge edge = ipc_edges.row(i);
        TV vi = ipc_vertices.row(edge[0]), vj = ipc_vertices.row(edge[1]);
        // if ((vi - vj).norm() < barrier_distance)
        //     std::cout << "edge " << edge.transpose() << " has length < 1e-6 " << std::endl;
    }
    
    if(USE_VIRTUAL_NODE)
        GenerateVirtualPoints();
    if(dim == 3 && USE_IMLS)
    {
        findProjectionIMLSMultiple3D(ipc_vertices,true,false);
    }
    if(dim == 2)
    {
        if(PULL_TEST && USE_IMLS)
        {
            findProjectionIMLSMultiple(ipc_vertices,true,false);
        }else if(USE_IMLS && !original)
        {
            findProjectionIMLS(ipc_vertices,true,false);
        }else
            findProjection(ipc_vertices, true, false);
        if(USE_NEW_FORMULATION) findProjectionIMLSSameSide(ipc_vertices,true,false);
    }
    
}

template <int dim>
void FEMSolver<dim>::updateIPC2DVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    int num_nodes_all = num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    for (int i = 0; i < num_nodes_all; i++)
        ipc_vertices.row(i) = deformed.segment<dim>(i * dim);
}

template <int dim>
T FEMSolver<dim>::computeCollisionFreeStepsize2D(const VectorXT& _u, const VectorXT& du)
{
    if(!BARRIER_ENERGY) return 1;

    // if(USE_FROM_STL) return 1;
    // if(dim == 2) return 1;

    // Eigen::MatrixXd current_position = ipc_vertices, 
    //     next_step_position = ipc_vertices;

    // for (int i = 0; i < num_nodes; i++)
    // {
    //     current_position.row(i) = undeformed.segment<dim>(i * dim) + _u.segment<dim>(i * dim);
    // }

    // findProjectionIMLSMultiple3D(current_position);

    // double min_step = 1.0;

    // for (int i = 0; i < boundary_info.size(); i++)
    // {
    //     if(boundary_info[i].dist < 0){
    //         std::cout<<boundary_info[i].slave_index<<" "<<boundary_info[i].dist<<std::endl;
    //         std::cout<<"error finding stepsize!"<<std::endl;
    //     }  
    // }
    
    // while(true)
    // {        
    //     bool step_ok = true;
    //     for (int i = 0; i < num_nodes; i++)
    //     {
    //         next_step_position.row(i) = undeformed.segment<dim>(i * dim) + _u.segment<dim>(i * dim) + min_step*du.segment<dim>(i * dim);
    //     }

    //     // for(int i=0; i<boundary_info.size(); ++i)
    //     // {
    //     //     if(deformed(3*boundary_info[i].slave_index+1) < 0.5 || deformed(3*boundary_info[i].slave_index+1) > 1.5)
    //     //     {
    //     //         min_step/=2;
    //     //         step_ok = false;
    //     //     }
    //     // }

    //     // if(!step_ok) break;
        
    //     findProjectionIMLSMultiple3D(next_step_position);

    //     // std::cout<<"distance: "<<result_values.transpose()<<std::endl;
    //     // std::cout<<"step size: "<<min_step<<std::endl;

    //     for(int i=0; i<boundary_info.size(); ++i)
    //     {
    //         if(USE_IMLS)
    //         {
    //             if(boundary_info[i].dist < 0)
    //             {
    //                 min_step/=2.;
    //                 step_ok = false;
    //                 if(min_step < 1e-10) {
    //                     //std::cout<<boundary_info[i].dist<<std::endl;
    //                     return 0.;
    //                 }
    //                 break;
    //             } 
    //         }
    //     }
    //     if(step_ok) break;
    // }

    double min_step = 1.0;
    int iter = 0;

    while(true)
    {
        VectorXa x = undeformed+_u+min_step*du;
        VectorXa bk = deformed;
        deformed = x;
        double temp;
        addFastRIMLSSCEnergy(temp);
        bool is_step_ok = true;
        for(int i=0; i<dist_info.size(); ++i)
        {
            if(dist_info[i].first > 0 && dist_info[i].second <= 0){
                std::cout<<dist_info[i].first<<" "<<dist_info[i].second<<std::endl;
                is_step_ok = false;
                break;
            }
        }
        if(is_step_ok || iter >= 30){break;} 
        if(iter >= 30){min_step = 0; break;} 
        iter++;
        min_step /= 2.0;
        deformed = bk;
    }
    
    return min_step;
}

template <int dim>
T FEMSolver<dim>::computeCollisionFreeStepsizeUnsigned(const VectorXT& _u, const VectorXT& du)
{
    // T ipc_step = computeCollisionFreeStepsize3D(_u, du);
    // Eigen::MatrixXd current_position = ipc_vertices, 
    //     next_step_position = ipc_vertices;
        
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
    //     next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    // }

    // //std::cout<<"du at 2415: "<<du.segment<3>(3*2415)<<std::endl;
    // std::cout<<"IPC3D result: "<<ipc_step<<std::endl;

    T ipc_step = 1;

    // ipc::Candidates c;
    // ipc::construct_collision_candidates(current_position, next_step_position, ipc_edges, ipc_faces, c,0, ipc::BroadPhaseMethod::BRUTE_FORCE);

    std::vector<bool> vertex_candidates(num_nodes,1);
    // for(int i=0; i<c.fv_candidates.size(); ++i)
    // {
    //     int p = c.fv_candidates[i].vertex_index;
    //     int v1 = ipc_faces(c.fv_candidates[i].face_index,0);
    //     int v2 = ipc_faces(c.fv_candidates[i].face_index,1);
    //     int v3 = ipc_faces(c.fv_candidates[i].face_index,2);
        
    //     if(geodist_close_matrix.coeff(p,v1) == 0 && geodist_close_matrix.coeff(p,v2) == 0 && geodist_close_matrix.coeff(p,v3) == 0)
    //     {
    //         std::cout<<"Vertex: "<<c.fv_candidates[i].vertex_index<<" Face: "<<c.fv_candidates[i].face_index<<" "<<ipc_faces(c.fv_candidates[i].face_index,1)<<" "<<ipc_faces(c.fv_candidates[i].face_index,2)<<" "<<ipc_faces(c.fv_candidates[i].face_index,0)<<std::endl;
    //         vertex_candidates[c.fv_candidates[i].vertex_index] = 1;
    //     }
            
    // }

    
    std::cout<<accumulate(vertex_candidates.begin(), vertex_candidates.end(),0)<<std::endl;

    // double min_step = 1.0;
    // int iter = 0;

    // while(true)
    // {
    //     VectorXa x = undeformed+_u+min_step*du;
    //     VectorXa bk = deformed;
    //     deformed = x;
    //     double temp;
    //     addFastIMLSSCEnergy(temp);
    //     bool is_step_ok = true;
    //     for(int i=0; i<dist_info.size(); ++i)
    //     {
    //         if(dist_info[i].first > 0 && dist_info[i].second <= 0){
    //             std::cout<<dist_info[i].first<<" "<<dist_info[i].second<<std::endl;
    //             is_step_ok = false;
    //             break;
    //         }
    //     }
    //     if(is_step_ok || iter >= 30){break;} 
    //     if(iter >= 30){min_step = 0; break;} 
    //     iter++;
    //     min_step /= 2.0;
    //     deformed = bk;
    // }
    Eigen::VectorXd max_step_size(num_nodes);
    for(int i=0; i<num_nodes; ++i)
    {
        max_step_size[i] = ipc_step;
    }

    double max_step = ipc_step;
    std::vector<int> previous_sign(num_nodes,0);

    double temp;
    if(USE_RIMLS)
        addFastRIMLSSCEnergy(temp);
    else
        addFastIMLSSCEnergy(temp,true);
    for(int i=0; i<dist_info.size(); ++i)
    {
        //std::cout<<"1111"<<dist_info[i].first<<" "<<dist_info[i].second<<std::endl;
        if(dist_info[i].first == 3216 || dist_info[i].first == 1551) std::cout<<"1111"<<dist_info[i].first<<" "<<dist_info[i].second<<std::endl;
        if(dist_info[i].first > 0 && dist_info[i].second > 0){
            previous_sign[dist_info[i].first] = 1;
        }
        else if(dist_info[i].first > 0 && dist_info[i].second < 0){
            previous_sign[dist_info[i].first] = -1;
        }
    }

    std::vector<int> potential_sign_pts;
    std::vector<int> potential_signs;

    VectorXa x = deformed+max_step*du;
    VectorXa bk = deformed;
    deformed = x;

    if(USE_RIMLS)
        addFastRIMLSSCEnergy(temp);
    else
        addFastIMLSSCEnergy(temp,true);
    deformed = bk;


    for(int i=0; i<dist_info.size(); ++i)
    {
        int index = dist_info[i].first;
        double distance;
        if(USE_RIMLS) evaluateUnsignedDistanceSqRIMLS(index,max_step,du,max_step,distance);
        else evaluateUnsignedDistanceSq(index,max_step,du,max_step,distance);
        
        if(dist_info[i].first > 0 && previous_sign[dist_info[i].first] == 0 && dist_info[i].second != 0)
        {
            // if(index == 1291)
            //     std::cout<<previous_sign[dist_info[i].first]<<" "<<dist_info[i].first<<" "<<dist_info[i].second<<" "<<distance<<std::endl;
            potential_sign_pts.push_back(dist_info[i].first);
            if(dist_info[i].second > 0) potential_signs.push_back(1);
            else if(dist_info[i].second < 0) potential_signs.push_back(-1);
        }
    }

    
    

    // Check previous sign
    int num_tests = potential_sign_pts.size();

    tbb::parallel_for(0, num_tests , [&](int i)
    //for(int i=0; i<num_tests; ++i)
    {
        int index = potential_sign_pts[i];
        int sign_start = 0;
        int sign_end = potential_signs[i];

        double start = 0;
        double end = max_step;

        while((end-start)*du.norm()>1e-12)
        {
            
            double mid = (start+end)/2.;
            double distance;
            int sign_mid = 0;
            if(USE_RIMLS) evaluateUnsignedDistanceSqRIMLS(index,mid,du,max_step,distance);
            else evaluateUnsignedDistanceSq(index,mid,du,max_step,distance);
            if(distance > 0) sign_mid = 1;
            if(distance < 0) sign_mid = -1;

            //if(index == 1291) std::cout<<index<<" "<<start<<" "<<sign_start<<" "<<mid<<" "<<distance<<" "<<end<<" "<<sign_end<<std::endl;

            if(sign_mid == 0)
            {
                start = mid;
            }
            else if(sign_mid == sign_end)
            {
                end = mid;
            }
            else 
            {
                previous_sign[index] = sign_mid;
                break;
            }
            previous_sign[index] = sign_end;
        }
    }
    );

    if(max_step < 1e-2)
    {
        max_step = 0.1;
        for(int i=0; i<num_nodes; ++i)
        {
            max_step_size[i] = max_step;
        }
    } 

    int iter = 0;
    double prev_step;

    while(true)
    {
        prev_step = max_step;
        VectorXa x = deformed+max_step*du;
        VectorXa bk = deformed;
        deformed = x;

        double temp;
        if(USE_RIMLS)
            addFastRIMLSSCEnergy(temp);
        else
            addFastIMLSSCEnergy(temp,true);
        
        deformed = bk;

        bool is_step_ok = true;
        std::vector<int> collision_points;

        // double dist;
        // evaluateUnsignedDistanceSq(1291,max_step,du,max_step,dist);
        // std::cout<<"1291 dist"<<dist<<std::endl;
        
        for(int i=0; i<dist_info.size(); ++i)
        {
            if(dist_info[i].first > 0){
                //if(previous_sign[dist_info[i].first] == 0 || previous_sign[dist_info[i].first]*dist_info[i].second > 0) continue;
                if(dist_info[i].second > 0) continue;
                std::cout<<dist_info[i].first<<" "<<dist_info[i].second<<" sign "<<previous_sign[dist_info[i].first]<<std::endl;
                collision_points.push_back(dist_info[i].first);
                is_step_ok = false;
            }
        }

        if(is_step_ok) break;
        int num_tests = collision_points.size();


        // tbb::parallel_for(0, num_tests , [&](int i)
        // {
        //     int k = collision_points[i];
        //     // define the starting point for the optimization
        //     double parameters[1] = {0.0};
        //     double max_step_local = min(max_step,max_step_size.minCoeff());
        //     // set up global optimizer options only i nitialization
        //     // is need to accept the default options
        //     pallas::Basinhopping::Options options;
        //     options.max_stagnant_iterations = 20;
        //     options.is_silent = true;
        //     options.minimum_cost = 1e-12;
        //     pallas::scoped_ptr<pallas::StepFunction> default_step(new pallas::DefaultStepFunction(max_step_local));
        //     options.set_step_function(default_step);


        //     // initialize a summary object to hold the
        //     // optimization details
        //     pallas::Basinhopping::Summary summary;
        //     // create a problem from your cost function
        //     Rosenbrock<dim>* ptr = new Rosenbrock<dim>();
        //     ptr->solver = this;
        //     ptr->index = k;
        //     ptr->du = du;
        //     ptr->ipc_stepsize = max_step_local;
        //     pallas::GradientProblem problem(ptr);

        //     pallas::Solve(options, problem, parameters, &summary);

        //     if((summary.final_cost) < 1e-10 && parameters[0] > 0 && parameters[0] < max_step_local) max_step_size(k) = 0.9*parameters[0];
        //     else max_step_size(k) = max_step_local;

        //     //std::cout<<k<<" out of "<<num_nodes<<". Final t = "<<parameters[0]<<" . Final cost = "<<summary.final_cost<<" max step: "<<max_step_local<<std::endl;

        // }
        // );

        
        

        max_step /= 2;

        
        // Eigen::Index index;
        // max_step = max_step_size.minCoeff(&index);
        // std::cout<<"Minimal Step Size Index: "<<index<<" max step "<<max_step<<std::endl;
        // if(max_step == prev_step && iter>1)
        // {
        //     max_step *= 0.5;
        // }
        
       
        iter ++;
    }

    
    
    return max_step;
}

template <int dim>
double FEMSolver<dim>::compute_barrier_potential2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side)
{
    tbb::enumerable_thread_specific<double> storage(0);

    if(!USE_NEW_FORMULATION || !eval_same_side)
    {
        tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundary_info.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                if(boundary_info[i].dist != 1e8)
                    local_potential += 
                    boundary_info[i].scale*compute_potential(barrier_distance, boundary_info[i].dist, boundary_info[i].slave_index);
            }
        });
    }
    else
    {
        tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundary_info.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_potential = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_potential += IMLS_param*
                boundary_info[i].scale*compute_potential_Sqaured_Norm(barrier_distance, boundary_info_same_side[i].dist, boundary_info_same_side[i].slave_index);
            }
        });
    }

    double potential = 0;
    for (const auto& local_potential : storage) {
        potential += local_potential;
    }
    // std::cout<<"potential: "<<potential<<std::endl;
    return potential;
}

template <int dim>
Eigen::VectorXd FEMSolver<dim>::compute_barrier_potential_gradient2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side)
{
    int num_nodes_all = num_nodes;
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    tbb::enumerable_thread_specific<Eigen::VectorXd> storage(
        Eigen::VectorXd::Zero(num_nodes_all*dim));

    // for(int i=0; i<boundary_info.size(); ++i)
    // {
    //     std::cout<<boundary_info[i].slave_index<<" "<<boundary_info[i].master_index_1<<" "<<boundary_info[i].master_index_2<<" "<<boundary_info[i].dist<<std::endl;
    //     std::cout<<boundary_info[i].dist_grad.transpose()<<std::endl;
    // }

    if(!USE_NEW_FORMULATION || !eval_same_side)
    {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), boundary_info.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_grad = storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    std::vector<long> vertices;
                    if(USE_IMLS)
                    {
                        //if(boundary_info[i].results.size() == 0) continue;
                        if(IMLS_3D_VIRTUAL)
                        {
                            vertices.push_back(FS(boundary_info[i].slave_index,0));
                            vertices.push_back(FS(boundary_info[i].slave_index,1));
                            vertices.push_back(FS(boundary_info[i].slave_index,2));
                        }
                        else
                            vertices.push_back(boundary_info[i].slave_index);
                        for(int j=0; j<boundary_info[i].results.size(); ++j)
                        {
                            vertices.push_back(boundary_info[i].results[j]);
                        }
                    } 
                    else
                        vertices = {boundary_info[i].slave_index, boundary_info[i].master_index_1, boundary_info[i].master_index_2};
                    //std::cout<<boundary_info[i].slave_index<<" "<<boundary_info[i].dist_grad.size()<<" "<<" "<<vertices.size()<<std::endl;
                    if(boundary_info[i].dist != 1e8)
                        local_gradient_to_global_gradient(
                        boundary_info[i].scale*compute_potential_gradient(barrier_distance, boundary_info[i].dist, boundary_info[i].dist_grad),
                        vertices, dim, local_grad, IMLS_3D_VIRTUAL);
                }
            });
    }else
    {
        tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundary_info_same_side.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_grad = storage.local();
            for (size_t i = r.begin(); i < r.end(); i++) {
                std::vector<long> vertices;
                if(USE_IMLS)
                {
                    //if(boundary_info[i].results.size() == 0) continue;
                    if(IMLS_3D_VIRTUAL)
                    {
                        vertices.push_back(FS(boundary_info_same_side[i].slave_index,0));
                        vertices.push_back(FS(boundary_info_same_side[i].slave_index,1));
                        vertices.push_back(FS(boundary_info_same_side[i].slave_index,2));
                    }
                    else
                        vertices.push_back(boundary_info_same_side[i].slave_index);
                    for(int j=0; j<boundary_info_same_side[i].results.size(); ++j)
                    {
                        vertices.push_back(boundary_info_same_side[i].results[j]);
                    }
                } 
                else
                    vertices = {boundary_info_same_side[i].slave_index, boundary_info_same_side[i].master_index_1, boundary_info_same_side[i].master_index_2};
                //std::cout<<boundary_info[i].slave_index<<" "<<boundary_info[i].dist_grad.size()<<" "<<" "<<vertices.size()<<std::endl;
                local_gradient_to_global_gradient(
                IMLS_param*boundary_info_same_side[i].scale*compute_potential_gradient_Sqaured_Norm(barrier_distance, boundary_info_same_side[i].dist, boundary_info_same_side[i].dist_grad),
                vertices, dim, local_grad, IMLS_3D_VIRTUAL);
            }
        });
    }
    

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes_all*dim);
    for (const auto& local_grad : storage) {
        grad += local_grad;
    }
    return grad;
}

template <int dim>
Eigen::SparseMatrix<double> FEMSolver<dim>::compute_barrier_potential_hessian2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side)
{
    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>>
        storage;

    // for(int i=0; i<boundary_info.size(); ++i)
    // {
    //     std::vector<long> vertices = {boundary_info[i].slave_index, boundary_info[i].master_index_1, boundary_info[i].master_index_2};
    //     std::cout<<vertices[0]<<" "<<vertices[1]<<" "<<vertices[2]<<std::endl;
    //     // std::cout<<boundary_info[i].dist_hess<<std::endl;
    //     // std::cout<<std::endl;
    //     std::cout<<compute_potential_hessian(barrier_distance, boundary_info[i].dist, boundary_info[i].dist_grad, boundary_info[i].dist_hess)<<std::endl;
    // }

    if(!USE_NEW_FORMULATION || !eval_same_side)
    {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), boundary_info.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_hess_triplets = storage.local();

                for (size_t i = r.begin(); i < r.end(); i++) {
                    std::vector<long> vertices;
                    if(USE_IMLS)
                    {
                        //if(boundary_info[i].results.size() == 0) continue;
                        vertices.push_back(boundary_info[i].slave_index);
                        for(int j=0; j<boundary_info[i].results.size(); ++j)
                        {
                            vertices.push_back(boundary_info[i].results[j]);
                        }
                    } 
                    else
                        vertices = {boundary_info[i].slave_index, boundary_info[i].master_index_1, boundary_info[i].master_index_2};
                    
                    if(!USE_SHELL)
                    {
                        local_hessian_to_global_triplets(
                        boundary_info[i].scale*compute_potential_hessian(barrier_distance, boundary_info[i].dist, boundary_info[i].dist_grad, boundary_info[i].dist_hess),
                        vertices, dim,
                        local_hess_triplets,IMLS_3D_VIRTUAL);
                    }else
                    {
                        if(boundary_info[i].dist != 1e8)
                        {
                            StiffnessMatrix temp = boundary_info[i].scale*compute_potential_hessian_sparse(barrier_distance, boundary_info[i].dist, boundary_info[i].dist_grad, boundary_info[i].dist_hess_s);
                            local_sparse_hessian_to_global_triplets(temp,vertices, dim,local_hess_triplets,IMLS_3D_VIRTUAL);
                        }
                            
                    }
                    
                    
                }
            });
    }
    else
    {
        tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), boundary_info_same_side.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            auto& local_hess_triplets = storage.local();

            for (size_t i = r.begin(); i < r.end(); i++) {
                std::vector<long> vertices;
                if(USE_IMLS)
                {
                    //if(boundary_info[i].results.size() == 0) continue;
                    vertices.push_back(boundary_info_same_side[i].slave_index);
                    for(int j=0; j<boundary_info_same_side[i].results.size(); ++j)
                    {
                        vertices.push_back(boundary_info_same_side[i].results[j]);
                    }
                } 
                else
                    vertices = {boundary_info_same_side[i].slave_index, boundary_info_same_side[i].master_index_1, boundary_info_same_side[i].master_index_2};
                local_hessian_to_global_triplets(
                IMLS_param*boundary_info_same_side[i].scale*compute_potential_hessian_Sqaured_Norm(barrier_distance, boundary_info_same_side[i].dist, boundary_info_same_side[i].dist_grad, boundary_info_same_side[i].dist_hess),
                vertices, dim,
                local_hess_triplets,IMLS_3D_VIRTUAL);
                
            }
        });
    }
    

    int num_nodes_all = num_nodes;
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    Eigen::SparseMatrix<double> hess(num_nodes_all*dim, num_nodes_all*dim);
    for (const auto& local_hess_triplets : storage) {
        Eigen::SparseMatrix<double> local_hess(num_nodes_all*dim, num_nodes_all*dim);
        local_hess.setFromTriplets(
            local_hess_triplets.begin(), local_hess_triplets.end());
        hess += local_hess;
    }
    // for(int i=0; i<num_nodes*dim; ++i)
    // {
    //     for(int j=0; j<num_nodes*dim; ++j)
    //     {
    //         std::cout<<hess.coeff(i,j)<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    return hess;
}

template <int dim>
void FEMSolver<dim>::findProjectionIMLS(Eigen::MatrixXd& ipc_vertices_deformed, bool re, bool stay)
{
    // Initialization
    if(USE_FROM_STL || TEST || SLIDING_TEST)
        samplePointsFromSTL();
    else
        samplePoints();
    buildConstraintsPointSet();

    if(re)
    {   
        // Build Query Matrix
        boundary_info.clear();
        Eigen::MatrixXd query_pts(slave_nodes.size(),2);
        double prev_len = 0, next_len = 0;

        if(USE_VIRTUAL_NODE)
        {
            query_pts.setZero(virtual_slave_nodes.size(),2);
            for(int i=0; i<virtual_slave_nodes.size(); ++i)
            {
                int i1 = virtual_slave_nodes[i].left_index;
                int i2 = virtual_slave_nodes[i].right_index;
                double pos = virtual_slave_nodes[i].eta;
                Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);
                query_pts.row(i) = p;

                if(i < virtual_slave_nodes.size() - 1)
                {
                    int i1p = virtual_slave_nodes[i+1].left_index;
                    int i2p = virtual_slave_nodes[i+1].right_index;
                    double posp = virtual_slave_nodes[i+1].eta;
                    Eigen::VectorXd pp = (1-posp)/2.0*ipc_vertices_deformed.row(i1p) + (1+posp)/2.0*ipc_vertices_deformed.row(i2p);

                    next_len = (p-pp).norm();
                }else{
                    next_len = 0;
                }

                point_segment_pair p1;

                p1.slave_index = -1;
                p1.master_index_1 = -1;
                p1.master_index_2 = -1;
                p1.index = 0;

                if(use_NTS_AR)
                    p1.scale = (prev_len+next_len)/2.0;
                else
                    p1.scale = 1;

                map_boundary_virtual[boundary_info.size()] = i;
                boundary_info.push_back(p1);
                prev_len = next_len;
            }
        }
        else
        {
            for(int i=0; i<slave_nodes.size(); ++i)
            {
                query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]);
                if(USE_NEW_FORMULATION) query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes);
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes.back(),0)) < 1e-5)
                //         query_pts(i,0) += 0.001;
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes[0],0)) < 1e-5)
                //         query_pts(i,0) -= 0.001;
                // if(CALCULATE_IMLS_PROJECTION)
                // {
                //     query_pts.row(i) = projectedPts.row(i);
                // }
                if(i < slave_nodes.size() - 1)
                {
                    Eigen::VectorXd p1 = ipc_vertices_deformed.row(slave_nodes[i]);
                    Eigen::VectorXd p2 = ipc_vertices_deformed.row(slave_nodes[i+1]);
                    if(USE_NEW_FORMULATION)
                    {
                        p1 = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes);
                        p2 = ipc_vertices_deformed.row(slave_nodes[i+1]+num_nodes);
                    }
                    // if(CALCULATE_IMLS_PROJECTION)
                    // {
                    //     p1 = ipc_vertices_deformed.row(num_nodes+i);
                    //     p2 = ipc_vertices_deformed.row(num_nodes+i+1);
                    // }
                    next_len = (p1-p2).norm();
                }else{
                    next_len = 0;
                }

                point_segment_pair p1;
                p1.slave_index = slave_nodes[i];
                if(USE_NEW_FORMULATION) p1.slave_index += num_nodes;
                // if(CALCULATE_IMLS_PROJECTION)
                // {
                //     p1.slave_index = i;
                // }
                p1.master_index_1 = -1;
                p1.master_index_2 = -1;
                p1.index = 0;

                if(use_NTS_AR)
                {
                    if(CALCULATE_IMLS_PROJECTION && IMLS_BOTH)
                    {
                        //std::cout<<"slave node Index: "<<slave_nodes[i]<<std::endl;
                        p1.scale = extendedAoC[slave_nodes[i]];
                    }
                        
                    else
                        p1.scale = RES*(prev_len+next_len)/2.0;
                }
                    
                else
                    p1.scale = 1;

                boundary_info.push_back(p1);
                prev_len = next_len;
            }
        }
        
        evaluateImplicitPotentialKR(query_pts, true, 0);


        if(IMLS_BOTH)
        {
            if(USE_VIRTUAL_NODE)
            {
                query_pts.setZero(virtual_master_nodes.size(),2);
                for(int i=0; i<virtual_master_nodes.size(); ++i)
                {
                    int i1 = virtual_master_nodes[i].left_index;
                    int i2 = virtual_master_nodes[i].right_index;
                    double pos = virtual_master_nodes[i].eta;
                    Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);
                    query_pts.row(i) = p;

                    if(i < virtual_master_nodes.size() - 1)
                    {
                        int i1p = virtual_master_nodes[i+1].left_index;
                        int i2p = virtual_master_nodes[i+1].right_index;
                        double posp = virtual_master_nodes[i+1].eta;
                        Eigen::VectorXd pp = (1-posp)/2.0*ipc_vertices_deformed.row(i1p) + (1+posp)/2.0*ipc_vertices_deformed.row(i2p);

                        next_len = (p-pp).norm();
                    }else{
                        next_len = 0;
                    }

                    point_segment_pair p1;

                    p1.slave_index = -1;
                    p1.master_index_1 = -1;
                    p1.master_index_2 = -1;
                    p1.index = 0;

                    if(use_NTS_AR)
                        p1.scale = (prev_len+next_len)/2.0;
                    else
                        p1.scale = 1;

                    map_boundary_virtual[boundary_info.size()] = i;
                    boundary_info.push_back(p1);
                    prev_len = next_len;
                }
            }
            else
            {
                query_pts.setZero(master_nodes.size(),2);
                prev_len = 0;
                next_len = 0;
                for(int i=0; i<master_nodes.size(); ++i)
                {
                    query_pts.row(i) = ipc_vertices_deformed.row(master_nodes[i]);
                    if(USE_NEW_FORMULATION) query_pts.row(i) = ipc_vertices_deformed.row(master_nodes[i]+num_nodes);
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes[0],0)) < 1e-5)
                    //     query_pts(i,0) += 0.001;
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes.back(),0)) < 1e-5)
                    //     query_pts(i,0) -= 0.001;

                    if(i < master_nodes.size() - 1)
                    {
                        Eigen::VectorXd p1 = ipc_vertices_deformed.row(master_nodes[i]);
                        Eigen::VectorXd p2 = ipc_vertices_deformed.row(master_nodes[i+1]);
                        if(USE_NEW_FORMULATION)
                        {
                            p1 = ipc_vertices_deformed.row(master_nodes[i]+num_nodes);
                            p2 = ipc_vertices_deformed.row(master_nodes[i+1]+num_nodes);
                        }
                        next_len = (p1-p2).norm();
                    }else{
                        next_len = 0;
                    }

                    point_segment_pair p1;
                    p1.slave_index = master_nodes[i];
                    if(USE_NEW_FORMULATION) p1.slave_index = master_nodes[i]+num_nodes;
                    p1.master_index_1 = -1;
                    p1.master_index_2 = -1;
                    p1.index = 1;

                    if(use_NTS_AR)
                    {
                        if(CALCULATE_IMLS_PROJECTION && IMLS_BOTH)
                            p1.scale = extendedAoC[master_nodes[i]];
                        else
                            p1.scale = RES*(prev_len+next_len)/2.0;
                    }
                        
                    else
                        p1.scale = 1;

                    boundary_info.push_back(p1);
                    //std::cout<<"Creation: "<<p1.slave_index<<" "<<p1.index<<std::endl;
                    prev_len = next_len;
                }
            }
            evaluateImplicitPotentialKR(query_pts, true, 1);
        }
    }
    else
    {
        // Build Query Matrix
        Eigen::MatrixXd query_pts(slave_nodes.size(),2);
        if(USE_VIRTUAL_NODE)
        {
            query_pts.setZero(virtual_slave_nodes.size(),2);
            for(int i=0; i<virtual_slave_nodes.size(); ++i)
            {
                int i1 = virtual_slave_nodes[i].left_index;
                int i2 = virtual_slave_nodes[i].right_index;
                double pos = virtual_slave_nodes[i].eta;
                Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);

                query_pts.row(i) = p;
            }
        }
        else
        {
            for(int i=0; i<slave_nodes.size(); ++i)
            {
                query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]);
                if(USE_NEW_FORMULATION) query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes); 
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes.back(),0)) < 1e-5)
                //         query_pts(i,0) += 0.001;
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes[0],0)) < 1e-5)
                //         query_pts(i,0) -= 0.001;
                // if(CALCULATE_IMLS_PROJECTION)
                //     query_pts.row(i) = ipc_vertices_deformed.row(num_nodes+i);
            }
        }
        
        evaluateImplicitPotentialKR(query_pts, true, 0);

        if(IMLS_BOTH)
        {
            if(USE_VIRTUAL_NODE)
            {
                query_pts.setZero(virtual_master_nodes.size(),2);
                for(int i=0; i<virtual_master_nodes.size(); ++i)
                {
                    int i1 = virtual_master_nodes[i].left_index;
                    int i2 = virtual_master_nodes[i].right_index;
                    double pos = virtual_master_nodes[i].eta;
                    Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);

                    query_pts.row(i) = p;
                }
            }
            else
            {
                query_pts.setZero(master_nodes.size(),2);
                for(int i=0; i<master_nodes.size(); ++i)
                {
                    query_pts.row(i) = ipc_vertices_deformed.row(master_nodes[i]);
                    if(USE_NEW_FORMULATION) query_pts.row(i) = ipc_vertices_deformed.row(master_nodes[i]+num_nodes); 
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes[0],0)) < 1e-5)
                    //     query_pts(i,0) += 0.001;
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes.back(),0)) < 1e-5)
                    //     query_pts(i,0) -= 0.001;
                }
            }
            evaluateImplicitPotentialKR(query_pts, true, 1);
        }
    }
}

template <int dim>
void FEMSolver<dim>::findProjectionIMLSMultiple(Eigen::MatrixXd& ipc_vertices_deformed, bool re, bool stay)
{
    assert(multiple_slave_nodes.size() == multiple_master_nodes.size());
    int pair_size = multiple_slave_nodes.size();
    if(re)
    {
        boundary_info.clear();
        if(use_multiple_pairs)
        {
            if(IMLS_BOTH) boundary_info_start.resize(2);
            else boundary_info_start.resize(1);
        }
    }
       

    for(int k=0; k<pair_size; ++k)
    {
        // Initialization
        samplePointsFromSTL(k);
        buildConstraintsPointSet();

        if(re)
        {   
            // Build Query Matrix
            Eigen::MatrixXd query_pts(multiple_slave_nodes[k].size(),2);
            double prev_len = 0, next_len = 0;
            boundary_info_start[0].push_back(boundary_info.size());
            for(int i=0; i<multiple_slave_nodes[k].size(); ++i)
            {
                
                query_pts.row(i) = ipc_vertices_deformed.row(multiple_slave_nodes[k][i]);
                if(i < multiple_slave_nodes[k].size() - 1)
                {
                    Eigen::VectorXd p1 = ipc_vertices_deformed.row(multiple_slave_nodes[k][i]);
                    Eigen::VectorXd p2 = ipc_vertices_deformed.row(multiple_slave_nodes[k][i+1]);
                    next_len = (p1-p2).norm();
                }else{
                    next_len = 0;
                }

                point_segment_pair p1;
                p1.slave_index = multiple_slave_nodes[k][i];
                p1.master_index_1 = -1;
                p1.master_index_2 = -1;
                p1.index = 0;

                if(use_NTS_AR)
                    p1.scale = RES*(prev_len+next_len)/2.0;
                else
                    p1.scale = 1;

                boundary_info.push_back(p1);
                prev_len = next_len;
            }
            evaluateImplicitPotentialKR(query_pts, true, 0, k);
            if(IMLS_BOTH)
            {
                query_pts.setZero(multiple_master_nodes[k].size(),2);
                prev_len = 0;
                next_len = 0;

                boundary_info_start[1].push_back(boundary_info.size());
                for(int i=0; i<multiple_master_nodes[k].size(); ++i)
                {
                    query_pts.row(i) = ipc_vertices_deformed.row(multiple_master_nodes[k][i]);

                    if(i < multiple_master_nodes[k].size() - 1)
                    {
                        Eigen::VectorXd p1 = ipc_vertices_deformed.row(multiple_master_nodes[k][i]);
                        Eigen::VectorXd p2 = ipc_vertices_deformed.row(multiple_master_nodes[k][i+1]);
                        next_len = (p1-p2).norm();
                    }else{
                        next_len = 0;
                    }

                    point_segment_pair p1;
                    p1.slave_index = multiple_master_nodes[k][i];
                    p1.master_index_1 = -1;
                    p1.master_index_2 = -1;
                    p1.index = 1;

                    if(use_NTS_AR)
                        p1.scale = RES*(prev_len+next_len)/2.0;
                    else
                        p1.scale = 1;

                    boundary_info.push_back(p1);
                    //std::cout<<"Creation: "<<p1.slave_index<<" "<<p1.index<<std::endl;
                    prev_len = next_len;
                }
                evaluateImplicitPotentialKR(query_pts, true, 1, k);
            }
        }
        else
        {
            // Build Query Matrix
            Eigen::MatrixXd query_pts(multiple_slave_nodes[k].size(),2);
            for(int i=0; i<multiple_slave_nodes[k].size(); ++i)
            {
                query_pts.row(i) = ipc_vertices_deformed.row(multiple_slave_nodes[k][i]);
            }
            
            evaluateImplicitPotentialKR(query_pts, true, 0, k);

            if(IMLS_BOTH)
            {
                query_pts.setZero(multiple_master_nodes[k].size(),2);
                for(int i=0; i<multiple_master_nodes[k].size(); ++i)
                {
                    query_pts.row(i) = ipc_vertices_deformed.row(multiple_master_nodes[k][i]);
                }
                evaluateImplicitPotentialKR(query_pts, true, 1, k);
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::findProjectionIMLSMultiple3D(Eigen::MatrixXd& ipc_vertices_deformed, bool re, bool stay)
{
    assert(dim == 3);
    assert(slave_nodes_3d.size() == master_nodes_3d.size());
    int pair_size = slave_nodes_3d.size();
    if(re)
    {
        boundary_info.clear();
        if(IMLS_BOTH) boundary_info_start_3d.resize(2);
        else boundary_info_start_3d.resize(1);
    }
       

    for(int k=0; k<pair_size; ++k)
    {
        // Initialization
        samplePointsFromSTL(k);
        buildConstraintsPointSet();

        if(re)
        {   
            // Build Query Matrix
            Eigen::MatrixXd query_pts(slave_nodes_3d[k].size(),dim);
            boundary_info_start_3d[0].push_back(boundary_info.size());

            if(IMLS_3D_VIRTUAL)
            {
                query_pts.resize(slave_surfaces_3d[k].size(),dim);
                for(int i=0; i<slave_surfaces_3d[k].size(); ++i)
                {
                    Eigen::VectorXd vp(3);
                    vp.setZero();

                    for(int l=0; l<3; ++l)
                    {
                        int ic = slave_surfaces_3d[k][i](l);
                        vp += deformed.segment<dim>(ic*dim);
                    }

                    vp.array() /= 3.0;

                    query_pts.row(i) = vp.transpose();

                    point_segment_pair p1;
                    p1.slave_index = slave_surfaces_global_index[k][i];
                    p1.master_index_1 = -1;
                    p1.master_index_2 = -1;
                    p1.index = 0;

                    if(use_NTS_AR)
                        p1.scale = 0.5*doublearea(slave_surfaces_global_index[k][i]);
                    else
                        p1.scale = 1;

                    boundary_info.push_back(p1);
                }
            }
            else
            {
                auto it = slave_nodes_3d[k].begin();
                for(int i=0; i<slave_nodes_3d[k].size(); ++i)
                {
                    ipc_vertices_deformed.row(it->first);
                    query_pts.row(i) = ipc_vertices_deformed.row(it->first);

                    point_segment_pair p1;
                    p1.slave_index = it->first;
                    p1.master_index_1 = -1;
                    p1.master_index_2 = -1;
                    p1.index = 0;

                    if(use_NTS_AR)
                        p1.scale = slave_nodes_area_3d[k][it->second];
                    else
                        p1.scale = 1;

                    boundary_info.push_back(p1);
                    it++;
                }
            }
            
            evaluateImplicitPotentialKR(query_pts, true, 0, k);
            if(IMLS_BOTH)
            {
                boundary_info_start_3d[1].push_back(boundary_info.size());
                if(IMLS_3D_VIRTUAL)
                {
                    query_pts.setZero(master_surfaces_3d[k].size(),dim);
                    for(int i=0; i<master_surfaces_3d[k].size(); ++i)
                    {
                        Eigen::VectorXd vp(3);
                        vp.setZero();

                        for(int l=0; l<3; ++l)
                        {
                            int ic = master_surfaces_3d[k][i](l);
                            vp += deformed.segment<dim>(ic*dim);
                        }

                        vp.array() /= 3.0;

                        query_pts.row(i) = vp.transpose();

                        point_segment_pair p1;
                        p1.slave_index = master_surfaces_global_index[k][i];
                        p1.master_index_1 = -1;
                        p1.master_index_2 = -1;
                        p1.index = 0;

                        if(use_NTS_AR)
                            p1.scale = 0.5*doublearea(master_surfaces_global_index[k][i]);
                        else
                            p1.scale = 1;

                        boundary_info.push_back(p1);
                    }
                }
                else
                {
                    query_pts.setZero(master_nodes_3d[k].size(),dim);
                    auto it = master_nodes_3d[k].begin();
                    for(int i=0; i<master_nodes_3d[k].size(); ++i)
                    {
                        query_pts.row(i) = ipc_vertices_deformed.row(it->first);

                        point_segment_pair p1;
                        p1.slave_index = it->first;
                        p1.master_index_1 = -1;
                        p1.master_index_2 = -1;
                        p1.index = 1;

                        if(use_NTS_AR)
                            p1.scale = master_nodes_area_3d[k][it->second];
                        else
                            p1.scale = 1;

                        boundary_info.push_back(p1);
                        //std::cout<<"Creation: "<<p1.slave_index<<" "<<p1.index<<std::endl;
                        it++;
                    }
                }
                evaluateImplicitPotentialKR(query_pts, true, 1, k);
            }
        }
        else
        {
            Eigen::MatrixXd query_pts(slave_nodes_3d[k].size(),dim);
            // Build Query Matrix
            if(IMLS_3D_VIRTUAL)
            {
                query_pts.setZero(slave_surfaces_3d[k].size(),dim);
                for(int i=0; i<slave_surfaces_3d[k].size(); ++i)
                {
                    Eigen::VectorXd vp(3);
                    vp.setZero();

                    for(int l=0; l<3; ++l)
                    {
                        int ic = slave_surfaces_3d[k][i](l);
                        vp += deformed.segment<dim>(ic*dim);
                    }

                    vp.array() /= 3.0;
                    query_pts.row(i) = vp.transpose();

                }
            }
            else
            {
                auto it = slave_nodes_3d[k].begin();
                for(int i=0; i<slave_nodes_3d[k].size(); ++i)
                {
                    query_pts.row(i) = ipc_vertices_deformed.row(it->first);
                    it++;
                }
            }
            evaluateImplicitPotentialKR(query_pts, true, 0, k);
        
            if(IMLS_BOTH)
            {
                if(IMLS_3D_VIRTUAL)
                {
                    query_pts.setZero(master_surfaces_3d[k].size(),dim);
                    for(int i=0; i<master_surfaces_3d[k].size(); ++i)
                    {
                        Eigen::VectorXd vp(3);
                        vp.setZero();

                        for(int l=0; l<3; ++l)
                        {
                            int ic = master_surfaces_3d[k][i](l);
                            vp += deformed.segment<dim>(ic*dim);
                        }

                        vp.array() /= 3.0;
                        query_pts.row(i) = vp.transpose();

                    }
                }
                else
                {
                    query_pts.setZero(master_nodes_3d[k].size(),dim);
                    auto it = master_nodes_3d[k].begin();
                    for(int i=0; i<master_nodes_3d[k].size(); ++i)
                    {
                        query_pts.row(i) = ipc_vertices_deformed.row(it->first);
                        it++;
                    }
                    
                }
                evaluateImplicitPotentialKR(query_pts, true, 1, k);
            }  
        }
    }
}

template <int dim>
void FEMSolver<dim>::findProjection(Eigen::MatrixXd& ipc_vertices_deformed, bool re, bool stay)
{
    //std::cout<<is_boundary.transpose()<<std::endl;
    // collision_candidates.resize(num_nodes,2);
    // collision_candidates.setZero();
    // tbb::parallel_for(0, num_nodes, [&](int i)
    // {
    //     if(is_boundary(i)){
    //         collision_candidates(i,0) = ipc_vertices_deformed(i,0);
    //         collision_candidates(i,1) = -1e-3;
    //     }
            
    // });

    if(re)
    {
         
        boundary_info.clear();
        useful_master_nodes.clear();
        if(BILATERAL){
            // Find effective master nodes;
            int slave_start = slave_nodes[0];
            int slave_end = slave_nodes.back();
            for(int i=0; i<master_nodes.size(); ++i){
                if(deformed(2*master_nodes[i]) >= deformed(2*slave_start))
                {
                    useful_master_nodes.push_back(master_nodes[i]);
                }
                if(deformed(2*master_nodes[i]) > deformed(2*slave_end)){
                    useful_master_nodes.pop_back();
                    break;
                } ;
            }

            double prev_len = 0, next_len = 0;

            for(int i=0; i<useful_master_nodes.size(); ++i)
            {
                if(i < useful_master_nodes.size() - 1)
                {
                    Eigen::VectorXd p1 = ipc_vertices_deformed.row(useful_master_nodes[i]);
                    Eigen::VectorXd p2 = ipc_vertices_deformed.row(useful_master_nodes[i+1]);
                    next_len = (p1-p2).norm();
                }else{
                    next_len = 0;
                }

                double min_dist = 1e5;
                int min_index = -1;
                Eigen::VectorXd p = ipc_vertices_deformed.row(useful_master_nodes[i]);
                for(int j=0; j<slave_segments.size(); ++j)
                {
                    Eigen::VectorXd v1 = ipc_vertices_deformed.row(slave_segments[j].first);
                    Eigen::VectorXd v2 = ipc_vertices_deformed.row(slave_segments[j].second); 
                    double dist = point2linedist(p,v1,v2, false);
                    //if(i == slave_nodes.size()-1)
                    // std::cout<<useful_master_nodes[i]<<" "<<slave_segments[j].first<<" "<<slave_segments[j].second<<" "<<dist<<std::endl;
                    // std::cout<<fabs(min_dist)-fabs(dist)<<" "<<dist<<std::endl;
                    if(fabs(min_dist) > fabs(dist) && (fabs(min_dist) - fabs(dist) > 1e-10))
                    {
                        min_dist = dist;
                        min_index = j;
                    }
                }
                
                int v1_min = slave_segments[min_index].first;
                int v2_min = slave_segments[min_index].second;
                //std::cout<<v1_min<<" "<<v2_min<<std::endl;

                VectorXvar x(6);
                x(0) = ipc_vertices_deformed(useful_master_nodes[i], 0);
                x(1) = ipc_vertices_deformed(useful_master_nodes[i], 1);
                x(2) = ipc_vertices_deformed(v1_min, 0);
                x(3) = ipc_vertices_deformed(v1_min, 1);
                x(4) = ipc_vertices_deformed(v2_min, 0);
                x(5) = ipc_vertices_deformed(v2_min, 1);
                var u = f_point2linedist(x,false);

                assert(double(u) == min_dist);

                Eigen::VectorXd g;
                Eigen::MatrixXd H = hessian(u, x, g);

                point_segment_pair p1;
                if(use_NTS_AR)
                    p1 = {useful_master_nodes[i], v1_min, v2_min, -1, RES*(prev_len+next_len)/2.0, min_dist, g, H, {}};
                else
                    p1 = {useful_master_nodes[i], v1_min, v2_min, -1, 1, min_dist, g, H, {}};

                boundary_info.push_back(p1);
                prev_len = next_len;
            }
        }
        
        double prev_len = 0, next_len = 0;
        if(USE_VIRTUAL_NODE)
        {

            for(int i=0; i<virtual_slave_nodes.size(); ++i)
            {
                int i1 = virtual_slave_nodes[i].left_index;
                int i2 = virtual_slave_nodes[i].right_index;
                double pos = virtual_slave_nodes[i].eta;
                Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);

                if(i < virtual_slave_nodes.size() - 1)
                {
                    int i1p = virtual_slave_nodes[i+1].left_index;
                    int i2p = virtual_slave_nodes[i+1].right_index;
                    double posp = virtual_slave_nodes[i+1].eta;
                    Eigen::VectorXd pp = (1-posp)/2.0*ipc_vertices_deformed.row(i1p) + (1+posp)/2.0*ipc_vertices_deformed.row(i2p);

                    next_len = (p-pp).norm();
                }else{
                    next_len = 0;
                }

                double min_dist = 1e5;
                int min_index = -1;

                for(int j=0; j<master_segments.size(); ++j)
                {
                    Eigen::VectorXd v1 = ipc_vertices_deformed.row(master_segments[j].first);
                    Eigen::VectorXd v2 = ipc_vertices_deformed.row(master_segments[j].second); 
                    double dist = point2linedist(p,v1,v2);
                    //if(i == slave_nodes.size()-1)
                    // std::cout<<slave_nodes[i]<<" "<<master_segments[j].first<<" "<<master_segments[j].second<<" "<<dist<<std::endl;
                    // std::cout<<fabs(min_dist)-fabs(dist)<<" "<<dist<<std::endl;
                    if(fabs(min_dist) > fabs(dist) && (fabs(min_dist) - fabs(dist) > 1e-10))
                    {
                        min_dist = dist;
                        min_index = j;
                    }
                }
                
                int v1_min = master_segments[min_index].first;
                int v2_min = master_segments[min_index].second;
                //std::cout<<v1_min<<" "<<v2_min<<std::endl;

                // if(slave_nodes[i] == 64 && min_dist <= 0)
                // {
                //     std::cout<<ipc_vertices_deformed.row(slave_nodes[i])<<std::endl;
                //     std::cout<<ipc_vertices_deformed.row(v1_min)<<std::endl;
                //     std::cout<<ipc_vertices_deformed.row(v2_min)<<std::endl;
                //     std::cout<<"___________________________________"<<std::endl;
                // }

                VectorXvar x(6);
                x(0) = p(0);
                x(1) = p(1);
                x(2) = ipc_vertices_deformed(v1_min, 0);
                x(3) = ipc_vertices_deformed(v1_min, 1);
                x(4) = ipc_vertices_deformed(v2_min, 0);
                x(5) = ipc_vertices_deformed(v2_min, 1);
                var u = f_point2linedist(x);

                assert(u == min_dist);

                Eigen::VectorXd g;
                Eigen::MatrixXd H = hessian(u, x, g);

                point_segment_pair p1;
                if(use_NTS_AR)
                    p1 = {-1, v1_min, v2_min,-1,RES*(prev_len+next_len)/2.0, min_dist, g, H};
                else
                    p1 = {-1, v1_min, v2_min,-1,1, min_dist, g, H};

                map_boundary_virtual[boundary_info.size()] = i;
                boundary_info.push_back(p1);

                prev_len = next_len;
                //std::cout<<slave_nodes[i]<<" "<<master_segments[min_index].first<<" "<<master_segments[min_index].second<<" "<<min_dist<<std::endl;
            }

            if(BILATERAL)
            {
                for(int i=0; i<virtual_master_nodes.size(); ++i)
                {
                    int i1 = virtual_master_nodes[i].left_index;
                    int i2 = virtual_master_nodes[i].right_index;
                    double pos = virtual_master_nodes[i].eta;
                    Eigen::VectorXd p = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);

                    if(i < virtual_master_nodes.size() - 1)
                    {
                        int i1p = virtual_master_nodes[i+1].left_index;
                        int i2p = virtual_master_nodes[i+1].right_index;
                        double posp = virtual_master_nodes[i+1].eta;
                        Eigen::VectorXd pp = (1-posp)/2.0*ipc_vertices_deformed.row(i1p) + (1+posp)/2.0*ipc_vertices_deformed.row(i2p);

                        next_len = (p-pp).norm();
                    }else{
                        next_len = 0;
                    }

                    double min_dist = 1e5;
                    int min_index = -1;

                    for(int j=0; j<slave_segments.size(); ++j)
                    {
                        Eigen::VectorXd v1 = ipc_vertices_deformed.row(slave_segments[j].first);
                        Eigen::VectorXd v2 = ipc_vertices_deformed.row(slave_segments[j].second); 
                        double dist = point2linedist(p,v1,v2,false);
                        //if(i == slave_nodes.size()-1)
                        //std::cout<<slave_nodes[i]<<" "<<master_segments[j].first<<" "<<master_segments[j].second<<" "<<dist<<std::endl;
                        // std::cout<<fabs(min_dist)-fabs(dist)<<" "<<dist<<std::endl;
                        // std::cout<<"Point: "<<i<<" segment: "<<j<<std::endl;
                        // std::cout<<p.transpose()<<std::endl;
                        // std::cout<<v1.transpose()<<std::endl;
                        // std::cout<<v2.transpose()<<std::endl;
                        if(fabs(min_dist) > fabs(dist) && (fabs(min_dist) - fabs(dist) > 1e-10))
                        {
                            min_dist = dist;
                            min_index = j;
                        }
                        //std::cout<<min_dist<<" "<<dist<<std::endl;
                    }
                    
                    int v1_min = slave_segments[min_index].first;
                    int v2_min = slave_segments[min_index].second;
                    //std::cout<<v1_min<<" "<<v2_min<<std::endl;

                    // if(slave_nodes[i] == 64 && min_dist <= 0)
                    // {
                    //     std::cout<<ipc_vertices_deformed.row(slave_nodes[i])<<std::endl;
                    //     std::cout<<ipc_vertices_deformed.row(v1_min)<<std::endl;
                    //     std::cout<<ipc_vertices_deformed.row(v2_min)<<std::endl;
                    //     std::cout<<"___________________________________"<<std::endl;
                    // }

                    VectorXvar x(6);
                    x(0) = p(0);
                    x(1) = p(1);
                    x(2) = ipc_vertices_deformed(v1_min, 0);
                    x(3) = ipc_vertices_deformed(v1_min, 1);
                    x(4) = ipc_vertices_deformed(v2_min, 0);
                    x(5) = ipc_vertices_deformed(v2_min, 1);
                    var u = f_point2linedist(x,false);

                    assert(u == min_dist);

                    Eigen::VectorXd g;
                    Eigen::MatrixXd H = hessian(u, x, g);

                    point_segment_pair p1;
                    if(use_NTS_AR)
                        p1 = {-1, v1_min, v2_min,-1,RES*(prev_len+next_len)/2.0, min_dist, g, H};
                    else
                        p1 = {-1, v1_min, v2_min,-1,1, min_dist, g, H};

                    map_boundary_virtual[boundary_info.size()] = i;
                    boundary_info.push_back(p1);

                    prev_len = next_len;
                    //std::cout<<slave_nodes[i]<<" "<<master_segments[min_index].first<<" "<<master_segments[min_index].second<<" "<<min_dist<<std::endl;
                }
            }

        }
        else
        {
            for(int i=0; i<slave_nodes.size(); ++i)
            {
                if(i < slave_nodes.size() - 1)
                {
                    Eigen::VectorXd p1 = ipc_vertices_deformed.row(slave_nodes[i]);
                    Eigen::VectorXd p2 = ipc_vertices_deformed.row(slave_nodes[i+1]);
                    next_len = (p1-p2).norm();
                }else{
                    next_len = 0;
                }

                double min_dist = 1e5;
                int min_index = -1;
                Eigen::VectorXd p = ipc_vertices_deformed.row(slave_nodes[i]);
                for(int j=0; j<master_segments.size(); ++j)
                {
                    Eigen::VectorXd v1 = ipc_vertices_deformed.row(master_segments[j].first);
                    Eigen::VectorXd v2 = ipc_vertices_deformed.row(master_segments[j].second); 
                    double dist = point2linedist(p,v1,v2);
                    //if(i == slave_nodes.size()-1)
                    // std::cout<<slave_nodes[i]<<" "<<master_segments[j].first<<" "<<master_segments[j].second<<" "<<dist<<std::endl;
                    // std::cout<<fabs(min_dist)-fabs(dist)<<" "<<dist<<std::endl;
                    if(fabs(min_dist) > fabs(dist) && (fabs(min_dist) - fabs(dist) > 1e-10))
                    {
                        min_dist = dist;
                        min_index = j;
                    }
                }
                
                int v1_min = master_segments[min_index].first;
                int v2_min = master_segments[min_index].second;
                //std::cout<<v1_min<<" "<<v2_min<<std::endl;

                // if(slave_nodes[i] == 64 && min_dist <= 0)
                // {
                //     std::cout<<ipc_vertices_deformed.row(slave_nodes[i])<<std::endl;
                //     std::cout<<ipc_vertices_deformed.row(v1_min)<<std::endl;
                //     std::cout<<ipc_vertices_deformed.row(v2_min)<<std::endl;
                //     std::cout<<"___________________________________"<<std::endl;
                // }

                VectorXvar x(6);
                x(0) = ipc_vertices_deformed(slave_nodes[i], 0);
                x(1) = ipc_vertices_deformed(slave_nodes[i], 1);
                x(2) = ipc_vertices_deformed(v1_min, 0);
                x(3) = ipc_vertices_deformed(v1_min, 1);
                x(4) = ipc_vertices_deformed(v2_min, 0);
                x(5) = ipc_vertices_deformed(v2_min, 1);
                var u = f_point2linedist(x);

                assert(u == min_dist);

                Eigen::VectorXd g;
                Eigen::MatrixXd H = hessian(u, x, g);

                point_segment_pair p1;
                if(use_NTS_AR)
                    p1 = {slave_nodes[i], v1_min, v2_min,-1,RES*(prev_len+next_len)/2.0, min_dist, g, H};
                else
                    p1 = {slave_nodes[i], v1_min, v2_min,-1,1, min_dist, g, H};
                boundary_info.push_back(p1);

                prev_len = next_len;
                //std::cout<<slave_nodes[i]<<" "<<master_segments[min_index].first<<" "<<master_segments[min_index].second<<" "<<min_dist<<std::endl;
            }
        }
        
    }else
    {
        for(int i=0; i<boundary_info.size(); ++i)
        {
            bool is_slave = true;
            if(BILATERAL)
            {
                if(!USE_VIRTUAL_NODE && i<useful_master_nodes.size()) is_slave = false;
                if(USE_VIRTUAL_NODE && i>=virtual_slave_nodes.size()) is_slave = false;
            }
                
            int p = boundary_info[i].slave_index;
            int v1 = boundary_info[i].master_index_1;
            int v2 = boundary_info[i].master_index_2;

            VectorXvar x(6);
            if(USE_VIRTUAL_NODE)
            {
                int i1 = virtual_slave_nodes[i].left_index;
                int i2 = virtual_slave_nodes[i].right_index;
                double pos = virtual_slave_nodes[i].eta;
                Eigen::VectorXd xp = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);

                if(BILATERAL && i>=virtual_slave_nodes.size())
                {
                    i1 = virtual_master_nodes[i-virtual_slave_nodes.size()].left_index;
                    i2 = virtual_master_nodes[i-virtual_slave_nodes.size()].right_index;
                    pos = virtual_master_nodes[i-virtual_slave_nodes.size()].eta;
                    xp = (1-pos)/2.0*ipc_vertices_deformed.row(i1) + (1+pos)/2.0*ipc_vertices_deformed.row(i2);
                }

                x(0) = xp(0);
                x(1) = xp(1);

            }
            else
            {
                x(0) = ipc_vertices_deformed(p, 0);
                x(1) = ipc_vertices_deformed(p, 1);
            }
            
            x(2) = ipc_vertices_deformed(v1, 0);
            x(3) = ipc_vertices_deformed(v1, 1);
            x(4) = ipc_vertices_deformed(v2, 0);
            x(5) = ipc_vertices_deformed(v2, 1);
            var u = f_point2linedist(x, is_slave);

            double min_dist_1 = u;
            Eigen::VectorXd g1;
            Eigen::MatrixXd H1 = hessian(u, x, g1);

            boundary_info[i].dist = min_dist_1;
            boundary_info[i].dist_grad = g1;
            boundary_info[i].dist_hess = H1;

            if(USE_VIRTUAL_NODE && !stay) return;

            int leftmost = master_nodes[0];
            int rightmost = master_nodes.back();
            if(BILATERAL){
                if(!USE_VIRTUAL_NODE && i<useful_master_nodes.size())
                {
                    leftmost = slave_nodes[0];
                    rightmost = slave_nodes.back();
                }

                if(USE_VIRTUAL_NODE && i>=virtual_slave_nodes.size())
                {
                    leftmost = slave_nodes[0];
                    rightmost = slave_nodes.back();
                }
            }
            
            bool success_1 = false;
            bool success_2 = false;

            VectorXvar temp(6);
            temp = x;
            //std::cout<<leftmost<< " "<<rightmost<<std::endl;

            //Right
            if(v1+1 <= rightmost){
                x(2) = ipc_vertices_deformed(v1+1, 0);
                x(3) = ipc_vertices_deformed(v1+1, 1);
                x(4) = ipc_vertices_deformed(v1, 0);
                x(5) = ipc_vertices_deformed(v1, 1);
                success_1 = true;
            }
            u = f_point2linedist(x, is_slave);
            double min_dist_2 = u;
            Eigen::VectorXd g2;
            Eigen::MatrixXd H2 = hessian(u, x, g2);

            x = temp;
            // Left
            if(v2-1 >= leftmost){
                x(2) = ipc_vertices_deformed(v2, 0);
                x(3) = ipc_vertices_deformed(v2, 1);
                x(4) = ipc_vertices_deformed(v2-1, 0);
                x(5) = ipc_vertices_deformed(v2-1, 1);
                success_2 = true;
            }

            u = f_point2linedist(x, is_slave);
            double min_dist_3 = u;
            Eigen::VectorXd g3;
            Eigen::MatrixXd H3 = hessian(u, x, g3);

            // if(p == 11 || p == 12)
            // {
            //     std::cout<<" -----------------------------------------"<<std::endl; 
            //     std::cout<<p<<" "<<v1<<" "<<v2<<" "<<success_1<<" "<<success_2<<std::endl;
            //     std::cout<<min_dist_1<<" "<<min_dist_2<<" "<<min_dist_3<<std::endl;
            //     std::cout<<" -----------------------------------------"<<std::endl; 
            // }

            if(fabs(min_dist_2) < fabs(min_dist_1) && fabs(min_dist_2) < fabs(min_dist_3) && success_1)
            {
                // std::cout<<" "<<p<<" "<<v1+1<<" "<<v1<<std::endl;
                // std::cout<<ipc_vertices_deformed.row(p)<<std::endl;
                // std::cout<<ipc_vertices_deformed.row(v1+1)<<std::endl;
                // std::cout<<ipc_vertices_deformed.row(v1)<<std::endl;

                // std::cout<<std::endl;

                boundary_info[i].master_index_1 = v1+1;
                boundary_info[i].master_index_2 = v1;
                boundary_info[i].dist = min_dist_2;
                boundary_info[i].dist_grad = g2;
                boundary_info[i].dist_hess = H2;
            }
            else if(fabs(min_dist_3) < fabs(min_dist_1) && fabs(min_dist_3) < fabs(min_dist_2) && success_2)
            {
                boundary_info[i].master_index_1 = v2;
                boundary_info[i].master_index_2 = v2-1;
                boundary_info[i].dist = min_dist_3;
                boundary_info[i].dist_grad = g3;
                boundary_info[i].dist_hess = H3;
            }
            else
            {
                boundary_info[i].dist = min_dist_1;
                boundary_info[i].dist_grad = g1;
                boundary_info[i].dist_hess = H1;
            }
        }
        
    } 
    //std::cout<<"Distance: "<<boundary_info[0].dist<<" "<<boundary_info[0].master_index_1<<" "<<boundary_info[0].master_index_2<<std::endl;
}

template <int dim>
void FEMSolver<dim>::addIPC2DEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = num_nodes;
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    if(dim == 3)
    {
        findProjectionIMLSMultiple3D(ipc_vertices_deformed);
    }
    if(dim == 2)
    {
        if(use_multiple_pairs)
        {
            findProjectionIMLSMultiple(ipc_vertices_deformed);
        }
        else if(USE_IMLS)
            findProjectionIMLS(ipc_vertices_deformed);
        else
            findProjection(ipc_vertices_deformed);
    }
    
    if(USE_VIRTUAL_NODE)
        contact_energy = barrier_weight * compute_vts_potential2D(ipc_vertices_deformed);
    else
        contact_energy = barrier_weight * compute_barrier_potential2D(ipc_vertices_deformed);

    energy += contact_energy;
}

template <int dim>
void FEMSolver<dim>::addIPC2DForceEntries(VectorXT& residual, double re)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = num_nodes;
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    if(dim == 3)
    {
        findProjectionIMLSMultiple3D(ipc_vertices_deformed);
    }
    if(dim == 2)
    {
         if(use_multiple_pairs)
        {
            findProjectionIMLSMultiple(ipc_vertices_deformed);
        }
        else if(USE_IMLS)
            findProjectionIMLS(ipc_vertices_deformed);
        else
            findProjection(ipc_vertices_deformed);
    }

    Eigen::VectorXd contact_gradient;
    if(USE_VIRTUAL_NODE)
        contact_gradient = barrier_weight * compute_vts_potential_gradient2D(ipc_vertices_deformed);
    else
        contact_gradient = barrier_weight * compute_barrier_potential_gradient2D(ipc_vertices_deformed);
    
    residual.segment(0, num_nodes_all * dim) += -contact_gradient.segment(0, num_nodes_all * dim);
}

template <int dim>
void FEMSolver<dim>::addIPC2DHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION) num_nodes_all += num_nodes;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    if(dim == 3)
    {
        findProjectionIMLSMultiple3D(ipc_vertices_deformed);
    }
    if(dim == 2)
    {
         if(use_multiple_pairs)
        {
            findProjectionIMLSMultiple(ipc_vertices_deformed);
        }
        else if(USE_IMLS)
            findProjectionIMLS(ipc_vertices_deformed);
        else
            findProjection(ipc_vertices_deformed);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Time difference (Build System Matrix) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    StiffnessMatrix contact_hessian;
    if(USE_VIRTUAL_NODE)
        contact_hessian = barrier_weight *  compute_vts_potential_hessian2D(ipc_vertices_deformed);
    else
         contact_hessian = barrier_weight *  compute_barrier_potential_hessian2D(ipc_vertices_deformed);

    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    //std::cout << "Time difference (Build System Matrix) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end).count() << "[ms]" << std::endl;    
    //std::cout<<contact_hessian.rows()<<" "<<contact_hessian.cols()<<std::endl;

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    //std::cout << "Time difference (Build System Matrix) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2).count() << "[ms]" << std::endl;
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void FEMSolver<dim>::compute1PKStress()
{
    #if USE_QUAD_ELEMENT
    OnePKStress.setZero(int(indices_quad.size()/(dim + 2)));
    for(int i=0; i<int(indices_quad.size()/(dim + 2)); ++i)
    {
        Eigen::MatrixXd x_deformed(4,2);
        Eigen::MatrixXd x_undeformed(4,2);
        for(int j=0; j<4; ++j)
        {
            for(int k=0; k<2; ++k)
            {
                int index = indices_quad(4*i+j);
                //std::cout<<index<<" ";
                x_deformed(j,k) = deformed(dim*index+k);
                x_undeformed(j,k) = undeformed(dim*index+k);
            }
           
        }
        //std::cout<<std::endl;
        double norm = compute1PKStressImplQuad(E, nu, x_deformed, x_undeformed);
        if(std::isnan(norm))
        {
            
            std::cout<<x_deformed<<std::endl;
            std::cout<<"-----------------------------"<<std::endl;
            std::cout<<x_undeformed<<std::endl;
            std::cout<<"-----------------------------"<<std::endl;
        }
        OnePKStress(i) = norm;
    }
    #else 
    OnePKStress.setZero(int(indices.size()/(dim + 1)));
    for(int i=0; i<int(indices.size()/(dim + 1)); ++i)
    {
        Eigen::MatrixXd x_deformed(3,2);
        Eigen::MatrixXd x_undeformed(3,2);
        for(int j=0; j<3; ++j)
        {
            for(int k=0; k<2; ++k)
            {
                int index = indices(3*i+j);
                x_deformed(j,k) = deformed(dim*index+k);
                x_undeformed(j,k) = undeformed(dim*index+k);
            }
           
        }
        double norm = compute1PKStressImpl(E, nu, x_deformed, x_undeformed);
        OnePKStress(i) = norm;
    }
    #endif
}

template <int dim>
void FEMSolver<dim>::computeCauchyStress()
{
    #if USE_QUAD_ELEMENT
    CauchyStressTensor.resize(int(indices_quad.size()/(dim + 2)));
    for(int i=0; i<int(indices_quad.size()/(dim + 2)); ++i)
    {
        Eigen::MatrixXd x_deformed(4,2);
        Eigen::MatrixXd x_undeformed(4,2);
        for(int j=0; j<4; ++j)
        {
            for(int k=0; k<2; ++k)
            {
                int index = indices_quad(4*i+j);
                x_deformed(j,k) = deformed(dim*index+k);
                x_undeformed(j,k) = undeformed(dim*index+k);
            }
           
        }
        //std::cout<<std::endl;
        Eigen::MatrixXd sigma = computeCauchyStressImplQuad(E, nu, x_deformed, x_undeformed);
        CauchyStressTensor[i] = sigma;
    }

    #endif

    // for(int i=0; i<boundary_info.size(); ++i)
    // {
    //     std::cout<<boundary_info[i].slave_index<<" "<<boundary_info[i].master_index_1<<" "<<boundary_info[i].master_index_2<<" "<<boundary_info[i].dist<<std::endl;
    // }

    // for(int i=0; i<master_segments.size(); ++i)
    // {
    //     std::cout<<master_segments[i].first<<" "<<master_segments[i].second<<std::endl;
    // }

    int contact_size = slave_segments.size();
    ContactTraction.resize(contact_size);
    for(int i=0; i<contact_size; ++i)
    {
        int i1 = slave_segments[i].first;
        int i2 = slave_segments[i].second;
        Eigen::VectorXd p1(2);
        p1<<deformed(2*i1), deformed(2*i1+1);
        Eigen::VectorXd p2(2);
        p2<<deformed(2*i2), deformed(2*i2+1);

        Eigen::VectorXd normal = EdgeNormal(p1,p2);
        ContactTraction[i] = fabs((CauchyStressTensor[slave_ele_indices[i]].transpose()*normal).dot(normal));
    }
}

template <int dim>
void FEMSolver<dim>::computeContactPressure()
{
    int contact_size = slave_segments.size();
    ContactPressure.setZero(contact_size);
    ContactLength.setZero(contact_size);
    ContactForce.setZero(num_nodes,dim);

    VectorXT residual(deformed.rows());
    // residual = f;
    residual.setZero();
    // VectorXT _u(deformed.rows());
    // _u.setZero();

    if(USE_MORTAR_METHOD)
        addMortarForceEntries(residual);
    else
    {
        if(USE_IPC_3D)
            addIPC3DForceEntries(residual);
        else if(USE_TRUE_IPC_2D)
            addIPC2DtrueForceEntries(residual);
        else
            addIPC2DForceEntries(residual);

        if(USE_NEW_FORMULATION)
        {
            addL2DistanceForceEntries(residual);
            addIMLSPenForceEntries(residual);
        }
    }
    // for(int i=0; i<boundary_info.size(); ++i)
    // {
    //     std::cout<<boundary_info[i].dist<<std::endl;
    // }
    double fx = 0;
    double fy = 0;
    double fz = 0;
    for(int i=0; i<num_nodes; ++i)
    {
        ContactForce.row(i) = residual.segment<dim>(dim*i);
        fx+=residual(dim*i);
        fy+=residual(dim*i+1);
        if(dim == 3)
            fz+=residual(dim*i+2);
    }

    std::ofstream file( "contactPressure.csv", std::ios::app ) ;
    // for(int i=0; i<slave_nodes.size(); ++i)
    // {
    //     double dxf = residual(2*slave_nodes[i]);
    //     double dyf = residual(2*slave_nodes[i]+1);
    //     std::cout<<"force on node "<<slave_nodes[i]<<": "<<residual(2*slave_nodes[i])<<" "<<residual(2*slave_nodes[i]+1)<<std::endl;
    //     file<<deformed(2*slave_nodes[i])<<","<<sqrt(dxf*dxf+dyf*dyf)<<std::endl;
    // }
        
    if(dim == 2)
    {
        for(int i=0; i<contact_size; ++i)
        {
            int i1 = slave_segments[i].first;
            int i2 = slave_segments[i].second;
            Eigen::VectorXd p1(2);
            p1<<deformed(2*i1), deformed(2*i1+1);
            Eigen::VectorXd p2(2);
            p2<<deformed(2*i2), deformed(2*i2+1);

            Eigen::VectorXd f1(2);
            f1<<residual(2*i1), residual(2*i1+1);
            Eigen::VectorXd f2(2);
            f2<<residual(2*i2), residual(2*i2+1);

            // int num_x_2 = WIDTH_2*SCALAR*RES;
            // ContactLength(i) = (WIDTH_2)/num_x_2;
            ContactLength(i) = (p1-p2).norm();
            if(i == 0)
                ContactPressure(i) = (f1.norm()+f2.norm()/2.)/(ContactLength(i));
            else if(i == contact_size-1)
                ContactPressure(i) = (f1.norm()/2.+f2.norm())/(ContactLength(i));
            else
                ContactPressure(i) = (f1.norm()/2.+f2.norm()/2.)/(ContactLength(i));
            std::cout<<"Pressure Between "<<i1<<" and "<<i2<<" is "<<ContactPressure(i)<<" with length: "<<ContactLength(i)<<std::endl;
            double p0 = 836.4102865;
            double a = 0.15222667;
            double p = 0;
            double r = (p1+p2)(0)/2.0;
            if(fabs(r) < a) p = p0*sqrt(1-(r/a)*(r/a));
            file<<r<<",,"<<ContactPressure(i)<<",,"<<p<<std::endl;
        }
        //std::cout<<"Average Pressure: "<<ContactPressure.sum()/ContactLength.sum()<<std::endl;
        std::cout<<ContactPressure.transpose()<<std::endl;
        Eigen::VectorXd ContactPressure_middle(contact_size-2);

        int num_nodes_upper = master_nodes.size();
        ContactPenetration.setZero(num_nodes_upper);
        for(int i=0; i<master_nodes.size(); ++i)
        {
            ContactPenetration(i) = deformed(dim*master_nodes[i]+1);
        }
        std::cout<<std::endl;
        double penetrations_avg = ContactPenetration.sum()/num_nodes_upper;
        for(int i=0; i<contact_size-2; ++i)
        {
            ContactPressure_middle(i) = ContactPressure(i+1);
        }
        std::cout<<"Pressure Var: "<<std::sqrt(computeVariance(ContactPressure))<<std::endl;
        std::cout<<"Penetration Var: "<<std::sqrt(computeVariance(ContactPenetration))<<std::endl;
        
        
    }
    else
    {
        // Compute per face area
        Eigen::MatrixXd Vs_deformed(num_nodes,dim);
        Eigen::MatrixXd Ns;
        Eigen::VectorXd As;
        // Eigen::VectorXd Z(3);
        // Z<<0,0,0;
        igl::per_face_normals_stable(VS,FS,Ns);
        igl::doublearea(VS,FS,As);

        for(int i=0; i<num_nodes; ++i)
        {
            Vs_deformed.row(i) = deformed.segment<dim>(dim*i);
        }
        Eigen::SparseMatrix<double> mass;
        igl::massmatrix(Vs_deformed,FS,igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);

        Eigen::MatrixXd Force_per_face(FS.rows(),3);
        Force_per_face.setZero();
        Eigen::VectorXd Force_per_face_norm(FS.rows(),1);
        Force_per_face_norm.setZero();

        for(int i=0; i<num_nodes; ++i)
        {
            Eigen::VectorXd f_i = ContactForce.row(i);
            if(f_i.norm() != 0) std::cout<<i<<" has force "<<f_i.transpose()<<std::endl;
            double area = 0;
            for(int j=0; j<vertex_triangle_indices[i].size(); ++j)
            {
                int face_id = vertex_triangle_indices[i][j];
                area += As(face_id);
            }

            for(int j=0; j<vertex_triangle_indices[i].size(); ++j)
            {
                int face_id = vertex_triangle_indices[i][j];
                Force_per_face.row(face_id) += f_i/double(vertex_triangle_indices[i].size());
                Force_per_face_norm(face_id) += f_i.norm()/double(vertex_triangle_indices[i].size());
            }
        }
        std::string file_name = "face_pressure_IMLS.csv";
        if(USE_IPC_3D) file_name = "face_pressure_IPC.csv";

        std::ofstream file( file_name, std::ios::app ) ;

        Eigen::VectorXd Pressure_per_face(FS.rows());
        
        // for(int i=0; i<FS.rows(); ++i)
        // {
        //     Force_per_face.row(i) =  2*Force_per_face.row(i).array()/As(i);
        //     Pressure_per_face(i) = Ns.row(i).dot(Force_per_face.row(i));
        //     Eigen::VectorXd c(3);
        //     c<<0,0,0;

        //     for(int j=0; j<3; ++j)
        //         c+= deformed.segment<3>(3*(FS(i,j)));
        //     c.array()/=3;

        //     double dist = sqrt(c(0)*c(0)+c(2)*c(2));
        //     std::cout<<"face: "<<i<<" pressure: "<<2*Force_per_face_norm(i)/As(i)<<std::endl;
        //     file<<i<<","<<dist<<","<<2*Force_per_face_norm(i)/As(i)<<std::endl;
        // }
        for(auto it = slave_nodes_3d[0].begin(); it!= slave_nodes_3d[0].end(); it++)
        {
            int node_index = it->first;
            Eigen::VectorXd p = Vs_deformed.row(node_index);
            double pressure = ContactForce.row(node_index).norm()/mass.coeff(node_index,node_index);
            double dist = sqrt(p(0)*p(0)+p(2)*p(2));
            file<<it->second<<","<<dist<<","<<pressure<<std::endl;
        }
    }
    

}

template <int dim>
void FEMSolver<dim>::displayBoundaryInfo()
{
    // if(USE_MORTAR_METHOD)
    // {
    //     for(int i=0; i<slave_nodes.size(); ++i)
    //     {
    //         std::cout<<slave_nodes[i]<<" "<<mortar.gap_functions[i].return_value<<std::endl;
    //     }
    // }
    // else
    // {
    //     //computeContactPressure();
    //     for(int i=0; i<boundary_info.size(); ++i)
    //     {
    //         if(boundary_info[i].dist == 1e8) continue;
    //         std::cout<<i<<" Boundary Index "<<boundary_info[i].slave_index<<" "<<boundary_info[i].master_index_1<<" "<<boundary_info[i].master_index_2<<" "<<boundary_info[i].dist<<std::endl;
    //         if(IMLS_3D_VIRTUAL)
    //         {
    //             int face_id = boundary_info[i].slave_index;
    //             Eigen::VectorXd vp(3);
    //             vp<<0,0,0;
    //             for(int j=0; j<3; ++j)
    //             {
    //                 vp += deformed.segment<dim>(dim*FS(face_id,j));
    //             }
    //             vp.array() /= 3;
    //             std::cout<<"slave pos: "<<vp.transpose()<<std::endl;
    //         }
    //         else if(!USE_VIRTUAL_NODE)
    //         {
    //             std::cout<<"slave pos: "<<deformed.segment<dim>(dim*boundary_info[i].slave_index).transpose()<<std::endl;
    //             //std::cout<<"Contact Force: "<<ContactForce.row(boundary_info[i].slave_index)<<std::endl;
    //         }
                
    //         if(USE_VIRTUAL_NODE)
    //         {
    //             if(map_boundary_virtual.find(i) != map_boundary_virtual.end())
    //             {
    //                 int i1 = virtual_slave_nodes[map_boundary_virtual[i]].left_index;
    //                 int i2 = virtual_slave_nodes[map_boundary_virtual[i]].right_index;
    //                 double pos = virtual_slave_nodes[map_boundary_virtual[i]].eta;
    //                 Eigen::VectorXd p = (1-pos)/2.0*deformed.segment<2>(2*i1) + (1+pos)/2.0*deformed.segment<2>(2*i2);

    //                 std::cout<<"slave pos: "<<p.transpose()<<std::endl;
    //             }
    //         }
            
            
    //         // Eigen::VectorXd p1 = deformed.segment<2>(2*boundary_info[i].master_index_1);
    //         // Eigen::VectorXd p2 = deformed.segment<2>(2*boundary_info[i].master_index_2);
    //         // std::cout<<v.transpose()<<std::endl;
    //         // std::cout<<boundary_info[i].dist_grad_transpose()<<std::endl;
    //         //std::cout<<boundary_info[i].dist_hess.transpose()<<std::endl;
    //         // std::cout<<p1.transpose()<<std::endl;
    //         // std::cout<<p2.transpose()<<std::endl;
    //         // std::cout<<point2linedist(v, p1, p2)<<std::endl;
    //     }
    // }
    T energy;
    addFastIMLSSCEnergy(energy);
    VectorXT f(num_nodes*dim);
    // f.setZero();
    // addFastIMLSSCForceEntries(f);
    // std::cout<<"Contact Force: "<<f.norm()<<std::endl;
    // for(int i=0; i<dist_info.size(); ++i)
    // {
    //     std::cout<<"Node: "<<dist_info[i].first<<" dist: "<< dist_info[i].second<<std::endl;
    //     if(dist_info[i].second<0){
    //         std::cout<<deformed.segment<3>(3*dist_info[i].first).transpose()<<std::endl;
    //         std::cout<<"Node: "<<dist_info[i].first<<" dist: "<< dist_info[i].second<<std::endl;
    //     } 
    // }
}

template <int dim>
void FEMSolver<dim>::checkDeformationGradient()
{
    std::cout<<"Check Deformation Gradient"<<std::endl;
    for(int j=0; j<num_ele_quad; ++j)
    {
        double lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        double mu = E / 2.0 / (1.0 + nu);

        Eigen::MatrixXd basis(4,2);
        basis<< -1.0/sqrt(3.0), -1.0/sqrt(3.0),
                1.0/sqrt(3.0), -1.0/sqrt(3.0),
                1.0/sqrt(3.0), 1.0/sqrt(3.0),
                -1.0/sqrt(3.0), 1.0/sqrt(3.0);

        double result = 0.0;

        Eigen::MatrixXd x(4,2);
        Eigen::MatrixXd X(4,2);
        for(int i=0; i<4; ++i)
        {
            x.row(i) = deformed.segment<2>(2*indices_quad(4*j+i));
            X.row(i) = undeformed.segment<2>(2*indices_quad(4*j+i));
        }

        for(int i=0; i<4; ++i)
        {
            Eigen::Matrix<double, 4, 2> dNdb;
            dNdb << -0.25*(1.0-basis(i,1)), -0.25*(1.0-basis(i,0)), 
                    0.25*(1.0-basis(i,1)), -0.25*(1.0+basis(i,0)),
                    0.25*(1.0+basis(i,1)), 0.25*(1.0+basis(i,0)),
                    -0.25*(1.0+basis(i,1)), 0.25*(1.0-basis(i,0));
            
            Eigen::MatrixXd dXdb = X.transpose() * dNdb;
            Eigen::MatrixXd dxdb = x.transpose() * dNdb;
            Eigen::MatrixXd F = dxdb * dXdb.inverse();

            double J = F.determinant();
            if(J < 0) std::cout<<"Flipped Triangle!!"<<std::endl;
        }
    }
}

template <int dim>
void FEMSolver<dim>::findContactMasterIPC(std::vector<double>& master_contact_nodes)
{
    
    std::unordered_map<int,int> temp;
    master_contact_nodes.clear();

    for(int i=0; i<boundary_info.size(); ++i)
    {
        int master_node_1 = boundary_info[i].master_index_1;
        int master_node_2 = boundary_info[i].master_index_2;

        if(temp.find(master_node_1) == temp.end())
        {
            temp[master_node_1] = temp.size();
        }

        if(temp.find(master_node_2) == temp.end())
        {
            temp[master_node_2] = temp.size();
        }
    }

    for(auto it = temp.begin(); it != temp.end(); it++)
    {
        master_contact_nodes.push_back(deformed(2*it->first+1));
    }
}

template <int dim>
double FEMSolver<dim>::checkMasterVariance()
{
    std::vector<double> master_contact_nodes;
    
    // if(USE_MORTAR_METHOD)
    // {
    //     mortar.findContactMaster(master_contact_nodes);
    // }
    // else
    //     findContactMasterIPC(master_contact_nodes);

    //Eigen::VectorXd master_y_data(master_contact_nodes.size());
    Eigen::VectorXd master_y_data(master_nodes.size());
    VectorXT residual_ipc(deformed.rows());
    residual_ipc.setZero();
    if(USE_MORTAR_METHOD)
        addMortarForceEntries(residual_ipc);
    else if(USE_TRUE_IPC_2D)
        addIPC2DtrueForceEntries(residual_ipc);
    else
        addIPC2DForceEntries(residual_ipc);
    if(USE_NEW_FORMULATION)
    {
        addL2DistanceForceEntries(residual_ipc);
        addIMLSPenForceEntries(residual_ipc);
    }

    
    std::vector<double> correct = {3000,6000,7000,6000,2000};
    
    
    for(int i=0; i<master_nodes.size(); ++i)
    {
        int j = i;
        if(USE_IMLS) j = master_nodes.size()-1-i;
        //std::cout<<master_contact_nodes[i]<<" ";
        master_y_data(i) = deformed(2*master_nodes[i]+1);
        std::cout<<correct[i]+residual_ipc(2*master_nodes[j]+1)<<std::endl;
    }
    std::cout<<std::endl;
    double var = computeVariance(master_y_data);
    std::cout<<"Master Y variance: "<<var<<std::endl;

    return var;
}


template class FEMSolver<2>;
template class FEMSolver<3>;