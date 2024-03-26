#include "../include/FEMSolver.h"
#include <cppad/example/cppad_eigen.hpp>
namespace 
{
    using CppAD::AD;
    using CppAD::pow;
    using CppAD::exp;
    using CppAD::sqrt;
    typedef CppAD::eigen_vector<AD<double>> a_vector;
    typedef Eigen::Matrix<AD<double>, 3, 1> a_vector3;

    // define y(x) = Poly(a, x) in the empty namespace
    AD<double> IMLSImpl(std::vector<std::vector<int>>& normal_pairs, std::vector<int>& results_index, const std::vector<AD<double>> &s, double radius)
    {     
        // int normal_num = 6;
        // std::vector<double> normal_weights = {0.1,0.1,0.3,0.3,0.1,0.1};

        // int Num = normal_pairs.size();
        // int num_all = s.size()/2-1;

        // CPPAD_TESTVECTOR(AD<double>) n(2*Num), pts(2*Num), pts_all(2*num_all);
        // CPPAD_TESTVECTOR(AD<double>) x(2);

        // x[0] = s[0];
        // x[1] = s[1];

        // for(int j=0; j<Num; ++j)
        // {
        //     pts[2*j] = s[2*results_index[j]+2];
        //     pts[2*j+1] = s[2*results_index[j]+3];
        // }

        // for(int j=0; j<num_all; ++j)
        // {
        //     pts_all[2*j] = s[2*j+2];
        //     pts_all[2*j+1] = s[2*j+3];
        // }

        // for(int j=0; j<Num; ++j)
        // {
        //     n[2*j] = 0;
        //     n[2*j+1] = 0;
        //     for(int k=0; k<normal_num; ++k)
        //     {
        //         int index_1 = normal_pairs[j][k];
        //         int index_2 = normal_pairs[j][k+1];

        //         AD<double> x0 = pts_all[2*index_1]; AD<double> y0 = pts_all[2*index_1+1];
        //         AD<double> x1 = pts_all[2*index_2]; AD<double> y1 = pts_all[2*index_2+1];

        //         AD<double> dx1 = x1-x0;
        //         AD<double> dy1 = y1-y0;
        //         AD<double> n1x = dy1/sqrt(dx1*dx1+dy1*dy1);
        //         AD<double> n1y = -dx1/sqrt(dx1*dx1+dy1*dy1);
        //         n[2*j] += normal_weights[k]*n1x;
        //         n[2*j+1] += normal_weights[k]*n1y;
        //     }
        // }

        // int max_iter = 3;
        // int max_iter_2 = 1;
        // double threshold = 1e-5;
        // double sigma_r = 0.5;
        // double sigma_n = 0.5;

        // std::vector<AD<double>> original(2);
        // original[0] = x[0]; original[1] = x[1];

        AD<double> f = 0, grad_fx = 0, grad_fy = 0;
        // while(true)
        // {
        //     //std::cout<<x[0]<<" "<<x[1]<<std::endl;
        //     for(int iter = 0; iter<max_iter; ++iter)
        //     {
        //         //std::cout<<iter_o<<" "<<iter<<std::endl;
        //         AD<double> sumW = 0, sumGwx = 0, sumGwy = 0, sumF = 0, sumGfx = 0, sumGfy = 0, sumNx = 0, sumNy = 0;
        //         for(int j=0; j<Num; ++j)
        //         {
        //             AD<double> pxx = x[0]-pts[2*j];
        //             AD<double> pxy = x[1]-pts[2*j+1];
        //             AD<double> fx = pxx*n[2*j] + pxy*n[2*j+1];

        //             AD<double> alpha = 1;
        //             if(iter > 0)
        //             {
        //                 AD<double> a1 = exp(-((fx-f)/sigma_r)*((fx-f)/sigma_r));
        //                 AD<double> norm_sqrt = (n[2*j]- grad_fx)*(n[2*j]- grad_fx) + (n[2*j+1]- grad_fy)*(n[2*j+1]- grad_fy);
        //                 AD<double> a2 = exp(-(norm_sqrt/sigma_n)*(1.0/sigma_n));
        //                 alpha = a1*a2;
        //             }

        //             AD<double> phi = pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 4);
        //             AD<double> dphidx = 4*pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 3)*(-1/(radius*radius))*2*pxx;
        //             AD<double> dphidy = 4*pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 3)*(-1/(radius*radius))*2*pxy;

        //             AD<double> w = alpha*phi;
        //             AD<double> grad_wx = alpha*dphidx;
        //             AD<double> grad_wy = alpha*dphidy;

        //             sumW += w;
        //             sumGwx += grad_wx;
        //             sumGwy += grad_wy;
        //             sumF += w*fx;
        //             sumGfx += grad_wx*fx;
        //             sumGfy += grad_wy*fx;
        //             sumNx += w*n[2*j];
        //             sumNy += w*n[2*j+1];

        //             // int j = results[i];
        //             // var res_dists_squared = (s(0)- s(2*j+2))*(s(0)- s(2*j+2)) + (s(1)- s(2*j+3))*(s(1)- s(2*j+3));
        //             // var phi_x = pow((1- (res_dists_squared)/(radius*radius)), 4);
        //             // lower += phi_x;
        //             // var dot_value = (s(0) - s(2*j+2))*n(2*j) + (s(1)- s(2*j+3))*n(2*j+1);
        //             // upper += dot_value*phi_x;
        //             //std::cout<<s[0]<< " "<<s[1]<< " "<<alpha<<" "<<phi<<std::endl;
        //         }
                
        //         f = sumF/(sumW+1e-7);
        //         grad_fx = (sumGfx-f*sumGwx+sumNx)/(sumW+1e-7);
        //         grad_fy = (sumGfy-f*sumGwy+sumNy)/(sumW+1e-7);

        //         //std::cout<<f<< " "<<grad_fx<<" "<<grad_fy<<std::endl;
        //     }
        //     x[0] = x[0]-f*grad_fx;
        //     x[1] = x[1]-f*grad_fy;

        //     //std::cout<<sqrt(f*f*grad_fx*grad_fx+f*f*grad_fy*grad_fy)<<std::endl;
        //     //if(sqrt(f*f*grad_fx*grad_fx+f*f*grad_fy*grad_fy) < 1e-3) break;
        //     break;
            
        // }
        //std::cout<<std::endl;
        return f;
    }

    AD<double> IMLSImpl3D(std::vector<std::vector<int>>& normal_pairs, std::vector<int>& results_index, const std::vector<AD<double>> &s, double radius, std::vector<std::vector<Eigen::VectorXd>> normal_pairs_3d, int index)
    {   
        // int Num = normal_pairs.size();
        // int num_all = s.size()/3-1;

        // std::vector<a_vector3> n(Num), pts(Num), pts_all(num_all);
        // a_vector3 x(3);

        // for(int i=0; i<3; ++i)
        //     x(i) = s[i];

        // for(int j=0; j<Num; ++j)
        // {
        //     for(int i=0; i<3; ++i)
        //         pts[j](i) = s[3*results_index[j]+3+i];
            
        // }

        // for(int j=0; j<num_all; ++j)
        // {
        //     for(int i=0; i<3; ++i)
        //         pts_all[j](i) = s[3*j+3+i];
        // }
        // // if(x(0) == 0 && x(2) == 0)
        // // {
        // //     std::cout<<"cur ptr: "<<x.transpose()<<std::endl;
        // // }

        // for(int j=0; j<Num; ++j)
        // {
        //     for(int i=0; i<3; ++i)
        //         n[j](i) = 0;

        //     AD<double> area = 0;
        //     a_vector3 normals;
        //     for(int i=0; i<3; ++i)
        //         normals(i) = 0;
            
        //     for(int tri=0; tri<normal_pairs_3d[j].size(); ++tri)
        //     {
        //         Eigen::VectorXd tri_ids = normal_pairs_3d[j][tri];
        //         std::vector<a_vector3> tri_pts(3);
        //         for(int k=0; k<3; ++k)
        //         {
        //             tri_pts[k]= pts_all[tri_ids(k)];
        //         }

        //         // if(x(0) == 0 && x(2) == 0)
        //         // {
        //         //    std::cout<<"tri: "<<tri<<std::endl;
        //         //    std::cout<<tri_pts[0].transpose()<<std::endl;
        //         //    std::cout<<tri_pts[1].transpose()<<std::endl;
        //         //    std::cout<<tri_pts[2].transpose()<<std::endl;
        //         // }

        //         a_vector3 u, v;
        //         u =  tri_pts[1] - tri_pts[0]; 
        //         v =  tri_pts[2] - tri_pts[0]; 

        //         AD<double> darea = 0.5*(u.cross(v)).norm();
        //         a_vector3 dnorm = u.cross(v);
        //         if(index == 0 && dnorm(1) < 0) dnorm = -dnorm;
        //         if(index == 1 && dnorm(1) > 0) dnorm = -dnorm;

        //         // std::cout<<"tri_ids: "<<tri_ids.transpose()<<std::endl;
        //         // std::cout<<"tri1: "<<tri_pts[0].transpose()<<std::endl;
        //         // std::cout<<"tri2: "<<tri_pts[1].transpose()<<std::endl;
        //         // std::cout<<"tri3: "<<tri_pts[2].transpose()<<std::endl;
        //         // std::cout<<"u: "<<u.transpose()<<std::endl;
        //         // std::cout<<"v: "<<v.transpose()<<std::endl;
        //         // std::cout<<"n: "<<dnorm.transpose()<<std::endl;
        //         dnorm.normalize();

        //         area += darea;
        //         normals += dnorm*darea;
        //     }
        //     n[j] = normals/area;
        //     // if(x(0) == 0 && x(2) == 0)
        //     // {
        //     //     std::cout<<n[j].transpose()<<std::endl;
        //     // }

        //     //std::cout<<n[j].transpose()<<std::endl;
        // }
        // if(x(0) == 0 && x(2) == 0)
        // {
        //     std::cout<<"Finished!"<<std::endl;
        // }

        int max_iter = 1;
        int max_iter_2 = 0;
        double threshold = 1e-5;
        double sigma_r = 0.5;
        double sigma_n = 0.5;

        AD<double> f = 0;
        // a_vector3 grad_f; grad_f(0) = 0; grad_f(1) = 0; grad_f(2) = 0;
        
        // for(int iter = 0; iter<max_iter; ++iter)
        // {
        //     //std::cout<<iter_o<<" "<<iter<<std::endl;
        //     AD<double> sumW = 0, sumF = 0;
        //     a_vector3 sumGw, sumGf, sumN;
        //     for(int i=0; i<3; ++i)
        //     {
        //         sumGw(i) = 0;
        //         sumGf(i) = 0;
        //         sumN(i) = 0;
        //     }

        //     for(int j=0; j<Num; ++j)
        //     {

        //         a_vector3 px;
        //         for(int i=0; i<3; ++i)
        //             px(i) = x(i)-pts[j](i);
        //         AD<double> fx = px.dot(n[j]);

        //         AD<double> alpha = 1;
        //         if(iter > 0)
        //         {
        //             AD<double> a1 = exp(-((fx-f)/sigma_r)*((fx-f)/sigma_r));
        //             AD<double> norm_sqrt = (n[j]-grad_f).squaredNorm();
        //             AD<double> a2 = exp(-(norm_sqrt/sigma_n)*(1.0/sigma_n));
        //             alpha = a1*a2;
        //         }

        //         AD<double> phi = pow((1- (px.squaredNorm())/(radius*radius)), 4);
        //         a_vector3 dphi;
        //         dphi = 4*pow((1- (px.squaredNorm())/(radius*radius)), 3)*(-1/(radius*radius))*2*px;
                

        //         AD<double> w = alpha*phi;
        //         a_vector3 grad_w;
        //         grad_w = alpha*dphi;

        //         sumW += w;
        //         sumGw += grad_w;
        //         sumF += w*fx;
        //         sumGf += grad_w*fx;
        //         sumN += w*n[j];
        //     }
            
        //     f = sumF/(sumW+1e-10);
        //     grad_f = (sumGf-f*sumGw+sumN)/(sumW+1e-10);
        // }
        // x -= f*grad_f;
        
        return f;
    }
}

template <int dim>
double FEMSolver<dim>::evaulateIMLSCPPAD(const Eigen::VectorXd& s, std::vector<std::vector<int>>& normal_pairs, std::vector<int>& results_index, double radius, Eigen::VectorXd& grad, Eigen::MatrixXd& hess, bool update, bool only_grad, std::vector<std::vector<Eigen::VectorXd>> normal_pairs_3d, int index)
{
    // using CppAD::AD;
    // using std::vector;
    // size_t n = s.size();

    // vector<AD<double> > X(n);
    // for(int i=0; i<n; i++)
    //     X[i] = s[i];

    // //std::cout<<X.size()<<std::endl;

    // CppAD::Independent(X);

    // size_t m = 1;              
    // vector<AD<double>> Y(m); 

    // if(dim == 2)
    //     Y[0] = IMLSImpl(normal_pairs, results_index, X, radius);
    // else
    //     Y[0] = IMLSImpl3D(normal_pairs, results_index, X, radius, normal_pairs_3d, index);
    // CppAD::ADFun<double> f(X, Y);

    // if(update)
    // {
    //     vector<double> jac(n); 
    //     vector<double> x(n);       
    //     for(int i=0; i<n; i++)
    //         x[i] = s[i];          
    //     jac  = f.Jacobian(x);

        
    //     vector<double> hes(n*n); 
    //     if(!only_grad)
    //     {
    //         hes = f.Hessian(x,0);
    //     }

    //     grad.setZero(n);
    //     hess.setZero(n,n);
    //     for(int i=0; i<n; ++i)
    //     {
    //         grad(i) = jac[i];
    //         if(!only_grad)
    //         {
    //             for(int j=0; j<n; ++j)
    //             {
    //                 hess(i,j) = hes[i*n+j];
    //             }
    //         }
    //     }
    // }
    

    //return Value(Y[0]);
    return 0;
}

template class FEMSolver<2>;
template class FEMSolver<3>;