#include "../include/MortarMethodDiff.h"
#include <cppad/example/cppad_eigen.hpp>
#include <iostream>
#include <Eigen/Dense>

// const std::vector<std::pair<double,double>> IntegrationPoints3 = {
//     std::pair<double,double>(5.0/9.0, -sqrt(3.0/5.0)),
//     std::pair<double,double>(8.0/9.0, 0.0),
//     std::pair<double,double>(5.0/9.0, +sqrt(3.0/5.0))
// };

// const std::vector<std::pair<double,double>> IntegrationPoints4 = {
//     std::pair<double,double>((18.0+sqrt(30.0))/36.0, +sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0))),
//     std::pair<double,double>((18.0+sqrt(30.0))/36.0, -sqrt(3.0/7.0 - 2.0/7.0 * sqrt(6.0/5.0))),
//     std::pair<double,double>((18.0-sqrt(30.0))/36.0, +sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0))),
//     std::pair<double,double>((18.0-sqrt(30.0))/36.0, -sqrt(3.0/7.0 + 2.0/7.0 * sqrt(6.0/5.0)))
// };

// const std::vector<std::pair<double,double>> IntegrationPoints5 = {
//     std::pair<double,double>(128.0/225.0, 0),
//     std::pair<double,double>((322.0+13*sqrt(70.0))/900.0, -sqrt(5.0 - 2.0 * sqrt(10.0/7.0))/3.0),
//     std::pair<double,double>((322.0+13*sqrt(70.0))/900.0, +sqrt(5.0 - 2.0 * sqrt(10.0/7.0))/3.0),
//     std::pair<double,double>((322.0-13*sqrt(70.0))/900.0, -sqrt(5.0 + 2.0 * sqrt(10.0/7.0))/3.0),
//     std::pair<double,double>((322.0-13*sqrt(70.0))/900.0, +sqrt(5.0 + 2.0 * sqrt(10.0/7.0))/3.0)
// };

// /* generate a random floating point number from min to max */

// double gap_stiffness = 1e8;
// double hi = 1;



// int nChoosek(int n, int k)
// {
//     if (k > n) return 0;
//     if (k * 2 > n) k = n-k;
//     if (k == 0) return 1;

//     int result = n;
//     for( int i = 2; i <= k; ++i ) {
//         result *= (n-i+1);
//         result /= i;
//     }
//     return result;
// }

// namespace 
// {
//     using CppAD::AD;
//     using CppAD::pow;
//     using CppAD::exp;
//     using CppAD::sqrt;
//     using CppAD::fabs;
//     using CppAD::pow;
//     typedef CppAD::eigen_vector<AD<double>> a_vector;
    
//     std::vector<AD<double>> calculateNormalsImpl(std::unordered_map<int,std::pair<int,int>>& segments, std::vector<AD<double>>& V)
//     {
//         std::vector<AD<double>> normals(V.size(),0);
//         for(int i=0; i<segments.size(); ++i)
//         {
//             int i1 = segments[i].first;
//             int i2 = segments[i].second;

//             AD<double> dx = V[2*i2]-V[2*i1];
//             AD<double> dy = V[2*i2+1]-V[2*i1+1];

//             std::vector<AD<double>> n(2);
//             n[0] = -dy/sqrt(dx*dx+dy*dy);
//             n[1] = dx/sqrt(dx*dx+dy*dy);

//             //std::cout<<i<<" out of "<<segments.size()<<std::endl;
//             // std::cout<<"V1: "<<V[2*i1]<<" "<<V[2*i1+1]<<std::endl;
//             // std::cout<<"V2: "<<V[2*i2]<<" "<<V[2*i2+1]<<std::endl;
//             //std::cout<<"Normal: "<<n[0]<<" "<<n[1]<<std::endl;
            
//             for(int j=0; j<2; ++j)
//             {
//                 int index = i1;
//                 if(j == 1) index = i2;

//                 normals[2*index] += n[0];
//                 normals[2*index+1] += n[1];
//             }
//         }

//         for(int i=0; i<normals.size()/2; ++i)
//         {

//             AD<double> x = normals[2*i];
//             AD<double> y = normals[2*i+1];

//             normals[2*i] = x/sqrt(x*x+y*y);
//             normals[2*i+1] = y/sqrt(x*x+y*y);

//             //std::cout<<normals[2*i]<<" "<<normals[2*i+1]<<std::endl;
//         }

//         return normals;
//     }
    
//     AD<double> projectSlaveToMasterImpl(std::vector<AD<double>> input)
//     {
//         std::vector<AD<double>> slave(2);
//         slave[0] = input[0]; slave[1] = input[1];
//         std::vector<AD<double>> n(2);
//         n[0] = input[2]; n[1] = input[3];
//         std::vector<AD<double>> master_1(2);
//         master_1[0] = input[4]; master_1[1] = input[5];
//         std::vector<AD<double>> master_2(2);
//         master_2[0] = input[6]; master_2[1] = input[7];


//         AD<double> upper = n[0]*(master_1[1] + master_2[1] - 2*slave[1]) - n[1]*(master_1[0] + master_2[0] - 2*slave[0]);
//         AD<double> lower = n[0]*(master_1[1] - master_2[1]) + n[1]*(master_2[0] - master_1[0]);

//         return upper/lower;
//     }

//     AD<double> projectMasterToSlaveImpl(std::vector<AD<double>> input)
//     {
//         std::vector<AD<double>> master(2);
//         master[0] = input[0]; master[1] = input[1];
//         std::vector<AD<double>> n1(2);
//         n1[0] = input[2]; n1[1] = input[3];
//         std::vector<AD<double>> n2(2);
//         n2[0] = input[4]; n2[1] = input[5];
//         std::vector<AD<double>> slave_1(2);
//         slave_1[0] = input[6]; slave_1[1] = input[7];
//         std::vector<AD<double>> slave_2(2);
//         slave_2[0] = input[8]; slave_2[1] = input[9];

//         if ((n1[0]-n2[0])*(n1[0]-n2[0])+(n1[1]-n2[1])*(n1[1]-n2[1]) < 1e-10)
//         {
//             AD<double> upper = -2*n1[0]*master[1] + n1[0]*slave_1[1] + n1[0]*slave_2[1] + 2*n1[1]*master[0] - n1[1]*slave_1[0] - n1[1]*slave_2[0];
//             AD<double> lower = n1[0]*slave_1[1] - n1[0]*slave_2[1] - n1[1]*slave_1[0] + n1[1]*slave_2[0];
//             return upper/(lower+1e-7);
//         }


//         AD<double> b_upper = 2*n1[0]*master[1] - 2*n1[0]*slave_1[1] - 2*n1[1]*master[0] + 2*n1[1]*slave_1[0] - 2*n2[0]*master[1] + 2*n2[0]*slave_2[1] + 2*n2[1]*master[0] - 2*n2[1]*slave_2[0];
//         AD<double> b_lower = n1[0]*slave_1[1] - n1[0]*slave_2[1] - n1[1]*slave_1[0] + n1[1]*slave_2[0] - n2[0]*slave_1[1] + n2[0]*slave_2[1] + n2[1]*slave_1[0] - n2[1]*slave_2[0];
//         AD<double> c_upper = -2*n1[0]*master[1] + n1[0]*slave_1[1] + n1[0]*slave_2[1] + 2*n1[1]*master[0] - n1[1]*slave_1[0] - n1[1]*slave_2[0] - 2*n2[0]*master[1] + n2[0]*slave_1[1] + n2[0]*slave_2[1] + 2*n2[1]*master[0] - n2[1]*slave_1[0] - n2[1]*slave_2[0];
//         AD<double> c_lower = n1[0]*slave_1[1] - n1[0]*slave_2[1] - n1[1]*slave_1[0] + n1[1]*slave_2[0] - n2[0]*slave_1[1] + n2[0]*slave_2[1] + n2[1]*slave_1[0] - n2[1]*slave_2[0];

//         AD<double> a = 1.0;
//         AD<double> b = b_upper / (b_lower+1e-7);
//         AD<double> c = c_upper / (c_lower+1e-7);
//         AD<double> d = b*b - 4*a*c;

//         if (d < 0)
//         {
//             // std::cerr<<"Mortar Method Error: negative discriminant "<<d<<std::endl;
//             // std::cerr<<"xm: "<< master[0]<<" "<< master[1]<<std::endl;
//             // std::cerr<<"xs1: "<< slave_1[0]<<" "<< slave_1[1]<<std::endl;
//             // std::cerr<<"xs2: "<< slave_2[0]<<" "<< slave_2[1]<<std::endl;
//             // std::cerr<<"n1: "<< n1[0]<<" "<< n1[1]<<std::endl;
//             // std::cerr<<"n2: "<< n2[0]<<" "<< n2[1]<<std::endl;
//             // return -100;
//             return 1e8;
//         }
//         AD<double> res1 = (-b + sqrt(d))/(2.0*a);
//         AD<double> res2 = (-b - sqrt(d))/(2.0*a);

//         //std::cout<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<res1<<" "<<res2<<"\n";

//         if(fabs(res1) < fabs(res2)) return res1;
//         else return res2;
//     }

//     a_vector calculateMortarGapEnergyImpl(a_vector input)
//     {
//         a_vector slave_2(2);
//         slave_2(0) = input(2); slave_2(1) = input(3);
//         a_vector slave_1(2);
//         slave_1(0) = input(0); slave_1(1) = input(1);
//         a_vector n1(2);
//         n1(0) = input(4); n1(1) = input(5);
//         a_vector n2(2);
//         n2(0) = input(6); n2(1) = input(7);
//         a_vector master_1(2);
//         master_1(0) = input(8); master_1(1) = input(9);
//         a_vector master_2(2);
//         master_2(0) = input(10); master_2(1) = input(11);

//         AD<double> coor1 = input(12);
//         AD<double> coor2 = input(13);

//         AD<double> J = sqrt((slave_2[0]-slave_1[0])*(slave_2[0]-slave_1[0]) + (slave_2[1]-slave_1[1])*(slave_2[1]-slave_1[1]))/2.0;
//         AD<double> s = fabs(coor2-coor1)/2.0;

//         a_vector ge(2);
//         ge(0) = 0.0; ge(1) = 0.0;

//         // std::cout<<"master point 1: "<<master_1(0)<<" "<<master_1(1)<<std::endl;
//         // std::cout<<"master point 2: "<<master_2(0)<<" "<<master_2(1)<<std::endl;
//         // std::cout<<"slave point 1: "<<slave_1(0)<<" "<<slave_1(1)<<std::endl;
//         // std::cout<<"slave point 2: "<<slave_2(0)<<" "<<slave_2(1)<<std::endl;


//         for(int j=0; j<IntegrationPoints5.size(); ++j)
//         {
//             double w = IntegrationPoints5[j].first;
//             double pos = IntegrationPoints5[j].second;

//             AD<double> coor_gauss = (1-pos)/2.0*coor1 + (1+pos)/2.0*coor2;
//             a_vector N(2);
//             N(0) = (1-coor_gauss)/2.0;
//             N(1) = (1+coor_gauss)/2.0;

//             a_vector n_gauss(2);
//             n_gauss(0) = N(0)*n1(0) + N(1)*n2(0);
//             n_gauss(1) = N(0)*n1(1) + N(1)*n2(1);
//             // AD<double> norm = sqrt(n_gauss(0)*n_gauss(0)+n_gauss(1)*n_gauss(1));
//             // n_gauss(0) = n_gauss(0)/norm;
//             // n_gauss(1) = n_gauss(1)/norm;

//             a_vector slave_gauss(2);
//             slave_gauss(0) = N(0)*slave_1(0) + N(1)*slave_2(0);
//             slave_gauss(1) = N(0)*slave_1(1) + N(1)*slave_2(1);

//             std::vector<AD<double>> input2(8);
//             input2[0] = slave_gauss(0); input2[1] = slave_gauss(1);
//             input2[2] = n_gauss(0); input2[3] = n_gauss(1);
//             input2[4] = master_1(0); input2[5] = master_1(1);
//             input2[6] = master_2(0); input2[7] = master_2(1);            

//             AD<double> master_coor = projectSlaveToMasterImpl(input2);
//             a_vector N_master(2);
//             N_master(0) = (1-master_coor)/2.0;
//             N_master(1) = (1+master_coor)/2.0;
//             a_vector master_gauss(2);
//             master_gauss(0) = N_master(0)*master_1(0) + N_master(1)*master_2(0);
//             master_gauss(1) = N_master(0)*master_1(1) + N_master(1)*master_2(1);
            
//             //ge += w*N*((master_gauss-slave_gauss).dot(n_gauss))*s*J;
//             AD<double> dist = 0;
//             if((master_gauss-slave_gauss).dot(n_gauss) < 0)
//             {
//                 dist = (master_gauss-slave_gauss).dot(n_gauss);
//                 ge += -gap_stiffness*w*N*(pow((master_gauss-slave_gauss).dot(n_gauss),3))*s*J;
//             }

//             // std::cout<<j<<" "<<dist<<std::endl;
//             // std::cout<<"master gauss point: "<<master_gauss(0)<<" "<<master_gauss(1)<<std::endl;
//             // std::cout<<"slave gauss point: "<<slave_gauss(0)<<" "<<slave_gauss(1)<<std::endl;
//             // std::cout<<"difference: "<<master_gauss(0)-slave_gauss(0)<<" "<<master_gauss(1)-slave_gauss(1)<<std::endl;
//             // std::cout<<"gauss normal: "<<n_gauss(0)<<" "<<n_gauss(1)<<std::endl;
                
//         }
//         return ge;
//     }

//     a_vector calculateMortarGapEnergyTestImpl(a_vector input, std::vector<double>& SamplePoints)
//     {
//         a_vector slave_2(2);
//         slave_2(0) = input(2); slave_2(1) = input(3);
//         a_vector slave_1(2);
//         slave_1(0) = input(0); slave_1(1) = input(1);
//         a_vector n1(2);
//         n1(0) = input(4); n1(1) = input(5);
//         a_vector n2(2);
//         n2(0) = input(6); n2(1) = input(7);
//         a_vector master_1(2);
//         master_1(0) = input(8); master_1(1) = input(9);
//         a_vector master_2(2);
//         master_2(0) = input(10); master_2(1) = input(11);

//         AD<double> coor1 = input(12);
//         AD<double> coor2 = input(13);

//         AD<double> J = sqrt((slave_2[0]-slave_1[0])*(slave_2[0]-slave_1[0]) + (slave_2[1]-slave_1[1])*(slave_2[1]-slave_1[1]))/2.0;
//         AD<double> s = fabs(coor2-coor1)/2.0;

//         a_vector ge(2);
//         ge(0) = 0.0; ge(1) = 0.0;

//         // std::cout<<"master point 1: "<<master_1(0)<<" "<<master_1(1)<<std::endl;
//         // std::cout<<"master point 2: "<<master_2(0)<<" "<<master_2(1)<<std::endl;
//         // std::cout<<"slave point 1: "<<slave_1(0)<<" "<<slave_1(1)<<std::endl;
//         // std::cout<<"slave point 2: "<<slave_2(0)<<" "<<slave_2(1)<<std::endl;

//         // Generate random samples between -1 and 1


//         for(int j=0; j<SamplePoints.size(); ++j)
//         {
//             double pos = SamplePoints[j];

//             AD<double> coor_gauss = (1-pos)/2.0*coor1 + (1+pos)/2.0*coor2;
//             a_vector N(2);
//             N(0) = (1-coor_gauss)/2.0;
//             N(1) = (1+coor_gauss)/2.0;

//             a_vector n_gauss(2);
//             n_gauss(0) = N(0)*n1(0) + N(1)*n2(0);
//             n_gauss(1) = N(0)*n1(1) + N(1)*n2(1);
//             // AD<double> norm = sqrt(n_gauss(0)*n_gauss(0)+n_gauss(1)*n_gauss(1));
//             // n_gauss(0) = n_gauss(0)/norm;
//             // n_gauss(1) = n_gauss(1)/norm;

//             a_vector slave_gauss(2);
//             slave_gauss(0) = N(0)*slave_1(0) + N(1)*slave_2(0);
//             slave_gauss(1) = N(0)*slave_1(1) + N(1)*slave_2(1);

//             std::vector<AD<double>> input2(8);
//             input2[0] = slave_gauss(0); input2[1] = slave_gauss(1);
//             input2[2] = n_gauss(0); input2[3] = n_gauss(1);
//             input2[4] = master_1(0); input2[5] = master_1(1);
//             input2[6] = master_2(0); input2[7] = master_2(1);            

//             AD<double> master_coor = projectSlaveToMasterImpl(input2);
//             a_vector N_master(2);
//             N_master(0) = (1-master_coor)/2.0;
//             N_master(1) = (1+master_coor)/2.0;
//             a_vector master_gauss(2);
//             master_gauss(0) = N_master(0)*master_1(0) + N_master(1)*master_2(0);
//             master_gauss(1) = N_master(0)*master_1(1) + N_master(1)*master_2(1);
            
//             //ge += w*N*((master_gauss-slave_gauss).dot(n_gauss))*s*J;
//             AD<double> dist = 0;
//             if((master_gauss-slave_gauss).dot(n_gauss) < 0)
//             {
//                 dist = (master_gauss-slave_gauss).dot(n_gauss);
//                 ge += -2.0/double(SamplePoints.size())*gap_stiffness*N*(pow((master_gauss-slave_gauss).dot(n_gauss),3))*s*J;
//             }

//             // std::cout<<j<<" "<<dist<<std::endl;
//             // std::cout<<"master gauss point: "<<master_gauss(0)<<" "<<master_gauss(1)<<std::endl;
//             // std::cout<<"slave gauss point: "<<slave_gauss(0)<<" "<<slave_gauss(1)<<std::endl;
//             // std::cout<<"difference: "<<master_gauss(0)-slave_gauss(0)<<" "<<master_gauss(1)-slave_gauss(1)<<std::endl;
//             // std::cout<<"gauss normal: "<<n_gauss(0)<<" "<<n_gauss(1)<<std::endl;
                
//         }
//         return ge;
//     }

//     AD<double> differentiable_clampImpl(std::vector<AD<double>> x)
//     {
//         double x_max = 1;
//         double x_min = -1;

//         int N = 2;

//         if(x[0]<x_min) return x_min;
//         if(x[0]>x_max) return x_max;

//         AD<double> y = (x[0]-x_min)/(x_max-x_min);

//         AD<double> result = 0;

//         for(int i=0; i<=N; ++i)
//         {
//             result += nChoosek(N+i,i) * nChoosek(2*N+1,N-i) * pow(-y,i);
//         }
//         result *= pow(y,N+1);

//         return result*(x_max-x_min) + x_min;
//     }
// }

// void all_info_clamp(all_info& input)
// {
//     if(input.return_value < -1.0 || input.return_value > 1.0)
//     {
//         if(input.return_value < -1.0)
//             input.return_value = -1.0;
//         else
//             input.return_value = 1.0;

//         input.return_value_grad.setZero();
//         input.return_value_hess.setZero();
//     }
// }

// void all_info_differentiable_clamp(all_info& input, bool update=false)
// {
//     using CppAD::AD;
//     using std::vector;
//     size_t n = 1;

//     vector<AD<double>> X(n);
//     X[0] = input.return_value;

//     CppAD::Independent(X);

//     size_t m = n;              
//     vector<AD<double>> Y(m); 
//     Y[0] = differentiable_clampImpl(X);
//     CppAD::ADFun<double> f(X, Y);

//     Eigen::VectorXd grad;
//     Eigen::MatrixXd hess;

//     if(update)
//     {
//         vector<double> jac(m*n); 
//         vector<double> x(n);       
//         x[0] = input.return_value;    

//         jac = f.Jacobian(x);
//         vector<double> hes(n*n); 
//         hes = f.Hessian(x,0);

//         grad.setZero(n);
//         hess.setZero(n,n);
//         for(int i=0; i<n; ++i)
//         {
//             grad(i) = jac[i];
//             for(int j=0; j<n; ++j)
//             {
//                 hess(i,j) = hes[i*n+j];
//             }
//         }
//         Eigen::VectorXd old_grad = input.return_value_grad;
//         Eigen::MatrixXd old_hess = input.return_value_hess;

//         input.return_value = Value(Y[0]);
//         input.return_value_grad = grad(0) * old_grad;
//         input.return_value_hess = hess(0,0) * old_grad*old_grad.transpose() + grad(0) * old_hess;
//     }else
//     {
//         input.return_value = Value(Y[0]);
//     }
// }

// void MortarMethodDiff::testcase()
// {
//     //Test CalculateNormals()
//     // V.setZero(3,2);
//     // V(0,0) = 7.0; V(0,1) = 7.0;
//     // V(1,0) = 4.0; V(1,1) = 3.0;
//     // V(2,0) = 0.0; V(2,1) = 0.0;

//     // segments.clear();
//     // segments[0] = std::pair<int,int>(0,1);
//     // segments[1] = std::pair<int,int>(1,2);

//     // calculateNormals(true);
//     // std::vector<all_info> value(6);

//     // for(int i=0; i<6; ++i)
//     // {
//     //     std::cout<<i<<normals[i].return_value<<std::endl;
//     //     value[i] = normals[i];
//     // }

    
//     // std::cout<<"---------------------Check Gradient--------------------"<<std::endl;
//     // double eps = 1e-5;
//     // for(int i=0; i<6; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals();
//     //     for(int j=0; j<6; ++j)
//     //     {
//     //         double new_value = normals[j].return_value;
//     //         std::cout<<i<<" "<<j<<" "<<(new_value-value[j].return_value)/eps<<" "<<value[j].return_value_grad(i)<<std::endl;
//     //     }
//     //     V(i/2,i%2) -= eps;
//     // }

//     // std::cout<<"---------------------Check Hessian--------------------"<<std::endl;
//     // for(int i=0; i<6; ++i)
//     // {
//     //     for(int j=0; j<6; ++j)
//     //     {
//     //         V(j/2,j%2) += eps;
//     //         calculateNormals(true);
//     //         for(int k=0; k<6; ++k)
//     //         {
//     //             Eigen::VectorXd new_value = normals[k].return_value_grad;
//     //             std::cout<<i<<" "<<j<<" "<<k<<" "<<(new_value(i)-value[k].return_value_grad(i))/eps<<" "<<value[k].return_value_hess(i,j)<<std::endl;

//     //         }
//     //         V(j/2,j%2) -= eps;
//     //     }
//     // }

//     //Test Projections
//     // V.setZero(5,2);
//     // V(0,0) = 0.0; V(0,1) = 0.0;
//     // V(1,0) = 4.0; V(1,1) = 3.0;
//     // V(2,0) = 7.0; V(2,1) = 7.0;
//     // V(3,0) = 4.0; V(3,1) = -2.0;
//     // V(4,0) = 5.0; V(4,1) = -1.0;

//     // segments.clear();
//     // segments[0] = std::pair<int,int>(0,1);
//     // segments[1] = std::pair<int,int>(1,2);
//     // segments[2] = std::pair<int,int>(3,4);

//     // normals.clear();
//     // calculateNormals(true);
//     // std::cout<<normals[0].return_value<<" "<<normals[1].return_value<<std::endl;
//     // std::cout<<normals[2].return_value<<" "<<normals[3].return_value<<std::endl;

//     // all_info value = projectMasterToSlave(3,0,1,0,1,true);
//     // std::cout<<value.return_value<<std::endl;

//     // std::cout<<"---------------------Check Gradient--------------------"<<std::endl;
//     // double eps = 1e-5;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals();
//     //     all_info new_value = projectMasterToSlave(3,0,1,0,1);
//     //     std::cout<<i<<" "<<(new_value.return_value-value.return_value)/eps<<" "<<value.return_value_grad(i)<<std::endl;
//     //     V(i/2,i%2) -= eps;
//     // }

//     // std::cout<<"---------------------Check Hessian--------------------"<<std::endl;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals(true);
//     //     all_info new_value = projectMasterToSlave(3,0,1,0,1,true);
//     //     for(int j=0; j<8; ++j)
//     //     {
//     //         std::cout<<i<<" "<<j<<" "<<(new_value.return_value_grad(j)-value.return_value_grad(j))/eps<<" "<<value.return_value_hess(i,j)<<std::endl;
//     //     }
//     //     V(i/2,i%2) -= eps;
//     // }

//     // Test Projections
//     // V.setZero(4,2);
//     // V(0,0) = 0.0; V(0,1) = 0.0;
//     // V(1,0) = -4.0; V(1,1) = -3.0;
//     // V(2,0) = 7.0; V(2,1) = 2.0;
//     // V(3,0) = 4.0; V(3,1) = -2.0;

//     // segments.clear();
//     // segments[0] = std::pair<int,int>(0,1);
//     // segments[1] = std::pair<int,int>(2,3);

//     // normals.clear();
//     // calculateNormals(true);
//     // //std::cout<<normals[0].return_value_grad.transpose()<<" "<<normals[1].return_value_grad.transpose()<<std::endl;

//     // all_info value = projectSlaveToMaster(0,0,2,3,true);
//     // std::cout<<value.return_value<<std::endl;

//     // std::cout<<"---------------------Check Gradient--------------------"<<std::endl;
//     // double eps = 1e-5;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals();
//     //     all_info new_value = projectSlaveToMaster(0,0,2,3);
//     //     std::cout<<i<<" "<<(new_value.return_value-value.return_value)/eps<<" "<<value.return_value_grad(i)<<std::endl;
//     //     V(i/2,i%2) -= eps;
//     // }

//     // std::cout<<"---------------------Check Hessian--------------------"<<std::endl;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals(true);
//     //     all_info new_value = projectSlaveToMaster(0,0,2,3,true);
//     //     for(int j=0; j<8; ++j)
//     //     {
//     //         std::cout<<i<<" "<<j<<" "<<(new_value.return_value_grad(j)-value.return_value_grad(j))/eps<<" "<<value.return_value_hess(i,j)<<std::endl;
//     //     }
//     //     V(i/2,i%2) -= eps;
//     // }

//     //Test Compute Segments
//     // V.setZero(4,2);
//     // V(0,0) = 3.2; V(0,1) = 4.0;
//     // V(1,0) = 1.0; V(1,1) = 2.1;
//     // V(2,0) = 1.8; V(2,1) = 1.7;
//     // V(3,0) = 0.0; V(3,1) = 1.0;    

//     // segments.clear();
//     // segments[0] = std::pair<int,int>(0,1);
//     // segments[1] = std::pair<int,int>(2,3);

//     // calculateNormals(true);
//     // std::cout<<normals[0].return_value<<" "<<normals[1].return_value<<std::endl;
//     // std::cout<<normals[2].return_value<<" "<<normals[3].return_value<<std::endl;

//     // slave_indices.clear();
//     // slave_indices.push_back(0);
//     // master_indices.clear();
//     // master_indices.push_back(1);

//     // calculateSegments(true);
//     // //std::cout<<seg_info[0].size()<<std::endl;
//     // segment_info_diff value = seg_info[0][0];
    
//     // std::cout<<"---------------------Check Gradient--------------------"<<std::endl;
//     // double eps = 1e-5;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals();
//     //     calculateSegments();
//     //     segment_info_diff new_value = seg_info[0][0];
//     //     for(int j=0; j<2; ++j)
//     //         std::cout<<i<<" "<<(new_value.xs[j].return_value-value.xs[j].return_value)/eps<<" "<<value.xs[j].return_value_grad(i)<<std::endl;
//     //     V(i/2,i%2) -= eps;
//     // }

//     // std::cout<<"---------------------Check Hessian--------------------"<<std::endl;
//     // for(int i=0; i<8; ++i)
//     // {
//     //     V(i/2,i%2) += eps;
//     //     calculateNormals(true);
//     //     calculateSegments(true);
//     //     segment_info_diff new_value = seg_info[0][0];
//     //     for(int j=0; j<8; ++j)
//     //     {
//     //         for(int k=0; k<2; ++k)
//     //             std::cout<<i<<" "<<j<<" "<<(new_value.xs[k].return_value_grad(j)-value.xs[k].return_value_grad(j))/eps<<" "<<value.xs[k].return_value_hess(i,j)<<std::endl;
//     //     }
//     //     V(i/2,i%2) -= eps;
//     // }

//     //Test Compute gap functions
//     V.setZero(6,2);
//     V(0,0) = 1.2; V(0,1) = 1.5;
//     V(1,0) = 2.3; V(1,1) = 1.9;
//     V(2,0) = 2.9; V(2,1) = 1.3;
//     V(3,0) = 0.0; V(3,1) = 2.1;
//     V(4,0) = 1.8; V(4,1) = 2.3;    
//     V(5,0) = 3.5; V(5,1) = 2.7;    

//     segments.clear();
//     segments[0] = std::pair<int,int>(0,1);
//     segments[1] = std::pair<int,int>(1,2);
//     segments[2] = std::pair<int,int>(3,4);
//     segments[3] = std::pair<int,int>(4,5);

//     slave_indices.clear();
//     slave_indices.push_back(2);
//     slave_indices.push_back(3);
//     master_indices.clear();
//     master_indices.push_back(0);
//     master_indices.push_back(1);

//     MortarMethod(true);
//     //std::cout<<seg_info[0].size()<<std::endl;
//     all_info value = gap_energy;
    
    
//     std::cout<<"---------------------Check Gradient--------------------"<<std::endl;
//     double eps = 1e-5;
//     for(int i=0; i<12; ++i)
//     {
//         V(i/2,i%2) += eps;
//         MortarMethod();
//         all_info new_value = gap_energy;
//         std::cout<<i<<" "<<(new_value.return_value-value.return_value)/eps<<" "<<value.return_value_grad(i)<<std::endl;
//         V(i/2,i%2) -= eps;
//     }

//     std::cout<<"---------------------Check Hessian--------------------"<<std::endl;
//     for(int i=0; i<12; ++i)
//     {
//         V(i/2,i%2) += eps;
//         MortarMethod(true);
//         all_info new_value = gap_energy;
//         for(int j=0; j<12; ++j)
//         {
//             std::cout<<i<<" "<<j<<" "<<(new_value.return_value_grad(j)-value.return_value_grad(j))/eps<<" "<<value.return_value_hess(i,j)<<std::endl;
//         }
//         V(i/2,i%2) -= eps;
//     }
// }

void MortarMethodDiff::calculateNormals(bool update)
{
//     using CppAD::AD;
//     using std::vector;
//     size_t n = 2*V.rows();

//     vector<AD<double>> X(n);
//     for(int i=0; i<n/2; i++)
//     {
//         X[2*i] = V(i,0);
//         X[2*i+1] = V(i,1);
//     }

//     CppAD::Independent(X);

//     size_t m = n;              
//     vector<AD<double>> Y(m); 
//     Y = calculateNormalsImpl(segments, X);
//     CppAD::ADFun<double> f(X, Y);

//     Eigen::VectorXd grad;
//     Eigen::MatrixXd hess;

//     if(update)
//     {
//         vector<double> jac(m*n); 
//         vector<double> x(n);       
//         for(int i=0; i<n/2; i++)
//         {
//             x[2*i] = V(i,0);
//             x[2*i+1] = V(i,1);
//         }     
//         jac = f.Jacobian(x);

//         for(int k=0; k<m; ++k)
//         {
//             vector<double> hes(n*n); 
//             hes = f.Hessian(x,k);

//             grad.setZero(n);
//             hess.setZero(n,n);
//             for(int i=0; i<n; ++i)
//             {
//                 grad(i) = jac[i+k*n];
//                 for(int j=0; j<n; ++j)
//                 {
//                     hess(i,j) = hes[i*n+j];
//                 }
//             }
//             normals[k] = {Value(Y[k]), grad, hess};
//         }
//     }else
//     {
//         for(int k=0; k<m; ++k)
//         {
//             normals[k] = {Value(Y[k]), grad, hess};
//         }
//     }
    
}

all_info MortarMethodDiff::projectSlaveToMaster(int slave, int normal, int master_1, int master_2, bool update)
{
//     using CppAD::AD;
//     using std::vector;
//     int n = 8;

//     vector<AD<double>> X(n);
//     X[0] = V(slave,0); X[1] = V(slave,1);
//     X[2] = normals[2*normal].return_value; X[3] = normals[2*normal+1].return_value;
//     X[4] = V(master_1,0); X[5] = V(master_1,1);
//     X[6] = V(master_2,0); X[7] = V(master_2,1);

//     CppAD::Independent(X);

//     int m = 1;              
//     vector<AD<double>> Y(m); 
//     Y[0] = projectSlaveToMasterImpl(X);
//     CppAD::ADFun<double> f(X, Y);

//     Eigen::VectorXd grad_global;
//     Eigen::MatrixXd hess_global;

//     Eigen::VectorXd grad;
//     Eigen::MatrixXd hess;

    all_info result;

//     if(update)
//     {
//         vector<double> jac(m*n); 
//         vector<double> x(n);       
//         x[0] = V(slave,0); x[1] = V(slave,1);
//         x[2] = normals[2*normal].return_value; x[3] = normals[2*normal+1].return_value;
//         x[4] = V(master_1,0); x[5] = V(master_1,1);
//         x[6] = V(master_2,0); x[7] = V(master_2,1);


//         jac = f.Jacobian(x);

//         vector<double> hes(n*n); 
//         hes = f.Hessian(x,0);

//         grad.setZero(n);
//         hess.setZero(n,n);
//         for(int i=0; i<n; ++i)
//         {
//             grad(i) = jac[i];
//             for(int j=0; j<n; ++j)
//             {
//                 hess(i,j) = hes[i*n+j];
//             }
//         }

//         //std::cout<<grad.transpose()<<std::endl;
//         //std::cout<<hess<<std::endl;

//         grad_global.setZero(2*V.rows());
//         hess_global.setZero(2*V.rows(), 2*V.rows());

//         // Partial a partial X
//         Eigen::VectorXd pa2pn1x(2*V.rows());
//         Eigen::VectorXd pa2pn2x(2*V.rows());
//         pa2pn1x.setZero();
//         pa2pn2x.setZero();

//         std::unordered_map<int,int> hm;
//         hm[0] = 2*slave;  hm[1] = 2*slave+1;
//         hm[4] = 2*master_1;  hm[5] = 2*master_1+1;
//         hm[6] = 2*master_2;  hm[7] = 2*master_2+1;

//         for(auto it = hm.begin(); it != hm.end(); it++)
//         {
//             grad_global(it->second) += grad(it->first);
//             pa2pn1x(it->second) = hess(2,it->first);
//             pa2pn2x(it->second) = hess(3,it->first);
//             for(auto it_2 = hm.begin(); it_2 != hm.end(); it_2++)
//             {
//                 hess_global(it->second,it_2->second) += hess(it->first,it_2->first);
//             }
//         }

//         grad_global += grad(2)*normals[2*normal].return_value_grad;
//         grad_global += grad(3)*normals[2*normal+1].return_value_grad;

//         hess_global += hess(2,2)*normals[2*normal].return_value_grad*normals[2*normal].return_value_grad.transpose();
//         hess_global += hess(3,3)*normals[2*normal+1].return_value_grad*normals[2*normal+1].return_value_grad.transpose();
        
//         Eigen::MatrixXd mat_temp;
//         mat_temp = hess(2,3)*normals[2*normal].return_value_grad*normals[2*normal+1].return_value_grad.transpose();
//         mat_temp += pa2pn1x*normals[2*normal].return_value_grad.transpose();
//         mat_temp += pa2pn2x*normals[2*normal+1].return_value_grad.transpose();
//         hess_global += (mat_temp+mat_temp.transpose());

//         hess_global += grad(2)*normals[2*normal].return_value_hess;
//         hess_global += grad(3)*normals[2*normal+1].return_value_hess;

//         result = {Value(Y[0]), grad_global, hess_global};
//     }else
//     {
//         result = {Value(Y[0]), grad_global, hess_global};
//     }

     return result;
}

all_info MortarMethodDiff::projectMasterToSlave(int master, int n1, int n2, int slave_1, int slave_2, bool update)
{
    // using CppAD::AD;
    // using std::vector;
    // int n = 10;

    // vector<AD<double>> X(n);
    // X[0] = V(master,0); X[1] = V(master,1);
    // X[2] = normals[2*n1].return_value; X[3] = normals[2*n1+1].return_value;
    // X[4] = normals[2*n2].return_value; X[5] = normals[2*n2+1].return_value;
    // X[6] = V(slave_1,0); X[7] = V(slave_1,1);
    // X[8] = V(slave_2,0); X[9] = V(slave_2,1);

    // CppAD::Independent(X);

    // int m = 1;              
    // vector<AD<double>> Y(m); 
    // Y[0] = projectMasterToSlaveImpl(X);
    // if(Y[0] == -100)
    // {
    //     std::cout<<"Wrong Points (master, slave1, slave2): ";
    //     std::cout<<master<<" "<< slave_1<<" "<< slave_2<<std::endl;
    //     std::cout<<"master "<<V(master,0)<<" "<<V(master,1)<<std::endl;
    //     std::cout<<"slave_1 "<<V(slave_1,0)<<" "<<V(slave_1,1)<<std::endl;
    //     std::cout<<"slave_2 "<<V(slave_2,0)<<" "<<V(slave_2,1)<<std::endl;
    //     std::cout<<"n1 "<<normals[2*n1].return_value<<" "<<normals[2*n1+1].return_value<<std::endl;
    //     std::cout<<"n2 "<<normals[2*n2].return_value<<" "<<normals[2*n2+1].return_value<<std::endl;
    // }
    // CppAD::ADFun<double> f(X, Y);

    // Eigen::VectorXd grad_global;
    // Eigen::MatrixXd hess_global;

    // Eigen::VectorXd grad;
    // Eigen::MatrixXd hess;

    all_info result;

    // if(update)
    // {
    //     vector<double> jac(m*n); 
    //     vector<double> x(n);       
    //     x[0] = V(master,0); x[1] = V(master,1);
    //     x[2] = normals[2*n1].return_value; x[3] = normals[2*n1+1].return_value;
    //     x[4] = normals[2*n2].return_value; x[5] = normals[2*n2+1].return_value;
    //     x[6] = V(slave_1,0); x[7] = V(slave_1,1);
    //     x[8] = V(slave_2,0); x[9] = V(slave_2,1);

    //     jac = f.Jacobian(x);

    //     vector<double> hes(n*n); 
    //     hes = f.Hessian(x,0);

    //     grad.setZero(n);
    //     hess.setZero(n,n);
    //     for(int i=0; i<n; ++i)
    //     {
    //         grad(i) = jac[i];
    //         for(int j=0; j<n; ++j)
    //         {
    //             hess(i,j) = hes[i*n+j];
    //         }
    //     }

    //     //std::cout<<grad.transpose()<<std::endl;
    //     //std::cout<<hess<<std::endl;

    //     grad_global.setZero(2*V.rows());
    //     hess_global.setZero(2*V.rows(), 2*V.rows());

    //     // Partial a partial X
    //     Eigen::VectorXd pa2pn1x(2*V.rows());
    //     Eigen::VectorXd pa2pn2x(2*V.rows());
    //     Eigen::VectorXd pa2pn3x(2*V.rows());
    //     Eigen::VectorXd pa2pn4x(2*V.rows());
    //     pa2pn1x.setZero();
    //     pa2pn2x.setZero();
    //     pa2pn3x.setZero();
    //     pa2pn4x.setZero();

    //     std::unordered_map<int,int> hm;
    //     hm[0] = 2*master;  hm[1] = 2*master+1;
    //     hm[6] = 2*slave_1;  hm[7] = 2*slave_1+1;
    //     hm[8] = 2*slave_2;  hm[9] = 2*slave_2+1;

    //     for(auto it = hm.begin(); it != hm.end(); it++)
    //     {
    //         grad_global(it->second) += grad(it->first);
    //         pa2pn1x(it->second) = hess(2,it->first);
    //         pa2pn2x(it->second) = hess(3,it->first);
    //         pa2pn3x(it->second) = hess(4,it->first);
    //         pa2pn4x(it->second) = hess(5,it->first);
    //         for(auto it_2 = hm.begin(); it_2 != hm.end(); it_2++)
    //         {
    //             hess_global(it->second,it_2->second) += hess(it->first,it_2->first);
    //         }
    //     }

    //     std::unordered_map<int,int> hm_2;
    //     hm_2[2] = 2*n1;  hm_2[3] = 2*n1+1;
    //     hm_2[4] = 2*n2;  hm_2[5] = 2*n2+1;

    //     for(auto it = hm_2.begin(); it != hm_2.end(); it++)
    //     {
    //         grad_global += grad(it->first)*normals[it->second].return_value_grad;
    //         hess_global += hess(it->first,it->first)*normals[it->second].return_value_grad*normals[it->second].return_value_grad.transpose();
    //         hess_global += grad(it->first)*normals[it->second].return_value_hess;
    //     }
        
    //     Eigen::MatrixXd mat_temp;
    //     mat_temp = pa2pn1x*normals[2*n1].return_value_grad.transpose();
    //     mat_temp += pa2pn2x*normals[2*n1+1].return_value_grad.transpose();
    //     mat_temp += pa2pn3x*normals[2*n2].return_value_grad.transpose();
    //     mat_temp += pa2pn4x*normals[2*n2+1].return_value_grad.transpose();

    //     for(int i=2; i<6; ++i)
    //     {
    //         for(int j=i+1; j<6; ++j)
    //         {
    //             mat_temp += hess(i,j)*normals[hm_2[i]].return_value_grad*normals[hm_2[j]].return_value_grad.transpose();
    //         }
    //     }
    //     hess_global += (mat_temp+mat_temp.transpose());

    //     result = {Value(Y[0]), grad_global, hess_global};
    // }else
    // {
    //     result = {Value(Y[0]), grad_global, hess_global};
    // }
    // return result;

    return result;
}

void MortarMethodDiff::calculateSegments(bool update)
{
    // seg_info.clear();
    // for(int i=0; i<slave_indices.size(); ++i)
    // {
    //     auto it = seg_info.find(slave_indices[i]);
    //     if(it == seg_info.end())
    //     {
    //         seg_info[slave_indices[i]] = std::vector<segment_info_diff>(0);
    //     }

    //     int slave_1 = segments[slave_indices[i]].first;
    //     int slave_2 = segments[slave_indices[i]].second;

    //     int n1 = segments[slave_indices[i]].first;
    //     int n2 = segments[slave_indices[i]].second;

    //     for(int j=0; j<master_indices.size(); ++j)
    //     {
    //         int master_1 = segments[master_indices[j]].first;
    //         int master_2 = segments[master_indices[j]].second;

    //         all_info stm1 = projectSlaveToMaster(slave_1,n1,master_1,master_2,update);
    //         all_info stm2 = projectSlaveToMaster(slave_2,n2,master_1,master_2,update);

    //         if(stm1.return_value > 1.0 && stm2.return_value > 1.0) continue;
    //         if(stm1.return_value < -1.0 && stm2.return_value < -1.0) continue;

    //         all_info mts1 = projectMasterToSlave(master_1,n1,n2,slave_1,slave_2,update);
    //         all_info mts2 = projectMasterToSlave(master_2,n1,n2,slave_1,slave_2,update);

    //         all_info_clamp(mts1);
    //         all_info_clamp(mts2);

    //         // all_info_differentiable_clamp(mts1, update);
    //         // all_info_differentiable_clamp(mts2, update);
    //         std::vector<all_info> res = {mts1,mts2};
            
    //         // std::cout<<slave_indices[i]<<" "<<master_indices[j]<<" "<<res.transpose()<<std::endl;
    //         //std::cout<<mts1.return_value<<" "<<mts2.return_value<<std::endl;
    //         if(fabs(res[1].return_value-res[0].return_value) < 1e-5) continue;
    //         seg_info[slave_indices[i]].push_back({master_indices[j],res});
    //     }
    // }
}

void MortarMethodDiff::calculateMortarGapEnergy(int slave_segment_id, std::vector<all_info>& gap_erengy_per_element, bool update)
{

    // typedef CppAD::eigen_vector<AD<double>> a_vector;
    // using CppAD::AD;
    // using std::vector;

    // gap_erengy_per_element.resize(2);
    // for(int i=0; i<2; ++i)
    // {
    //     if(update)
    //     {
    //         gap_erengy_per_element[i].return_value_grad.setZero(2*V.rows());
    //         gap_erengy_per_element[i].return_value_hess.setZero(2*V.rows(),2*V.rows());
    //     }
    // }
    
    // int slave_1 = segments[slave_segment_id].first;
    // int slave_2 = segments[slave_segment_id].second;
    // Eigen::Vector2d slave_1_value = V.row(segments[slave_segment_id].first);
    // Eigen::Vector2d slave_2_value = V.row(segments[slave_segment_id].second);

    // int n1 = segments[slave_segment_id].first;
    // int n2 = segments[slave_segment_id].second;
    // Eigen::Vector2d n1_value = {normals[2*n1].return_value, normals[2*n1+1].return_value};
    // Eigen::Vector2d n2_value = {normals[2*n2].return_value, normals[2*n2+1].return_value};

    // std::vector<segment_info_diff> slave_projections = seg_info[slave_segment_id];

    // for(int a=0; a<slave_projections.size(); ++a)
    // {
    //     int master_segment_id = slave_projections[a].master_id;
    //     int master_1 = segments[master_segment_id].first;
    //     int master_2 = segments[master_segment_id].second;
    //     Eigen::Vector2d master_1_value = V.row(segments[master_segment_id].first);
    //     Eigen::Vector2d master_2_value = V.row(segments[master_segment_id].second);

    //     double coor1 = slave_projections[a].xs[0].return_value;
    //     double coor2 = slave_projections[a].xs[1].return_value;
    //     int n = 14;

    //     a_vector X(n);
    //     X(0) = slave_1_value(0); X(1) = slave_1_value(1);
    //     X(2) = slave_2_value(0); X(3) = slave_2_value(1);
    //     X(4) = n1_value(0); X(5) = n1_value(1);
    //     X(6) = n2_value(0); X(7) = n2_value(1);
    //     X(8) = master_1_value(0); X(9) = master_1_value(1);
    //     X(10) = master_2_value(0); X(11) = master_2_value(1);
    //     X(12) = coor1; X(13) = coor2;

    //     CppAD::Independent(X);

    //     int m = 2;              
    //     a_vector Y(m); 
    //     if(use_mortar_IMLS)
    //     {
    //         Y = calculateMortarGapEnergyTestImpl(X,SamplePoints);
    //     }
    //     else
    //         Y = calculateMortarGapEnergyImpl(X);
    //     CppAD::ADFun<double> f(X, Y);

    //     Eigen::VectorXd grad_global(2*V.rows());
    //     Eigen::MatrixXd hess_global(2*V.rows(),2*V.rows());

    //     Eigen::VectorXd grad;
    //     Eigen::MatrixXd hess;

    //     for(int b=0; b<2; ++b)
    //     {
    //         if(update)
    //         {
    //             Eigen::VectorXd jac(m*n); 
    //             Eigen::VectorXd x(n);       
    //             x(0) = slave_1_value(0); x(1) = slave_1_value(1);
    //             x(2) = slave_2_value(0); x(3) = slave_2_value(1);
    //             x(4) = n1_value(0); x(5) = n1_value(1);
    //             x(6) = n2_value(0); x(7) = n2_value(1);
    //             x(8) = master_1_value(0); x(9) = master_1_value(1);
    //             x(10) = master_2_value(0); x(11) = master_2_value(1);
    //             x(12) = coor1; x(13) = coor2;

    //             jac = f.Jacobian(x);

    //             Eigen::VectorXd hes(n*n); 
    //             hes = f.Hessian(x,b);

    //             grad.setZero(n);
    //             hess.setZero(n,n);
    //             for(int i=0; i<n; ++i)
    //             {
    //                 grad(i) = jac(i+b*n);
    //                 for(int j=0; j<n; ++j)
    //                 {
    //                     hess(i,j) = hes(i*n+j);
    //                 }
    //             }

    //             grad_global.setZero(2*V.rows());
    //             hess_global.setZero(2*V.rows(), 2*V.rows());

    //             // Partial a partial X
    //             Eigen::VectorXd pa2pn1x(2*V.rows());
    //             Eigen::VectorXd pa2pn2x(2*V.rows());
    //             Eigen::VectorXd pa2pn3x(2*V.rows());
    //             Eigen::VectorXd pa2pn4x(2*V.rows());
    //             Eigen::VectorXd pa2pc1x(2*V.rows());
    //             Eigen::VectorXd pa2pc2x(2*V.rows());
    //             pa2pn1x.setZero();
    //             pa2pn2x.setZero();
    //             pa2pn3x.setZero();
    //             pa2pn4x.setZero();
    //             pa2pc1x.setZero();
    //             pa2pc2x.setZero();

    //             std::unordered_map<int,int> hm;
    //             hm[0] = 2*slave_1;  hm[1] = 2*slave_1+1;
    //             hm[2] = 2*slave_2;  hm[3] = 2*slave_2+1;
    //             hm[8] = 2*master_1;  hm[9] = 2*master_1+1;
    //             hm[10] = 2*master_2;  hm[11] = 2*master_2+1;

    //             for(auto it = hm.begin(); it != hm.end(); it++)
    //             {
    //                 grad_global(it->second) += grad(it->first);
    //                 pa2pn1x(it->second) = hess(4,it->first);
    //                 pa2pn2x(it->second) = hess(5,it->first);
    //                 pa2pn3x(it->second) = hess(6,it->first);
    //                 pa2pn4x(it->second) = hess(7,it->first);
    //                 pa2pc1x(it->second) = hess(12,it->first);
    //                 pa2pc2x(it->second) = hess(13,it->first);
    //                 for(auto it_2 = hm.begin(); it_2 != hm.end(); it_2++)
    //                 {
    //                     hess_global(it->second,it_2->second) += hess(it->first,it_2->first);
    //                 }
    //             }

    //             std::unordered_map<int,int> hm_2;
    //             hm_2[4] = 2*n1;  hm_2[5] = 2*n1+1;
    //             hm_2[6] = 2*n2;  hm_2[7] = 2*n2+1;

    //             for(auto it = hm_2.begin(); it != hm_2.end(); it++)
    //             {
    //                 grad_global += grad(it->first)*normals[it->second].return_value_grad;
    //                 hess_global += hess(it->first,it->first)*normals[it->second].return_value_grad*normals[it->second].return_value_grad.transpose();
    //                 hess_global += grad(it->first)*normals[it->second].return_value_hess;
    //             }

    //             grad_global += grad(12)*slave_projections[a].xs[0].return_value_grad;
    //             grad_global += grad(13)*slave_projections[a].xs[1].return_value_grad;
    //             hess_global += hess(12,12)*slave_projections[a].xs[0].return_value_grad*slave_projections[a].xs[0].return_value_grad.transpose();
    //             hess_global += hess(13,13)*slave_projections[a].xs[1].return_value_grad*slave_projections[a].xs[1].return_value_grad.transpose();
    //             hess_global += grad(12)*slave_projections[a].xs[0].return_value_hess;
    //             hess_global += grad(13)*slave_projections[a].xs[1].return_value_hess;
                
    //             Eigen::MatrixXd mat_temp;
    //             mat_temp = pa2pn1x*normals[2*n1].return_value_grad.transpose();
    //             mat_temp += pa2pn2x*normals[2*n1+1].return_value_grad.transpose();
    //             mat_temp += pa2pn3x*normals[2*n2].return_value_grad.transpose();
    //             mat_temp += pa2pn4x*normals[2*n2+1].return_value_grad.transpose();
    //             mat_temp += pa2pc1x*slave_projections[a].xs[0].return_value_grad.transpose();
    //             mat_temp += pa2pc2x*slave_projections[a].xs[1].return_value_grad.transpose();

    //             for(int i=4; i<8; ++i)
    //             {
    //                 for(int j=i+1; j<8; ++j)
    //                 {
    //                     mat_temp += hess(i,j)*normals[hm_2[i]].return_value_grad*normals[hm_2[j]].return_value_grad.transpose();
    //                 }
    //                 mat_temp += hess(i,12)*normals[hm_2[i]].return_value_grad*slave_projections[a].xs[0].return_value_grad.transpose();
    //                 mat_temp += hess(i,13)*normals[hm_2[i]].return_value_grad*slave_projections[a].xs[1].return_value_grad.transpose();
    //             }
    //             hess_global += (mat_temp+mat_temp.transpose());

    //             gap_erengy_per_element[b].return_value += Value(Y[b]);
    //             gap_erengy_per_element[b].return_value_grad += grad_global;
    //             gap_erengy_per_element[b].return_value_hess += hess_global;
    //         }else
    //         {
    //             gap_erengy_per_element[b].return_value += Value(Y[b]);
    //         }
    //     }
    // }
}

void MortarMethodDiff::MortarMethod(bool update)
{
    // for(int i=0; i<slave_indices.size(); ++i)
    // {
    //     slave_nodes.push_back(segments[slave_indices[i]].first);
    //     slave_nodes.push_back(segments[slave_indices[i]].second);
    // }

    // for(int i=0; i<master_indices.size(); ++i)
    // {
    //     master_nodes.push_back(segments[master_indices[i]].first);
    //     master_nodes.push_back(segments[master_indices[i]].second);
    // }

    // std::sort(slave_nodes.begin(), slave_nodes.end());
    // std::sort(master_nodes.begin(), master_nodes.end());

    // auto last = std::unique(slave_nodes.begin(), slave_nodes.end());
    // slave_nodes.erase(last, slave_nodes.end());
    // last = std::unique(master_nodes.begin(), master_nodes.end());
    // master_nodes.erase(last, master_nodes.end());

    // calculateNormals(update);
    // calculateSegments(update);

    // int size = V.rows();
    // gap_functions.resize(size);

    // for(int i=0; i<size; ++i)
    // {
    //     gap_functions[i].return_value = 0;
    //     if(update)
    //     {
    //         gap_functions[i].return_value_grad.setZero(2*size);
    //         gap_functions[i].return_value_hess.setZero(2*size,2*size);
    //     }
    // }
    

    // for(int i=0; i<slave_indices.size(); ++i)
    // {
    //     std::vector<all_info> ge;
    //     calculateMortarGapEnergy(slave_indices[i],ge,update);

    //     int i1 = segments[slave_indices[i]].first;
    //     int i2 = segments[slave_indices[i]].second;

    //     gap_functions[i1].return_value += ge[0].return_value;
    //     gap_functions[i2].return_value += ge[1].return_value;
    //     if(update)
    //     {
    //         gap_functions[i1].return_value_grad += ge[0].return_value_grad;
    //         gap_functions[i2].return_value_grad += ge[1].return_value_grad;
    //         gap_functions[i1].return_value_hess += ge[0].return_value_hess;
    //         gap_functions[i2].return_value_hess += ge[1].return_value_hess;
    //     }
    // }

    // // Compute total gap energy
    // gap_energy.return_value = 0;
    // gap_energy.return_value_grad.setZero(2*size);
    // gap_energy.return_value_hess.setZero(2*size,2*size);

    // Eigen::VectorXd dgedgi(size);
    // Eigen::VectorXd d2gedgi2(size);
    // dgedgi.setZero();
    // d2gedgi2.setZero();
    // for(int i=0; i<size; ++i)
    // {
    //     // std::cout<<i<<" "<<gap_functions[i].return_value<<std::endl;
    //     // if(gap_functions[i].return_value < 0)
    //     // {
    //     //     gap_energy.return_value += - gap_stiffness * gap_functions[i].return_value * gap_functions[i].return_value * gap_functions[i].return_value;
    //     //     dgedgi(i) = -3 * gap_stiffness * gap_functions[i].return_value * gap_functions[i].return_value;
    //     //     d2gedgi2(i) = -6 * gap_stiffness * gap_functions[i].return_value;
    //     //     if(update)
    //     //     {
    //     //         gap_energy.return_value_grad += dgedgi(i)*gap_functions[i].return_value_grad;
    //     //         gap_energy.return_value_hess += dgedgi(i)*gap_functions[i].return_value_hess + d2gedgi2(i)*gap_functions[i].return_value_grad*gap_functions[i].return_value_grad.transpose();
    //     //     }
    //     // }

    //     gap_energy.return_value += gap_functions[i].return_value;
    //     if(update)
    //     {
    //         gap_energy.return_value_grad += gap_functions[i].return_value_grad;
    //         gap_energy.return_value_hess += gap_functions[i].return_value_hess;
    //     }
    // }
}

void MortarMethodDiff::findContactMaster(std::vector<double>& master_contact_nodes)
{
    // std::unordered_map<int, int> temp;
    // master_contact_nodes.clear();
    // for(auto it = seg_info.begin(); it != seg_info.end(); it++)
    // {
    //     std::vector<segment_info_diff> master_eles = it->second;
    //     for(int i=0; i<master_eles.size(); ++i)
    //     {
    //         int master_id = master_eles[i].master_id;
    //         int master_node_1 = segments[master_id].first;
    //         int master_node_2 = segments[master_id].second;

    //         if(temp.find(master_node_1) == temp.end())
    //         {
    //             temp[master_node_1] = temp.size();
    //         }

    //         if(temp.find(master_node_2) == temp.end())
    //         {
    //             temp[master_node_2] = temp.size();
    //         }
    //     }
    // }

    // for(auto it = temp.begin(); it != temp.end(); it++)
    // {
    //     master_contact_nodes.push_back(V(it->first,1));
    // }
}

