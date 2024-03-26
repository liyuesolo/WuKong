// Sample points from polygonal mesh
// Evaluate Implicit potential function
// Visualize in the original problem

#include <igl/winding_number.h>
#include <igl/writeOBJ.h>
#include "../include/FEMSolver.h"
#include <igl/bounding_box_diagonal.h>
#include <Eigen/SparseCholesky>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <igl/copyleft/cgal/points_inside_component.h>
#include <chrono>
#include <tbb/parallel_for.h>

using namespace autodiff;


double eps_scale = 0.03;
double polyDegree = 2;
double wendlandRadius = 1.5;

// int normal_num = 2;
// std::vector<double> normal_weights = {0.5,0.5};

int normal_num = 6;
std::vector<double> normal_weights = {0.1,0.1,0.3,0.3,0.1,0.1};


template<int dim>
Eigen::VectorXd FEMSolver<dim>::convertToGlobalGradient(Eigen::VectorXd& grad,std::vector<int>& useful_index,int index)
{
    int num_sample_points_no_more = map_sample_to_deform[index].size();
    Eigen::VectorXd res(dim+num_sample_points_no_more*dim);
    res.setZero();

    for(int i=0; i<=useful_index.size(); ++i)
    {
        // if(index == 1)
        // {
        //     std::cout<<i<<" "<<grad.size()/2<<" "<<map_sample_to_deform_with_scale[index].size()<<std::endl;
        //     std::cout<<"again"<<std::endl;
        // } 
        int no_more_index = 0;
        double scale = 1;
        if(i!=0)
        {
            int sample_pts_index = useful_index[i-1];
            no_more_index = map_sample_to_deform_with_scale[index][sample_pts_index].first;
            scale = map_sample_to_deform_with_scale[index][sample_pts_index].second;
        }

        int next = no_more_index + 1;
        if(next >=  num_sample_points_no_more) next-=num_sample_points_no_more;

        if(i!=0)
        {
            no_more_index++;
            next++;
        }

        for(int j=0; j<dim; ++j)
        {
            res(dim*no_more_index+j) += scale*grad(dim*i+j);
            res(dim*next+j) += (1-scale)*grad(dim*i+j);
        }

    }
    return res;
}

template<int dim>
Eigen::MatrixXd FEMSolver<dim>::convertToGlobalHessian(Eigen::MatrixXd& hess,std::vector<int>& useful_index,int index)
{
    int num_sample_points_no_more = map_sample_to_deform[index].size();
    Eigen::MatrixXd res(dim+num_sample_points_no_more*dim, dim+num_sample_points_no_more*dim);
    res.setZero();


    for(int i=0; i<=useful_index.size(); ++i)
    {
        for(int j=0; j<=useful_index.size(); ++j)
        {   
            int no_more_index_1 = 0, no_more_index_2 = 0; 
            double scale_1 = 1, scale_2 = 1;
            if(i!=0)
            {
                int sample_pts_index = useful_index[i-1];
                no_more_index_1 = map_sample_to_deform_with_scale[index][sample_pts_index].first;
                scale_1 = map_sample_to_deform_with_scale[index][sample_pts_index].second;
            }

            if(j!=0)
            {
                int sample_pts_index = useful_index[j-1];
                no_more_index_2 = map_sample_to_deform_with_scale[index][sample_pts_index].first;
                scale_2 = map_sample_to_deform_with_scale[index][sample_pts_index].second;
            }

            int next_1 = no_more_index_1 + 1;
            if(next_1 >=  num_sample_points_no_more) next_1-=num_sample_points_no_more;
            int next_2 = no_more_index_2 + 1;
            if(next_2 >=  num_sample_points_no_more) next_2-=num_sample_points_no_more;

            if(i!=0)
            {
                no_more_index_1++;
                next_1++;
            }
            if(j!=0)
            {
                no_more_index_2++;
                next_2++;
            }

            for(int a=0; a<dim; ++a)
            {
                for(int b=0; b<dim; ++b)
                {
                    res(dim*no_more_index_1+a, dim*no_more_index_2+b) += scale_1*scale_2*hess(dim*i+a, dim*j+b);
                    res(dim*next_1+a, dim*no_more_index_2+b) += (1-scale_1)*scale_2*hess(dim*i+a, dim*j+b);
                    res(dim*no_more_index_1+a, dim*next_2+b) += scale_1*(1-scale_2)*hess(dim*i+a, dim*j+b);
                    res(dim*next_1+a, dim*next_2+b) += (1-scale_1)*(1-scale_2)*hess(dim*i+a, dim*j+b);
                }
            }
        }
    }
    return res;
}

double computeWeightFunction(Eigen::VectorXd& x, Eigen::VectorXd& s)
{
    double eps = 1e-5;
    assert(x.size() == s.size());
    double r = (x-s).norm();
    return 1.0/(r*r+eps*eps);
}

var computeSignedDistance(const ArrayXvar& s, const ArrayXvar& n, double radius)
{
    int Num = s.size()/2-1;

    var upper = 0.0;
    var lower = 0.0;
    for(int j=0; j<Num; ++j)
    {

        var res_dists_squared = (s(0)- s(2*j+2))*(s(0)- s(2*j+2)) + (s(1)- s(2*j+3))*(s(1)- s(2*j+3));
        var phi_x = pow((1- (res_dists_squared)/(radius*radius)), 4);
        lower += phi_x;
        var dot_value = (s(0) - s(2*j+2))*n(2*j) + (s(1)- s(2*j+3))*n(2*j+1);
        upper += dot_value*phi_x;
    }
    return upper/lower;
}



var computeSignedDistanceAll(const ArrayXvar& s, std::vector<int>& results, double radius)
{
    int Num = s.size()/2-1;
    var upper = 0.0;
    var lower = 0.0;
    var hi = radius;

    int normal_num = 6;
    std::vector<double> normal_weights = {0.1,0.1,0.3,0.3,0.1,0.1};

    ArrayXvar n(2*Num), pts(2*Num);

    for(int j=0; j<Num; ++j)
    {
        pts(2*j) = s(2*j+2);
        pts(2*j+1) = s(2*j+3);
    }

    for(int j=0; j<Num; ++j)
    {
        n(2*j) = 0;
        n(2*j+1) = 0;
        for(int k=0; k<normal_num; ++k)
        {
            int index_1 = j-normal_num/2+k;
            if(index_1 < 0) index_1 += Num;
            if(index_1 >= Num) index_1 -= Num;

            int index_2 = index_1+1;
            if(index_2 < 0) index_2 += Num;
            if(index_2 >= Num) index_2 -= Num;

            var x0 = pts(2*index_1); var y0 = pts(2*index_1+1);
            var x1 = pts(2*index_2); var y1 = pts(2*index_2+1);

            var dx1 = x1-x0;
            var dy1 = y1-y0;
            var n1x = dy1/sqrt(dx1*dx1+dy1*dy1);
            var n1y = -dx1/sqrt(dx1*dx1+dy1*dy1);
            n(2*j) += normal_weights[k]*n1x;
            n(2*j+1) += normal_weights[k]*n1y;
        }
    }

    for(int i=0; i<results.size(); ++i)
    {
        int j = results[i];
        var res_dists_squared = (s(0)- s(2*j+2))*(s(0)- s(2*j+2)) + (s(1)- s(2*j+3))*(s(1)- s(2*j+3));
        //var u = sqrt(res_dists_squared)/hi;
        //var phi_x = 1./exp(u+-u);
        var phi_x = pow((1- (res_dists_squared)/(radius*radius)), 4);
        //var phi_x = exp(-(res_dists_squared)/(radius*radius));
        lower += phi_x;
        var dot_value = (s(0) - s(2*j+2))*n(2*j) + (s(1)- s(2*j+3))*n(2*j+1);
        upper += dot_value*phi_x;
    }
    return upper/lower;
}

var computeSignedRIMLS(const ArrayXvar& s, std::vector<int>& results, double radius)
{
    int Num = s.size()/2-1;
    var upper = 0.0;
    var lower = 0.0;

    ArrayXvar n(2*Num), pts(2*Num);
    ArrayXvar x(2);
    x(0) = s(0);
    x(1) = s(1);

    for(int j=0; j<Num; ++j)
    {
        pts(2*j) = s(2*j+2);
        pts(2*j+1) = s(2*j+3);
    }

    int normal_num = 6;
    std::vector<double> normal_weights = {0.1,0.1,0.3,0.3,0.1,0.1};

    for(int j=0; j<Num; ++j)
    {
        n(2*j) = 0;
        n(2*j+1) = 0;
        for(int k=0; k<normal_num; ++k)
        {
            int index_1 = j-normal_num/2+k;
            if(index_1 < 0) index_1 += Num;
            if(index_1 >= Num) index_1 -= Num;

            int index_2 = index_1+1;
            if(index_2 < 0) index_2 += Num;
            if(index_2 >= Num) index_2 -= Num;

            var x0 = pts(2*index_1); var y0 = pts(2*index_1+1);
            var x1 = pts(2*index_2); var y1 = pts(2*index_2+1);

            var dx1 = x1-x0;
            var dy1 = y1-y0;
            var n1x = dy1/sqrt(dx1*dx1+dy1*dy1);
            var n1y = -dx1/sqrt(dx1*dx1+dy1*dy1);
            n(2*j) += normal_weights[k]*n1x;
            n(2*j+1) += normal_weights[k]*n1y;
        }
    }

    int max_iter = 1;
    int max_iter_2 = 0;
    var threshold = 1e-5;
    var sigma_r = 0.5;
    var sigma_n = 0.5;

    var f = 0, grad_fx = 0, grad_fy = 0;
    for(int iter = 0; iter<max_iter; ++iter)
    {
        //std::cout<<iter_o<<" "<<iter<<std::endl;
        var sumW = 0, sumGwx = 0, sumGwy = 0, sumF = 0, sumGfx = 0, sumGfy = 0, sumNx = 0, sumNy = 0;
        for(int i=0; i<results.size(); ++i)
        {
            int j = results[i];
            var pxx = x(0)-pts(2*j);
            var pxy = x(1)-pts(2*j+1);
            var fx = pxx*n(2*j) + pxy*n(2*j+1);

            var alpha = 1;
            if(iter > 0)
            {
                var a1 = exp(-((fx-f)/sigma_r)*((fx-f)/sigma_r));
                var norm = sqrt((n(2*j)- grad_fx)*(n(2*j)- grad_fx) + (n(2*j+1)- grad_fy)*(n(2*j+1)- grad_fy));
                var a2 = exp(-(norm/sigma_n)*(norm/sigma_n));
                alpha = a1*a2;
            }

            var phi = pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 4);
            var dphidx = 4*pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 3)*(-1/(radius*radius))*2*pxx;
            var dphidy = 4*pow((1- (pxx*pxx+pxy*pxy)/(radius*radius)), 3)*(-1/(radius*radius))*2*pxy;

            var w = alpha*phi;
            var grad_wx = alpha*dphidx;
            var grad_wy = alpha*dphidy;

            sumW += w;
            sumGwx += grad_wx;
            sumGwy += grad_wy;
            sumF += w*fx;
            sumGfx += grad_wx*fx;
            sumGfy += grad_wy*fx;
            sumNx += w*n(2*j);
            sumNy += w*n(2*j+1);

            // int j = results[i];
            // var res_dists_squared = (s(0)- s(2*j+2))*(s(0)- s(2*j+2)) + (s(1)- s(2*j+3))*(s(1)- s(2*j+3));
            // var phi_x = pow((1- (res_dists_squared)/(radius*radius)), 4);
            // lower += phi_x;
            // var dot_value = (s(0) - s(2*j+2))*n(2*j) + (s(1)- s(2*j+3))*n(2*j+1);
            // upper += dot_value*phi_x;
        }

        f = sumF/sumW;
        grad_fx = (sumGfx-f*sumGwx+sumNx)/sumW;
        grad_fy = (sumGfy-f*sumGwy+sumNy)/sumW;
    }
    x(0) -= f*grad_fx;
    x(1) -= f*grad_fy;
    //std::cout<<"done"<<std::endl;
    return f;
}

template<int dim>
void FEMSolver<dim>:: updateHashDataStructure(double radius, int index)
{
    Eigen::VectorXd pts(constrained_points[index].rows()*dim);
    for(int i=0; i<constrained_points[index].rows(); ++i)
    {
        pts.segment<dim>(dim*i) = constrained_points[index].row(i);
    }
    SH_data[index].build(2*radius, pts);
}

template<int dim>
void FEMSolver<dim>::find_neighbor_pts_SH(Eigen::VectorXd& query_pt, double radius, std::vector<int>& results, std::vector<double>& res_dists, int index)
{
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::vector<int> temp;
    SH_data[index].getOneRingNeighbors(query_pt, temp);
    for(int i=0; i<temp.size(); ++i)
    {   
        double cur_dist = (constrained_points[index].row(temp[i])-query_pt.transpose()).norm();
        if(cur_dist <= radius)
        {
            res_dists.push_back(cur_dist);
            results.push_back(temp[i]);
        }
        
    }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ns]" << std::endl;
}


template<int dim>
void FEMSolver<dim>:: find_neighbor_pts(Eigen::VectorXd& query_pt, double radius, std::vector<int>& results, std::vector<double>& res_dists, int index) {
    int num_pts = sample_points[index].size();
    // std::cout<<index<<std::endl;
    // std::cout<<constrained_points[index].rows()<<" "<<constrained_points[index].cols()<<" "<<num_pts<<std::endl;
    for (int i = 0; i < num_pts; i++) {
        int max_iter = 1;
        if(!USE_IMLS)
            max_iter = 3;
        for (int a = 0; a < max_iter; a++) {
            Eigen::VectorXd cur_ptr = constrained_points[index].row(a * num_pts + i);
            double cur_dist = (cur_ptr - query_pt).norm();
            // std::cout<<"sample point "<<cur_ptr.transpose()<<" "<<query_pt.transpose()<<std::endl;
            // std::cout<<i<<" "<<"cur dist "<<cur_dist<<" radius "<<radius<<std::endl;
            if (cur_dist <= radius) {
                results.push_back(a * num_pts + i);
                res_dists.push_back(cur_dist);
            }
        }
    }
}

template<int dim>
bool FEMSolver<dim>::checkInsideRectangle(Eigen::VectorXd q, int index)
{
    //std::cout<<corners[1].size()<<std::endl;
    // std::cout<<"size: "<<corners.size()<<" index: "<<index<<std::endl;
    // for(int i=0; i<corners[index].size(); ++i)
    // {
    //     std::cout<<corners[index][i].transpose()<<std::endl;
    // }
    // Eigen::VectorXd p1 = corners[index][0];
    // Eigen::VectorXd p2 = corners[index][1];

    // if(q(0) >= p1(0) && q(0) <= p2(0) && q(1) >= p1(1) && q(1) <= p2(1)) return true;
    // else return false;
    return false;
}

template<int dim>
bool FEMSolver<dim>::checkInsideRectangleFromSTL(Eigen::VectorXd q, int index)
{
    Eigen::VectorXd result;
    Eigen::MatrixXd ptr(1,3);
    std::vector<int> boundary = (index == 0) ? master_nodes : slave_nodes;
    int nvert = boundary.size();

    double testx = q(0);
    double testy = q(1);

    int i,j,c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        double vertxi = deformed(2*boundary[i]);
        double vertyi = deformed(2*boundary[i]+1);
        double vertxj = deformed(2*boundary[j]);
        double vertyj = deformed(2*boundary[j]+1);

        if ( ((vertyi>testy) != (vertyj>testy)) &&
        (testx < (vertxj-vertxi) * (testy-vertyi) / (vertyj-vertyi) + vertxi))
        c = !c;
    }
    return c;
    // ptr(0,0) = q(0);
    // ptr(0,1) = q(1);
    // ptr(0,2) = q(2);
    // // std::cout<<index<<std::endl;
    // std::cout<<ptr.row(0)<<std::endl;

    // for(int i=0; i<Vs[index].rows(); ++i)
    // {
    //     std::cout<<Vs[index].row(i)<<std::endl;
    // }

    // std::cout<<" ------------------------------"<<std::endl;

    // for(int i=0; i<Fs[index].rows(); ++i)
    // {
    //     std::cout<<Fs[index].row(i)<<std::endl;
    // }

    // ptr(0,0) = 0;
    // ptr(0,1) = 0;
    // ptr(0,2) = 0;

    // igl::copyleft::cgal::points_inside_component(Vs[index],Fs[index],ptr,result);
    //return true;
}

template<int dim>
bool FEMSolver<dim>::checkInsideMeshFromSTL3D(Eigen::VectorXd& q, int index, int pair_id)
{
    assert(dim == 3);
    std::unordered_map<int,int> effective_master_nodes = master_nodes_3d[pair_id];
    if(index == 1) effective_master_nodes = slave_nodes_3d[pair_id];
    std::vector<Eigen::VectorXi> effective_master_surfaces = master_surfaces_3d[pair_id];
    if(index == 1) effective_master_surfaces = slave_surfaces_3d[pair_id];

    int num_pts = effective_master_nodes.size();

    Eigen::MatrixXd V_test(num_pts,dim);
    for(auto it = effective_master_nodes.begin(); it!=effective_master_nodes.end(); it++)
    {
        V_test.row(it->second) = deformed.segment<dim>(dim*it->first);
    }

    Eigen::MatrixXi F_test(effective_master_surfaces.size(),3);
    for(int i=0; i<effective_master_surfaces.size(); ++i)
    {
        for(int j=0; j<3; ++j)
            F_test(i,j) = effective_master_nodes[effective_master_surfaces[i](j)];
    }


    Eigen::VectorXd W;
    Eigen::MatrixXd Q = q.transpose();
    igl::winding_number(V_test,F_test,Q,W);
    // igl::writeOBJ("winding.obj",V_test,F_test);

    //std::cout<<"winding number of "<<q.transpose()<<" is "<<W(0)<<std::endl;

    if(W(0) > 0) return true;
    else return false;
}

template<int dim>
int FEMSolver<dim>::brute_force_NN(Eigen::VectorXd& query_pt, int index) {
    int num_pts = sample_points[index].size();
    double min_sqared_dist = -1;
    int min_index = -1;
    for (int i = 0; i < num_pts; i++) {
        Eigen::VectorXd diff = sample_points[index][i] - query_pt;
        double squared_dist = diff(0) * diff(0) + diff(1) * diff(1);
        if (min_sqared_dist == -1 || min_sqared_dist > squared_dist) {
            min_index = i;
            min_sqared_dist = squared_dist;
        }
    }
    return min_index;
}

template<int dim>
void FEMSolver<dim>::samplePoints()
{
    result_values.resize(object_num);
    SH_data.resize(object_num);
    if(!NO_SAMPLING)
    {
        corners.resize(1);
        sample_points.resize(1);
        sample_normals.resize(1);
        for(int i=0; i<boundary_segments.size(); ++i)
        {
        
            Eigen::VectorXd prev = boundary_segments[i];
            Eigen::VectorXd next = boundary_segments[(i+1)%boundary_segments.size()];

            sample_points[0].push_back(prev);
            sample_normals[0].push_back(boundary_normals[i] + boundary_normals[(i-1)%boundary_segments.size()]);
            corners[0].push_back(prev);
            for(int j=1; j<sample_res; ++j)
            {
                //std::cout<<i<<" "<<j<<std::endl;
                Eigen::VectorXd temp = ((sample_res-j)*prev+j*next)/sample_res;
                sample_points[0].push_back(temp);
                sample_normals[0].push_back(boundary_normals[i]);
            }
        }
    }
    else
    {
        int num_x = WIDTH_1*RES+1, num_y = HEIGHT_1*RES+1;
        int num_x_2 = WIDTH_2*SCALAR*RES+1, num_y_2 = HEIGHT_2*SCALAR*RES+1;

        map_sample_to_deform.resize(object_num);
        map_sample_to_deform_with_scale.resize(object_num);
            
        sample_points.resize(object_num);
        sample_normals.resize(object_num);
        corners.resize(object_num);

        Eigen::VectorXd prev, next;

        for(int i=0; i<object_num; ++i)
        {
            sample_points[i].clear();
            sample_normals[i].clear();
            map_sample_to_deform[i].clear();
            map_sample_to_deform_with_scale[i].clear();
            corners[i].clear();
        }
            

        sample_points[0].push_back(deformed.segment<2>(2*0));
        sample_normals[0].push_back(Eigen::Vector2d(-1.0,-1.0).normalized());
        map_sample_to_deform[0].push_back(0);
        if(true)
            map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));
        corners[0].push_back(sample_points[0].back());
        for(int i=1; i<num_x; ++i)
        {

            if(USE_MORE_POINTS)
            {
                prev = sample_points[0].back();
                next = deformed.segment<2>(2*(i+0*num_x));
                for(int l=1; l<sample_res; ++l)
                {
                    //std::cout<<i<<" "<<j<<std::endl;
                    Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                    sample_points[0].push_back(temp);
                    sample_normals[0].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
                    map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
                }
            }

            if(i == num_x-1) continue;

            sample_points[0].push_back(deformed.segment<2>(2*(i+0*num_x)));
            sample_normals[0].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
            map_sample_to_deform[0].push_back(i+0*num_x);
            if(true)
                map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));

            // if(i == num_x-2 && USE_MORE_POINTS)
            // {
            //     prev = sample_points[0].back();
            //     next = deformed.segment<2>(2*(num_x-1));
            //     for(int l=1; l<sample_res; ++l)
            //     {
            //         //std::cout<<i<<" "<<j<<std::endl;
            //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
            //         sample_points[0].push_back(temp);
            //         sample_normals[0].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
            //         map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
            //     }
            // }
        }

        sample_points[0].push_back(deformed.segment<2>(2*(num_x-1)));
        sample_normals[0].push_back(Eigen::Vector2d(1.0,-1.0).normalized());
        map_sample_to_deform[0].push_back(num_x-1);
        if(true)
            map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));
        //corners[0].push_back(sample_points[0].back());
        for(int j=1; j<num_y; ++j)
        {
            if(USE_MORE_POINTS)
            {
                prev = sample_points[0].back();
                next = deformed.segment<2>(2*(num_x-1+j*num_x));
                for(int l=1; l<sample_res; ++l)
                {
                    //std::cout<<i<<" "<<j<<std::endl;
                    Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                    sample_points[0].push_back(temp);
                    sample_normals[0].push_back(Eigen::Vector2d(1.0,0.0).normalized());
                    map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
                }
            }

            if(j == num_y-1) continue;
            
            sample_points[0].push_back(deformed.segment<2>(2*(num_x-1+j*num_x)));
            sample_normals[0].push_back(Eigen::Vector2d(1.0,0.0).normalized());
            map_sample_to_deform[0].push_back(num_x-1+j*num_x);
            if(true)
                map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));

            // if(j==num_y-2 && USE_MORE_POINTS)
            // {
            //     prev = sample_points[0].back();
            //     next = deformed.segment<2>(2*(num_x*num_y-1));
            //     for(int l=1; l<sample_res; ++l)
            //     {
            //         //std::cout<<i<<" "<<j<<std::endl;
            //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
            //         sample_points[0].push_back(temp);
            //         sample_normals[0].push_back(Eigen::Vector2d(1.0,0.0).normalized());
            //         map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
            //     }
            // }
        }


        sample_points[0].push_back(deformed.segment<2>(2*(num_x*num_y-1)));
        sample_normals[0].push_back(Eigen::Vector2d(1.0,1.0).normalized());
        map_sample_to_deform[0].push_back(num_x*num_y-1);
        if(true)
            map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));
        corners[0].push_back(sample_points[0].back());
        for(int i=num_x-2; i>=0; --i)
        {
            if(USE_MORE_POINTS)
            {
                prev = sample_points[0].back();
                next = deformed.segment<2>(2*(i+(num_y-1)*num_x));
                for(int l=1; l<sample_res; ++l)
                {
                    //std::cout<<i<<" "<<j<<std::endl;
                    Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                    sample_points[0].push_back(temp);
                    sample_normals[0].push_back(Eigen::Vector2d(0.0,1.0).normalized());
                    map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
                }
            }
            if(i == 0) continue;

            sample_points[0].push_back(deformed.segment<2>(2*(i+(num_y-1)*num_x)));
            sample_normals[0].push_back(Eigen::Vector2d(0.0,1.0).normalized());
            map_sample_to_deform[0].push_back(i+(num_y-1)*num_x);
            if(true)
                map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));

            // if(i==1 && USE_MORE_POINTS)
            // {
            //     prev = sample_points[0].back();
            //     next = deformed.segment<2>(2*((num_y-1)*num_x));
            //     for(int l=1; l<sample_res; ++l)
            //     {
            //         //std::cout<<i<<" "<<j<<std::endl;
            //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
            //         sample_points[0].push_back(temp);
            //         sample_normals[0].push_back(Eigen::Vector2d(0.0,1.0).normalized());
            //         map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
            //     }
            // }
        }


        sample_points[0].push_back(deformed.segment<2>(2*((num_y-1)*num_x)));
        sample_normals[0].push_back(Eigen::Vector2d(-1.0,1.0).normalized());
        map_sample_to_deform[0].push_back((num_y-1)*num_x);
        if(true)
            map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));
        //corners[0].push_back(sample_points[0].back());
        for(int j=num_y-2; j>=0; --j)
        {
            if(USE_MORE_POINTS )
            {
                prev = sample_points[0].back();
                next = deformed.segment<2>(2*(0+j*num_x));
                for(int l=1; l<sample_res; ++l)
                {
                    //std::cout<<i<<" "<<j<<std::endl;
                    Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                    sample_points[0].push_back(temp);
                    sample_normals[0].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                    map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
                }
            }

            if(j == 0) continue;

            sample_points[0].push_back(deformed.segment<2>(2*(0+j*num_x)));
            sample_normals[0].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
            map_sample_to_deform[0].push_back(0+j*num_x);
            if(true)
                map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));

            // if(j==1 && USE_MORE_POINTS)
            // {
            //     prev = sample_points[0].back();
            //     next = deformed.segment<2>(2*0);
            //     for(int l=1; l<sample_res; ++l)
            //     {
            //         //std::cout<<i<<" "<<j<<std::endl;
            //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
            //         sample_points[0].push_back(temp);
            //         sample_normals[0].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
            //         map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
            //     }
            // }
        }

        if(IMLS_BOTH|| USE_NEW_FORMULATION)
        {
            sample_points[1].push_back(deformed.segment<2>(2*((num_y_2-1)*num_x_2+num_x*num_y)));
            sample_normals[1].push_back(Eigen::Vector2d(-1.0,1.0).normalized());
            map_sample_to_deform[1].push_back((num_y_2-1)*num_x_2+num_x*num_y);
            if(true)
                map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
            
            //corners[1].push_back(sample_points[1].back());
            for(int j=num_y_2-2; j>=0; --j)
            {
                if(USE_MORE_POINTS)
                {
                    prev = sample_points[1].back();
                    next = deformed.segment<2>(2*(0+j*num_x_2+num_x*num_y));
                    for(int l=1; l<sample_res; ++l)
                    {
                        //std::cout<<i<<" "<<j<<std::endl;
                        Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                        sample_points[1].push_back(temp);
                        sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                        map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                    }
                }
                if(j == 0) continue;

                sample_points[1].push_back(deformed.segment<2>(2*(0+j*num_x_2+num_x*num_y)));
                sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                map_sample_to_deform[1].push_back(0+j*num_x_2+num_x*num_y);
                if(true)
                    map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
                
                // if(j == 1 && USE_MORE_POINTS)
                // {
                //     prev = sample_points[1].back();
                //     next = deformed.segment<2>(2*(0+num_x*num_y));
                //     for(int l=1; l<sample_res; ++l)
                //     {
                //         //std::cout<<i<<" "<<j<<std::endl;
                //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                //         sample_points[1].push_back(temp);
                //         sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                //         map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                //     }
                // }
            }

            sample_points[1].push_back(deformed.segment<2>(2*(0+num_x*num_y)));
            sample_normals[1].push_back(Eigen::Vector2d(-1.0,-1.0).normalized());
            map_sample_to_deform[1].push_back(0+num_x*num_y);
            if(true)
                map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
            corners[1].push_back(sample_points[1].back());
            for(int i=1; i<num_x_2; ++i)
            {
                if(USE_MORE_POINTS)
                {
                    prev = sample_points[1].back();
                    next = deformed.segment<2>(2*(i+0*num_x_2+num_x*num_y));
                    for(int l=1; l<sample_res; ++l)
                    {
                        //std::cout<<i<<" "<<j<<std::endl;
                        Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                        sample_points[1].push_back(temp);
                        sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                        map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                    }
                }
                if(i == num_x_2-1) continue;

                sample_points[1].push_back(deformed.segment<2>(2*(i+0*num_x_2+num_x*num_y)));
                sample_normals[1].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
                map_sample_to_deform[1].push_back(i+0*num_x_2+num_x*num_y);
                if(true)
                    map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));

                // if(i == num_x_2-2 && USE_MORE_POINTS)
                // {
                //     prev = sample_points[1].back();
                //     next = deformed.segment<2>(2*(num_x_2-1+num_x*num_y));
                //     for(int l=1; l<sample_res; ++l)
                //     {
                //         //std::cout<<i<<" "<<j<<std::endl;
                //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                //         sample_points[1].push_back(temp);
                //         sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                //         map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                //     }
                // }
            }

            sample_points[1].push_back(deformed.segment<2>(2*(num_x_2-1+num_x*num_y)));
            sample_normals[1].push_back(Eigen::Vector2d(1.0,-1.0).normalized());
            map_sample_to_deform[1].push_back(num_x_2-1+num_x*num_y);
            if(true)
                map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
            // corners[1].push_back(sample_points[1].back());

            for(int j=1; j<num_y_2; ++j)
            {
                if(USE_MORE_POINTS)
                {
                    prev = sample_points[1].back();
                    next = deformed.segment<2>(2*(num_x_2-1+j*num_x_2+num_x*num_y));
                    for(int l=1; l<sample_res; ++l)
                    {
                        //std::cout<<i<<" "<<j<<std::endl;
                        Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                        sample_points[1].push_back(temp);
                        sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                        map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                    }
                }

                if(j == num_y_2-1) continue;

                sample_points[1].push_back(deformed.segment<2>(2*(num_x_2-1+j*num_x_2+num_x*num_y)));
                sample_normals[1].push_back(Eigen::Vector2d(1.0,0.0).normalized());
                map_sample_to_deform[1].push_back(num_x_2-1+j*num_x_2+num_x*num_y);
                if(true)
                    map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
                
                // if(j == num_y_2-2 && USE_MORE_POINTS)
                // {
                //     prev = sample_points[1].back();
                //     next = deformed.segment<2>(2*(num_x_2*num_y_2-1+num_x*num_y));
                //     for(int l=1; l<sample_res; ++l)
                //     {
                //         //std::cout<<i<<" "<<j<<std::endl;
                //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                //         sample_points[1].push_back(temp);
                //         sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                //         map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                //     }
                // }
            }

            sample_points[1].push_back(deformed.segment<2>(2*(num_x_2*num_y_2-1+num_x*num_y)));
            sample_normals[1].push_back(Eigen::Vector2d(1.0,1.0).normalized());
            map_sample_to_deform[1].push_back(num_x_2*num_y_2-1+num_x*num_y);
            corners[1].push_back(sample_points[1].back());

            if(true)
                map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));
            for(int i=num_x_2-2; i>=0; --i)
            {
                if(USE_MORE_POINTS)
                {
                    prev = sample_points[1].back();
                    next = deformed.segment<2>(2*(i+(num_y_2-1)*num_x_2+num_x*num_y));
                    for(int l=1; l<sample_res; ++l)
                    {
                        //std::cout<<i<<" "<<j<<std::endl;
                        Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                        sample_points[1].push_back(temp);
                        sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                        map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                    }
                }

                if(i == 0) continue;

                sample_points[1].push_back(deformed.segment<2>(2*(i+(num_y_2-1)*num_x_2+num_x*num_y)));
                sample_normals[1].push_back(Eigen::Vector2d(0.0,1.0).normalized());
                map_sample_to_deform[1].push_back(i+(num_y_2-1)*num_x_2+num_x*num_y);
                if(true)
                    map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));

                // if(i == 1 && USE_MORE_POINTS)
                // {
                //     prev = sample_points[1].back();
                //     next = deformed.segment<2>(2*((num_y_2-1)*num_x_2+num_x*num_y));
                //     for(int l=1; l<sample_res; ++l)
                //     {
                //         //std::cout<<i<<" "<<j<<std::endl;
                //         Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                //         sample_points[1].push_back(temp);
                //         sample_normals[1].push_back(Eigen::Vector2d(-1.0,0.0).normalized());
                //         map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                //     }
                // }
            }
        }
    }
    // for(int i=0; i<object_num; ++i)
    // {
    //     std::cout<<"object num: "<<i<<" has size "<<sample_points[i].size()<<std::endl;
    // }

    // for(int i=0; i<sample_points[1].size(); ++i)
    // {
    //     std::cout<<sample_points[1][i].transpose()<<std::endl;
    // }
}

template<int dim>
void FEMSolver<dim>::samplePointsFromSTL(int pair_id)
{
    result_values.resize(object_num);
    SH_data.resize(object_num);

    map_sample_to_deform.resize(object_num);
    map_sample_to_deform_with_scale.resize(object_num);
        
    sample_points.resize(object_num);
    sample_normals.resize(object_num);
    corners.resize(object_num);

    Eigen::VectorXd prev, next;

    for(int i=0; i<object_num; ++i)
    {
        sample_points[i].clear();
        sample_normals[i].clear();
        map_sample_to_deform[i].clear();
        map_sample_to_deform_with_scale[i].clear();
        corners[i].clear();
    }

    std::vector<int> effective_slave_nodes = slave_nodes;
    std::vector<int> effective_master_nodes = master_nodes;

    if(dim == 2 && use_multiple_pairs)
    {
        effective_slave_nodes = multiple_slave_nodes[pair_id];
        effective_master_nodes = multiple_master_nodes[pair_id];
    }
    else if(dim == 3)
    {
        effective_slave_nodes.resize(slave_nodes_3d[pair_id].size());
        effective_master_nodes.resize(master_nodes_3d[pair_id].size());
        for(auto it = slave_nodes_3d[pair_id].begin(); it != slave_nodes_3d[pair_id].end(); ++it)
        {
            effective_slave_nodes[it->second] = it->first ;
        }
        for(auto it = master_nodes_3d[pair_id].begin(); it != master_nodes_3d[pair_id].end(); ++it)
        {
            effective_master_nodes[it->second] = it->first;
        }
    }

    // std::cout<<"Effective Slave Nodes: ";
    // for(int i=0; i<multiple_slave_nodes[pair_id].size(); ++i)
    // {
    //     std::cout<<" "<<multiple_slave_nodes[pair_id][i]<<" ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"Effective Master Nodes: ";
    // for(int i=0; i<multiple_master_nodes[pair_id].size(); ++i)
    // {
    //     std::cout<<" "<<multiple_master_nodes[pair_id][i]<<" ";
    // }
    // std::cout<<std::endl;

    
    for(int i=0; i<effective_master_nodes.size(); ++i)
    {
        effective_master_nodes[i];
        //deformed.segment<dim>(dim*effective_master_nodes[i]);
        sample_points[0].push_back(deformed.segment<dim>(dim*effective_master_nodes[i]));
        if(dim == 2)
            sample_normals[0].push_back(Eigen::Vector2d(-1.0,-1.0).normalized());
        map_sample_to_deform[0].push_back(effective_master_nodes[i]);
        map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,1.));

        if(USE_MORE_POINTS && dim == 2)
        {
            prev = sample_points[0].back();
            next = deformed.segment<dim>(dim*(effective_master_nodes[(i+1)%effective_master_nodes.size()]));
            for(int l=1; l<sample_res; ++l)
            {
                Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                sample_points[0].push_back(temp);
                if(dim == 2)
                    sample_normals[0].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
                map_sample_to_deform_with_scale[0].push_back(std::pair<int, double>(map_sample_to_deform[0].size()-1,double(sample_res-l)/double(sample_res)));
            }
        }
    }
    if(IMLS_BOTH)
    {
        for(int i=0; i<effective_slave_nodes.size(); ++i)
        {
            sample_points[1].push_back(deformed.segment<dim>(dim*effective_slave_nodes[i]));
            if(dim == 2)
                sample_normals[1].push_back(Eigen::Vector2d(-1.0,-1.0).normalized());
            map_sample_to_deform[1].push_back(effective_slave_nodes[i]);
            map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,1.));

            if(USE_MORE_POINTS && dim == 2)
            {
                prev = sample_points[1].back();
                next = deformed.segment<dim>(dim*(effective_slave_nodes[(i+1)%effective_slave_nodes.size()]));
                for(int l=1; l<sample_res; ++l)
                {
                    Eigen::VectorXd temp = ((sample_res-l)*prev+l*next)/sample_res;
                    sample_points[1].push_back(temp);
                    if(dim == 2) 
                        sample_normals[1].push_back(Eigen::Vector2d(0.0,-1.0).normalized());
                    map_sample_to_deform_with_scale[1].push_back(std::pair<int, double>(map_sample_to_deform[1].size()-1,double(sample_res-l)/double(sample_res)));
                }
            }
        }
    }
}

template<int dim>
void FEMSolver<dim>::buildConstraintsPointSet()
{
   

    constrained_points.resize(object_num);
    constrained_values.resize(object_num);
    constrained_normals.resize(object_num);

    for(int k=0; k<object_num; ++k)
    {
        //std::cout<<"sample points num: "<<sample_points[k].size()<<std::endl;
        int num_pts = sample_points[k].size();
        constrained_points[k].resize(3 * num_pts, dim);
        constrained_values[k].resize(3 * num_pts, 1);
        constrained_normals[k].resize(3 * num_pts, dim);

        if(use_Kernel_Regression)
        {
            constrained_points[k].resize(num_pts, dim);
            constrained_values[k].resize(num_pts, 1);
            constrained_normals[k].resize(num_pts, dim);
        }

        Eigen::MatrixXd original_pts, positive_pts, negative_pts;
        original_pts.resize(num_pts, dim);
        positive_pts.resize(num_pts, dim);
        negative_pts.resize(num_pts, dim);

        double bbd = 0;
        Eigen::MatrixXd P(num_pts, dim);
        for(int i=0; i<num_pts; ++i)
        {
            P.row(i) = sample_points[k][i];
        }

        if(!use_Kernel_Regression)
        {
            bbd = igl::bounding_box_diagonal(P);
        }
       

        //std::cout<<num_pts<<" "<<sample_normals.size()<<std::endl;
        

        for (int i = 0; i < num_pts; i++) {
            //std::cout<<i<<" "<<std::endl;
            constrained_points[k].row(i) = sample_points[k][i];
            original_pts.row(i) = sample_points[k][i];
            constrained_values[k](i, 0) = 0;

            if(!NO_SAMPLING)
                constrained_normals[k].row(i) = sample_normals[k][i];

            if(use_Kernel_Regression) continue;

            double eps = eps_scale * bbd;
            Eigen::VectorXd candidate_p2;


            while (true) {

                candidate_p2 = sample_points[k][i] + eps * sample_normals[k][i];

                int search_index = brute_force_NN(candidate_p2, k);
                //std::cout<<search_index<<" "<<i<<std::endl;
                if (search_index == i) break;
                else eps /= 2;
                Eigen::VectorXd test = P.row(i);
            }


            constrained_points[k].row(i + num_pts) = candidate_p2;
            positive_pts.row(i) = candidate_p2;
            constrained_values[k](i + num_pts) = eps;
            constrained_normals[k].row(i + num_pts) = sample_normals[k][i];


            eps = eps_scale * bbd;
            Eigen::VectorXd candidate_p3;
            while (true) {
                candidate_p3 = sample_points[k][i] - eps * sample_normals[k][i];
                int search_index = brute_force_NN(candidate_p3,k);
                if (search_index == i) break;
                else eps /= 2;
            }

            constrained_points[k].row(i + 2 * num_pts) = candidate_p3;
            negative_pts.row(i) = candidate_p3;
            constrained_values[k](i + 2 * num_pts) = -eps;
            constrained_normals[k].row(i + 2 * num_pts) = sample_normals[k][i];

        }
    }
    //std::cout<<constrained_points[0].rows()<<std::endl;
    // for(int i=0; i<constrained_points[0].rows();++i)
    // {
    //     std::cout<<"Index: "<<i<<" point: "<<constrained_points[0].row(i)<<std::endl;
    // }
}

template<int dim>
void FEMSolver<dim>::evaluateImplicitPotential(Eigen::MatrixXd& xs, int index)
{
    int num_sample_points = sample_points[index].size();
    int num_evaluate_points = xs.rows();
    result_values[index].setZero(num_evaluate_points,1);

    auto bb_min = xs.colwise().minCoeff().eval();
    auto bb_max = xs.colwise().maxCoeff().eval();

    Eigen::RowVector2d dims = bb_max - bb_min;
    double radius = dims.norm() * wendlandRadius;

    for(int i=0; i<num_evaluate_points; ++i)
    {
        Eigen::VectorXd cur_ptr = xs.row(i);
        std::vector<int> results;
        std::vector<double> res_dists;
        find_neighbor_pts(cur_ptr, radius, results, res_dists, index);

        int Num = results.size();
        std::vector<int> coeff = { 1,3,6 };

        //std::cout<<i<<" "<<num_evaluate_points<<" "<<Num<<std::endl;

        if (Num < coeff[polyDegree]) {
            if(cur_ptr(1) < HEIGHT_1)
                result_values[index](i,0) = 100000000.0;
            else
                result_values[index](i,0) = 100000000.0;
            continue;
        }else{
            // Build W(x)
            Eigen::MatrixXd w_x_pi = Eigen::MatrixXd::Zero(Num, Num);
            for (int j = 0; j < Num; j++) {
                w_x_pi(j, j) = pow((1 - res_dists[j] / radius), 4) * (4 * res_dists[j] / radius + 1);
            }

            // Build B(pn)
            Eigen::MatrixXd b_pi;
            Eigen::VectorXd b_x;

            if (polyDegree == 0) 
            {
                b_pi = Eigen::MatrixXd::Zero(Num, 1);
                b_x = Eigen::VectorXd::Zero(1);
                for (int j = 0; j < Num; j++) {
                    b_pi(j, 0) = 1;
                }
                b_x << 1;
            }
            else if (polyDegree == 1) 
            {
                b_pi = Eigen::MatrixXd::Zero(Num, 3);
                b_x = Eigen::VectorXd::Zero(3);
                for (int j = 0; j < Num; j++) {
                    b_pi(j, 0) = 1;
                    b_pi(j, 1) = constrained_points[index](results[j], 0);
                    b_pi(j, 2) = constrained_points[index](results[j], 1);
                }
                b_x << 1, cur_ptr(0), cur_ptr(1);
            }
            else if (polyDegree == 2) 
            {
                b_pi = Eigen::MatrixXd::Zero(Num, 6);
                b_x = Eigen::VectorXd::Zero(6);
                for (int j = 0; j < Num; j++) {
                    b_pi(j, 0) = 1;
                    b_pi(j, 1) = constrained_points[index](results[j], 0);
                    b_pi(j, 2) = constrained_points[index](results[j], 1);
                    b_pi(j, 3) = constrained_points[index](results[j], 0)*constrained_points[index](results[j], 0);
                    b_pi(j, 4) = constrained_points[index](results[j], 1)*constrained_points[index](results[j], 1);
                    b_pi(j, 5) = constrained_points[index](results[j], 0)*constrained_points[index](results[j], 1);
                }
                b_x << 1, cur_ptr(0), cur_ptr(1), cur_ptr(0)*cur_ptr(0), cur_ptr(1)*cur_ptr(1), cur_ptr(0)*cur_ptr(1);
            }

            Eigen::VectorXd f_pi = Eigen::VectorXd::Zero(Num);
            for (int j = 0; j < Num; j++) {
                f_pi(j) = constrained_values[index](results[j]) + (cur_ptr.transpose() - constrained_points[index].row(results[j])).dot(constrained_normals[index].row(results[j]));
            }

            Eigen::MatrixXd A = w_x_pi * b_pi;
            Eigen::VectorXd b = w_x_pi * f_pi;
            Eigen::VectorXd c = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            
            result_values[index](i,0) = c.dot(b_x);
        }
    }
}

template<int dim>
void FEMSolver<dim>::evaluateImplicitPotentialKR(Eigen::MatrixXd& xs, bool update, int index, int pair_id, bool eval_same_side)
{
    int num_sample_points = sample_points[index].size();
    int num_evaluate_points = xs.rows();
    result_values[index].setZero(num_evaluate_points,1);
    result_grad.resize(num_evaluate_points);
    result_hess.resize(num_evaluate_points);

    auto bb_min = xs.colwise().minCoeff().eval();
    auto bb_max = xs.colwise().maxCoeff().eval();

    samples_grad.clear();
    samples_hess.clear();
    Eigen::MatrixXd Zeros;
    Zeros.setZero(dim,dim);
    Eigen::VectorXd Zeros2;
    Zeros2.setZero(dim);

    //std::cout<<"num of evaluated points: "<<num_evaluate_points<<std::endl;

    for(int i=0; i<VS.rows(); ++i)
    {
        samples_grad.push_back(Zeros2);
        samples_hess.push_back(Zeros);
    }

    //Eigen::RowVector2d dims = bb_max - bb_min;

    // Build current V and F
    if(USE_FROM_STL && dim == 2)
    {
        for(int i=0; i<Vs[0].rows(); ++i)
        {
            Vs[0](i,0) = deformed(dim*i);
            Vs[0](i,1) = deformed(dim*i+1);
            if(dim == 2)
                Vs[0](i,2) = 0;
            else
                Vs[0](i,1) = deformed(dim*i+2);
        }

        for(int i=0; i<Vs[1].rows(); ++i)
        {
            Vs[1](i,0) = deformed(dim*(i+Vs[0].rows()));
            Vs[1](i,1) = deformed(dim*(i+Vs[0].rows())+1);
            if(dim == 2)
                Vs[0](i,2) = 0;
            else
                Vs[0](i,1) = deformed(dim*(i+Vs[0].rows())+2);
        }
    }
    

    //double radius = dims.norm() * wendlandRadius;
    //std::cout<< dims.norm() * wendlandRadius<<std::endl;

    double radius = 0.01;
    if(SLIDING_TEST) radius = 0.05;
    if(PULL_TEST) radius = 1.5;
    if(IMLS_BOTH)
        radius = 0.5;
    if(USE_FROM_STL)
    {
        if(USE_SHELL) radius = 0.05;
        else if(TEST_CASE == 0) radius = 0.2;
        else if(TEST_CASE == 2) radius = 0.1;
        else if(TEST_CASE == 1) radius = 0.1; //0.08
        else if(TEST_CASE == 3) radius  = 0.2;
    }
    //if(USE_MORE_POINTS) radius = 0.1;
    //if(index == 1) radius /= SCALAR;

    //updateHashDataStructure(radius,index);

    // tbb::parallel_for(tbb::blocked_range<int>(0,xs.rows()),
    //                    [&](tbb::blocked_range<int> r)
    // {
    //for(int i=r.begin(); i<r.end(); ++i)
    for(int i=0; i<xs.rows(); ++i)
    {
        //std::cout<<i<<" out of "<<num_evaluate_points<<std::endl;
        Eigen::VectorXd cur_ptr = xs.row(i);

        std::vector<int> results;
        std::vector<double> res_dists;
        find_neighbor_pts(cur_ptr, radius, results, res_dists, index);
        int Num = results.size();
        //std::cout<<"Num of nerighbors: "<<Num<<std::endl;

        Eigen::VectorXd dist_grad(dim*(num_sample_points+1));
        Eigen::MatrixXd dist_hess(dim*(num_sample_points+1),dim*(num_sample_points+1));
        if(!USE_SHELL)
        {
            dist_grad.setZero();
            dist_hess.setZero();
        }
        StiffnessMatrix dist_hess_s(dim*(num_sample_points+1),dim*(num_sample_points+1));
        if(USE_SHELL)
        {
            dist_hess_s.setZero();
        }
        

        // std::cout<<"cur point: "<<cur_ptr.transpose()<<std::endl;
        // for(int j=0; j<results.size(); ++j)
        // {
        //     std::cout<<results[j]<<std::endl;
        // }

        // if(update)
        // {
        //     if(boundary_info[i].index != index) continue;
        //     std::cout<<"node index: "<<boundary_info[i].slave_index<<std::endl;
        // }

        //std::cout<<i<<" Fuck "<<constrained_points[index].rows()<<std::endl;

        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        if(Num == 0)
        {
            // TODO: Solve this
            //std::cout<<index<<std::endl;
            bool inside = false;
            Eigen::VectorXd cur_ptr_3d(3);
            cur_ptr_3d(0) = cur_ptr(0);
            cur_ptr_3d(1) = cur_ptr(1);
            cur_ptr_3d(2) = 0;
            
            // if(USE_FROM_STL && dim == 2) inside = checkInsideRectangleFromSTL(cur_ptr_3d, index);
            // else if(dim == 2) inside = checkInsideRectangle(cur_ptr, index);
            // else if(dim == 3) inside = checkInsideMeshFromSTL3D(cur_ptr,index,pair_id);

            if(inside || SLIDING_TEST)
                result_values[index](i,0) = -100000000.0;
            else
                result_values[index](i,0) = 100000000.0;

            if(update)
            {
                int new_index;
                if(dim == 3)
                    new_index = boundary_info_start_3d[index][pair_id]+i;
                if(dim == 2)
                {
                    if(use_multiple_pairs)
                        new_index = boundary_info_start[index][pair_id]+i;
                    else
                        new_index = index*slave_nodes.size()+i;
                }
                
                if(USE_VIRTUAL_NODE) new_index = index*virtual_slave_nodes.size()+i;
                if(USE_NEW_FORMULATION && eval_same_side) new_index = (1-index)*slave_nodes.size()+i;
                if(USE_VIRTUAL_NODE && USE_NEW_FORMULATION && eval_same_side) new_index = (1-index)*virtual_slave_nodes.size()+i;
                if(!USE_NEW_FORMULATION || !eval_same_side)
                {
                    boundary_info[new_index].dist = result_values[index](i,0);
                    // boundary_info[new_index].dist_grad = dist_grad;
                    // //std::cout<<"node index: "<<boundary_info[new_index].slave_index<<" size: "<<dist_grad.size()<<std::endl;
                    // if(!USE_SHELL)
                    //     boundary_info[new_index].dist_hess = dist_hess;
                    // else
                    //     boundary_info[new_index].dist_hess_s = dist_hess_s;
                    // boundary_info[new_index].results = {};
                }
                else
                {
                    boundary_info_same_side[new_index].dist = result_values[index](i,0);
                    boundary_info_same_side[new_index].dist_grad = dist_grad;
                    //std::cout<<"node index: "<<boundary_info[new_index].slave_index<<" size: "<<dist_grad.size()<<std::endl;
                    if(!USE_SHELL)
                        boundary_info[new_index].dist_hess = dist_hess;
                    else
                        boundary_info[new_index].dist_hess_s = dist_hess_s;
                    boundary_info_same_side[new_index].results = {};
                }
                // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

                // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;
                
            }
        }
        else
        {
            // if(Num > 1)
            //     std::cout<<i<<" "<<Num<<std::endl;
            Eigen::VectorXd s(dim*num_sample_points+dim);
            s.setZero();
            //std::cout<<i<<" Fuck "<<constrained_points[index].rows()<<std::endl;

            std::unordered_map<int, int> hm;
            std::vector<std::vector<int>> normal_pairs(Num, std::vector<int>(normal_num+1));
            std::vector<std::vector<Eigen::VectorXd>> normal_pairs3D(Num);


            std::vector<int> useful_index;
            std::vector<int> results_index(Num);
            int hm_size = 0;
            for(int j=0; j<Num; ++j)
            {
                if (hm.find(results[j]) == hm.end())
                {
                    hm[results[j]] = hm_size;
                    hm_size++;
                    useful_index.push_back(results[j]);
                }
                results_index[j] = hm[results[j]];

                if(dim == 2)
                {
                    for(int k = -normal_num/2; k<=normal_num/2; ++k)
                    {
                        int normal_index = results[j] + k;
                        if(normal_index < 0) normal_index += num_sample_points;
                        else if(normal_index >= num_sample_points) normal_index -= num_sample_points;
                        if (hm.find(normal_index) == hm.end())
                        {
                            hm[normal_index] = hm_size;
                            hm_size++;
                            useful_index.push_back(normal_index);
                        }
                        //std::cout<<k+normal_num/2<<" "<<normal_index<<" "<<hm[normal_index]<<std::endl;
                        normal_pairs[j][k+normal_num/2] = hm[normal_index];
                    }
                }
                else if(dim == 3)
                {
                    // Use another way to calculate the "Contributing Vertex"
                    int global_index = map_sample_to_deform[index][results[j]];
                    
                    // if(cur_ptr(0) == 0 && cur_ptr(2) == 0)
                    //     std::cout<<"Closest Face of "<<cur_ptr.transpose()<<std::endl;
                    for(int k=0; k<vertex_triangle_indices[global_index].size(); ++k)
                    {
                        int face_id = vertex_triangle_indices[global_index][k];
                        // if(cur_ptr(0) == 0 && cur_ptr(2) == 0)
                        // {
                        //     std::cout<<FS.row(face_id)<<std::endl;
                        // }

                        Eigen::VectorXd tri(3);
                        bool success = true;
                        for(int l=0; l<3; ++l)
                        {
                            int vertex_index = -1;
                            if(index == 0)
                            {   
                                if(master_nodes_3d[pair_id].find(FS(face_id,l)) != master_nodes_3d[pair_id].end())
                                    vertex_index = master_nodes_3d[pair_id][FS(face_id,l)];
                                else
                                {
                                    int new_index = boundary_info_start_3d[index][pair_id]+i;
                                    int node_index = 0;
                                    if(!USE_NEW_FORMULATION || !eval_same_side) 
                                        node_index = boundary_info[new_index].slave_index;
                                    else
                                        node_index = boundary_info_same_side[new_index].slave_index;
                                    std::cout<<"Undeformed: "<<undeformed.segment<dim>(dim*node_index).transpose()<<std::endl;
                                    std::cout<<"u: "<<u.segment<dim>(dim*node_index).transpose()<<std::endl;
                                    std::cout<<"Deformed: "<<deformed.segment<dim>(dim*node_index).transpose()<<std::endl;
                                    std::cout<<"Search point: "<<cur_ptr.transpose()<<" in face: "<<face_id<<" cannot find: "<<FS(face_id,l)<<std::endl;
                                }
                                    
                            }  
                            else
                            {
                                if(slave_nodes_3d[pair_id].find(FS(face_id,l)) != slave_nodes_3d[pair_id].end())
                                    vertex_index = slave_nodes_3d[pair_id][FS(face_id,l)];
                            }
                            if(vertex_index == -1)
                            {
                                success = false;
                                break;
                            } 

                            if(hm.find(vertex_index) == hm.end())
                            {
                                hm[vertex_index] = hm_size;
                                hm_size++;
                                useful_index.push_back(vertex_index);
                            }
                            tri(l) = hm[vertex_index];
                        }
                        if(success)
                            normal_pairs3D[j].push_back(tri);
                    }
                }
                
            }
            
            // VectorXvar n(2*num_sample_points);
            s(0) = cur_ptr(0);
            s(1) = cur_ptr(1);
            if(dim == 3)
                s(2) = cur_ptr(2);

            for(int j=0; j<hm.size(); ++j)
            {
                //std::cout<<index<<" "<<constrained_points[index].rows()<<" "<<useful_index[j]<<std::endl;
                s(dim*j+dim) = constrained_points[index](useful_index[j],0);
                s(dim*j+dim+1) = constrained_points[index](useful_index[j],1);
                if(dim == 3)
                    s(dim*j+dim+2) = constrained_points[index](useful_index[j],2);

                // n(2*j) = constrained_normals(j,0);
                // n(2*j+1) = constrained_normals(j,1);
            }
        

            
            double u = evaulateIMLSCPPAD(s,normal_pairs,results_index,radius,dist_grad,dist_hess,update,0,normal_pairs3D,index);

            // VectorXvar s(2*num_sample_points+2);
            // // VectorXvar n(2*num_sample_points);
            // s(0) = cur_ptr(0);
            // s(1) = cur_ptr(1);
            // for(int j=0; j<num_sample_points; ++j)
            // {
            //     s(2*j+2) = constrained_points[index](j,0);
            //     s(2*j+3) = constrained_points[index](j,1);
            //     // n(2*j) = constrained_normals(j,0);
            //     // n(2*j+1) = constrained_normals(j,1);
            // }
                
            // var u = computeSignedDistanceAll(s,results,radius);


            // std::cout<<u<<std::endl;
            // std::cout<<dist_grad.transpose()<<std::endl;
            // std::cout<<dist_hess<<std::endl;


            // std::cout<<"cur point: "<<cur_ptr.transpose()<<std::endl;
            if(update)
            {
                int new_index;
                if(dim == 3)
                    new_index = boundary_info_start_3d[index][pair_id]+i;
                if(dim == 2)
                {
                    if(use_multiple_pairs)
                        new_index = boundary_info_start[index][pair_id]+i;
                    else if(USE_VIRTUAL_NODE)
                        new_index = index*virtual_slave_nodes.size()+i;
                    else    
                        new_index = index*slave_nodes.size()+i;
                }
                    
                //if(boundary_info[new_index].index != index) continue;
                if(USE_NEW_FORMULATION && eval_same_side) new_index = (1-index)*slave_nodes.size()+i;
                if(USE_VIRTUAL_NODE && USE_NEW_FORMULATION && eval_same_side) new_index = (1-index)*virtual_slave_nodes.size()+i;


                if(!USE_NEW_FORMULATION || !eval_same_side)
                {
                    boundary_info[new_index].dist = (double)u;
            
                    convertToGlobalGradient(dist_grad,useful_index,index);
                    
                    //std::cout<<new_index<<" "<<boundary_info.size()<<std::endl;
                    boundary_info[new_index].dist_grad = convertToGlobalGradient(dist_grad,useful_index,index);
                    if(!USE_SHELL)
                        boundary_info[new_index].dist_hess = convertToGlobalHessian(dist_hess,useful_index,index);
                    else
                        boundary_info[new_index].dist_hess_s = convertToGlobalHessian(dist_hess,useful_index,index).sparseView();
                    
                    // if(u < 0)
                    // {
                    //     for(int j=0; j<Num; ++j)
                    //     {
                    //         std::cout<<"global Index: "<<map_sample_to_deform[index][results[j]]<<std::endl;
                    //         for(int l=0; l<normal_pairs3D[j].size(); ++l)
                    //         {
                    //             std::cout<<"Normal Contribution: ";
                    //             for(int fi = 0; fi<3; ++fi)
                    //                 std::cout<<map_sample_to_deform[index][normal_pairs3D[j][l](fi)]<<" ";
                    //             std::cout<<std::endl;
                    //         }
                    //     }
                    //     std::cout<<"Useful Index: ";
                    //     for(int j=0; j<useful_index.size(); ++j)
                    //     {
                    //         std::cout<<map_sample_to_deform[index][useful_index[j]]<<" ";
                    //     }
                    //     std::cout<<std::endl;
                    //     std::cout<<"Dist grad: "<<dist_grad.transpose()<<std::endl;
                    // } 

                    // if(u < 0)
                    // {
                    //     std::cout<<"gradient: ";
                    //     std::cout<<boundary_info[new_index].dist_grad.transpose()<<std::endl;
                    // }
                        
                    //dist_hess = hessian(u, s, dist_grad);

                                    
                    // boundary_info[new_index].dist_grad = dist_grad;
                    // boundary_info[new_index].dist_hess = dist_hess;

                    boundary_info[new_index].results.clear();
                    //std::cout<<"dist grad size "<<dist_grad.size()<<std::endl;
                    for(int j=0; j<map_sample_to_deform[index].size(); ++j)
                    {
                        //std::cout<<"connecting points: "<<VS.row(map_sample_to_deform[results[j]])<<std::endl;
                        boundary_info[new_index].results.push_back(map_sample_to_deform[index][j]);
                        //std::cout<<map_sample_to_deform[index][j]<<" ";
                    }
                    //std::cout<< boundary_info[new_index].dist_hess.rows()/3<<" "<<boundary_info[new_index].results.size()<<std::endl;

                    //std::cout<<std::endl;
                }
                else
                {
                    boundary_info_same_side[new_index].dist = (double)u;
            
                    convertToGlobalGradient(dist_grad,useful_index,index);
                    //std::cout<<new_index<<" "<<boundary_info.size()<<std::endl;
                    boundary_info_same_side[new_index].dist_grad = convertToGlobalGradient(dist_grad,useful_index,index);
                    boundary_info_same_side[new_index].dist_hess = convertToGlobalHessian(dist_hess,useful_index,index);
                    boundary_info_same_side[new_index].results.clear();
                    for(int j=0; j<map_sample_to_deform[index].size(); ++j)
                    {
                        //std::cout<<"connecting points: "<<VS.row(map_sample_to_deform[results[j]])<<std::endl;
                        boundary_info_same_side[new_index].results.push_back(map_sample_to_deform[index][j]);
                        //std::cout<<map_sample_to_deform[index][j]<<" ";
                    }
                }
            }
            result_values[index](i,0) = (double)u;

        }
        // result_grad[i] = dist_grad;
        // result_hess[i] = dist_hess;
        //std::cout<<"iter: "<<i<<std::endl;
    }
// });
    
    //std::cout<<result_values[0].rows()<<std::endl;
}

template<int dim>
void FEMSolver<dim>::evaluateImplicitPotentialKRGradient(Eigen::MatrixXd& ps, std::vector<Eigen::VectorXd>& dfsdps, Eigen::VectorXd& fs, bool calculate_grad, int index)
{
    index = 1;

    int num_sample_points = sample_points[index].size();
    int num_evaluate_points = ps.rows();

    result_values[index].setZero(num_evaluate_points,1);
    result_grad.resize(num_evaluate_points);
    result_hess.resize(num_evaluate_points);

    // auto bb_min = ps.colwise().minCoeff().eval();
    // auto bb_max = ps.colwise().maxCoeff().eval();

    fs.setZero(num_evaluate_points);
    dfsdps.resize(num_evaluate_points);

    samples_grad.clear();
    samples_hess.clear();
    Eigen::MatrixXd Zeros;
    Zeros.setZero(2,2);
    Eigen::VectorXd Zeros2;
    Zeros2.setZero(2);

    //std::cout<<"num of evaluated points: "<<num_evaluate_points<<std::endl;

    for(int i=0; i<VS.rows(); ++i)
    {
        samples_grad.push_back(Zeros2);
        samples_hess.push_back(Zeros);
    }

    //Eigen::RowVector2d dims = bb_max - bb_min;

    double radius = 0.6;

    for(int i=0; i<num_evaluate_points; ++i)
    {
        Eigen::VectorXd cur_ptr = ps.row(i);

        std::vector<int> results;
        std::vector<double> res_dists;
        find_neighbor_pts(cur_ptr, radius, results, res_dists, index);
        int Num = results.size();

        Eigen::VectorXd dist_grad(2*(num_sample_points+1));
        Eigen::MatrixXd dist_hess(2*(num_sample_points+1),2*(num_sample_points+1));
        dist_grad.setZero();
        dist_hess.setZero();

        dfsdps[i].setZero(2);
            
        if(Num == 0)
        {
            fs(i) = 100000000.0;
            dfsdps[i] = Zeros2;
            std::cerr<<"Error: Too far away from the surface!"<<std::endl;
        }
        else
        {
            Eigen::VectorXd s(2*num_sample_points+2);

            std::unordered_map<int, int> hm;
            std::vector<std::vector<int>> normal_pairs(Num, std::vector<int>(normal_num+1));
            std::vector<int> useful_index;
            std::vector<int> results_index(Num);
            int hm_size = 0;
            for(int j=0; j<Num; ++j)
            {
                if (hm.find(results[j]) == hm.end())
                {
                    hm[results[j]] = hm_size;
                    hm_size++;
                    useful_index.push_back(results[j]);
                }
                results_index[j] = hm[results[j]];
                for(int k = -normal_num/2; k<=normal_num/2; ++k)
                {
                    int normal_index = results[j] + k;
                    if(normal_index < 0) normal_index += num_sample_points;
                    else if(normal_index >= num_sample_points) normal_index -= num_sample_points;
                    if (hm.find(normal_index) == hm.end())
                    {
                        hm[normal_index] = hm_size;
                        hm_size++;
                        useful_index.push_back(normal_index);
                    }
                    normal_pairs[j][k+normal_num/2] = hm[normal_index];
                }
            }
            
            s(0) = cur_ptr(0);
            s(1) = cur_ptr(1);

            for(int j=0; j<hm.size(); ++j)
            {
                s(2*j+2) = constrained_points[index](useful_index[j],0);
                s(2*j+3) = constrained_points[index](useful_index[j],1);
            }
            
            double u = evaulateIMLSCPPAD(s,normal_pairs,results_index,radius,dist_grad,dist_hess,1,1);
            fs(i) = u;
            dfsdps[i] = dist_grad.segment<2>(0);

        }
    }
}

template<int dim>
void FEMSolver<dim>::testIMLS(Eigen::MatrixXd& xs, int index)
{
    // sample_points.clear();
    // sample_normals.clear();
    if(USE_FROM_STL || TEST || SLIDING_TEST || PULL_TEST)
    {
        if(use_multiple_pairs) samplePointsFromSTL(0);
        else samplePointsFromSTL(0);
    }
        
    else
        samplePoints();
    buildConstraintsPointSet();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    if(use_Kernel_Regression)
    {
        if(use_multiple_pairs)
            evaluateImplicitPotentialKR(xs,false,index,0);
        else
            evaluateImplicitPotentialKR(xs,false,index,0);
    }    
    else
    {
        evaluateImplicitPotential(xs,index);
    } 
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
}

template<int dim>
void FEMSolver<dim>::computeSlaveCenter()
{
    int num_slave_nodes = slave_nodes.size();
    slave_center.setZero(2);
    for(int i=0; i<num_slave_nodes; ++i)
    {
        slave_center(0) += deformed(2*slave_nodes[i]);
        slave_center(1) += deformed(2*slave_nodes[i]+1);
    }
        
    slave_center/=num_slave_nodes;
}

template<int dim>
void FEMSolver<dim>::testDerivativeIMLS()
{
    object_num = 1;
    sample_points.resize(1);

    sample_points[0].push_back(Eigen::Vector2d(0.0,0.0));
    sample_points[0].push_back(Eigen::Vector2d(0.25,0.0));
    sample_points[0].push_back(Eigen::Vector2d(0.5,0.0));
    sample_points[0].push_back(Eigen::Vector2d(0.75,0.0));
    sample_points[0].push_back(Eigen::Vector2d(1.0,0.0));
    sample_points[0].push_back(Eigen::Vector2d(1.0,1.0));
    sample_points[0].push_back(Eigen::Vector2d(0.0,1.0));

    buildConstraintsPointSet();

    Eigen::MatrixXd xs(1,2);
    xs.setZero();

    double x_start = 0;
    double x_end = 1;
    double y = 0.0;
    double eps = 1e-7;
    for(int i=0; i<1000; ++i)
    {
        xs(0,0) = ((1000-i)*x_start+i*x_end)/1000;
        xs(0,1) = y;

        boundary_info.resize(1);
        evaluateImplicitPotentialKR(xs,true);

        Eigen::VectorXd grad = boundary_info[0].dist_grad;
        Eigen::MatrixXd hess = boundary_info[0].dist_hess;

        std::cout<<grad.transpose()<<std::endl;
        std::cout<<hess<<std::endl;

        for(int j=0; j<8; ++j)
        {
            sample_points[0][j/2](j%2) += eps;
            buildConstraintsPointSet();
            evaluateImplicitPotentialKR(xs,true);
            Eigen::VectorXd grad_p = boundary_info[0].dist_grad;
            for(int k=0; k<10; ++k)
                std::cout<<"pos: "<<xs(0)<<" index "<<j+2<<" "<<k<<" "<< hess(j+2,k)<< " " << (grad_p(k)-grad(k))/eps<<std::endl;
            sample_points[0][j/2](j%2) -= eps;
        }
    }

    object_num = 2;
}



template class FEMSolver<2>;
template class FEMSolver<3>;