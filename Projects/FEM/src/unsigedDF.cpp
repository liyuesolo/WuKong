#include "../include/FEMSolver.h"
#include "../include/VecMatDef.h"

template <int dim>
bool Rosenbrock<dim>::Evaluate(const double* parameters,
                          double* cost,
                          double* gradient) const 
{
    const double t = parameters[0];
    double temp;
    if(USE_RIMLS)
        cost[0] = solver->evaluateUnsignedDistanceSqRIMLS(index,t,du,ipc_stepsize,temp);
    else
        cost[0] = solver->evaluateUnsignedDistanceSq(index,t,du,ipc_stepsize,temp);
    if (gradient != NULL) {
        if(USE_RIMLS)
            gradient[0] = solver->evaluateUnsignedDistanceSqGradientRIMLS(index,t,du,ipc_stepsize);
        else
            gradient[0] = solver->evaluateUnsignedDistanceSqGradient(index,t,du,ipc_stepsize);
    }
    return true;
}

template <int dim>
double MyOptimization<dim>:: evaluateSample( const boost::numeric::ublas::vector<double> &query )
{
    const double t = query[0];
    double temp;
    double result = solver->evaluateUnsignedDistanceSq(index,t,du,1.,temp);
    return result;
}

double sign(double t)
{
    if (t<0) return -1.;
    else if (t>0) return 1.;
    else return 0.;
}


template <int dim>
double FEMSolver<dim>::evaluateUnsignedDistanceSq(int i, double t, const VectorXa& du, double ipc_stepsize, double& distance)
{
    //if(t >= 1 || t < 0) return 1;
    double t0 = t;
    t = min(t,ipc_stepsize);
    t = max(t,0.);
    VectorXa deformed_cur = deformed+t*du;
    // Tree local_accelerator;
    // local_accelerator.clear();

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     if(is_surface_vertex[i] == 0) continue;
    //     Eigen::VectorXd point = deformed.segment<dim>(dim*i);
    //     local_accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    // }

    std::vector<int> local_current_index;

    Vector3a xi = deformed_cur.segment<3>(3*i);

    //Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

    // std::vector<std::tuple<Point_d, int>> points_query;
    // accelerator.search(std::back_inserter(points_query), fs);

    //for(int k=0; k<points_query.size(); ++k)
    for(int k=0; k<num_nodes; ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(3*k);
        //int index = std::get<1>(points_query[k]);
        int index = k;
        if(is_surface_vertex[index] && (geodist_close_matrix.coeff(i,index) == 0) && (ck-xi).norm()<=radius)
        {
            local_current_index.push_back(index);
        }     
    }

    //if(local_current_index.size() == 0) return 1.+fabs(t-t0);
    if(local_current_index.size() == 0)
    {
        distance = 0;
        return 1.+fabs(t-t0);
    } 

    AScalar fx = 0;
    AScalar gx = 0;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        int valence = nodes_adjacency[local_current_index[k]].size();
        if(valence > 11) std::cout<<"WTF!!!!"<<std::endl;
        VectorXa vs;

        if(valence <= 6)
        {
            vs.resize(18);
            for(int a=0; a<6; ++a)
            {
                vs.segment<3>(a*3) = ck;
            }
        }else
        {
            vs.resize(33);
            for(int a=0; a<11; ++a)
            {
                vs.segment<3>(a*3) = ck;
            }
        }

        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
        }
        
        if((ck-xi).norm()<=radius) {
            if(valence <= 6)
            {
                fx += fx_func(xi, vs, ck, radius);
                gx += gx_func(xi, ck, radius);
            }else
            {
                fx += fx_func11(xi, vs, ck, radius);
                gx += gx_func11(xi, ck, radius);
            }
        }
        // std::cout<<xi.transpose()<<std::endl;
        // std::cout<<vs.transpose()<<std::endl;
        // std::cout<<ck.transpose()<<std::endl;
        // std::cout<<radius<<std::endl;
        // std::cout<<compute_normal(xi, vs, ck, radius)<<std::endl;
    }

    //if(abs(gx)<1e-6) return 1.+fabs(t-t0);
    if(abs(gx)<1e-6)
    {
        distance = 0;
        return 1.+fabs(t-t0);
    } 

    AScalar dist = fx/(gx*radius);

    //if(i ==2415) std::cout<<t<<" "<<dist<<std::endl;
    //std::cout<<dist<<" "<<fx<<" "<<gx<<std::endl;
    //return log(dist*dist)+fabs(t-t0);
    //return dist*dist+fabs(t-t0);
    distance = fx/gx;
    return fabs(dist)+fabs(t-t0);
}

template <int dim>
double FEMSolver<dim>::evaluateUnsignedDistanceSqGradient(int i, double t, const VectorXa& du, double ipc_stepsize)
{
    if(t >= ipc_stepsize || t <= 0.) return sign(t);
    t = min(t,ipc_stepsize);
    t = max(t,0.);
    VectorXa deformed_cur = deformed+t*du;
    // Tree local_accelerator;
    // local_accelerator.clear();

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     if(is_surface_vertex[i] == 0) continue;
    //     Eigen::VectorXd point = deformed.segment<dim>(dim*i);
    //     local_accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    // }

	std::vector<int> local_current_index;

    Vector3a xi = deformed_cur.segment<3>(3*i);

    //Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

    // std::vector<std::tuple<Point_d, int>> points_query;
    // accelerator.search(std::back_inserter(points_query), fs);

    //for(int k=0; k<points_query.size(); ++k)
    for(int k=0; k<num_nodes; ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(3*k);
        //int index = std::get<1>(points_query[k]);
        int index = k;
        if(is_surface_vertex[index] && (geodist_close_matrix.coeff(i,index) == 0) && (ck-xi).norm()<=radius)
        {
            local_current_index.push_back(index);
        }     
    }

    if(local_current_index.size() == 0) return sign(t);

    VectorXa gradient(num_nodes*dim);
    gradient.setZero();
    VectorXa sum_dfdx(dim*num_nodes);
    sum_dfdx.setZero();
    VectorXa sum_dgdx(dim*num_nodes);
    sum_dgdx.setZero();

    AScalar fx = 0;
    AScalar gx = 0;

    VectorXa ele_dfdx; 
    VectorXa ele_dgdx;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        int valence = nodes_adjacency[local_current_index[k]].size();
        Eigen::VectorXi valence_indices;
        VectorXa vs;

        if(valence <= 6)
        {
            ele_dfdx.resize(24);
            ele_dgdx.resize(24);
            vs.resize(18);
            valence_indices.resize(8);
            for(int a=0; a<6; ++a)
            {
                vs.segment<3>(a*3) = ck;
                valence_indices(a+2)=(local_current_index[k]);
            }
        }else
        {
            ele_dfdx.resize(39);
            ele_dgdx.resize(39);
            vs.resize(33);
            valence_indices.resize(13);
            for(int a=0; a<11; ++a)
            {
                vs.segment<3>(a*3) = ck;
                valence_indices(a+2)=(local_current_index[k]);
            }
        }
    

        valence_indices(0)=(i);
        valence_indices(1)=(local_current_index[k]);

    
        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
            valence_indices(a+2) = nodes_adjacency[local_current_index[k]][a];
        }
        
        if((ck-xi).norm()<=radius) {
            if(valence <= 6)
            {
                ele_dfdx = dfdx_func(xi, vs, ck, radius);
                ele_dgdx = dgdx_func(xi, ck, radius);
                fx += fx_func(xi, vs, ck, radius);
                gx += gx_func(xi, ck, radius);
            }else
            {
                ele_dfdx = dfdx_func11(xi, vs, ck, radius);
                ele_dgdx = dgdx_func11(xi, ck, radius);
                fx += fx_func11(xi, vs, ck, radius);
                gx += gx_func11(xi, ck, radius);
            }
            IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
            IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
        }
    }
    AScalar dist = fx/(gx);

    if(abs(gx) < 1e-6) return sign(t);
    gradient.setZero();
    //gradient = 2*(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2)*dist);

    if(dist > 0)
        gradient = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2)*pow(radius,1));
    else
        gradient = -(gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2)*pow(radius,1));

    //if(i ==2415) std::cout<<t<<" gradient "<<gradient.dot(du)<<std::endl;
    return gradient.dot(du);
}

template <int dim>
double FEMSolver<dim>::evaluateUnsignedDistanceSqRIMLS(int i, double t, const VectorXa& du, double ipc_stepsize, double& distance)
{
    //if(t >= 1 || t < 0) return 1;
    double t0 = t;
    t = min(t,ipc_stepsize);
    t = max(t,0.);
    VectorXa deformed_cur = deformed+t*du;
    // Tree local_accelerator;
    // local_accelerator.clear();

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     if(is_surface_vertex[i] == 0) continue;
    //     Eigen::VectorXd point = deformed.segment<dim>(dim*i);
    //     local_accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    // }

    std::vector<int> local_current_index;

    Vector3a xi = deformed_cur.segment<3>(3*i);

    //Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

    // std::vector<std::tuple<Point_d, int>> points_query;
    // accelerator.search(std::back_inserter(points_query), fs);

    //for(int k=0; k<points_query.size(); ++k)
    for(int k=0; k<num_nodes; ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(3*k);
        //int index = std::get<1>(points_query[k]);
        int index = k;
        if(is_surface_vertex[index] && (geodist_close_matrix.coeff(i,index) == 0) && (ck-xi).norm()<=1.2*radius)
        {
            local_current_index.push_back(index);
        }     
    }

    //if(local_current_index.size() == 0) return 1.+fabs(t-t0);
    if(local_current_index.size() == 0)
    {
        distance = 0;
        return 1.+fabs(t-t0);
    }


    AScalar fx = 0;
    AScalar gx = 0;
    VectorXa N(3); N.setZero();
    VectorXa GF(3); GF.setZero();
    VectorXa GW(3); GW.setZero();
    AScalar fx2 = 0;
    AScalar gx2 = 0;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        int valence = nodes_adjacency[local_current_index[k]].size();
        if(valence > 11) std::cout<<"WTF!!!!"<<std::endl;
        VectorXa vs;

        vs.resize(33);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
        }


        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
        }

        if((ck-xi).norm()<=radius)
        {
            fx += fx_func11(xi, vs, ck, radius);
            gx += gx_func11(xi, ck, radius);
            GF(0) += sumGF1_func(xi, vs, ck, radius); GF(1) += sumGF2_func(xi, vs, ck, radius); GF(2) += sumGF3_func(xi, vs, ck, radius);
            GW(0) += sumGW1_func(xi, vs, ck, radius); GW(1) += sumGW2_func(xi, vs, ck, radius); GW(2) += sumGW3_func(xi, vs, ck, radius);
            N(0) += sumN1_func(xi, vs, ck, radius); N(1) += sumN2_func(xi, vs, ck, radius); N(2) += sumN3_func(xi, vs, ck, radius);
        }
    }


    if(abs(gx)<1e-6)
    {
        distance = 0;
        return 1.+fabs(t-t0);
    } 

    AScalar f = fx/gx;
    VectorXa grad_f = (GF-f*GW+N)/gx;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        int valence = nodes_adjacency[local_current_index[k]].size();
        VectorXa vs;

        vs.resize(33);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
        }


        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
        }

        if((ck-xi).norm()<=radius)
        {
            fx2 += fx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
            gx2 += gx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
        }
    }

    if(abs(gx2)<=1e-6)
    {
        distance = 0;
        return 1.+fabs(t-t0);
    } 
    AScalar dist = fx2/(gx2*radius);
    distance = fx2/gx2;

    //if(i ==2415) std::cout<<t<<" "<<dist<<std::endl;
    //std::cout<<dist<<" "<<fx<<" "<<gx<<std::endl;
    //return log(dist*dist)+fabs(t-t0);
    //return dist*dist+fabs(t-t0);
    return fabs(dist)+fabs(t-t0);
}

template <int dim>
double FEMSolver<dim>::evaluateUnsignedDistanceSqGradientRIMLS(int i, double t, const VectorXa& du, double ipc_stepsize)
{
    if(t >= ipc_stepsize || t <= 0.) return sign(t);
    t = min(t,ipc_stepsize);
    t = max(t,0.);
    VectorXa deformed_cur = deformed+t*du;
    // Tree local_accelerator;
    // local_accelerator.clear();

    // for(int i=0; i<num_nodes; ++i)
    // {
    //     if(is_surface_vertex[i] == 0) continue;
    //     Eigen::VectorXd point = deformed.segment<dim>(dim*i);
    //     local_accelerator.insert(std::make_tuple(Point_d(point[0], point[1], point[2]), i));
    // }

	std::vector<int> local_current_index;

    Vector3a xi = deformed_cur.segment<3>(3*i);

    //Fuzzy_sphere fs(Point_d(xi[0], xi[1], xi[2]), radius+radius*0.2, radius*0.2);

    // std::vector<std::tuple<Point_d, int>> points_query;
    // accelerator.search(std::back_inserter(points_query), fs);

    //for(int k=0; k<points_query.size(); ++k)
    for(int k=0; k<num_nodes; ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(3*k);
        //int index = std::get<1>(points_query[k]);
        int index = k;
        if(is_surface_vertex[index] && (geodist_close_matrix.coeff(i,index) == 0) && (ck-xi).norm()<=radius)
        {
            local_current_index.push_back(index);
        }     
    }

    if(local_current_index.size() == 0) return sign(t);

    VectorXa gradient(num_nodes*dim);
    gradient.setZero();
    VectorXa sum_dfdx(dim*num_nodes);
    sum_dfdx.setZero();
    VectorXa sum_dgdx(dim*num_nodes);
    sum_dgdx.setZero();
    VectorXa sum_dGF1dx(dim*num_nodes);
    sum_dGF1dx.setZero();
    VectorXa sum_dGF2dx(dim*num_nodes);
    sum_dGF2dx.setZero();
    VectorXa sum_dGF3dx(dim*num_nodes);
    sum_dGF3dx.setZero();
    VectorXa sum_dGW1dx(dim*num_nodes);
    sum_dGW1dx.setZero();
    VectorXa sum_dGW2dx(dim*num_nodes);
    sum_dGW2dx.setZero();
    VectorXa sum_dGW3dx(dim*num_nodes);
    sum_dGW3dx.setZero();
    VectorXa sum_dN1dx(dim*num_nodes);
    sum_dN1dx.setZero();
    VectorXa sum_dN2dx(dim*num_nodes);
    sum_dN2dx.setZero();
    VectorXa sum_dN3dx(dim*num_nodes);
    sum_dN3dx.setZero();
    VectorXa sum_df2dx(dim*num_nodes);
    sum_df2dx.setZero();
    VectorXa sum_dg2dx(dim*num_nodes);
    sum_dg2dx.setZero();

    //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;
    AScalar fx = 0;
    AScalar gx = 0;
    VectorXa N(3); N.setZero();
    VectorXa GF(3); GF.setZero();
    VectorXa GW(3); GW.setZero();
    AScalar fx2 = 0;
    AScalar gx2 = 0;
    VectorXa ele_dfdx;
    VectorXa ele_dgdx;
    VectorXa ele_df2dx;
    VectorXa ele_dg2dx;
    VectorXa ele_dGF1dx,ele_dGF2dx,ele_dGF3dx;
    VectorXa ele_dGW1dx,ele_dGW2dx,ele_dGW3dx;
    VectorXa ele_dN1dx,ele_dN2dx,ele_dN3dx;
    //std::cout<<i<<" out of "<<slave_nodes_3d[0].size()<<std::endl;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        //f(i == 1567) std::cout<<k<<" k out of "<<local_current_index.size()<<std::endl;
        int valence = nodes_adjacency[local_current_index[k]].size();
        Eigen::VectorXi valence_indices;
        VectorXa vs;

        ele_dfdx.resize(39);
        ele_dgdx.resize(39);
        vs.resize(33);
        valence_indices.resize(13);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
            valence_indices(a+2)=(local_current_index[k]);
        }


        valence_indices(0)=(i);
        valence_indices(1)=(local_current_index[k]);


        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
            valence_indices(a+2) = nodes_adjacency[local_current_index[k]][a];
        }

        if((ck-xi).norm()<=radius) {
            ele_dfdx = dfdx_func11(xi, vs, ck, radius);
            ele_dgdx = dgdx_func11(xi, ck, radius);
            ele_dGF1dx = dsumGF1dx_func(xi, vs, ck, radius); ele_dGF2dx = dsumGF2dx_func(xi, vs, ck, radius); ele_dGF3dx = dsumGF3dx_func(xi, vs, ck, radius);
            ele_dGW1dx = dsumGW1dx_func(xi, vs, ck, radius); ele_dGW2dx = dsumGW2dx_func(xi, vs, ck, radius); ele_dGW3dx = dsumGW3dx_func(xi, vs, ck, radius);
            ele_dN1dx = dsumN1dx_func(xi, vs, ck, radius); ele_dN2dx = dsumN2dx_func(xi, vs, ck, radius); ele_dN3dx = dsumN3dx_func(xi, vs, ck, radius);
            fx += fx_func11(xi, vs, ck, radius);
            gx += gx_func11(xi, ck, radius);
            GF(0) += sumGF1_func(xi, vs, ck, radius); GF(1) += sumGF2_func(xi, vs, ck, radius); GF(2) += sumGF3_func(xi, vs, ck, radius);
            GW(0) += sumGW1_func(xi, vs, ck, radius); GW(1) += sumGW2_func(xi, vs, ck, radius); GW(2) += sumGW3_func(xi, vs, ck, radius);
            N(0) += sumN1_func(xi, vs, ck, radius); N(1) += sumN2_func(xi, vs, ck, radius); N(2) += sumN3_func(xi, vs, ck, radius);
            IMLS_local_gradient_to_global_gradient(ele_dfdx,valence_indices,dim,sum_dfdx);
            IMLS_local_gradient_to_global_gradient(ele_dgdx,valence_indices,dim,sum_dgdx);
            IMLS_local_gradient_to_global_gradient(ele_dGF1dx,valence_indices,dim,sum_dGF1dx);
            IMLS_local_gradient_to_global_gradient(ele_dGF2dx,valence_indices,dim,sum_dGF2dx);
            IMLS_local_gradient_to_global_gradient(ele_dGF3dx,valence_indices,dim,sum_dGF3dx);
            IMLS_local_gradient_to_global_gradient(ele_dGW1dx,valence_indices,dim,sum_dGW1dx);
            IMLS_local_gradient_to_global_gradient(ele_dGW2dx,valence_indices,dim,sum_dGW2dx);
            IMLS_local_gradient_to_global_gradient(ele_dGW3dx,valence_indices,dim,sum_dGW3dx);
            IMLS_local_gradient_to_global_gradient(ele_dN1dx,valence_indices,dim,sum_dN1dx);
            IMLS_local_gradient_to_global_gradient(ele_dN2dx,valence_indices,dim,sum_dN2dx);
            IMLS_local_gradient_to_global_gradient(ele_dN3dx,valence_indices,dim,sum_dN3dx);

        }
    }
    if(abs(gx)<1e-6) return sign(t);

    AScalar f = fx/gx;
    VectorXa grad_f = (GF-f*GW+N)/gx;

    VectorXa dfdx = (gx*sum_dfdx-fx*sum_dgdx)/(pow(gx,2));
    // A = GF/gx
    VectorXa dA1dx = (gx*sum_dGF1dx-GF(0)*sum_dgdx)/(pow(gx,2)); VectorXa dA2dx = (gx*sum_dGF2dx-GF(1)*sum_dgdx)/(pow(gx,2)); VectorXa dA3dx = (gx*sum_dGF3dx-GF(2)*sum_dgdx)/(pow(gx,2));
    // B = GW/gx
    VectorXa dB1dx = (gx*sum_dGW1dx-GW(0)*sum_dgdx)/(pow(gx,2)); VectorXa dB2dx = (gx*sum_dGW2dx-GW(1)*sum_dgdx)/(pow(gx,2)); VectorXa dB3dx = (gx*sum_dGW3dx-GW(2)*sum_dgdx)/(pow(gx,2));
    // C = N/gx
    VectorXa dC1dx = (gx*sum_dN1dx-N(0)*sum_dgdx)/(pow(gx,2)); VectorXa dC2dx = (gx*sum_dN2dx-N(1)*sum_dgdx)/(pow(gx,2)); VectorXa dC3dx = (gx*sum_dN3dx-N(2)*sum_dgdx)/(pow(gx,2));

    VectorXa dgrad_f1dx = dA1dx - dfdx*(GW(0)/gx) - f*dB1dx + dC1dx;
    VectorXa dgrad_f2dx = dA2dx - dfdx*(GW(1)/gx) - f*dB2dx + dC2dx;
    VectorXa dgrad_f3dx = dA3dx - dfdx*(GW(2)/gx) - f*dB3dx + dC3dx;

    for(int k=0; k<local_current_index.size(); ++k)
    {
        Vector3a ck = deformed_cur.segment<3>(local_current_index[k]*3);
        int valence = nodes_adjacency[local_current_index[k]].size();
        Eigen::VectorXi valence_indices;
        VectorXa vs;

        ele_df2dx.resize(43); ele_dg2dx.resize(43);
        vs.resize(33);
        valence_indices.resize(13);
        for(int a=0; a<11; ++a)
        {
            vs.segment<3>(a*3) = ck;
            valence_indices(a+2)=(local_current_index[k]);
        }


        valence_indices(0)=(i);
        valence_indices(1)=(local_current_index[k]);


        for(int a=0; a<nodes_adjacency[local_current_index[k]].size(); ++a)
        {
            vs.segment<3>(a*3) = deformed_cur.segment<3>(nodes_adjacency[local_current_index[k]][a]*3);
            valence_indices(a+2) = nodes_adjacency[local_current_index[k]][a];
        }

        AScalar df2df = 0; AScalar dg2df = 0;
        Vector3a df2dgrad_f; Vector3a dg2dgrad_f;
        if((ck-xi).norm()<=radius)
        {
            // df2dx contains df2/df df2/dgrad_f1 ...
            VectorXa temp1 = dfdx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
            VectorXa temp2 = dgdx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
            df2df = temp1(39);  dg2df = temp2(39);
            df2dgrad_f = temp1.segment<3>(40); dg2dgrad_f = temp2.segment<3>(40);
            ele_df2dx = temp1.segment<39>(0); ele_dg2dx = temp2.segment<39>(0);
            fx2 += fx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
            gx2 += gx_func_2(xi, vs, ck, radius,f,grad_f,sigma_r,sigma_n);
            IMLS_local_gradient_to_global_gradient(ele_df2dx,valence_indices,dim,sum_df2dx);
            IMLS_local_gradient_to_global_gradient(ele_dg2dx,valence_indices,dim,sum_dg2dx);
            sum_df2dx += (df2df * dfdx + df2dgrad_f(0) * dgrad_f1dx + df2dgrad_f(1) * dgrad_f2dx + df2dgrad_f(2) * dgrad_f3dx);
            sum_dg2dx += (dg2df * dfdx + dg2dgrad_f(0) * dgrad_f1dx + dg2dgrad_f(1) * dgrad_f2dx + dg2dgrad_f(2) * dgrad_f3dx);
        }
    }

    AScalar dist = fx2/gx2;

    if(abs(gx2) <= 1e-6) return sign(t);
               

    if(dist > 0)
        gradient = (gx2*sum_df2dx-fx2*sum_dg2dx)/(pow(gx2,2)*pow(radius,1));
    else
        gradient = -(gx2*sum_df2dx-fx2*sum_dg2dx)/(pow(gx2,2)*pow(radius,1));

    return gradient.dot(du);
}

template class FEMSolver<2>;
template class FEMSolver<3>;
template class Rosenbrock<2>;
template class Rosenbrock<3>;
template class MyOptimization<2>;
template class MyOptimization<3>;