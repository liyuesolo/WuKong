#include "../include/FEMSolver.h"

template <int dim>
void FEMSolver<dim>::findProjectionIMLSSameSide(Eigen::MatrixXd& ipc_vertices_deformed, bool re, bool stay)
{
    // Initialization
    if(USE_FROM_STL || TEST)
        samplePointsFromSTL();
    else
        samplePoints();
    buildConstraintsPointSet();

    if(re)
    {   
        // Build Query Matrix
        boundary_info_same_side.clear();
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

                map_boundary_virtual_same_side[boundary_info_same_side.size()] = i;
                boundary_info_same_side.push_back(p1);
                prev_len = next_len;
            }
        }
        else
        {
            for(int i=0; i<slave_nodes.size(); ++i)
            {
                query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes);
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
                    Eigen::VectorXd p1 = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes);
                    Eigen::VectorXd p2 = ipc_vertices_deformed.row(slave_nodes[i+1]+num_nodes);
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

                boundary_info_same_side.push_back(p1);
                prev_len = next_len;
            }
        }
        
        evaluateImplicitPotentialKR(query_pts, true, 1, -1, true);


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

                    map_boundary_virtual_same_side[boundary_info_same_side.size()] = i;
                    boundary_info_same_side.push_back(p1);
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
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes[0],0)) < 1e-5)
                    //     query_pts(i,0) += 0.001;
                    // if(fabs(query_pts(i,0)-ipc_vertices_deformed(slave_nodes.back(),0)) < 1e-5)
                    //     query_pts(i,0) -= 0.001;

                    if(i < master_nodes.size() - 1)
                    {
                        Eigen::VectorXd p1 = ipc_vertices_deformed.row(master_nodes[i]);
                        Eigen::VectorXd p2 = ipc_vertices_deformed.row(master_nodes[i+1]);
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

                    boundary_info_same_side.push_back(p1);
                    //std::cout<<"Creation: "<<p1.slave_index<<" "<<p1.index<<std::endl;
                    prev_len = next_len;
                }
            }
            evaluateImplicitPotentialKR(query_pts, true, 0, -1, true);
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
                query_pts.row(i) = ipc_vertices_deformed.row(slave_nodes[i]+num_nodes); 
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes.back(),0)) < 1e-5)
                //         query_pts(i,0) += 0.001;
                // if(fabs(query_pts(i,0)-ipc_vertices_deformed(master_nodes[0],0)) < 1e-5)
                //         query_pts(i,0) -= 0.001;
                // if(CALCULATE_IMLS_PROJECTION)
                //     query_pts.row(i) = ipc_vertices_deformed.row(num_nodes+i);
            }
        }
        
        evaluateImplicitPotentialKR(query_pts, true, 1, -1, true);

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
            evaluateImplicitPotentialKR(query_pts, true, 0, -1, true);
        }
    }
    // for(int i=0; i<boundary_info_same_side.size(); ++i)
    //     std::cout<<"dist_grad size: "<<boundary_info_same_side[i].dist_grad.size()<<" "<<boundary_info_same_side[i].dist<<std::endl;
}

template <int dim>
void FEMSolver<dim>::addIMLSPenEnergy(T& energy)
{
    T IMLS_Pen_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = 2*num_nodes;

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    findProjectionIMLSSameSide(ipc_vertices_deformed);

    double phi;
    if(USE_VIRTUAL_NODE)
        phi = compute_vts_potential2D(ipc_vertices_deformed,true);
    else
        phi = compute_barrier_potential2D(ipc_vertices_deformed,true);

    energy += phi;
}

template <int dim>
void FEMSolver<dim>::addIMLSPenForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = 2*num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    findProjectionIMLSSameSide(ipc_vertices_deformed);

    Eigen::VectorXd contact_gradient;
    if(USE_VIRTUAL_NODE) contact_gradient = compute_vts_potential_gradient2D(ipc_vertices_deformed, true);
    else contact_gradient = compute_barrier_potential_gradient2D(ipc_vertices_deformed, true);
    
    residual.segment(0, num_nodes_all * dim) += -contact_gradient.segment(0, num_nodes_all * dim);
}

template <int dim>
void FEMSolver<dim>::addIMLSPenHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    int num_nodes_all = 2*num_nodes;
    //if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();

    for (int i = 0; i < num_nodes_all; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);
    }

    findProjectionIMLSSameSide(ipc_vertices_deformed);
    
    StiffnessMatrix contact_hessian;
    if(USE_VIRTUAL_NODE) contact_hessian = compute_vts_potential_hessian2D(ipc_vertices_deformed, true);
    else contact_hessian = compute_barrier_potential_hessian2D(ipc_vertices_deformed, true);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void FEMSolver<dim>::addL2DistanceEnergy(T& energy)
{
    T L2_energy = 0.0;

    std::vector<int> ndofs(num_nodes,0);
    auto it = slave_nodes_3d[0].begin();
    for(int i=0; i<slave_nodes_3d[0].size(); ++i)
    {
        ndofs[it->first] = 1;
        it++;
    }

    for(int i=0; i<num_nodes; ++i)
        if(ndofs[i] == 0)
            L2_energy += 0.5* L2D_param * (deformed.segment<dim>(i*dim) - deformed.segment<dim>((i+num_nodes)*dim)).squaredNorm();

    energy += L2_energy;
}

template <int dim>
void FEMSolver<dim>::addL2DistanceForceEntries(VectorXT& residual)
{
    int num_nodes_all = 2*num_nodes;
    Eigen::VectorXd L2_gradient(num_nodes_all*dim);

    std::vector<int> ndofs(num_nodes,0);
    auto it = slave_nodes_3d[0].begin();
    for(int i=0; i<slave_nodes_3d[0].size(); ++i)
    {
        ndofs[it->first] = 1;
        it++;
    }

    for(int i=0; i<num_nodes; ++i){
        if(ndofs[i] == 0)
        {
            L2_gradient.segment<dim>(i*dim) = L2D_param * (deformed.segment<dim>(i*dim) - deformed.segment<dim>((i+num_nodes)*dim));
            L2_gradient.segment<dim>((i+num_nodes)*dim) = -L2D_param * (deformed.segment<dim>(i*dim) - deformed.segment<dim>((i+num_nodes)*dim));
        }
    }    
    
    residual.segment(0, num_nodes_all * dim) += -L2_gradient.segment(0, num_nodes_all * dim);
}

template <int dim>
void FEMSolver<dim>::addL2DistanceHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    int num_nodes_all = 2*num_nodes;
    StiffnessMatrix L2_hessian(dim*num_nodes_all,dim*num_nodes_all);
    std::vector<Eigen::Triplet<double>> L2_hessian_triplets;
    
    std::vector<int> ndofs(num_nodes,0);
    auto it = slave_nodes_3d[0].begin();
    for(int i=0; i<slave_nodes_3d[0].size(); ++i)
    {
        ndofs[it->first] = 1;
        it++;
    }

    for(int i=0; i<num_nodes; ++i){
        if(ndofs[i] == 0)
        {
            for(int j=0; j<dim;++j)
            {
                L2_hessian_triplets.emplace_back(dim * i + j, dim * i + j, L2D_param);
                L2_hessian_triplets.emplace_back(dim * (i+num_nodes) + j, dim * i + j, -L2D_param);
                L2_hessian_triplets.emplace_back(dim * i + j, dim * (i+num_nodes) + j, -L2D_param);
                L2_hessian_triplets.emplace_back(dim * (i+num_nodes) + j, dim * (i+num_nodes) + j, L2D_param);
            }
        }
    }
    L2_hessian.setFromTriplets(L2_hessian_triplets.begin(), L2_hessian_triplets.end());
    //std::cout<<L2_hessian<<std::endl;

    std::vector<Entry> L2_entries = entriesFromSparseMatrix(L2_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), L2_entries.begin(), L2_entries.end());
}

template class FEMSolver<2>;
template class FEMSolver<3>;