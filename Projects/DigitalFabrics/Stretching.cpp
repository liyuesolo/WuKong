#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingK(std::vector<Entry>& entry_K)
{
    for (auto& rod : Rods)
    {
        rod->iterateSegmentsWithOffset([&](int node_i, int node_j, Offset offset_i, Offset offset_j)
        {
            TV xi, xj, Xi, Xj, dXi, dXj;
            rod->x(node_i, xi); rod->x(node_j, xj);
            rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj);

            std::vector<TV> x(4);
            x[0] = xi; x[1] = xj; x[2] = Xi; x[3] = Xj;

            T J[8][8];
            memset(J, 0, sizeof(J));
            #include "Maple/YarnStretchingJ.mcg"

            std::vector<int> nodes = { node_i, node_j };
            std::vector<TV> dXdu = { dXi, dXj };

            std::vector<Offset> offsets = { offset_i, offset_j };
            
            for(int k = 0; k < nodes.size(); k++)
                for(int l = 0; l < nodes.size(); l++)
                    for(int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            {
                                entry_K.push_back(Entry(offsets[k][i], offsets[l][j] + j, -J[k*dim + i][l * dim + j]));
                                
                                entry_K.push_back(Entry(offsets[k][i], offsets[l][dim], -J[k*dim + i][2 * dim + l * dim + j] * dXdu[l][j]));

                                entry_K.push_back(Entry(offsets[k][dim], offsets[l][j] , -J[2 * dim + k * dim + i][l * dim + j] * dXdu[k][i]));

                                // dX/du ^T d2e/dX2 dXdu                    
                                entry_K.push_back(Entry(offsets[k][dim], 
                                                        offsets[l][dim], 
                                                        -J[2 * dim + k*dim + i][2 * dim + l * dim + j] * dXdu[l][j] * dXdu[k][i]));
                            }
        });   
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Entry>& entry_K)
{

    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);

        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, true, false);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        T J[8][8];
        memset(J, 0, sizeof(J));
        #include "Maple/YarnStretchingJ.mcg"
        // for(int k = 0; k < nodes.size(); k++)
        //     for(int l = 0; l < nodes.size(); l++)
        //         for(int i = 0; i < dim; i++)
        //             for (int j = 0; j < dim; j++)
        //                 {
        //                     entry_K.push_back(Entry(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));
        //                 }


        for(int k = 0; k < nodes.size(); k++)
            for(int l = 0; l < nodes.size(); l++)
                for(int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                        {
                            entry_K.push_back(Entry(nodes[k] * dof + i, nodes[l] * dof + j, -J[k*dim + i][l * dim + j]));

                            entry_K.push_back(Entry(nodes[k] * dof + i, nodes[l] * dof + dim + yarn_type, -J[k*dim + i][2 * dim + l * dim + j] * dXdu[l][j]));

                            entry_K.push_back(Entry(nodes[k] * dof + dim + yarn_type, nodes[l] * dof + j, -J[2 * dim + k * dim + i][l * dim + j] * dXdu[k][i]));

                            // dX/du ^T d2e/dX2 dXdu                    
                            entry_K.push_back(Entry(nodes[k] * dof + dim + yarn_type, 
                                                                nodes[l] * dof + dim + yarn_type, 
                                                                -J[2 * dim + k*dim + i][2 * dim + l * dim + j] * dXdu[l][j] * dXdu[k][i]));
                            
                        }
    }
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);

        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, true, true);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        Vector<T, 8> F;
        F.setZero();
        #include "Maple/YarnStretchingF.mcg"

        // sum de/dX d2X/du2
        for(int d = 0; d < dim; d++)
        {
            entry_K.push_back(Entry(nodes[0] * dof + dim + yarn_type, 
                                                nodes[0] * dof + dim + yarn_type, -F[0*dim + 2*dim + d] * d2Xdu2[0][d]));
            entry_K.push_back(Entry(nodes[1] * dof + dim + yarn_type, 
                                                nodes[1] * dof + dim + yarn_type, -F[1*dim + 2*dim + d] * d2Xdu2[1][d]));
        }
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingForce(Eigen::Ref<VectorXT> residual)
{
    for (auto& rod : Rods)
    {
        rod->iterateSegmentsWithOffset([&](int node_i, int node_j, Offset offset_i, Offset offset_j)
        {
            std::cout << "node i " << node_i << " node j " << node_j << std::endl;
            TV xi, xj, Xi, Xj, dXi, dXj;
            rod->x(node_i, xi); rod->x(node_j, xj);
            rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj);

            std::vector<TV> x(4);
            x[0] = xi; x[1] = xj; x[2] = Xi; x[3] = Xj;

            Vector<T, 8> F;
            F.setZero();
            #include "Maple/YarnStretchingF.mcg"

            residual.template segment<dim>(offset_i[0]) += F.template segment<dim>(0);
            residual.template segment<dim>(offset_j[0]) += F.template segment<dim>(dim);

            residual(offset_i[dim]) += F.template segment<dim>(2*dim).dot(dXi);
            residual(offset_j[dim]) += F.template segment<dim>(2*dim + dim).dot(dXj);
        });
    }
}
template<class T, int dim>
void EoLRodSim<T, dim>::addStretchingForce(Eigen::Ref<const DOFStack> q_temp,
     Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;

   
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++)
    {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);
        
        
        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, true, false);

        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        Vector<T, 8> F;
        F.setZero();
        #include "Maple/YarnStretchingF.mcg"
        
        int cnt = 0;
        for (int node : nodes)
        {
            // -de/dx
            residual.col(node).template segment<dim>(0) += F.template segment<dim>(cnt*dim);
            // -de/dX dXdu
            for(int d = 0; d < dim; d++)
            {
                residual(dim + yarn_type, node) += F[cnt*dim + 2*dim + d] * dXdu[cnt][d];
            }
            cnt++;
        } 
    }
    if(print_force_mag)
        std::cout << "stretching norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addStretchingEnergy()
{
    T E = 0;
    for (auto& rod : Rods)
    {
        rod->iterateSegments([&](int node_i, int node_j)
        {
            TV xi, xj, Xi, Xj;
            rod->x(node_i, xi); rod->x(node_j, xj);
            rod->X(node_i, Xi); rod->X(node_j, Xj);

            std::vector<TV> x(4);
            x[0] = xi; x[1] = xj; x[2] = Xi; x[3] = Xj;

            T V[1];
            #include "Maple/YarnStretchingV.mcg"
            E += V[0];
            
        });
    }
    return E;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addStretchingEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    // std::cout << "compute stretching energy" << std::endl;
    VectorXT rod_energy(n_rods);
    rod_energy.setZero();

    // // tbb::parallel_for(0, n_rods, [&](int rod_idx){
    for (int rod_idx = 0; rod_idx < n_rods; rod_idx++) {
        int node0 = rods.col(rod_idx)[0];
        int node1 = rods.col(rod_idx)[1];
        TV x0 = q_temp.col(node0).template segment<dim>(0);
        TV x1 = q_temp.col(node1).template segment<dim>(0);

        std::vector<TV> X; std::vector<TV> dXdu; std::vector<TV> d2Xdu2;
        int yarn_type = rods.col(rod_idx)[2];
        std::vector<int> nodes = { node0, node1 };
        // std::cout << "getMaterialPositions " <<  node0 << " " << node1 << " " << q_temp(dim + yarn_type, node0) << " " << q_temp(dim + yarn_type, node1) << std::endl;
        getMaterialPositions(q_temp, nodes, X, yarn_type, dXdu, d2Xdu2, false, false);
        // std::cout << "getMaterialPositions done" << std::endl;
        std::vector<TV> x(4);
        x[0] = x0; x[1] = x1; x[2] = X[0]; x[3] = X[1];

        T V[1];
        #include "Maple/YarnStretchingV.mcg"
        rod_energy[rod_idx] += V[0];

    // });
    }

    // std::cout << "compute stretching energy done" << std::endl;
    
    return rod_energy.sum();
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;