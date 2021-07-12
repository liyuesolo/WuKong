#include "EoLRodSim.h"
using std::abs;

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingK(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for (auto& rod : Rods)
    {
        rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
        {
            TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
            rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
            rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

            std::vector<TV> x(6);
            x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
            
            std::vector<int> nodes = { node_i, node_j, node_k };
            std::vector<TV> dXdu = { dXi, dXj, dXk };

            std::vector<Offset> offsets = { offset_i, offset_j, offset_k };
            
            
            // set to 20 for 3D, if 2D, only 12x12 if filled
            T J[20][20];
            memset(J, 0, sizeof(J));
            if constexpr (dim == 2)
            {
                #include "Maple/YarnBendDiscreteRestCurvatureJ.mcg"
            }
            else
            {
                int dof_theta1 = rod->theta_dof_start_offset + (second - 1);

                std::vector<int> theta_dofs = {dof_theta1, dof_theta1 + 1};
                
                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                std::vector<TV> u_rest(2);
                u_rest[0] = rod->reference_frame_us[second - 1];
                u_rest[1] = rod->reference_frame_us[second];
                T B[2][2];
                B[0][0] = rod->B[0][0]; B[0][1] = rod->B[0][1];
                B[1][0] = rod->B[1][0]; B[1][1] = rod->B[1][1];
                #include "Maple/RodBending3DJ.mcg" 
                for(int k = 0; k < nodes.size(); k++)
                    for(int l = 0; l < nodes.size(); l++)
                    {
                        for(int i = 0; i < dim; i++)
                            for (int j = 0; j < 2; j++)
                            {
                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], theta_dofs[j], -J[k*dim + i][18 + j]));
                                entry_K.push_back(Eigen::Triplet<T>(theta_dofs[j], offsets[l][j], -J[18 + j][l * dim + i]));
                            }
                    }
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                        entry_K.push_back(Eigen::Triplet<T>(theta_dofs[i], theta_dofs[j], -J[18 + i][18 + j]));
                        
                }
                
            }

            for(int k = 0; k < nodes.size(); k++)
                for(int l = 0; l < nodes.size(); l++)
                    for(int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            {
                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], offsets[l][j], -J[k*dim + i][l * dim + j]));

                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], offsets[l][dim], -J[k*dim + i][3 * dim + l * dim + j] * dXdu[l][j]));

                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][dim], offsets[l][j], -J[3 * dim + k * dim + i][l * dim + j] * dXdu[k][i]));

                                
                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][dim], 
                                                                    offsets[l][dim], 
                                                                    -J[3 * dim + k*dim + i][3 * dim + l * dim + j] * dXdu[l][j] * dXdu[k][i]));

                            }
        });
    }

    for (auto& rod : Rods)
    {
        rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
        {
            TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk, ddXi, ddXj, ddXk;
            rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
            rod->XdXddX(node_i, Xi, dXi, ddXi); rod->XdXddX(node_j, Xj, dXj, ddXj); rod->XdXddX(node_k, Xk, dXk, ddXk);

            std::vector<TV> x(6);
            x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
            

            Vector<T, 20> F;
            F.setZero();
            if constexpr (dim == 2)
            {
                #include "Maple/YarnBendDiscreteRestCurvatureF.mcg"
            }
            else if constexpr (dim == 3)
            {
                
                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                std::vector<TV> u_rest(2);
                u_rest[0] = rod->reference_frame_us[second - 1];
                u_rest[1] = rod->reference_frame_us[second];
                T B[2][2];
                B[0][0] = rod->B[0][0]; B[0][1] = rod->B[0][1];
                B[1][0] = rod->B[1][0]; B[1][1] = rod->B[1][1];

                #include "Maple/RodBending3DF.mcg"
                
            }
            
            for(int d = 0; d < dim; d++)
            {
                entry_K.push_back(Eigen::Triplet<T>(offset_i[dim], 
                                    offset_i[dim], -F[0*dim + 3*dim + d] * ddXi[d]));
                entry_K.push_back(Eigen::Triplet<T>(offset_j[dim], 
                                    offset_j[dim], -F[1*dim + 3*dim + d] * ddXj[d]));
                entry_K.push_back(Eigen::Triplet<T>(offset_k[dim], 
                                    offset_k[dim], -F[2*dim + 3*dim + d] * ddXk[d]));
            }    

        });
    }
    if(!add_pbc_bending)
        return;
}

template<class T, int dim>
void EoLRodSim<T, dim>::addBendingForce(Eigen::Ref<VectorXT> residual)
{
    VectorXT residual_cp = residual;
    for (auto& rod : Rods)
    {
        rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
        {
            TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
            rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
            rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

            std::vector<TV> x(6);
            x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
            

            Vector<T, 20> F;
            F.setZero();
            if constexpr (dim == 2)
            {
                #include "Maple/YarnBendDiscreteRestCurvatureF.mcg"
            }
            else if constexpr (dim == 3)
            {
                
                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                std::vector<TV> u_rest(2);
                u_rest[0] = rod->reference_frame_us[second - 1];
                u_rest[1] = rod->reference_frame_us[second];
                T B[2][2];
                B[0][0] = rod->B[0][0]; B[0][1] = rod->B[0][1];
                B[1][0] = rod->B[1][0]; B[1][1] = rod->B[1][1];

                #include "Maple/RodBending3DF.mcg"
                
                residual.template segment<2>(rod->theta_dof_start_offset + (second - 1)) += 
                    F.template segment<2>(18);
            }
            

            residual.template segment<dim>(offset_i[0]) += F.template segment<dim>(0);
            residual.template segment<dim>(offset_j[0]) += F.template segment<dim>(dim);
            residual.template segment<dim>(offset_k[0]) += F.template segment<dim>(dim + dim);

            residual(offset_i[dim]) += F.template segment<dim>(3*dim).dot(dXi);
            residual(offset_j[dim]) += F.template segment<dim>(3*dim + dim).dot(dXj);
            residual(offset_k[dim]) += F.template segment<dim>(3*dim + 2*dim).dot(dXk);
        });
    }
    if(!add_pbc_bending)
    {
        if (print_force_mag)
            std::cout << "bending force " << (residual - residual_cp).norm() << std::endl;
        return;
    }

}

template<class T, int dim>
T EoLRodSim<T, dim>::addBendingEnergy()
{
    T energy = 0.0;
    for (auto& rod : Rods)
    {
        T energy_current = energy;
        rod->iterate3Nodes([&](int node_i, int node_j, int node_k, int second){
                TV xi, xj, xk, Xi, Xj, Xk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->X(node_i, Xi); rod->X(node_j, Xj); rod->X(node_k, Xk);

                std::vector<TV> x(6);
                x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;

                T V[1];
                if constexpr (dim == 2)
                {
                    #include "Maple/YarnBendDiscreteRestCurvatureV.mcg"
                }
                else if constexpr (dim == 3)
                {
                    T theta0 = rod->reference_angles[second - 1];
                    T theta1 = rod->reference_angles[second];
                    std::vector<TV> u_rest(2);
                    u_rest[0] = rod->reference_frame_us[second - 1];
                    u_rest[1] = rod->reference_frame_us[second];
                    T B[2][2];
                    B[0][0] = rod->B[0][0]; B[0][1] = rod->B[0][1];
                    B[1][0] = rod->B[1][0]; B[1][1] = rod->B[1][1];
                    #include "Maple/RodBending3DV.mcg"
                    
                }
                energy += V[0];
            });
        
    }
    
    if(!add_pbc_bending)
        return energy;
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;