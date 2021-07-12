#include "EoLRodSim.h"
#include "EoLRodBendingAndTwisting.h"

template<class T, int dim>
T EoLRodSim<T, dim>::add3DBendingAndTwistingEnergy()
{
    if constexpr (dim == 3)
    {
        T energy = 0.0;
        for (auto& rod : Rods)
        {
            T energy_current = energy;
            rod->iterate3Nodes([&](int node_i, int node_j, int node_k, int second){
                TV xi, xj, xk, Xi, Xj, Xk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->X(node_i, Xi); rod->X(node_j, Xj); rod->X(node_k, Xk);

                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                TV referenceNormal1 = rod->reference_frame_us[second - 1];
                TV referenceNormal2 = rod->reference_frame_us[second];
                Matrix<T, 2, 2> B = rod->bending_coeffs;

                
                energy += computeRodBendingAndTwistEnergy(B, rod->kt, 0.0, referenceNormal1,
                    referenceNormal2, 0.0, xk, xi, xj, Xk, Xi, Xj, theta0, theta1);
                });
            
        }
        return energy;
    }
    return 0;
}
template<class T, int dim>
void EoLRodSim<T, dim>::add3DBendingAndTwistingForce(Eigen::Ref<VectorXT> residual)
{
    if constexpr (dim == 3)
    {
        for (auto& rod : Rods)
        {
            rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
            {
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                TV referenceNormal1 = rod->reference_frame_us[second - 1];
                TV referenceNormal2 = rod->reference_frame_us[second];
                Matrix<T, 2, 2> B = rod->bending_coeffs;

                Vector<T, 20> F;
                computeRodBendingAndTwistEnergyGradient(B, rod->kt, 0.0, referenceNormal1,
                    referenceNormal2, 0.0, xk, xi, xj, Xk, Xi, Xj, theta0, theta1, F);
                
                F *= -1.0;

                residual.template segment<dim>(offset_k[0]) += F.template segment<dim>(0);
                residual.template segment<dim>(offset_i[0]) += F.template segment<dim>(dim);
                residual.template segment<dim>(offset_j[0]) += F.template segment<dim>(dim + dim);

                residual(offset_k[dim]) += F.template segment<dim>(3*dim).dot(dXk);
                residual(offset_i[dim]) += F.template segment<dim>(3*dim + dim).dot(dXi);
                residual(offset_j[dim]) += F.template segment<dim>(3*dim + 2*dim).dot(dXj);

                residual.template segment<2>(rod->theta_dof_start_offset + (second - 1)) += 
                    F.template segment<2>(18);
            });
        }
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::add3DBendingAndTwistingK(std::vector<Entry>& entry_K)
{
    if constexpr (dim == 3)
    {
        for (auto& rod : Rods)
        {
            rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
            {
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

                std::vector<int> nodes = { node_k, node_i, node_j };
                std::vector<TV> dXdu = { dXk, dXi, dXj };

                std::vector<Offset> offsets = { offset_k, offset_i, offset_j };

                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];

                int dof_theta1 = rod->theta_dof_start_offset + (second - 1);
                
                std::vector<int> theta_dofs = {dof_theta1, dof_theta1 + 1};

                // std::cout << dof_theta1 << " " << dof_theta1 + 1 << std::endl;
                // std::getchar();

                TV referenceNormal1 = rod->reference_frame_us[second - 1];
                TV referenceNormal2 = rod->reference_frame_us[second];
                Matrix<T, 2, 2> B = rod->bending_coeffs;

                Matrix<T, 20, 20> J;
                
                computeRodBendingAndTwistEnergyHessian(B, rod->kt, 0.0, referenceNormal1,
                    referenceNormal2, 0.0, xk, xi, xj, Xk, Xi, Xj, theta0, theta1, J);

                J *= -1.0;
                for(int k = 0; k < nodes.size(); k++)
                    for(int l = 0; l < nodes.size(); l++)
                        for(int i = 0; i < dim; i++)
                            for (int j = 0; j < dim; j++)
                                {
                                    entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], offsets[l][j], -J(k*dim + i, l * dim + j)));

                                    entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], offsets[l][dim], -J(k*dim + i, 3 * dim + l * dim + j) * dXdu[l][j]));

                                    entry_K.push_back(Eigen::Triplet<T>(offsets[k][dim], offsets[l][j], -J(3 * dim + k * dim + i, l * dim + j) * dXdu[k][i]));

                                    
                                    entry_K.push_back(Eigen::Triplet<T>(offsets[k][dim], 
                                                                        offsets[l][dim], 
                                                                        -J(3 * dim + k*dim + i, 3 * dim + l * dim + j) * dXdu[l][j] * dXdu[k][i]));

                                }
                for(int k = 0; k < nodes.size(); k++)
                    for(int l = 0; l < nodes.size(); l++)
                    {
                        for(int i = 0; i < dim; i++)
                            for (int j = 0; j < 2; j++)
                            {
                                entry_K.push_back(Eigen::Triplet<T>(offsets[k][i], theta_dofs[j], -J(k*dim + i, 18 + j)));
                                entry_K.push_back(Eigen::Triplet<T>(theta_dofs[j], offsets[l][j], -J(18 + j, l * dim + i)));
                            }
                    }
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        entry_K.push_back(Eigen::Triplet<T>(theta_dofs[i], theta_dofs[j], -J(18 + i, 18 + j)));
            });
            
        }

        for (auto& rod : Rods)
        {   
            // continue;
            rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                Offset offset_i, Offset offset_j, Offset offset_k, int second)
            {
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk, ddXi, ddXj, ddXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdXddX(node_i, Xi, dXi, ddXi); rod->XdXddX(node_j, Xj, dXj, ddXj); rod->XdXddX(node_k, Xk, dXk, ddXk);

                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                TV referenceNormal1 = rod->reference_frame_us[second - 1];
                TV referenceNormal2 = rod->reference_frame_us[second];
                Matrix<T, 2, 2> B = rod->bending_coeffs;

                Vector<T, 20> F;
                computeRodBendingAndTwistEnergyGradient(B, rod->kt, 0.0, referenceNormal1,
                    referenceNormal2, 0.0, xk, xi, xj, Xk, Xi, Xj, theta0, theta1, F);
                
                F *= -1.0;

                for(int d = 0; d < dim; d++)
                {
                    entry_K.push_back(Eigen::Triplet<T>(offset_i[dim], 
                                        offset_i[dim], -F[1*dim + 3*dim + d] * ddXi[d]));
                    entry_K.push_back(Eigen::Triplet<T>(offset_j[dim], 
                                        offset_j[dim], -F[2*dim + 3*dim + d] * ddXj[d]));
                    entry_K.push_back(Eigen::Triplet<T>(offset_k[dim], 
                                        offset_k[dim], -F[0*dim + 3*dim + d] * ddXk[d]));
                }
            });
        }
    }
    
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;