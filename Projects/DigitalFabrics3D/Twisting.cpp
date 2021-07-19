#include "EoLRodSim.h"
template<class T, int dim>
void EoLRodSim<T, dim>::addTwistingK(std::vector<Entry>& entry_K)
{
    if constexpr (dim == 3)
    {
        for (auto& rod : Rods)
        {
            rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                    Offset offset_i, Offset offset_j, Offset offset_k, int second, bool is_crossing)
            {
                T kt = rod->kt;
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

                std::vector<TV> x(6);
                x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
                
                std::vector<int> nodes = { node_i, node_j, node_k };
                std::vector<TV> dXdu = { dXi, dXj, dXk };

                std::vector<Offset> offsets = { offset_i, offset_j, offset_k };

                T J[20][20];
                memset(J, 0, sizeof(J));
                
                int dof_theta1 = rod->theta_dof_start_offset + (theta - 1) * 2;

                std::vector<int> theta_dofs = {dof_theta1, dof_theta1 + 1};
                
                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                std::vector<TV> u_rest(2);
                u_rest[0] = rod->reference_frame_us[second - 1];
                u_rest[1] = rod->reference_frame_us[second];
                
                // #include "Maple/RodTwisting3DJ.mcg"
                
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
                    Offset offset_i, Offset offset_j, Offset offset_k, int second, bool is_crossing)
            {
                T kt = rod->kt;
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk, ddXi, ddXj, ddXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdXddX(node_i, Xi, dXi, ddXi); rod->XdXddX(node_j, Xj, dXj, ddXj); rod->XdXddX(node_k, Xk, dXk, ddXk);

                std::vector<TV> x(6);
                x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
                

                Vector<T, 20> F;
                F.setZero();
                
                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                std::vector<TV> u_rest(2);
                u_rest[0] = rod->reference_frame_us[second - 1];
                u_rest[1] = rod->reference_frame_us[second];
                   
                // #include "Maple/RodTwisting3DF.mcg"
                
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
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::addTwistingForce(Eigen::Ref<VectorXT> residual)
{
    if constexpr (dim == 3)
    {
        VectorXT residual_cp = residual;
        for (auto& rod : Rods)
        {
            rod->iterate3NodesWithOffsets([&](int node_i, int node_j, int node_k, 
                    Offset offset_i, Offset offset_j, Offset offset_k, int second, bool is_crossing)
            {
                T kt = rod->kt;
                TV xi, xj, xk, Xi, Xj, Xk, dXi, dXj, dXk;
                rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                rod->XdX(node_i, Xi, dXi); rod->XdX(node_j, Xj, dXj); rod->XdX(node_k, Xk, dXk);

                std::vector<TV> x(6);
                x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;
                

                Vector<T, 20> F;
                F.setZero();
                
                // #include "Maple/RodTwisting3DF.mcg"

                T theta0 = rod->reference_angles[second - 1];
                T theta1 = rod->reference_angles[second];
                TV tangent = (xj - xi).normalized();
                TV prev_tangent = (xi - xk).normalized();
                
                TV d1_hat = rod->reference_frame_us[second-1];
                TV d1 = parallelTransportOrthonormalVector(d1_hat, prev_tangent, tangent);

                TV d2 = tangent.cross(d1);

                // #include "Maple/RodTwisting3DV.mcg"
                T inv_len = 1.0 / ((Xj - Xi).norm() + (Xi - Xk).norm());
                TV kb;
                rod->curvatureBinormal(prev_tangent, tangent, kb);

                T dTwistdtheta0 = -1;
                T dTwistdtheta1 = 1;
                TV dTwistdxj = 0.5 / (xj - xi).norm() * kb;
                TV dTwistdxk = -0.5 / (xi - xk).norm() * kb;

                TV dTwistdxi = -(dTwistdxj + dTwistdxk);


                

                // TV dmidei = ((TM::Identity() - (tangent * tangent.transpose())) / (xj - xi).norm()) *
                //     ((d1_hat.dot(tangent) *  / (1 + ))- 
                    
                //     );


                



                int theta_dof0 = rod->theta_dof_start_offset + (second - 1) * 2;
                int theta_dof1 = theta_dof0 + 1;
                T ref_twist = rod->computeReferenceTwist(tangent, prev_tangent, second);

                residual.template segment<dim>(offset_i[0]) += -kt * inv_len * (theta1 - theta0 + ref_twist) * dTwistdxi;
                residual.template segment<dim>(offset_j[0]) += -kt * inv_len * (theta1 - theta0 + ref_twist) * dTwistdxj;
                residual.template segment<dim>(offset_k[0]) += -kt * inv_len * (theta1 - theta0 + ref_twist) * dTwistdxk;

                residual.template segment<2>(theta_dof0) += -kt * inv_len * (theta1 - theta0 + ref_twist) * TV2(dTwistdtheta0, dTwistdtheta1);

            });
        }
        if (print_force_mag)
            std::cout << "twisting force " << (residual - residual_cp).norm() << std::endl;
    }
}


template<class T, int dim>
T EoLRodSim<T, dim>::addTwistingEnergy()
{
    if constexpr (dim == 3)
    {
        T energy = 0.0;
        for (auto& rod : Rods)
        {
            T energy_current = energy;
            rod->iterate3Nodes([&](int node_i, int node_j, int node_k, int second, bool is_crossing){
                    T kt = rod->kt;
                    TV xi, xj, xk, Xi, Xj, Xk;
                    rod->x(node_i, xi); rod->x(node_j, xj); rod->x(node_k, xk);
                    rod->X(node_i, Xi); rod->X(node_j, Xj); rod->X(node_k, Xk);

                    std::vector<TV> x(6);
                    x[0] = xi; x[1] = xj; x[2] = xk; x[3] = Xi; x[4] = Xj; x[5] = Xk;

                    T V[1];
                    T theta0 = rod->reference_angles[second - 1];
                    T theta1 = rod->reference_angles[second];
                    TV tangent = (xj - xi).normalized();
                    TV prev_tangent = (xi - xk).normalized();
                    
                    // #include "Maple/RodTwisting3DV.mcg"
                    T inv_len = 1.0 / ((Xj - Xi).norm() + (Xi - Xk).norm());
                    T ref_twist = rod->computeReferenceTwist(tangent, prev_tangent, second);
                    V[0] += 0.5 * kt * inv_len * std::pow(theta1 - theta0 + ref_twist, 2);
                    // std::cout << V[0] << std::endl;
                    energy += V[0];
                });
            
        }
    }
}


template class EoLRodSim<double, 2>;
template class EoLRodSim<double, 3>;