#include "EoLRodSim.h"

// template<class T, int dim>
// void EoLRodSim<T, dim>::addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
// {
//     iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref, const auto& Tij){
        
//         for (int i = 0; i < dof; i++)
//         {
//             if (ni == ni_ref && nj == nj_ref && Tij.norm() > 1e-6)
//             {
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj * dof + i, k_pbc));
//             }
//             else
//             {
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni_ref * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj_ref * dof + i, k_pbc));

//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni_ref * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj_ref * dof + i, -k_pbc));

//                 entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni_ref * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj_ref * dof + i, -k_pbc));

//                 entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni * dof + i, k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni_ref * dof + i, -k_pbc));
//                 entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj_ref * dof + i, k_pbc));
//             }
            
//         }
//     });
// }

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    // iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref, const auto& Tij){
        
    //     for (int i = 0; i < dof; i++)
    //     {
    //         if (ni == ni_ref && nj == nj_ref && Tij.norm() > 1e-6)
    //         {
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj * dof + i, k_pbc));
    //         }
    //         else
    //         {
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni_ref * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj_ref * dof + i, k_pbc));

    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni_ref * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj_ref * dof + i, -k_pbc));

    //             entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni_ref * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj_ref * dof + i, -k_pbc));

    //             entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni * dof + i, k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni_ref * dof + i, -k_pbc));
    //             entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj_ref * dof + i, k_pbc));
    //         }
            
    //     }
    // });

    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);

        T dij = (xj - xi).dot(strain_dir);
        TM Hessian = strain_dir * strain_dir.transpose();
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < dim; j++)
            {
                entry_K.push_back(Eigen::Triplet<T>(node_i * dof + i, node_i * dof + j, k_pbc * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_i * dof + i, node_j * dof + j, -k_pbc * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_j * dof + i, node_i * dof + j, -k_pbc * Hessian(i, j)));
                entry_K.push_back(Eigen::Triplet<T>(node_j * dof + i, node_j * dof + j, k_pbc * Hessian(i, j)));
            }
        }
    });
}

// template<class T, int dim>
// void EoLRodSim<T, dim>::addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
// {
//     iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref, const auto& Tij){
        
//         TVDOF qi = q_temp.col(ni);
//         TVDOF qj = q_temp.col(nj);
//         TVDOF qi_ref = q_temp.col(ni_ref);
//         TVDOF qj_ref = q_temp.col(nj_ref);
//         TVDOF dij;
//         if (ni == ni_ref && nj == nj_ref && Tij.norm() > 1e-6)
//             dij = qi - qj -Tij;
//         else
//         {
//             dij = qi - qj - (qi_ref - qj_ref);
//             residual.col(ni_ref) += k_pbc * (dij);
//             residual.col(nj_ref) += -k_pbc * (dij);
//         }
//         residual.col(ni) += -k_pbc * (dij);
//             residual.col(nj) += k_pbc * (dij);
//     });
// }

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);

        T dij = (xj - xi).dot(strain_dir);
        
        residual.col(node_i).template segment<dim>(0) += k_pbc * strain_dir * (dij - Dij);
        residual.col(node_j).template segment<dim>(0) += -k_pbc * strain_dir * (dij - Dij);
    });
    // iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref, const auto& Tij){
        
    //     TVDOF qi = q_temp.col(ni);
    //     TVDOF qj = q_temp.col(nj);
    //     TVDOF qi_ref = q_temp.col(ni_ref);
    //     TVDOF qj_ref = q_temp.col(nj_ref);
    //     TVDOF dij;
    //     if (ni == ni_ref && nj == nj_ref && Tij.norm() > 1e-6)
    //         dij = qi - qj -Tij;
    //     else
    //     {
    //         dij = qi - qj - (qi_ref - qj_ref);
    //         residual.col(ni_ref) += k_pbc * (dij);
    //         residual.col(nj_ref) += -k_pbc * (dij);
    //     }
    //     residual.col(ni) += -k_pbc * (dij);
    //         residual.col(nj) += k_pbc * (dij);
    // });
}

// template<class T, int dim>
// T EoLRodSim<T, dim>::addPBCEnergy(Eigen::Ref<const DOFStack> q_temp)
// {
//     T energy_pbc = 0.0;
//     iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref, const auto& Tij){
//         TVDOF qi = q_temp.col(ni);
//         TVDOF qj = q_temp.col(nj);
//         TVDOF qi_ref = q_temp.col(ni_ref);
//         TVDOF qj_ref = q_temp.col(nj_ref);
//         TVDOF dij;
//         if (ni == ni_ref && nj == nj_ref && Tij.norm() > 1e-6)
//             dij = qi - qj - Tij;
//         else
//             dij = qi - qj - (qi_ref - qj_ref);
        
//         energy_pbc += 0.5 * k_pbc * dij.dot(dij);
//     });
//     // std::cout << "periodic bc energy: " << energy_pbc << std::endl;
//     return energy_pbc;
// }

template<class T, int dim>
T EoLRodSim<T, dim>::addPBCEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy_pbc = 0.0;
    iteratePBCStrainData([&](int node_i, int node_j, TV strain_dir, T Dij){
        TV xi = q_temp.col(node_i).template segment<dim>(0);
        TV xj = q_temp.col(node_j).template segment<dim>(0);

        T dij = (xj - xi).dot(strain_dir);
        
        energy_pbc += 0.5 * k_pbc * (dij - Dij) * (dij - Dij);
    });

    // std::cout << "periodic bc energy: " << energy_pbc << std::endl;
    return energy_pbc;
}



template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;