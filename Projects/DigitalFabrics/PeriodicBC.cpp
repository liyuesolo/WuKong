#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
    iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref){
        for (int i = 0; i < dof; i++)
        {
            entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, ni_ref * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni * dof + i, nj_ref * dof + i, k_pbc));

            entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, ni_ref * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj * dof + i, nj_ref * dof + i, -k_pbc));

            entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, ni_ref * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(ni_ref * dof + i, nj_ref * dof + i, -k_pbc));

            entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni * dof + i, k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, ni_ref * dof + i, -k_pbc));
            entry_K.push_back(Eigen::Triplet<T>(nj_ref * dof + i, nj_ref * dof + i, k_pbc));
        }
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref){
        // cout4Nodes(ni, nj, ni_ref, nj_ref);
        TVDOF qi = q_temp.col(ni);
        TVDOF qj = q_temp.col(nj);
        TVDOF qi_ref = q_temp.col(ni_ref);
        TVDOF qj_ref = q_temp.col(nj_ref);
        TVDOF dij = qi - qj - (qi_ref - qj_ref);
        
        residual.col(ni) += -k_pbc * (dij);
        residual.col(nj) += k_pbc * (dij);
        residual.col(ni_ref) += k_pbc * (dij);
        residual.col(nj_ref) += -k_pbc * (dij);
    });
}

template<class T, int dim>
T EoLRodSim<T, dim>::addPBCEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy_pbc = 0.0;
    iteratePBCPairs([&](int ni, int nj, int ni_ref, int nj_ref){
        TVDOF qi = q_temp.col(ni);
        TVDOF qj = q_temp.col(nj);
        TVDOF qi_ref = q_temp.col(ni_ref);
        TVDOF qj_ref = q_temp.col(nj_ref);
        TVDOF dij = qi - qj - (qi_ref - qj_ref);

        energy_pbc += 0.5 * k_pbc * dij.dot(dij);
    });
    
    return energy_pbc;
}
template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;