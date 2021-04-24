#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{
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
}


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
    return energy_pbc;
}



template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;