#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegK(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for (int i = 0; i < n_nodes; i++)
    {
        entry_K.push_back(Eigen::Triplet<T>(i * dof + dim, i * dof + dim, ke));
        entry_K.push_back(Eigen::Triplet<T>(i * dof + dim + 1, i * dof + dim + 1, ke));
    }
}
template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    tbb::parallel_for(0, n_nodes, [&](int i){
        TV2 delta_eularian = q_temp.col(i).template segment<2>(dim) - q0.col(i).template segment<2>(dim);
        residual.col(i).template segment<2>(dim) += -ke * delta_eularian;
    });
}

template<class T, int dim>
T EoLRodSim<T, dim>::addEulerianRegEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    VectorXT energy(n_nodes);
    energy.setZero();
    tbb::parallel_for(0, n_nodes, [&](int i){
        TV2 delta_eularian = q_temp.col(i).template segment<2>(dim) - q0.col(i).template segment<2>(dim);
        energy[i] += 0.5 * ke * delta_eularian.dot(delta_eularian);
    });
    return energy.sum();
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;