#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegK(std::vector<Eigen::Triplet<T>>& entry_K)
{
    for (int i = 0; i < n_nodes; i++)
    {
        entry_K.push_back(Eigen::Triplet<T>(i * dof + dim, i * dof + dim, ke));
        entry_K.push_back(Eigen::Triplet<T>(i * dof + dim + 1, i * dof + dim + 1, ke));
        // int yarn_type = rods.col(i)[2];
        // entry_K.push_back(Eigen::Triplet<T>(i * dof + dim + yarn_type, i * dof + dim + yarn_type, ke));
    }
}
template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;
    tbb::parallel_for(0, n_nodes, [&](int i){
        TV2 delta_eularian = q_temp.col(i).template segment<2>(dim) - q0.col(i).template segment<2>(dim);
        residual.col(i).template segment<2>(dim) += -ke * delta_eularian;
        // int yarn_type = rods.col(i)[2];
        // T delta_eularian = q_temp(yarn_type + dim, i) - q0(yarn_type + dim, i);
        // residual(dim + yarn_type, i) += -ke * delta_eularian;
    });
    // std::cout << "Eulerian penalty norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addEulerianRegEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    VectorXT energy(n_nodes);
    energy.setZero();
    tbb::parallel_for(0, n_nodes, [&](int i){
        int yarn_type = rods.col(i)[2];
        // T delta_eularian = q_temp(yarn_type + dim, i) - q0(yarn_type + dim, i);
        // energy[i] += 0.5 * ke * delta_eularian * delta_eularian;
        TV2 delta_eularian = q_temp.col(i).template segment<2>(dim) - q0.col(i).template segment<2>(dim);
        energy[i] += 0.5 * ke * delta_eularian.dot(delta_eularian);
    });
    // std::cout << "|u|: " << energy.sum() << std::endl;
    return energy.sum();
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;