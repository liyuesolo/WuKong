#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegK(std::vector<Eigen::Triplet<T>>& entry_K)
{
    // for (int i = 0; i < n_nodes; i++)
    // {
    //     entry_K.push_back(Eigen::Triplet<T>(i * dof + dim, i * dof + dim, ke));
    //     entry_K.push_back(Eigen::Triplet<T>(i * dof + dim + 1, i * dof + dim + 1, ke));
    //     // int yarn_type = rods.col(i)[2];
    //     // entry_K.push_back(Eigen::Triplet<T>(i * dof + dim + yarn_type, i * dof + dim + yarn_type, ke));
    // }

    // iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        
    //     if (left != -1 && right != -1 && top != -1 && bottom != -1)
    //     {
    //         entry_K.push_back(Eigen::Triplet<T>(middle * dof + dim, middle * dof + dim, ke));
    //         entry_K.push_back(Eigen::Triplet<T>(middle * dof + dim + 1, middle * dof + dim + 1, ke));
    //     }
    // });

    iterateSlidingNodes([&](int node_id){
        entry_K.push_back(Eigen::Triplet<T>(node_id * dof + dim, node_id * dof + dim, ke));
        entry_K.push_back(Eigen::Triplet<T>(node_id * dof + dim + 1, node_id * dof + dim + 1, ke));
    });

}
template<class T, int dim>
void EoLRodSim<T, dim>::addEulerianRegForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;

    // iterateYarnCrossingsSerial([&](int middle, int bottom, int top, int left, int right){
        
    //     if (left != -1 && right != -1 && top != -1 && bottom != -1)
    //     {
    //         TV2 delta_eularian = q_temp.col(middle).template segment<2>(dim) - q0.col(middle).template segment<2>(dim);
    //         residual.col(middle).template segment<2>(dim) += -ke * delta_eularian;
    //     }
    // });

    iterateSlidingNodes([&](int node_id){
        TV2 delta_eularian = q_temp.col(node_id).template segment<2>(dim) - q0.col(node_id).template segment<2>(dim);
        residual.col(node_id).template segment<2>(dim) += -ke * delta_eularian;
    });

    if(print_force_mag)
        std::cout << "Eulerian penalty norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addEulerianRegEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();

    iterateSlidingNodes([&](int node_id){
        TV2 delta_eularian = q_temp.col(node_id).template segment<2>(dim) - q0.col(node_id).template segment<2>(dim);
        crossing_energy[node_id] += 0.5 * ke * delta_eularian.dot(delta_eularian);
    });
    return crossing_energy.sum();
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;