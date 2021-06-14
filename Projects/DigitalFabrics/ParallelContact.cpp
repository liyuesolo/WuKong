#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K)
{

    iterateSlidingNodes([&](int node_id){
        auto singleDirHessian = [&, node_id](int dir)
            {
                T delta_u = std::abs(q_temp(dir, node_id) - q0(dir, node_id));
                T tunnel_R = dir == dim ? tunnel_u : tunnel_v;
                if (delta_u < tunnel_R)
                    return;
                entry_K.push_back(Entry(node_id * dof + dir, node_id * dof + dir, k_yc));
            };
            singleDirHessian(dim);
            singleDirHessian(dim+1);
    });
}

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual)
{
    DOFStack residual_cp = residual;
    
    iterateSlidingNodes([&](int node_id){
        auto singleDirGrad = [&, node_id](int dir)
            {
                T delta_u = q_temp(dir, node_id) - q0(dir, node_id);
                T tunnel_R = dir == dim ? tunnel_u : tunnel_v;
                if (std::abs(delta_u) < tunnel_R)
                    return;
                if (delta_u < 0)
                    residual(dir, node_id) += -k_yc * (q_temp(dir, node_id) - q0(dir, node_id) + tunnel_R);
                else
                    residual(dir, node_id) += -k_yc * (q_temp(dir, node_id) - q0(dir, node_id) - tunnel_R);
                // std::cout << "node " << middle << " " << delta_u << " " << tunnel_R << std::endl;   
            };
        singleDirGrad(dim);
        singleDirGrad(dim + 1);
    });
    if (print_force_mag)
        std::cout << "contact force norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addParallelContactEnergy(Eigen::Ref<const DOFStack> q_temp)
{
    T energy = 0.0;
    
    VectorXT crossing_energy(n_nodes);
    crossing_energy.setZero();
    iterateSlidingNodes([&](int node_id){
        auto singleDirEnergy = [&, node_id](int dir)
        {
            T tunnel_R = dir == dim ? tunnel_u : tunnel_v;
            T delta_u = q_temp(dir, node_id) - q0(dir, node_id);
            
            if (std::abs(delta_u) < tunnel_R)
                return;
            if (delta_u < 0)
                crossing_energy[node_id] += 0.5 * k_yc * std::pow(q_temp(dir, node_id) - q0(dir, node_id) + tunnel_R, 2);
            else
                crossing_energy[node_id] += 0.5 * k_yc * std::pow(q_temp(dir, node_id) - q0(dir, node_id) - tunnel_R, 2);
            
            // std::cout << "node " << node_id << std::endl;
        };
        singleDirEnergy(dim);
        singleDirEnergy(dim+1);
    });
    
    energy += crossing_energy.sum();
    return energy;
}


template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;