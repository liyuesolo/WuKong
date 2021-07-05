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
void EoLRodSim<T, dim>::addParallelContactK(std::vector<Entry>& entry_K)
{
    for (auto& crossing : rod_crossings)
    {
        int node_idx = crossing->node_idx;
        std::vector<int> rods_involved = crossing->rods_involved;
        std::vector<Vector<T, 2>> sliding_ranges = crossing->sliding_ranges;

        int cnt = 0;
        for (int rod_idx : rods_involved)
        {
            Offset offset;
            Rods[rod_idx]->getEntry(node_idx, offset);
            T u, U;
            Rods[rod_idx]->u(node_idx, u);
            Rods[rod_idx]->U(node_idx, U);
            T delta_u = (u - U);
            Range range = sliding_ranges[cnt];
            // 0 is the positive side sliding range
            if(delta_u > 0 && range[0] > 1e-6)
            {
                entry_K.push_back(Entry(offset[dim], offset[dim], k_yc));
            }
            else if (delta_u < 0 && range[1] > 1e-6)
            {
                entry_K.push_back(Entry(offset[dim], offset[dim], k_yc));
            }
            cnt++;
        }
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactForce(Eigen::Ref<VectorXT> residual)
{
    VectorXT residual_cp = residual;
    for (auto& crossing : rod_crossings)
    {
        int node_idx = crossing->node_idx;
        std::vector<int> rods_involved = crossing->rods_involved;
        std::vector<Vector<T, 2>> sliding_ranges = crossing->sliding_ranges;

        int cnt = 0;
        for (int rod_idx : rods_involved)
        {
            Offset offset;
            Rods[rod_idx]->getEntry(node_idx, offset);
            T u, U;
            Rods[rod_idx]->u(node_idx, u);
            Rods[rod_idx]->U(node_idx, U);
            T delta_u = (u - U);
            Range range = sliding_ranges[cnt];
            // 0 is the positive side sliding range
            if(delta_u > 0 && range[0] > 1e-6)
            {
                residual[offset[dim]] += -k_yc * (delta_u - range[0]);
            }
            else if (delta_u < 0 && range[1] > 1e-6)
            {
                residual[offset[dim]] += -k_yc * (delta_u + range[1]);
            }
            cnt++;
        }
    }
    if (print_force_mag)
        std::cout << "contact force norm: " << (residual - residual_cp).norm() << std::endl;
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
                    residual(dir, node_id) += -k_yc * (delta_u + tunnel_R);
                else
                    residual(dir, node_id) += -k_yc * (delta_u - tunnel_R);
                // std::cout << "node " << middle << " " << delta_u << " " << tunnel_R << std::endl;   
            };
        singleDirGrad(dim);
        singleDirGrad(dim + 1);
    });
    if (print_force_mag)
        std::cout << "contact force norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addParallelContactEnergy()
{
    T energy = 0.0;
    for (auto& crossing : rod_crossings)
    {
        int node_idx = crossing->node_idx;
        std::vector<int> rods_involved = crossing->rods_involved;
        std::vector<Vector<T, 2>> sliding_ranges = crossing->sliding_ranges;

        int cnt = 0;
        for (int rod_idx : rods_involved)
        {
            Offset offset;
            Rods[rod_idx]->getEntry(node_idx, offset);
            T u, U;
            Rods[rod_idx]->u(node_idx, u);
            Rods[rod_idx]->U(node_idx, U);
            T delta_u = (u - U);
            Range range = sliding_ranges[cnt];
            // 0 is the positive side sliding range
            if(delta_u > 0 && range[0] > 1e-6)
            {
                energy += 0.5 * k_yc * std::pow(delta_u - range[0], 2);
            }
            else if (delta_u < 0 && range[1] > 1e-6)
            {
                energy += 0.5 * k_yc * std::pow(delta_u + range[1], 2);
            }
            cnt++;
        }
    }
    return energy;
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

            // std::vector<TV> X, X0, dXdu, d2Xdu2;
            // getMaterialPositions(q0, {node_id}, X0, dir, dXdu, d2Xdu2, false, false );
            // getMaterialPositions(q_temp, {node_id}, X, dir, dXdu, d2Xdu2, false, false );

            // T delta_u = (X[0] - X0[0]).norm();
            // std::cout << "delta u " << delta_u << " tunnel R " << tunnel_R << " dim " << dir << std::endl;
            if (std::abs(delta_u) < tunnel_R)
                return;
            if (delta_u < 0)
                // crossing_energy[node_id] += 0.5 * k_yc * std::pow(q_temp(dir, node_id) - q0(dir, node_id) + tunnel_R, 2);
                crossing_energy[node_id] += 0.5 * k_yc * std::pow(delta_u + tunnel_R, 2);
            else
                crossing_energy[node_id] += 0.5 * k_yc * std::pow(delta_u - tunnel_R, 2);
            
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