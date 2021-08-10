#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addParallelContactK(std::vector<Entry>& entry_K)
{
    if (!add_contact_penalty)
        return;
    for (auto& crossing : rod_crossings)
    {
        if (crossing->is_fixed)
            continue;
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
            if(delta_u >= range[0])
            {
                entry_K.push_back(Entry(offset[dim], offset[dim], k_yc));
            }
            else if (delta_u <= -range[1])
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
    if (!add_contact_penalty)
        return;
    VectorXT residual_cp = residual;
    for (auto& crossing : rod_crossings)
    {
        if (crossing->is_fixed)
            continue;
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
            // if(rod_idx == 1)
            //     std::cout << range.transpose() << " " << delta_u << std::endl;
            // 0 is the positive side sliding range
            if(delta_u >= range[0])
            {
                // std::cout<< delta_u  << " " << range[0] << std::endl;
                residual[offset[dim]] += -k_yc * (delta_u - range[0]);
            }
            // 1 is the sliding range along the negative direction
            else if (delta_u <= -range[1])
            {
                // std::cout<< delta_u  << " " << range[1] << std::endl;
                residual[offset[dim]] += -k_yc * (delta_u + range[1]);
            }
            cnt++;
        }
    }
    if (print_force_mag)
        std::cout << "contact force norm: " << (residual - residual_cp).norm() << std::endl;
}

template<class T, int dim>
T EoLRodSim<T, dim>::addParallelContactEnergy()
{
    if (!add_contact_penalty)
        return 0.0;
    T energy = 0.0;
    for (auto& crossing : rod_crossings)
    {
        if (crossing->is_fixed)
            continue;
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
            // std::cout << delta_u << " " << range.transpose() << std::endl;
            if(delta_u >= range[0])
            {
                // std::cout << delta_u << " " << u << " " << U << std::endl;
                energy += 0.5 * k_yc * std::pow(delta_u - range[0], 2);
            }
            else if (delta_u <= -range[1])
            {
                // std::cout << delta_u << " " << u << " " << U << std::endl;
                energy += 0.5 * k_yc * std::pow(delta_u + range[1], 2);
            }
            cnt++;
        }
    }
    return energy;
}



template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;