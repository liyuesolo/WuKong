#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addBCStretchingTest()
{
    for(int node_id = 0; node_id < n_nodes; node_id++)
    {
        TV2 uv = q.col(node_id).template segment<2>(dim);
        if (std::abs(uv[0]) < 1e-6)
        {
            TVDOF target, mask;
            target.setZero();
            mask.setZero();
            mask.template segment<dim>(0).setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        if (std::abs(uv[1] - 1) < 1e-6)
        {
            TVDOF target, mask;
            target.setZero();
            target[dim - 1] = 0.2;
            mask.setZero();
            mask.template segment<dim>(0).setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
    }
}

template class EoLRodSim<double, 3>;