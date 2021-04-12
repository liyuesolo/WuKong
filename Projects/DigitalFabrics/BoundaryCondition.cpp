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
            target = q.col(node_id);
            mask.setZero();
            mask.template segment<dim>(0).setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        if (std::abs(uv[0] - 1) < 1e-6)
        {
            TVDOF target, mask;
            target = q.col(node_id);
            target[dim - 1] += 0.2;
            mask.setZero();
            mask.template segment<dim>(0) = TV::Ones();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
    }
    std::cout << "# Dirichlet constraint: " << dirichlet_data.size() << std::endl;
}

template class EoLRodSim<double, 3>;