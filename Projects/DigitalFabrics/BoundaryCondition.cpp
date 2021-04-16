#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::addBCStretchingTest()
{
    for(int node_id = 0; node_id < n_nodes; node_id++)
    {
        TV2 uv = q.col(node_id).template segment<2>(dim);
        TVDOF target, mask;
        if (std::abs(uv[0]) < 1e-6)
        {
            target.setZero();
            target[0] = -0.2;
            mask.setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        else if (std::abs(uv[0] - 1) < 1e-6)
        {
            target.setZero();
            target[0] = 0.2;
            mask.setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        else
        {
            // target.setZero();
            // mask.setOnes();
            // mask.template segment<dim>(0).setZero();
            // dirichlet_data[node_id] = std::make_pair(target, mask);
        }

    }
    std::cout << "# Dirichlet constraint: " << dirichlet_data.size() << std::endl;
}

// template<class T, int dim>
// void EoLRodSim<T, dim>::addBCStretchingTest()
// {
//     for(int node_id = 0; node_id < n_nodes; node_id++)
//     {
//         TV2 uv = q.col(node_id).template segment<2>(dim);
//         TVDOF target, mask;
//         if (std::abs(uv[0]) < 1e-6 && uv[1] == 0.5 )
//         {
//             target.setZero();
//             target[0] = -0.2;
//             mask.setOnes();
//             dirichlet_data[node_id] = std::make_pair(target, mask);
//         }
//         else if (std::abs(uv[0] - 1) < 1e-6 && uv[1] == 0.5 )
//         {
//             target.setZero();
//             target[0] = 0.2;
//             mask.setOnes();
//             mask.template segment<dim>(0).setZero();
//             dirichlet_data[node_id] = std::make_pair(target, mask);
//         }
//         // else
//         // {
//         //     target.setZero();
//         //     mask.setOnes();
//         //     mask.template segment<dim>(0).setZero();
//         //     dirichlet_data[node_id] = std::make_pair(target, mask);
//         // }

//     }
//     std::cout << "# Dirichlet constraint: " << dirichlet_data.size() << std::endl;
// }

template<class T, int dim>
void EoLRodSim<T, dim>::addBCShearingTest()
{
    for(int node_id = 0; node_id < n_nodes; node_id++)
    {
        TV2 uv = q.col(node_id).template segment<2>(dim);
        TVDOF target, mask;
        if (std::abs(uv[0]) < 1e-6)
        {
            target.setZero();
            // target[0] = -0.2;
            mask.setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        else if (std::abs(uv[0] - 1) < 1e-6)
        {
            target.setZero();
            if constexpr (dim == 3)
                target[2] = 0.2;
            else
                target[0] = -0.1;
            mask.setOnes();
            dirichlet_data[node_id] = std::make_pair(target, mask);
        }
        else
        {
            // target.setZero();
            // mask.setOnes();
            // mask.template segment<dim>(0).setZero();
            // dirichlet_data[node_id] = std::make_pair(target, mask);
        }

    }
    std::cout << "# Dirichlet constraint: " << dirichlet_data.size() << std::endl;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;