#include "FEMSolver.h"

template<class T, int dim>
void FEMSolver<T, dim>::fixAxisEnd(int axis)
{
    // std::cout << "fix end" << std::endl;
    for (int i = 0; i < num_nodes; i++)
    {
        TV pos = undeformed.template segment<dim>(i * dim);
        if (pos[axis] < min_corner[axis] + 1e-6)
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = 0.0;
        }
    }
    // std::cout << "fix end done" << std::endl;
}

template<class T, int dim>
void FEMSolver<T, dim>::addDirichletLambda(std::function<bool(const TV&, TV&)> node_helper)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV delta;
        if (node_helper(undeformed.template segment<dim>(i * dim), delta))
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = delta[d];
            
        }
    }
    
}

template<class T, int dim>
void FEMSolver<T, dim>::addNeumannLambda(
    std::function<bool(const TV&, TV&)> node_helper,
    VectorXT& f)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV force;
        if (node_helper(undeformed.template segment<dim>(i * dim), force))
        {
            f.template segment<dim>(i * dim) += force;
            // std::cout << "add force to node " << i << std::endl;
        }
    }
    // std::cout << f << std::endl;
}

template class FEMSolver<float, 2>;
template class FEMSolver<float, 3>;
template class FEMSolver<double, 2>;
template class FEMSolver<double, 3>;