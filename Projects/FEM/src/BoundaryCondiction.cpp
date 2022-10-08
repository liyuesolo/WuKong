#include <igl/copyleft/tetgen/tetrahedralize.h>
#include "../include/FEMSolver.h"

template <int dim>
void FEMSolver<dim>::dragMiddle()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if ((x[0] > center[0] - 0.1 && x[0] < center[0] + 0.1) 
            && (x[2] < min_corner[2] + 1e-6))
        {
            dirichlet_data[i * dim + 2] = -1;
            f[i * dim + 2] = -100.0;
        }
    }
}

template <int dim>
void FEMSolver<dim>::applyCompression(int dir, T percent)
{
    use_penalty = true;
    penalty_pairs.clear();
    
    T region = 0.01;
    for (int i = 0; i < num_nodes; i++)
    {
        T dx = max_corner[dir] - min_corner[dir];
        TV x = undeformed.segment<3>(i * dim);
        if (x[dir] < min_corner[dir] + region * dx)
        {
            TV target = x;
            target[dir] += dx * percent;
            for (int d = 0; d < dim; d++)
                penalty_pairs.push_back(std::make_pair(i * dim + d, target[d]));
        }
        if (x[dir] > max_corner[dir] - region * dx)
        {
            TV target = x;
            target[dir] -= dx * percent;
            for (int d = 0; d < dim; d++)
                penalty_pairs.push_back(std::make_pair(i * dim + d, target[d]));
        }
    }
}

// template class FEMSolver<2>;
template class FEMSolver<3>;