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
// template class FEMSolver<2>;
template class FEMSolver<3>;