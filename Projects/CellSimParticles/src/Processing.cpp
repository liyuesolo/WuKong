#include "../include/CellSim.h"


template <int dim>
void CellSim<dim>::removeVtxInsideYolk()
{
    if constexpr (dim == 3)
    {
        // iterateCellParallel([&](in))
    }
}

template class CellSim<2>;
template class CellSim<3>;