#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::setUniaxialStrain(TV displacement)
{
    pbc_translation[0] = TVDOF::Zero();
    pbc_translation[0][0] = displacement[0];
    pbc_translation[0][1] = displacement[1];
    pbc_translation[0][2] = -1.;
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;