#ifndef HOMOGENIZATION_H
#define HOMOGENIZATION_H

#include "EoLRodSim.h"

template<class T, int dim>
class EoLRodSim;

template<class T, int dim>
class Homogenization
{
    using TVDOF = Vector<T, dim+2>;
    using DOFStack = Matrix<T, dim + 2, Eigen::Dynamic>;
    using IV2 = Vector<int, 2>;
    using TV = Vector<T, dim>;

public:
    EoLRodSim<T, dim>& eol_sim;
public:
    Homogenization(EoLRodSim<T, dim>& sim) : eol_sim(sim) {}
    ~Homogenization() {}


    T YoungsModulusFromUniaxialStrain()
    {
        
    }
};

#endif