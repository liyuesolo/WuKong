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
    EoLRodSim<T, dim>& sim;
public:
    Homogenization(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~Homogenization() {}

    void initalizeSim()
    {
        sim.buildPlanePeriodicBCScene3x3();
    }

    T YoungsModulusFromUniaxialStrain()
    {
        TV displacement = TV::Zero();
        displacement[0] = -1.2;
        displacement[1] = -0.1;
        sim.setUniaxialStrain(displacement);
        return 1.0;
    }
};

#endif