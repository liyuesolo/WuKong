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
    using TV2 = Vector<T, 2>;
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;
    using CDoF2D = Vector<T, 6>;
    using CHessian2D = Matrix<T, 6, 6>;
    using ComplianceTensor = Matrix<T, 3 * (dim - 1), 3 * (dim - 1)>;
    using TVEntry = Vector<T, 3 * (dim - 1)>;

    using ComplianceTensorFull = Matrix<T, dim * dim, dim * dim>;
    using TVEntryFull = Vector<T, dim * dim>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;


public:
    EoLRodSim<T, dim>& sim;
    
public:
    Homogenization(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~Homogenization() {}

    void initalizeSim();

    void marcoMaterialParametersFitting();
    void materialParametersFromUniaxialStrain(T theta, T s, TV2& E_nu);
    void computeYoungsModulusPoissonRatioBatch();

    void computeMacroStressStrain(TM& stress_marco, TM& strain_marco);
    
    void fitComplianceTensor();
    void fitComplianceFullTensor();
};

#endif