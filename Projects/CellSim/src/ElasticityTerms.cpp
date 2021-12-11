#include "../include/VertexModel.h"
#include "../include/autodiff/Elasticity.h"

void VertexModel::addElasticityEnergy(T& energy)
{
    T elastic_potential = 0.0;
    iterateFixedTetsSerial([&](Matrix<T, 3, 4>& x_deformed, 
        Matrix<T, 3, 4>& x_undeformed, VtxList& indices)
    {
        T ei = 0.0;
        compute3DNeoHookeanEnergyEnu(E, nu, x_deformed, x_undeformed, ei);
        elastic_potential += ei;
    });
    energy += elastic_potential;
}

void VertexModel::addElasticityForceEntries(VectorXT& residual)
{
    iterateFixedTetsSerial([&](Matrix<T, 3, 4>& x_deformed, 
        Matrix<T, 3, 4>& x_undeformed, VtxList& indices)
    {
        Vector<T, 12> dedx;
        compute3DNeoHookeanEnergyEnuGradient(E, nu, x_deformed, x_undeformed, dedx);
        addForceEntry<12>(residual, indices, -dedx);
    });
}

void VertexModel::addElasticityHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    iterateFixedTetsSerial([&](Matrix<T, 3, 4>& x_deformed, 
        Matrix<T, 3, 4>& x_undeformed, VtxList& indices)
    {
        Matrix<T, 12, 12> hessian;
        compute3DNeoHookeanEnergyEnuHessian(E, nu, x_deformed, x_undeformed, hessian);
        if (projectPD)
            projectBlockPD<12>(hessian);
        addHessianEntry<12>(entries, indices, hessian);
    });
}