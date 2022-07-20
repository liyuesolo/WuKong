#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

void FEMSolver::addElastsicPotential(T& energy)
{
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    iterateElementsParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        // T ei = computeNeoHookeanStrainEnergy(x_deformed, x_undeformed);
        T ei;
        computeLinear2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
    });
    energy += energies_neoHookean.sum();
}

void FEMSolver::addElasticForceEntries(VectorXT& residual)
{
    iterateElementsSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        Vector<T, 6> dedx;
        // computeNeoHookeanStrainEnergyGradient(x_deformed, x_undeformed, dedx);
        computeLinear2DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
        
        addForceEntry<6>(residual, indices, -dedx);
    });
}

void FEMSolver::addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    iterateElementsSerial([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        Matrix<T, 6, 6> hessian;
        // computeNeoHookeanStrainEnergyHessian(x_deformed, x_undeformed, hessian);
        computeLinear2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
        if (project_PD)
            projectBlockPD<6>(hessian);
        
        addHessianEntry<6>(entries, indices, hessian);
    });
}

