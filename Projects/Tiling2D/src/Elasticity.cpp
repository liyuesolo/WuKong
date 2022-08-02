#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

void FEMSolver::addElastsicPotential(T& energy)
{
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    
    iterateElementsParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
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
        computeLinear2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
        if (project_PD)
            projectBlockPD<6>(hessian);
        
        addHessianEntry<6>(entries, indices, hessian);
    });
}

T FEMSolver::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    Matrix<T, 3, 2> dNdb;
        dNdb << -1.0, -1.0, 
            1.0, 0.0,
            0.0, 1.0;
           
    VectorXT step_sizes = VectorXT::Zero(num_ele);

    iterateElementsParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        TM dXdb = x_undeformed.transpose() * dNdb;
        TM dxdb = x_deformed.transpose() * dNdb;
        TM A = dxdb * dXdb.inverse();
        T a, b, c, d;
        a = 0;
        b = A.determinant();
        c = A.diagonal().sum();
        d = 0.8;

        T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
        if (t < 0 || t > 1) t = 1;
            step_sizes(tet_idx) = t;
    });
    return step_sizes.minCoeff();
}

