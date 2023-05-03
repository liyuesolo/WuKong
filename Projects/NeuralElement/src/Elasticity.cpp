#include "../include/FEMSolver.h"
#include "../autodiff/FEMEnergy.h"


template<int dim>
void FEMSolver<dim>::addElastsicPotential(T& energy)
{
    VectorXT energies_neoHookean(num_ele);
    energies_neoHookean.setZero();
    if constexpr (dim == 2)
    {
        if (quadratic)
            iterateQuadElementsParallel([&](const QuadEleNodes& x_deformed, 
                const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
            {
                T ei;
                computeQuadratic2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
                energies_neoHookean[tet_idx] += ei;
            });
        else
            iterateElementsParallel([&](const EleNodes& x_deformed, 
                const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
            {
                T ei;
                computeLinear2DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
                energies_neoHookean[tet_idx] += ei;
            });
    }
    energy += energies_neoHookean.sum();
}



template<int dim>
void FEMSolver<dim>::addElasticForceEntries(VectorXT& residual)
{
    if constexpr (dim == 2)
    {
        if (quadratic)
            iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
                const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
            {

                Vector<T, 12> dedx;
                computeQuadratic2DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
                
                addForceEntry<12>(residual, indices, -dedx);
            });
        else
            iterateElementsSerial([&](const EleNodes& x_deformed, 
                const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
            {
                Vector<T, 6> dedx;
                computeLinear2DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
                
                addForceEntry<6>(residual, indices, -dedx);
            });
    }
    
    
}

template<int dim>
void FEMSolver<dim>::addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    if constexpr (dim == 2)
    {
        if (quadratic)
            iterateQuadElementsSerial([&](const QuadEleNodes& x_deformed, 
                const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int tet_idx)
            {
                Matrix<T, 12, 12> hessian;
                computeQuadratic2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
                if (project_PD)
                    projectBlockPD<12>(hessian);
                
                addHessianEntry<12>(entries, indices, hessian);
            });
        else
            iterateElementsSerial([&](const EleNodes& x_deformed, 
                const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
            {
                Matrix<T, 6, 6> hessian;
                computeLinear2DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
                // std::cout << hessian << std::endl;
                // std::getchar();
                if (project_PD)
                    projectBlockPD<6>(hessian);
                
                addHessianEntry<6>(entries, indices, hessian);
            });
    }
}


template<int dim>
T FEMSolver<dim>::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    if constexpr (dim == 2)
    {
        if (quadratic)
        {
            auto computedNdb = [&](const TV &xi)
            {
                Matrix<T, 6, 2> dNdb;
                dNdb(0, 0) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
                dNdb(1, 0) = 4.0 * xi[0] - 1.0;
                dNdb(2, 0) = 0.0;
                dNdb(3, 0) = 4.0 * xi[1];
                dNdb(4, 0) = -4.0 * xi[1];
                dNdb(5, 0) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[0];

                dNdb(0, 1) = -(1.0 - 2.0 * xi[0] - 2.0 * xi[1]) - 2.0 * (1.0 - xi[0] - xi[1]);
                dNdb(1, 1) = 0.0;
                dNdb(2, 1) = 4.0 * xi[1] - 1.0;
                dNdb(3, 1) = 4.0 * xi[0];
                dNdb(4, 1) = 4.0 * (1.0 - xi[0] - xi[1]) - 4.0 * xi[1];
                dNdb(5, 1) = -4.0 * xi[0];
                return dNdb;
            };

            auto gauss2DP2Position = [&](int idx)
            {
                if (idx == 0)
                    return TV(T(1.0 / 6.0), T(1.0 / 6.0));
                else if (idx == 1)
                    return TV(T(2.0 / 3.0), T(1.0 / 6.0));
                return TV(T(1.0 / 6.0), T(2.0 / 3.0));
                
            };
            VectorXT step_sizes = VectorXT::Zero(num_ele * 3);
            iterateQuadElementsParallel([&](const QuadEleNodes& x_deformed, 
                const QuadEleNodes& x_undeformed, const QuadEleIdx& indices, int ele_idx)
            {
                for (int idx = 0; idx < 3; idx++)
                {
                    TV xi = gauss2DP2Position(idx);
                    Matrix<T, 6, 2> dNdb = computedNdb(xi);
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
                        step_sizes(ele_idx * 3 + idx) = t;
                } 
            });
            return step_sizes.minCoeff();
        }
        else
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
    }
    
}

template class FEMSolver<2>;
template class FEMSolver<3>;