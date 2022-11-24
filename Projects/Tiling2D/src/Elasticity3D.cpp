#include "../include/HexFEMSolver.h"
#include "../include/autodiff/HexFEM.h"
void HexFEMSolver::addPlainStrainElastsicPotential(T& energy)
{
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        T ei;
        if (stvk)
            computeHexFEMPlainStrainStVKEnergy(KL_stiffness, KL_stiffness_shear, lambda, mu, x.transpose(), X.transpose(), ei);
        else
            computeHexFEMPlainStrainNHEnergy(KL_stiffness, lambda, mu, x.transpose(), X.transpose(), ei);
        // std::cout << ei << std::endl;
        // std::getchar();
        energy += ei;
    });
}

void HexFEMSolver::addPlainStrainElasticForceEntries(VectorXT& residual)
{
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
    
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);

        Vector<T, 24> dedx;
        if (stvk)
            computeHexFEMPlainStrainStVKEnergyGradient(KL_stiffness, KL_stiffness_shear, lambda, mu, x.transpose(), X.transpose(), dedx);
        else
            computeHexFEMPlainStrainNHEnergyGradient(KL_stiffness, lambda, mu, x.transpose(), X.transpose(), dedx);
        addForceEntry<24>(residual, nodal_indices, -dedx);
        
    });
}

void HexFEMSolver::addPlainStrainElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    iterateHexElementSerial([&](int cell_idx){
        Vector<int, 8> nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        Matrix<T, 24, 24> hessian;
        if (stvk)
            computeHexFEMPlainStrainStVKEnergyHessian(KL_stiffness, KL_stiffness_shear, lambda, mu, x.transpose(), X.transpose(), hessian);
        else
            computeHexFEMPlainStrainNHEnergyHessian(KL_stiffness, lambda, mu, x.transpose(), X.transpose(), hessian);
        addHessianEntry<24>(entries, nodal_indices, hessian);
        
    });
}

void HexFEMSolver::addNHElastsicPotential(T& energy)
{
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        if (stvk)
            energy += computeHexFEMStVKEnergy(lambda, mu, x.transpose(), X.transpose());
        else
            energy += computeHexFEMNeoHookeanEnergy(lambda, mu, x.transpose(), X.transpose());
    });
}
void HexFEMSolver::addNHElasticForceEntries(VectorXT& residual)
{
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);

        Vector<T, 24> dedx;
        if (stvk)
            computeHexFEMStVKEnergyGradient(lambda, mu, x.transpose(), X.transpose(), dedx);
        else
            computeHexFEMNeoHookeanEnergyGradient(lambda, mu, x.transpose(), X.transpose(), dedx);
        addForceEntry<24>(residual, nodal_indices, -dedx);
        
    });
}
void HexFEMSolver::addNHElasticHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    iterateHexElementSerial([&](int cell_idx){
        Vector<int, 8> nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        Matrix<T, 24, 24> hessian;

        if (stvk)
            computeHexFEMStVKEnergyHessian(lambda, mu, x.transpose(), X.transpose(), hessian);
        else
            computeHexFEMNeoHookeanEnergyHessian(lambda, mu, x.transpose(), X.transpose(), hessian);

        addHessianEntry<24>(entries, nodal_indices, hessian);
    });
}

void HexFEMSolver::checkGreenStrain()
{
    auto shapeFunction = [&](const TV& xi)
    {
        Vector<T, 8> basis;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                    basis[4*(i-1) + 2 * (j-1) + k - 1] = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
        return basis;
    };

    auto dNdxi = [&](const TV& xi)
    {
        Matrix<T, 8, 3> dN_dxi;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                {
                    int basis_id = 4*(i-1) + 2 * (j-1) + k - 1;
                    dN_dxi(basis_id, 0) = ((pow(-1.0, i)) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 1) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        (pow(-1.0, j)) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 2) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * (pow(-1.0, k)) ) / 8.0;
                }
        return dN_dxi;
    };

    int cnt = 0;
    iterateHexElementSerial([&](int cell_idx)
    {
        if (cnt > 0 )
            return;
        cnt++;
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
    
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);

        for (int i = 1; i < 3; i++)
        {
            for (int j = 1; j < 3; j++)
            {
                for (int k = 1; k < 3; k++)
                {
                    TV xi(pow(-1.0, i) / sqrt(3.0), pow(-1.0, j) / sqrt(3.0), pow(-1.0, k) / sqrt(3.0));
                    Matrix<T, 8, 3> dNdb = dNdxi(xi);
                    
                    TM dXdb = X.transpose() * dNdb;
                    TM dxdb = x.transpose() * dNdb;
                    
                    TM defGrad = dxdb * dXdb.inverse();
                    TM GreenStrain = 0.5 * (defGrad.transpose() * defGrad - TM::Identity());
                    std::cout << GreenStrain << std::endl;
                    std::cout << std::endl;
                    break;
                }
                break;
            }
            break;
        }
        return;
        
    });
}

T HexFEMSolver::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    auto shapeFunction = [&](const TV& xi)
    {
        Vector<T, 8> basis;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                    basis[4*(i-1) + 2 * (j-1) + k - 1] = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
        return basis;
    };

    auto dNdxi = [&](const TV& xi)
    {
        Matrix<T, 8, 3> dN_dxi;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                {
                    int basis_id = 4*(i-1) + 2 * (j-1) + k - 1;
                    dN_dxi(basis_id, 0) = ((pow(-1.0, i)) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 1) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        (pow(-1.0, j)) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 2) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * (pow(-1.0, k)) ) / 8.0;
                }
        return dN_dxi;
    };

    VectorXT step_sizes = VectorXT::Zero(indices.rows());
    iterateHexElementSerial([&](int cell_idx){
        Vector<int, 8> nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        int cnt = 0;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                {
                    TV xi(pow(-1.0, i) / sqrt(3.0), pow(-1.0, j) / sqrt(3.0), pow(-1.0, k) / sqrt(3.0));
                    Matrix<T, 8, 3> dNdb = dNdxi(xi);
                    
                    TM dXdb = X.transpose() * dNdb;
                    TM dxdb = x.transpose() * dNdb;
                    
                    TM A = dxdb * dXdb.inverse();
                    T a, b, c, d;
                    a = A.determinant();
                    b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
                    c = A.diagonal().sum();
                    d = 0.8;

                    T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
                    if (t < 0 || t > 1) t = 1;
                        step_sizes(cell_idx * 8 + cnt) = t;
                    cnt++;
                }
    });
    return step_sizes.minCoeff();
}