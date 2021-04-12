#include "EoLRodSim.h"

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradient(Eigen::Ref<DOFStack> dq)
{
    T epsilon = 1e-4;
    DOFStack gradient(dof, n_nodes);
    gradient.setZero();
    std::cout << "checkGradient computeResidual" << std::endl;
    computeResidual(gradient, dq);
    DOFStack gradient_FD(dof, n_nodes);
    gradient_FD.setZero();

    for(int n_node = 0; n_node < n_nodes; n_node++)
    {
        for (int d = 0; d < dof; ++d)
        {
            dq(d, n_node) += epsilon;
            T E0 = computeTotalEnergy(dq);
            dq(d, n_node) -= 2.0 * epsilon;
            T E1 = computeTotalEnergy(dq);
            dq(d, n_node) += epsilon;
            gradient_FD(d, n_node) = (E1 - E0) / (2*epsilon);
            std::cout << n_node << " " << gradient_FD(d, n_node) << " " << gradient(d, n_node) << std::endl;
            std::getchar();
        }
        
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessian(Eigen::Ref<DOFStack> dq)
{
    T epsilon = 1e-4;
    std::vector<Eigen::Triplet<T>> entry_K;
    buildSystemMatrix(entry_K, dq);
    Eigen::SparseMatrix<T> A(n_nodes * dof, n_nodes * dof);
    A.setFromTriplets(entry_K.begin(), entry_K.end());      

    for(int n_node = 0; n_node < n_nodes; n_node++)
    {
        for (int d = 0; d < dof; ++d)
        {
            dq(d, n_node) += epsilon;
            DOFStack g0(dof, n_nodes), g1(dof, n_nodes);
            g0.setZero(); g1.setZero();
            computeResidual(g0, dq);
            dq(d, n_node) -= 2.0 * epsilon;
            computeResidual(g1, dq);
            dq(d, n_node) += epsilon;
            DOFStack row_FD = (g1 - g0) / (2 * epsilon);
            for(int i = 0; i < n_nodes; i++)
            {
                std::cout << n_node << " " << row_FD(d, i) << " " << A.coeff(i * dof + d, n_node * dof + d) << std::endl;
                std::getchar();
            }
        }
        
    }

}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;