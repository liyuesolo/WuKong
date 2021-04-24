#include "EoLRodSim.h"


template<class T, int dim>
void EoLRodSim<T, dim>::runDerivativeTest()
{
    // test which term
    add_regularizor = false;
    add_stretching=false;
    add_penalty =false;
    add_bending = true;
    add_shearing = false;
    add_pbc = false;
    add_eularian_reg = false;

    DOFStack dq(dof, n_nodes);
    dq.setZero();
    if (add_pbc)
    {
        q(1, 0) += 0.1;
        q(0, 0) += 0.1;
        q(0, 1) -= 0.1;
        q(0, 1) -= 0.1;
        q(0, 15) += 0.1;
        q(1, 8) += 0.1;
        // q(1, 14) -= 0.1;
        q(0, 9) += 0.1;
        q(1, 9) += 0.1;
        // q(0, 10) += 0.1;
        // q(0, 6) += 0.1;
        // q(1, 6) -= 0.1;
        
    }
    else
    {
        q(0, 3) += 0.1;
        q(2, 3) += 0.1;
        q(3, 1) += 0.1;
        q(1, 4) += 0.1;
        q(2, 2) += 0.1;
        q(1, 3) += 0.1;
        q(1, 0) += 0.1;

        q(1, 0) += 0.1;
        q(0, 0) += 0.1;
        q(0, 1) -= 0.1;
        q(1, 1) -= 0.1;
        q(0, 15) += 0.1;
        q(1, 8) += 0.1;
        // q(1, 14) -= 0.1;
        q(0, 9) += 0.1;
        q(1, 9) += 0.1;
    }

    checkGradient(dq);
    checkHessian(dq);
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradient(Eigen::Ref<DOFStack> dq)
{
    T epsilon = 1e-5;
    DOFStack gradient(dof, n_nodes);
    gradient.setZero();
    std::cout << "===================== checkGradient =====================" << std::endl;
    std::cout << "current state vector delta " << (dq).transpose() << std::endl;
    computeResidual(gradient, dq);
    DOFStack gradient_FD(dof, n_nodes);
    gradient_FD.setZero();
    int cnt = 0;
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
            if( gradient_FD(d, n_node) == 0 && gradient(d, n_node) == 0)
                continue;
            std::cout << n_node << " " << gradient_FD(d, n_node) << " " << gradient(d, n_node) << std::endl;
            std::getchar();
            cnt++;
        }   
    }
    // if(!cnt)
        // std::cout << "Gradient all correct" << std::endl;
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessian(Eigen::Ref<DOFStack> dq)
{
    T epsilon = 1e-7;
    std::vector<Eigen::Triplet<T>> entry_K;
    buildSystemMatrix(entry_K, dq);
    Eigen::SparseMatrix<T> A(n_nodes * dof, n_nodes * dof);
    A.setFromTriplets(entry_K.begin(), entry_K.end());      
    // std::cout << A << std::endl;
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
                if(A.coeff(n_node * dof + d, i * dof + d) == 0 && row_FD(d, i) == 0)
                    continue;
                std::cout << "node i: " << n_node << " node j: " << i << " dof: " << d << " " << row_FD(d, i) << " " << A.coeff(n_node * dof + d, i * dof + d) << std::endl;
                std::getchar();
            }
        }
        
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;