#include "EoLRodSim.h"


template<class T, int dim>
void EoLRodSim<T, dim>::runDerivativeTest()
{
    // test which term
    // print_force_mag = true;
    add_regularizor = false;
    add_stretching=false;
    add_penalty =false;
    add_bending = false;
    add_shearing = true;
    add_pbc = false;
    add_eularian_reg = false;

    DOFStack dq(dof, n_nodes);
    dq.setZero();
    if (true || add_pbc)
    {
        // q(0, 3) += 0.1;
        // q(2, 3) += 0.1;
        // q(3, 1) += 0.1;
        // q(1, 4) += 0.1;
        // q(2, 2) += 0.1;
        // q(1, 3) += 0.1;
        // q(1, 0) += 0.1;

        // q(1, 0) += 0.1;
        // q(0, 0) += 0.1;
        // q(0, 1) -= 0.1;
        // q(1, 1) -= 0.1;
        // q(0, 15) += 0.1;
        // q(1, 8) += 0.1;
        // // q(1, 14) -= 0.1;
        // q(0, 9) += 0.1;
        // q(1, 9) += 0.1;
        
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
        q(1, 14) -= 0.1;
        q(0, 9) += 0.1;
        q(1, 9) += 0.1;
        q(0, 21) += 0.1;
        q(1, 6) += 0.1;
        q(1, 18) += 0.1;
        q(1, 15) -= 0.1;
    }

    checkGradient(dq);
    checkHessian(dq);
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradientSecondOrderTerm(Eigen::Ref<DOFStack> dq)
{
    std::cout << "===================== checkGradient =====================" << std::endl;
    DOFStack gradient(dof, n_nodes);
    gradient.setZero();
    computeResidual(gradient, dq);
    gradient *= -1;
    const auto& gd = Eigen::Map<const VectorXT>(gradient.data(), gradient.size());
    T E0 = computeTotalEnergy(dq);
    DOFStack dx(dof, n_nodes);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        const auto& delta_x = Eigen::Map<const VectorXT>(dx.data(), dx.size());
        T E1 = computeTotalEnergy(dq + dx);
        T dE = E1 - E0;
        
        dE -= gd.dot(delta_x);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessianHigherOrderTerm(Eigen::Ref<DOFStack> dq)
{
    std::cout << "===================== check Hessian =====================" << std::endl;
    std::vector<Eigen::Triplet<T>> entry_K;
    buildSystemMatrix(entry_K, dq);
    StiffnessMatrix A(n_nodes * dof, n_nodes * dof);
    A.setFromTriplets(entry_K.begin(), entry_K.end());
    
    DOFStack gradient(dof, n_nodes);
    gradient.setZero();
    computeResidual(gradient, dq);
    gradient *= -1;
    const auto& f0 = Eigen::Map<const VectorXT>(gradient.data(), gradient.size());
    
    DOFStack dx(dof, n_nodes);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        const auto& delta_x = Eigen::Map<const VectorXT>(dx.data(), dx.size());
        DOFStack g1(dof, n_nodes);
        g1.setZero();
        computeResidual(g1, dq + dx);
        g1 *= -1;
        const auto& f1 = Eigen::Map<const VectorXT>(g1.data(), g1.size());
        
        T df_norm = (f0 + (A * delta_x) - f1).norm();
        std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradient(Eigen::Ref<DOFStack> dq)
{
    checkGradientSecondOrderTerm(dq);
    return;
    add_regularizor = false;
    add_stretching=false;
    add_penalty =false;
    add_bending = false;
    add_shearing = true;
    add_pbc = false;
    add_eularian_reg = false;
    T epsilon = 1e-6;
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
            // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
            gradient_FD(d, n_node) = (E1 - E0) / (2*epsilon);
            if( gradient_FD(d, n_node) == 0 && gradient(d, n_node) == 0)
                continue;
            // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
            //     continue;
            std::cout << n_node << " dof " << d << " " << gradient_FD(d, n_node) << " " << gradient(d, n_node) << std::endl;
            std::getchar();
            cnt++;
        }   
    }
    // if(!cnt)
        // std::cout << "Gradient all correct" << std::endl;
    
    add_stretching=true;
    add_bending = true;
    add_shearing = true;
    add_pbc = true;
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessian(Eigen::Ref<DOFStack> dq)
{
    checkHessianHigherOrderTerm(dq);
    // return;
    
    T epsilon = 1e-8;
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
                // if (std::abs( A.coeff(n_node * dof + d, i * dof + d) - row_FD(d, i)) < 1e-4)
                    // continue;
                std::cout << "node i: " << n_node << " node j: " << i << " dof: " << d << " " << row_FD(d, i) << " " << A.coeff(n_node * dof + d, i * dof + d) << std::endl;
                std::getchar();
            }
        }
        
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;