#include "EoLRodSim.h"


template<class T, int dim>
void EoLRodSim<T, dim>::runDerivativeTest()
{
    // test which term
    // print_force_mag = true;
    run_diff_test = true;
    add_regularizor = false;
    add_stretching=false;
    add_penalty =false;
    add_bending = false;
    add_shearing = false;
    add_pbc = true;
    add_eularian_reg = false;

    DOFStack dq(dof, n_dof);
    dq.setZero();
    if (true )
    {
        // dq(0, 3) += 0.01;
        // dq(2, 3) += 0.01;
        // dq(3, 1) += 0.01;
        // dq(1, 4) += 0.01;
        // dq(2, 2) += 0.01;
        // dq(1, 3) += 0.01;
        // dq(1, 0) += 0.01;

        // dq(1, 0) += 0.01;
        // dq(0, 0) += 0.01;
        // dq(0, 1) -= 0.01;
        // dq(1, 1) -= 0.01;
        // dq(0, 15) += 0.01;
        // dq(1, 8) += 0.01;
        // // q(1, 14) -= 0.1;
        // dq(0, 9) += 0.01;
        // dq(1, 9) += 0.01;
        
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

    checkGradient(Eigen::Map<VectorXT>(dq.data(), dq.size()));
    checkHessian(Eigen::Map<VectorXT>(dq.data(), dq.size()));
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradientSecondOrderTerm(Eigen::Ref<VectorXT> dq)
{
    std::cout << "===================== checkGradient =====================" << std::endl;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;
    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(gradient, dq, lambdas, kappa);
    
    gradient *= -1;
    T E0 = computeTotalEnergy(dq, lambdas, kappa);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        T E1 = computeTotalEnergy(dq + dx, lambdas, kappa);
        T dE = E1 - E0;
        
        dE -= gradient.dot(dx);
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
void EoLRodSim<T, dim>::checkHessianHigherOrderTerm(Eigen::Ref<VectorXT> dq)
{
    std::cout << "===================== check Hessian =====================" << std::endl;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;

    StiffnessMatrix A;
    buildSystemMatrix(dq, A, kappa);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(f0, dq, lambdas, kappa);
    f0 *= -1;
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof);
        f1.setZero();
        computeResidual(f1, dq + dx, lambdas, kappa);
        f1 *= -1;
        T df_norm = (f0 + (A * dx) - f1).norm();
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
void EoLRodSim<T, dim>::checkGradient(Eigen::Ref<VectorXT> dq)
{
    checkGradientSecondOrderTerm(dq);
    return;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();
    std::cout << "===================== checkGradient =====================" << std::endl;
    // std::cout << "current state vector delta " << (dq).transpose() << std::endl;
    computeResidual(gradient, dq, lambdas, kappa);
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        T E0 = computeTotalEnergy(dq, lambdas, kappa);
        dq(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy(dq, lambdas, kappa);
        dq(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (2*epsilon);
        // if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            // continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    // if(!cnt)
        // std::cout << "Gradient all correct" << std::endl;
    
    // add_stretching=true;
    // add_bending = true;
    // add_shearing = true;
    // add_pbc = true;
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessian(Eigen::Ref<VectorXT> dq)
{
    // checkHessianHigherOrderTerm(dq);
    // return;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;
    T epsilon = 1e-2;
    StiffnessMatrix A;
    buildSystemMatrix(dq, A, kappa);

    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        
        {
            dq(dof_i) += epsilon;
            VectorXT g0(n_dof), g1(n_dof);
            g0.setZero(); g1.setZero();
            computeResidual(g0, dq, lambdas, kappa);
            dq(dof_i) -= 2.0 * epsilon;
            computeResidual(g1, dq, lambdas, kappa);
            dq(dof_i) += epsilon;
            VectorXT row_FD = (g1 - g0) / (2 * epsilon);
            for(int i = 0; i < n_dof; i++)
            {
                if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                    continue;
                // if (std::abs( A.coeff(n_node * dof + d, i * dof + d) - row_FD(d, i)) < 1e-4)
                    // continue;
                std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
                    << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
                    << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
                std::getchar();
            }
        }
        
    }
}

template class EoLRodSim<double, 3>;
template class EoLRodSim<double, 2>;