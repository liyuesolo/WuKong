#include "EoLRodSim.h"


template<class T, int dim>
void EoLRodSim<T, dim>::derivativeTest()
{
    run_diff_test = true;
    add_regularizor = false;
    add_stretching = false;
    add_penalty =false;
    add_bending = false;
    add_shearing = false;
    add_pbc = false;
    add_contact_penalty = true;
    add_eularian_reg = true;
    deformed_states /= unit;
    VectorXT dq(W.cols());
    // VectorXT dq(W.rows());
    dq.setRandom();
    dq *= 1.0 / dq.norm();
    for (int i = 0; i < dq.rows(); i++)
        dq(i) = dq(i) * 0.5 + 0.5;
    dq *= 0.01;
    dq(2) += 1.0;
    dq(3) += 1.0;
    testGradient(dq);
    testHessian(dq);
}

template<class T, int dim>
void EoLRodSim<T, dim>::runDerivativeTest()
{
    // test which term
    // print_force_mag = true;
    run_diff_test = true;
    add_regularizor = false;
    add_stretching = false;
    add_penalty =false;
    add_bending = true;
    add_shearing = false;
    add_pbc = false;
    add_contact_penalty = false;
    add_eularian_reg = true;
    

    DOFStack dq(dof, n_dof);
    dq.setZero();
    // std::cout << tunnel_R << std::endl;
    if (true )
    {
        dq(0, 3) += 0.01;
        dq(2, 3) += 0.01;
        dq(2, 1) += 0.01;
        dq(1, 4) += 0.01;
        dq(2, 4) += 0.01;
        dq(2, 2) += 0.01;
        dq(1, 3) += 0.01;
        
        // dq(1, 0) += 0.01;
        // dq(0, 0) += 0.01;
        dq(0, 1) -= 0.01;
        dq(1, 1) -= 0.01;
        dq(0, 6) += 0.01;
        dq(1, 8) += 0.01;
        // q(1, 14) -= 0.1;
        dq(0, 9) += 0.01;
        dq(1, 9) += 0.01;
        // dq(2, 20) += 0.1;
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
    dq *= 0.03;
    checkGradient(Eigen::Map<VectorXT>(dq.data(), dq.size()));
    checkHessian(Eigen::Map<VectorXT>(dq.data(), dq.size()));
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkMaterialPositionDerivatives()
{
    T epsilon = 1e-6;
    if (new_frame_work)
    {
        for (auto& rod : Rods)
        {
            rod->iterateSegmentsWithOffset([&](int node_i, int node_j, Offset offset_i, Offset offset_j)
            {
                // std::cout << "node i " << node_i << " node j " << node_j << std::endl;
                TV X1, X0, dX0;
                rod->XdX(node_i, X0, dX0);
                deformed_states[offset_i[dim]] += epsilon;
                rod->X(node_i, X1);
                deformed_states[offset_i[dim]] -= epsilon;
                for (int d = 0; d < dim; d++)
                {
                    std::cout << "dXdu: " << (X1[d] - X0[d]) / epsilon << " " << dX0[d] << std::endl;
                }
                // std::getchar();
            });
        }
    }
    else
    {
        int yarn_type = 0;
        for (int i = 0; i < n_nodes; i++)
        {
            std::vector<int> nodes = { i };
            std::vector<TV> X0, X1; std::vector<TV> dXdu, dXdu1; std::vector<TV> d2Xdu2, dummy2;
            
            getMaterialPositions(q, nodes, X0, yarn_type, dXdu, d2Xdu2, true, true);

            TV x0 = X0[0];
            q(dim, i) += epsilon;
            getMaterialPositions(q, nodes, X1, yarn_type, dXdu1, dummy2, true, false);
            TV x1 = X1[0];

            for (int d = 0; d < dim; d++)
            {
                std::cout << "dXdu: " << (x1[d] - x0[d]) / epsilon << " " << dXdu[0][d] << std::endl;
                std::cout << "d2Xdu2 " << (dXdu1[0][d] - dXdu[0][d]) / epsilon << " " << d2Xdu2[0][d] << std::endl;
                std::getchar();
            }
            q(dim, i) -= epsilon;
        }
    }
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
    for(int i = 0; i < n_dof; i++) dx[i] += 0.5;
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
void EoLRodSim<T, dim>::testGradient(Eigen::Ref<VectorXT> dq)
{
    T epsilon = 1e-6;
    n_dof = W.cols();
    // n_dof = W.rows();
    
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeGradient(gradient, dq);
    // std::cout << gradient << std::endl;
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    
    // dq.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(dq);
        
        dq(dof_i) -= 1.0 * epsilon;
        T E1 = computeTotalEnergy(dq);
        std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (1*epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkGradient(Eigen::Ref<VectorXT> dq)
{
    checkGradientSecondOrderTerm(dq);
    // return;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();
    std::cout << "===================== checkGradient =====================" << std::endl;
    // std::cout << "current state vector delta " << (dq) << std::endl;
    computeResidual(gradient, dq, lambdas, kappa);
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        T E0 = computeTotalEnergy(dq, lambdas, kappa);
        // dq(dof_i) -= 2.0 * epsilon;
        // T E1 = computeTotalEnergy(dq, lambdas, kappa);
        // dq(dof_i) += epsilon;

        dq(dof_i) -= 1.0 * epsilon;
        T E1 = computeTotalEnergy(dq, lambdas, kappa);
        
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        // gradient_FD(dof_i) = (E1 - E0) / (2*epsilon);
        gradient_FD(dof_i) = (E1 - E0) / (1*epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
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
void EoLRodSim<T, dim>::testHessian(Eigen::Ref<VectorXT> dq)
{
    T epsilon = 1e-6;
    StiffnessMatrix A;
    buildSystemDoFMatrix(dq, A);
    n_dof = W.cols();
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        dq(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        computeGradient(g0, dq);

        dq(dof_i) -= 1.0 * epsilon;
        computeGradient(g1, dq);
            
        VectorXT row_FD = (g1 - g0) / (epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            // if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-4)
            //     continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout <<" FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
}

template<class T, int dim>
void EoLRodSim<T, dim>::checkHessian(Eigen::Ref<VectorXT> dq)
{
    checkHessianHigherOrderTerm(dq);
    // return;
    DOFStack lambdas(dof, n_pb_cons);
    lambdas.setOnes();
    T kappa = 1.5;
    T epsilon = 1e-5;
    StiffnessMatrix A;
    buildSystemMatrix(dq, A, kappa);

    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        {
            dq(dof_i) += epsilon;
            VectorXT g0(n_dof), g1(n_dof);
            g0.setZero(); g1.setZero();
            computeResidual(g0, dq, lambdas, kappa);
            // dq(dof_i) -= 2.0 * epsilon;
            // computeResidual(g1, dq, lambdas, kappa);
            // dq(dof_i) += epsilon;
            // VectorXT row_FD = (g1 - g0) / (2 * epsilon);

            dq(dof_i) -= 1.0 * epsilon;
            computeResidual(g1, dq, lambdas, kappa);
            
            VectorXT row_FD = (g1 - g0) / (epsilon);

            for(int i = 0; i < n_dof; i++)
            {
                if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                    continue;
                // if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-4)
                //     continue;
                // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
                //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
                //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
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