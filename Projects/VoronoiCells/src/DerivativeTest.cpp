#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::checkTotalGradient(bool perturb)
{
    run_diff_test = true;
    int n_dof = deformed.rows();
    VectorXT gradient(n_dof); gradient.setZero();
    T epsilon = 1e-6;
    delta_u = VectorXT::Zero(undeformed.rows());
    
    updateCurrentState();
    traceGeodesics();
    computeResidual(gradient);
    gradient *= -1.0;

    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        delta_u(dof_i) += epsilon;
        updateCurrentState();
        T E0 = computeTotalEnergy();
        
        delta_u(dof_i) -= 2.0 * epsilon;
        updateCurrentState();
        T E1 = computeTotalEnergy();
        delta_u(dof_i) += epsilon;
        updateCurrentState();
        std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E0 - E1) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(dof_i) - gradient(dof_i)) < 1e-3 * std::abs(gradient(dof_i)))
        //     continue;
        std::cout << " dof " << dof_i << " FD " << gradient_FD(dof_i) << " symbolic " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    run_diff_test = false;
}

void IntrinsicSimulation::checkTotalGradientScale(bool perturb)
{
    run_diff_test = true;
    int n_dof = deformed.rows();

    VectorXT gradient(n_dof); gradient.setZero();
    delta_u = VectorXT::Zero(n_dof);
    if (perturb)
    {
        // delta_u.setRandom();
        // delta_u *= 1.0 / delta_u.norm();
        // delta_u *= 0.01;
    }
    VectorXT delta_u_backup = delta_u;
    updateCurrentState();
    traceGeodesics();

    computeResidual(gradient);
    gradient *= -1.0;

    T E0 = computeTotalEnergy();
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {    
        delta_u = delta_u_backup + dx;
        updateCurrentState();
        // traceGeodesics();
        T E1 = computeTotalEnergy();
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
    run_diff_test = false;
}

void IntrinsicSimulation::checkTotalHessianScale(bool perturb)
{
    // run_diff_test = true;
    // int n_dof = deformed.rows();
    
    // delta_u = VectorXT::Zero(undeformed.rows());
    // if (perturb)
    // {
    //     delta_u.setRandom();
    //     delta_u *= 1.0 / delta_u.norm();
    //     delta_u *= 0.01;
    // }

    // StiffnessMatrix A(n_dof, n_dof);
    // buildSystemMatrix(A);

    // VectorXT f0(n_dof);
    // f0.setZero();
    // computeResidual(u, f0);
    // f0 *= -1;
    
    // VectorXT dx(n_dof);
    // dx.setRandom();
    // dx *= 1.0 / dx.norm();
    
    // dx *= 0.1;
    // T previous = 0.0;
    // for (int i = 0; i < 10; i++)
    // {
        
    //     VectorXT f1(n_dof);
    //     f1.setZero();
    //     computeResidual(u + dx, f1);
    //     f1 *= -1;
    //     T df_norm = (f0 + (A * dx) - f1).norm();
    //     std::cout << "df_norm " << df_norm << std::endl;
    //     if (i > 0)
    //     {
    //         std::cout << (previous/df_norm) << std::endl;
    //     }
    //     previous = df_norm;
    //     dx *= 0.5;
    // }
    // run_diff_test = false;
}

void IntrinsicSimulation::checkTotalHessian(bool perturb)
{
    T epsilon = 1e-6;
    run_diff_test = true;
    int n_dof = deformed.rows();
    
    delta_u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        delta_u.setRandom();
        delta_u *= 1.0 / delta_u.norm();
        delta_u *= 0.01;
    }
    updateCurrentState();
    traceGeodesics();
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(A);
    
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        delta_u(dof_i) += epsilon;
        updateCurrentState();
        traceGeodesics();
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        
        computeResidual(g0);
        g0 *= -1.0; 
        
        delta_u(dof_i) -= 2.0 * epsilon;
        updateCurrentState();
        traceGeodesics();
        computeResidual(g1); 
        g1 *= -1.0;
        
        delta_u(dof_i) += epsilon;
        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(i, dof_i) == 0 && row_FD(i) == 0)
                continue;
            // if (std::abs( A.coeff(i, dof_i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
            //     continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Hessian Diff Test Passed" << std::endl;
    run_diff_test = false;
}
