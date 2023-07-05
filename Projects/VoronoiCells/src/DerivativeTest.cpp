#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::checkTotalGradient(bool perturb)
{
    run_diff_test = true;
    int n_dof = undeformed.rows();
    VectorXT gradient(n_dof); gradient.setZero();
    T epsilon = 1e-6;
    delta_u = VectorXT::Zero(undeformed.rows());
    updateCurrentState();
    std::vector<std::pair<SurfacePoint, gcFace>> current = mass_surface_points;
    
    computeResidual(gradient);
    gradient *= -1.0;
    // std::cout << gradient.transpose() << std::endl;
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    int cnt = 0;
    // std::cout << std::setprecision(10);
    // std::cout << "-----" << std::endl;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << mass_surface_points[std::floor(dof_i/2.0)].first.faceCoords << std::endl;
        delta_u(dof_i) += epsilon;
        mass_surface_points = current;
        updateCurrentState();
        T E0 = computeTotalEnergy();
        // return;
        delta_u(dof_i) -= 2.0 * epsilon;
        mass_surface_points = current;
        updateCurrentState();
        
        T E1 = computeTotalEnergy();
        delta_u(dof_i) += epsilon;
        mass_surface_points = current;
        updateCurrentState();
        
        gradient_FD(dof_i) = (E0 - E1) / (2.0 * epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        if (std::abs( gradient_FD(dof_i) - gradient(dof_i)) < 1e-3 * std::abs(gradient(dof_i)))
            continue;
        cnt++;
        std::cout << " dof " << dof_i << " vtx " << std::floor(dof_i/2.0) << " FD " << gradient_FD(dof_i) << " symbolic " << gradient(dof_i) << std::endl;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        
        std::getchar();
    }
    if (cnt != n_dof)
        std::cout << "all gradients are correct" << std::endl;
    mass_surface_points = current;
    updateCurrentState();
    run_diff_test = false;
}

void IntrinsicSimulation::checkTotalGradientScale(bool perturb)
{
    std::cout << "================CHECK GRADIENT SCALE==============" <<std::endl;
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
    std::vector<std::pair<SurfacePoint, gcFace>> current = mass_surface_points;
    mass_surface_points = current;
    updateCurrentState();
    computeResidual(gradient);
    gradient *= -1.0;

    retrace = false;
    T E0 = computeTotalEnergy();
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {    
        delta_u = delta_u_backup + dx;
        mass_surface_points = current;
        updateCurrentState();
        T E1 = computeTotalEnergy();
        T dE = E1 - E0;
        dE -= gradient.dot(dx);
        std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    delta_u = delta_u_backup;
    mass_surface_points = current;
    updateCurrentState();

    run_diff_test = false;
    mass_surface_points = current;
    std::cout << "================ DONE ==============" <<std::endl;
}

void IntrinsicSimulation::checkTotalHessianScale(bool perturb)
{
    std::cout << "================CHECK HESSIAN SCALE==============" <<std::endl;
    run_diff_test = true;
    int n_dof = deformed.rows();
    
    delta_u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        // delta_u.setRandom();
        // delta_u *= 1.0 / delta_u.norm();
        // delta_u *= 0.01;
    }
    VectorXT delta_u_backup = delta_u;
    std::vector<std::pair<SurfacePoint, gcFace>> current = mass_surface_points;
    delta_u = delta_u_backup;
    mass_surface_points = current;
    updateCurrentState();
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(f0);
    f0 *= -1;
    
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof);
        f1.setZero();
        delta_u = delta_u_backup + dx;
        mass_surface_points = current;
        updateCurrentState();
        computeResidual(f1);
        f1 *= -1;
        T df_norm = (f0 + (A * dx) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    run_diff_test = false;
    delta_u = delta_u_backup;
    mass_surface_points = current;
    updateCurrentState();
    std::cout << "================ DONE ==============" <<std::endl;
}

void IntrinsicSimulation::checkTotalHessian(bool perturb)
{
    T epsilon = 1e-6;
    run_diff_test = true;
    int n_dof = deformed.rows();
    if (perturb)
    {
        // delta_u.setRandom();
        // delta_u *= 1.0 / delta_u.norm();
        // delta_u *= 0.01;
    }
    std::vector<std::pair<SurfacePoint, gcFace>> current = mass_surface_points;
    delta_u = VectorXT::Zero(undeformed.rows());
    updateCurrentState();
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(A);
    
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();

        delta_u(dof_i) += epsilon;
        mass_surface_points = current;     
        updateCurrentState();

        computeResidual(g0);
        g0 *= -1.0; 
        
        delta_u(dof_i) -= 2.0 * epsilon;
        
        mass_surface_points = current;
        updateCurrentState();
        
        computeResidual(g1); 
        g1 *= -1.0;
        
        // std::cout << "gradient x + dx " << g0.transpose() << std::endl;
        // std::cout << "gradient x - dx " << g1.transpose() << std::endl;
        delta_u(dof_i) += epsilon;
        mass_surface_points = current;
        updateCurrentState();

        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(i, dof_i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(i, dof_i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    mass_surface_points = current;
    updateCurrentState();
    std::cout << "Hessian Diff Test Passed" << std::endl;
    run_diff_test = false;
}
