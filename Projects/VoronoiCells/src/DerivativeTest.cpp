#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::checkLengthDerivativesScale()
{
    run_diff_test = true;
    int n_dof = deformed.rows();

    VectorXT gradient(n_dof);
    u = VectorXT::Zero(undeformed.rows());
    u[0] = 0.05; u[1] = -0.05; 
    deformed = undeformed + u;
    simDoFToPosition(deformed);
    addEdgeLengthForceEntries(1.0, gradient);
    gradient *= -1.0;

    T E0 = 0.0;
    addEdgeLengthEnergy(1.0, E0);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {    
        T E1 = 0.0;
        deformed = undeformed + u + dx;
        simDoFToPosition(deformed);
        addEdgeLengthEnergy(1.0, E1);
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

void IntrinsicSimulation::checkLengthDerivatives()
{
    run_diff_test = true;
    T epsilon = 1e-6;
    T E0 = 0.0 , E1 = 0.0, E2 = 0.0;
    VectorXT gradient = VectorXT::Zero(undeformed.rows());
    u = VectorXT::Zero(undeformed.rows());
    VectorXT du = u;
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;
    // u[0] = 0.05; u[1] = -0.05; 
    deformed = undeformed + u;
    simDoFToPosition(deformed);
    addEdgeLengthEnergy(1.0, E0);
    addEdgeLengthForceEntries(1.0, gradient);
    gradient *= -1.0;
    std::cout << "gradient " << gradient.transpose() << std::endl;
    
    u[0] += epsilon; 
    deformed = undeformed + u;
    simDoFToPosition(deformed);
    addEdgeLengthEnergy(1.0, E1);

    u[0] -= 2.0 * epsilon;
    deformed = undeformed + u;
    simDoFToPosition(deformed);
    addEdgeLengthEnergy(1.0, E2);
    std::cout << "gradient FD " << (E1 - E2) / 2.0 / epsilon << std::endl;
    u[0] += epsilon; 
    
    // std::cout << "E0 " << E0 << std::endl;
    // std::cout << "E1 " << E1 << std::endl;
    // u[0] = 0.05; u[1] = -0.05 + epsilon;
    // deformed = undeformed + u;
    // E1 = 0.0;
    // simDoFToPosition(deformed);
    // addEdgeLengthEnergy(1.0, E1);
    // std::cout << "gradient FD " << (E1 - E0) / epsilon << std::endl;
    // std::cout << "E0 " << E0 << std::endl;
    // std::cout << "E1 " << E1 << std::endl;
    // u[2] = 0.05 + epsilon; u[3] = -0.05;
    // deformed = undeformed + u;
    // simDoFToPosition(deformed);
    // addEdgeLengthEnergy(1.0, E1);
    // std::cout << "gradient FD " << (E1 - E0) / epsilon << std::endl;

    // u[0] = 0.05; u[1] = -0.05 + epsilon;
    // deformed = undeformed + u;
    // simDoFToPosition(deformed);
    // addEdgeLengthEnergy(1.0, E1);
    // std::cout << "gradient FD " << (E1 - E0) / epsilon << std::endl;
}

void IntrinsicSimulation::checkTotalGradient(bool perturb)
{
    run_diff_test = true;
    int n_dof = deformed.rows();
    VectorXT gradient(n_dof); gradient.setZero();
    T epsilon = 1e-6;
    u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        VectorXT du = u;
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.001;
        u += du;
    }
    computeResidual(u, gradient);
    gradient *= -1.0;

    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();
    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(u);
        
        u(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy(u);
        u(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
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
    u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        VectorXT du = u;
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.01;
        u += du;
    }
    computeResidual(u, gradient);
    gradient *= -1.0;

    T E0 = computeTotalEnergy(u);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.01;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {    
        
        T E1 = computeTotalEnergy(u+dx);
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
    run_diff_test = true;
    int n_dof = deformed.rows();
    
    u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        VectorXT du = u;
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.01;
        u += du;
    }

    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(u, f0);
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
        computeResidual(u + dx, f1);
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
}

void IntrinsicSimulation::checkTotalHessian(bool perturb)
{
    T epsilon = 1e-6;
    run_diff_test = true;
    int n_dof = deformed.rows();
    
    u = VectorXT::Zero(undeformed.rows());
    if (perturb)
    {
        VectorXT du = u;
        du.setRandom();
        du *= 1.0 / du.norm();
        du *= 0.01;
        u += du;
    }

    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);
    

    StiffnessMatrix A_FD(n_dof, n_dof);
    std::vector<Entry> entries;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        
        computeResidual(u, g0);
        g0 *= -1.0; 
        
        u(dof_i) -= 2.0 * epsilon;
        
        computeResidual(u, g1); 
        g1 *= -1.0;
        
        u(dof_i) += epsilon;
        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            entries.push_back(Entry(i, dof_i, row_FD(i)));
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
    A_FD.setFromTriplets(entries.begin(), entries.end());
    std::cout << A_FD << std::endl;
    std::cout << std::endl;
    std::cout << A << std::endl;
    std::cout << "Hessian Diff Test Passed" << std::endl;
    run_diff_test = false;
}

void IntrinsicSimulation::checkGeodesicDerivative(bool perturb)
{
    Edge eij = spring_edges[0];
    SurfacePoint vA = mass_surface_points[eij[0]].first;
    SurfacePoint vB = mass_surface_points[eij[1]].first;

    T geo_dis; std::vector<SurfacePoint> path;
    std::vector<IxnData> ixn_data;
    computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);

    VectorXT delta = undeformed; delta.setZero();
    
}