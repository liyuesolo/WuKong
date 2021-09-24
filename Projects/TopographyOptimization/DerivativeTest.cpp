#include "FEMSolver.h"

template<class T, int dim>
void FEMSolver<T, dim>::derivativeTest()
{
    VectorXT du(num_nodes * dim);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    u += du;
    checkTotalGradient();
    checkTotalHessian();
}

template<class T, int dim>
void FEMSolver<T, dim>::checkTotalGradient()
{
    run_diff_test = true;
    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    int n_dof = num_nodes * dim;
    T epsilon = 1e-7;
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(u, gradient);
    
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(u);
        
        u(dof_i) -= 1.0 * epsilon;
        T E1 = computeTotalEnergy(u);
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (1*epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    run_diff_test = false;

}

template<class T, int dim>
void FEMSolver<T, dim>::checkTotalHessian()
{
    std::cout << "======================== CHECK HESSIAN ========================" << std::endl;
    run_diff_test = true;
    T epsilon = 1e-7;
    int n_dof = num_nodes * dim;
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);

    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        computeResidual(u, g0);

        u(dof_i) -= 1.0 * epsilon;
        computeResidual(u, g1);
            
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
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    run_diff_test = false;
}

// template class FEMSolver<float, 2>;
// template class FEMSolver<float, 3>;
// template class FEMSolver<double, 2>;
template class FEMSolver<double, 3>;