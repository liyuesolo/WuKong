#include "../include/FEMSolver.h"
template <int dim>
void FEMSolver<dim>::checkTotalGradient(bool perturb)
{
    run_diff_test = true;
    
    std::cout << "======================== CHECK GRADIENT 2nd Scale ========================" << std::endl;
    T epsilon = 1e-8;

    int num_nodes_all = num_nodes;
    if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION)num_nodes_all += additional_dof;
    VectorXT du(num_nodes_all * dim);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;
    else u.setZero();

    int n_dof = num_nodes_all * dim;

    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(u, gradient, true);
    
    gradient *= -1;
    T E0 = computeTotalEnergy(u);
    VectorXT dx(n_dof);
    dx.setZero();
    for (int i = 0; i < n_dof; i++)
    {
        dx(i) = epsilon;
        T E1 = computeTotalEnergy(u + dx);
        T E2 = computeTotalEnergy(u - dx);
        T dE = E1 - E2;
        
        std::cout<<i/dim<<" "<<i%dim << " analytical " << gradient(i) <<" numerical "<<dE/(2.*epsilon)<< std::endl;
        dx(i) = 0;
    }
    run_diff_test = false;
}


template <int dim>
void FEMSolver<dim>::checkTotalHessian(bool perturb)
{
    run_diff_test = true;
    
    std::cout << "======================== CHECK HESSIAN 2nd Scale ========================" << std::endl;
    T epsilon = 1e-7;

    int num_nodes_all = num_nodes;
    if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION)num_nodes_all += additional_dof;
    VectorXT du(num_nodes_all * dim);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.001;
    if (perturb)
        u += du;

    int n_dof = num_nodes_all * dim;

    StiffnessMatrix A(n_dof, n_dof);
    
    buildSystemMatrix(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(u, f0);
    f0 *= -1;

    VectorXT dx(n_dof);
    dx.setZero();
    
    for (int i = 0; i < n_dof; i++)
    {
        dx(i) = epsilon;
        
        VectorXT f1(n_dof);
        f1.setZero();
        computeResidual(u + dx, f1, true);
        f1 *= -1;

        for(int j=0; j<n_dof; j++)
        {
            if(A.coeff(i,j) !=0 && (A.coeff(i,j)-(f1-f0)(j)/epsilon)/A.coeff(i,j) > 1e-3)
                std::cout<<i/dim<<" "<<i%dim<<" "<<j/dim<<" "<<j%dim << " analytical " << A.coeff(i,j) <<" numerical "<<(f1-f0)(j)/epsilon<< std::endl;
        }
        dx(i) = 0;
    }
    run_diff_test = false;
}


template <int dim>
void FEMSolver<dim>::checkTotalGradientScale(bool perturb)
{
    
    run_diff_test = true;
    
    std::cout << "======================== CHECK GRADIENT 2nd Scale ========================" << std::endl;
    T epsilon = 1e-8;

    int num_nodes_all = num_nodes;
    if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION)num_nodes_all += additional_dof;
    VectorXT du(num_nodes_all * dim);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.00001;
    if (perturb)
        u += du;
    
    int n_dof = num_nodes_all * dim;

    VectorXT gradient(n_dof);
    gradient.setZero();
    computeResidual(u, gradient, true);
    
    gradient *= -1;
    T E0 = computeTotalEnergy(u);
    VectorXT dx(n_dof);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        T E1 = computeTotalEnergy(u + dx);
        T dE = E1 - E0;
        //std::cout<<dx.transpose()<<std::endl;
        std::cout << "E0 " << E0 <<" E1 "<<E1<< std::endl;
        std::cout << "dE " << dE <<" dE' "<<gradient.dot(dx)<< std::endl;

        // double sum = 0;
        // std::cout<<gradient.size()<<" "<<dx.size()<<std::endl;
        // for(int i=0; i<n_dof; ++i)
        // {
        //     sum+=gradient(i)*dx(i);
        //     std::cout<<gradient(i)<<" "<<dx(i)<<" "<<sum<<std::endl;
        // }
        // std::cout<<sum<<std::endl;

        // std::cout<<gradient.transpose()<<std::endl;
        // std::cout<<"-------------------------------"<<std::endl;
        // std::cout<<dx.transpose()<<std::endl;
        // std::cout<<"-------------------------------"<<std::endl;

        dE -= gradient.dot(dx);
        if (i > 0)
        {
            std::cout << (previous/dE) << std::endl;
        }
        previous = dE;
        dx *= 0.5;
    }
    run_diff_test = false;
}

template <int dim>
void FEMSolver<dim>::checkTotalHessianScale(bool perturb)
{
    std::cout << "===================== check Hessian 2nd Scale =====================" << std::endl;
    run_diff_test = true;
    

    int num_nodes_all = num_nodes;
    if(CALCULATE_IMLS_PROJECTION) num_nodes_all += projectedPts.rows();
    if(USE_NEW_FORMULATION)num_nodes_all += additional_dof;
    VectorXT du(num_nodes_all * dim);
    du.setRandom();
    du *= 1.0 / du.norm();
    du *= 0.00001;
    if (perturb)
        u += du;
    
    int n_dof = num_nodes_all * dim;
    StiffnessMatrix A(n_dof, n_dof);
    
    buildSystemMatrix(u, A);

    VectorXT f0(n_dof);
    f0.setZero();
    computeResidual(u, f0);
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
        computeResidual(u + dx, f1, true);
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
    run_diff_test = false;
}

template class FEMSolver<2>;
template class FEMSolver<3>;