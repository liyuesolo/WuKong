#include "../include/FEMSolver.h"

template <int dim>
void FEMSolver<dim>::addSlidingSpringEnergy(T& energy)
{
    T spring_energy = 0.0;

    Eigen::VectorXd p1;
    p1 = deformed.segment<dim>(dim*(2*(sliding_res+1)));
    Eigen::VectorXd p2 = p1;
    p2(0) = r2*cos(theta2);
    p2(1) = -r2*sin(theta2);
    spring_energy += 0.5* sliding_stiffness * (p1-p2).squaredNorm();

    energy += spring_energy;
}

template <int dim>
void FEMSolver<dim>::addSlidingSpringForceEntries(VectorXT& residual)
{
    int num_nodes_all = num_nodes;
    Eigen::VectorXd L2_gradient(num_nodes_all*dim);

    int index = 2*(sliding_res+1);
    Eigen::VectorXd p1;
    p1 = deformed.segment<dim>(dim*(index));
    Eigen::VectorXd p2 = p1;
    p2(0) = r2*cos(theta2);
    p2(1) = -r2*sin(theta2);

    L2_gradient.segment<dim>(index*dim) = sliding_stiffness * (p1-p2);
        
    
    residual.segment(0, num_nodes_all * dim) += -L2_gradient.segment(0, num_nodes_all * dim);
}

template <int dim>
void FEMSolver<dim>::addSlidingSpringHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    int num_nodes_all = num_nodes;
    StiffnessMatrix L2_hessian(dim*num_nodes_all,dim*num_nodes_all);
    std::vector<Eigen::Triplet<double>> L2_hessian_triplets;
    int index = 2*(sliding_res+1);

    for(int j=0; j<dim;++j)
    {
        L2_hessian_triplets.emplace_back(dim * index + j, dim * index + j, sliding_stiffness);
    }
    L2_hessian.setFromTriplets(L2_hessian_triplets.begin(), L2_hessian_triplets.end());
    //std::cout<<L2_hessian<<std::endl;

    std::vector<Entry> L2_entries = entriesFromSparseMatrix(L2_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), L2_entries.begin(), L2_entries.end());
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringEnergy2(T& energy)
{
    T spring_energy = 0.0;
    for(int i=0; i<spring_indices.size(); ++i)
    {
        Eigen::VectorXd p1 = deformed.segment<dim>(dim*spring_indices[i]);
        Eigen::VectorXd p2 = spring_ends[i];
        spring_energy += 0.5* virtual_spring_stiffness * (p1-p2).squaredNorm();
    }
    energy += spring_energy;
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringForceEntries2(VectorXT& residual)
{
    int num_nodes_all = num_nodes;
    Eigen::VectorXd L2_gradient(num_nodes_all*dim);
    L2_gradient.setZero();

   for(int i=0; i<spring_indices.size(); ++i)
    {
        Eigen::VectorXd p1 = deformed.segment<dim>(dim*spring_indices[i]);
        Eigen::VectorXd p2 = spring_ends[i];
        L2_gradient.segment<dim>(dim*spring_indices[i]) = virtual_spring_stiffness * (p1-p2);
        //std::cout<<virtual_spring_stiffness<<" "<<p1(0)<<" "<<p2(0)<<std::endl;
        std::cout<<"Spring Force: "<<virtual_spring_stiffness * (p1-p2)(0)<<std::endl;
    }   
    
    residual.segment(0, num_nodes_all * dim) += -L2_gradient.segment(0, num_nodes_all * dim);
}

template <int dim>
void FEMSolver<dim>::addVirtualSpringHessianEntries2(std::vector<Entry>& entries,bool project_PD)
{
    int num_nodes_all = num_nodes;
    StiffnessMatrix L2_hessian(dim*num_nodes_all,dim*num_nodes_all);
    std::vector<Eigen::Triplet<double>> L2_hessian_triplets;
    L2_hessian_triplets.clear();

    for(int i=0; i<spring_indices.size(); ++i)
    {
        //std::cout<<i<<" "<<spring_indices[i]<<std::endl;
        Eigen::VectorXd p1 = deformed.segment<dim>(dim*spring_indices[i]);
        Eigen::VectorXd p2 = spring_ends[i];
        for(int j=0; j<dim; ++j)
            L2_hessian_triplets.emplace_back(dim*spring_indices[i]  + j, dim*spring_indices[i] + j, virtual_spring_stiffness);
    }  
    L2_hessian.setFromTriplets(L2_hessian_triplets.begin(), L2_hessian_triplets.end());
    //std::cout<<L2_hessian<<std::endl;

    std::vector<Entry> L2_entries = entriesFromSparseMatrix(L2_hessian.block(0, 0, num_nodes_all * dim , num_nodes_all * dim));
    //std::cout<<contact_hessian<<std::endl;
    
    entries.insert(entries.end(), L2_entries.begin(), L2_entries.end());
}

template class FEMSolver<2>;
template class FEMSolver<3>;