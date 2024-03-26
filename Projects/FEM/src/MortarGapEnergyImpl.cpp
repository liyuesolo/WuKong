#include "../include/FEMSolver.h"

template <typename DerivedLocalGrad, typename DerivedGrad>
void local_gradient_to_global_gradient(
    const Eigen::MatrixBase<DerivedLocalGrad>& local_grad,
    const std::vector<long>& ids,
    int dim,
    Eigen::PlainObjectBase<DerivedGrad>& grad)
{
    for (int i = 0; i < ids.size(); i++) {
        grad.segment(dim * ids[i], dim);
        local_grad.segment(dim * i, dim);
        grad.segment(dim * ids[i], dim) += local_grad.segment(dim * i, dim);
    }

}

template <typename Derived>
void local_hessian_to_global_triplets(
    const Eigen::MatrixBase<Derived>& local_hessian,
    const std::vector<long>& ids,
    int dim,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    for (int i = 0; i < ids.size(); i++) {
        for (int j = 0; j < ids.size(); j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    triplets.emplace_back(
                        dim * ids[i] + k, dim * ids[j] + l,
                        local_hessian(dim * i + k, dim * j + l));
                }
            }
        }
    }
}

template <int dim>
void FEMSolver<dim>::initializeMortarInformation()
{
    int num_slave_node = slave_nodes.size();
    int num_master_node = master_nodes.size();

    Eigen::MatrixXd V((num_slave_node+num_master_node),2);
    for(int i=0; i<num_slave_node; ++i)
    {
        V(i,0) = deformed(2*slave_nodes[i]);
        V(i,1) = deformed(2*slave_nodes[i]+1);
    }
    for(int i=0; i<num_master_node; ++i)
    {
        V(i+num_slave_node,0) = deformed(2*master_nodes[i]);
        V(i+num_slave_node,1) = deformed(2*master_nodes[i]+1);
    }

    //std::cout<<V<<std::endl;

    // for(int i=0; i<num_slave_node; ++i)
    // {
    //     std::cout<<mortar.normals[2*i].return_value<<" "<<mortar.normals[2*i+1].return_value<<std::endl;
    // }

    std::unordered_map<int,std::pair<int,int>> segments;
    std::vector<int> slave_indices;
    std::vector<int> master_indices;
    for(int i=0; i<slave_segments.size(); ++i)
    {
        segments[i] = std::pair<int,int>(i+1,i);
        slave_indices.push_back(i);
    }
    for(int i=0; i<master_segments.size(); ++i)
    {
        segments[i+slave_segments.size()] = std::pair<int,int>(i+num_slave_node,i+num_slave_node+1);
        master_indices.push_back(i+slave_segments.size());
    }

    mortar.initialization(V,slave_indices,master_indices,segments);
}

template <int dim>
void FEMSolver<dim>::updateMortarInformation()
{
    int num_slave_node = slave_nodes.size();
    int num_master_node = master_nodes.size();

    Eigen::MatrixXd V((num_slave_node+num_master_node),2);
    for(int i=0; i<num_slave_node; ++i)
    {
        V(i,0) = deformed(2*slave_nodes[i]);
        V(i,1) = deformed(2*slave_nodes[i]+1);
    }
    for(int i=0; i<num_master_node; ++i)
    {
        V(i+num_slave_node,0) = deformed(2*master_nodes[i]);
        V(i+num_slave_node,1) = deformed(2*master_nodes[i]+1);
    }
    // std::cout<<V<<std::endl;
    mortar.updateVertices(V);
}


template <int dim>
void FEMSolver<dim>::addMortarEnergy(T& energy)
{
    updateMortarInformation();
    mortar.MortarMethod();
    
    energy += mortar.gap_energy.return_value;
}

template <int dim>
void FEMSolver<dim>::addMortarForceEntries(VectorXT& residual)
{
    updateMortarInformation();
    mortar.MortarMethod(true);
    Eigen::VectorXd local_grad = mortar.gap_energy.return_value_grad;
    std::vector<long> vertices;
    for(int i=0; i<slave_nodes.size(); ++i)
    {
        vertices.push_back(slave_nodes[i]);
    }
    for(int i=0; i<master_nodes.size(); ++i)
    {
        vertices.push_back(master_nodes[i]);
    }

    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_nodes*dim);
    local_gradient_to_global_gradient(local_grad,vertices,dim,grad);

    residual.segment(0, num_nodes * dim) += -grad.segment(0, num_nodes * dim);
}

template <int dim>
void FEMSolver<dim>::addMortarHessianEntries(std::vector<Entry>& entries, bool project_PD)
{
    updateMortarInformation();
    mortar.MortarMethod(true);

    Eigen::MatrixXd local_hess = mortar.gap_energy.return_value_hess;
    std::vector<long> vertices;
    for(int i=0; i<slave_nodes.size(); ++i)
    {
        vertices.push_back(slave_nodes[i]);
    }
    for(int i=0; i<master_nodes.size(); ++i)
    {
        vertices.push_back(master_nodes[i]);
    }

    Eigen::SparseMatrix<double> hess(num_nodes*dim, num_nodes*dim);
    std::vector<Eigen::Triplet<double>> local_hess_triplets;

    local_hessian_to_global_triplets(local_hess, vertices, dim, local_hess_triplets);
    hess.setFromTriplets(local_hess_triplets.begin(), local_hess_triplets.end());

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(hess.block(0, 0, num_nodes * dim , num_nodes * dim));
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template class FEMSolver<2>;
template class FEMSolver<3>;