#include <ipc/ipc.hpp>
#include "../include/VertexModel.h"

void VertexModel::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    try
    {
        contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    }
    catch(const std::runtime_error& e)
    {
        std::cout << "error catch " << std::endl;
        std::cerr << e.what() << '\n';
    }
    
    energy += contact_energy;
    
}
void VertexModel::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    VectorXT contact_gradient;

    try
    {
        contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << e.what() << '\n';
    }

    residual.segment(0, basal_vtx_start * 3) += -contact_gradient;
}

void VertexModel::addIPCHessianEntries(std::vector<Entry>& entries, 
    bool projectPD)
{
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, projectPD);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}