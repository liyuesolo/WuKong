#include <ipc/ipc.hpp>
#include "../include/FEMSolver.h"

void FEMSolver::computeIPCRestData()
{
    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    
    std::vector<Edge> edges;
    ipc_faces.resize(num_surface_faces, 3);
    for (int i = 0; i < num_surface_faces; i++)
    {
        ipc_faces.row(i) = surface_indices.segment<3>(i * 3);
        for (int j = 0; j < 3; j++)
        {
            int k = (j + 1) % 3;
            Edge ei(ipc_faces(i, j), ipc_faces(i, k));
            auto find_iter = std::find_if(edges.begin(), edges.end(), 
                [&ei](const Edge e)->bool {return (ei[0] == e[0] && ei[1] == e[1] ) 
                    || (ei[0] == e[1] && ei[1] == e[0]); });
            if (find_iter == edges.end())
            {
                edges.push_back(ei);
            }
        }
    }
    ipc_edges.resize(edges.size(), 2);
    for (int i = 0; i < edges.size(); i++)
        ipc_edges.row(i) = edges[i];    
    
}

T FEMSolver::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    
    Eigen::MatrixXd current_position(num_nodes, 3), 
        next_step_position(num_nodes, 3);
        
    for (int i = 0; i < num_nodes; i++)
    {
        // current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
        current_position.row(i) = undeformed.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    }
    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}

void FEMSolver::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = deformed.segment<3>(i * 3);
}
void FEMSolver::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, 
    ipc_edges, ipc_faces, ipc_constraints, barrier_distance);

    energy += contact_energy;
}
void FEMSolver::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);

    residual.segment(0, num_nodes * 3) += -contact_gradient;

}
void FEMSolver::addIPCHessianEntries(std::vector<Entry>& entries)
{
    Eigen::MatrixXd ipc_vertices_deformed(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, false);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}