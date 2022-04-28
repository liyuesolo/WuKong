#include <ipc/ipc.hpp>
#include "../include/VertexModel2D.h"

void VertexModel2D::buildIPCRestData()
{
    ipc_edges.resize(edges.size(), 2);
    ipc_vertices.resize(num_nodes, 2);
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices.row(i) = deformed.segment<2>(i * 2);
    }
    for (int i = 0; i < edges.size(); i++)
    {
        ipc_edges.row(i) = edges[i];
    }
}

void VertexModel2D::addIPCEnergy(T& energy)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices.row(i) = deformed.segment<2>(i * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);
    
    T energy_ipc = ipc_barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);

    energy += energy_ipc;    
}


void VertexModel2D::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices.row(i) = deformed.segment<2>(i * 2);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);
    
    VectorXT contact_gradient = ipc_barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);
    
    residual -= contact_gradient;
}

void VertexModel2D::addIPCHessianEntries(std::vector<Entry>& entries)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
     for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices.row(i) = deformed.segment<2>(i * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = ipc_barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance, false);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

void VertexModel2D::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        ipc_vertices.row(i) = deformed.segment<2>(i * 2);
    });
}

T VertexModel2D::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    
    Eigen::MatrixXd current_position(num_nodes, 2), 
        next_step_position(num_nodes, 2);
        
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        current_position.row(i) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2);
        next_step_position.row(i) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2) + du.segment<2>(i * 2);
    });

    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}