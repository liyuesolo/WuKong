#include <ipc/ipc.hpp>
#include "../include/VertexModel2D.h"

bool fake_3d = false;

void VertexModel2D::buildIPCRestData()
{
    if (fake_3d)
    {
        ipc_edges.resize(edges.size() + n_cells, 2);
        // ipc_edges.resize(edges.size(), 2);
        ipc_vertices.resize(num_nodes, 3);
        ipc_faces.resize(n_cells *  2, 3);
        ipc_vertices.setZero();

        for (int i = 0; i < num_nodes; i++)
        {
            ipc_vertices.row(i).segment<2>(0) = undeformed.segment<2>(i * 2);
        }
        for (int i = 0; i < edges.size(); i++)
        {
            ipc_edges.row(i) = edges[i];
        }

        iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            ipc_faces.row(cell_idx * 2 + 0) = IV3(indices[0], indices[1], indices[2]);
            ipc_faces.row(cell_idx * 2 + 1) = IV3(indices[0], indices[2], indices[3]);
            ipc_edges.row(edges.size() + cell_idx) = Edge(indices[0], indices[2]);
        });
    }   
    else
    {
        ipc_edges.resize(edges.size(), 2);
        ipc_vertices.resize(num_nodes, 2);
        for (int i = 0; i < num_nodes; i++)
        {
            ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
        }
        for (int i = 0; i < edges.size(); i++)
        {
            ipc_edges.row(i) = edges[i];
        }

        // ipc_faces.resize(n_cells *  2, 3);
        // iterateCellSerial([&](VtxList& indices, int cell_idx)
        // {
        //     ipc_faces.row(cell_idx * 2 + 0) = IV3(indices[0], indices[1], indices[2]);
        //     ipc_faces.row(cell_idx * 2 + 1) = IV3(indices[0], indices[2], indices[3]);
        //     // ipc_edges.row(edges.size() + cell_idx) = Edge(indices[0], indices[2]);
        // });
    }
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;
}

void VertexModel2D::addIPCEnergy(T& energy)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);
    
    T energy_ipc = ipc_barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);

    if (add_ipc_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            ipc_barrier_distance, ipc_barrier_weight, 0.5, ipc_friction_constraints
        );
        T friction_energy = ipc::compute_friction_potential<T>(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, 1e-5
        );
        energy += friction_energy;
    }

    energy += energy_ipc;    
}


void VertexModel2D::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);

    VectorXT contact_gradient = ipc_barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);

    if (add_ipc_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            ipc_barrier_distance, ipc_barrier_weight, 0.5, ipc_friction_constraints
        );
        VectorXT friction_energy_gradient = ipc::compute_friction_potential_gradient(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, 1e-5
        );
        std::cout << "friction force " << friction_energy_gradient.norm() << std::endl;
        residual += -friction_energy_gradient;
    }
    
    int dim = fake_3d ? 3 : 2;
    for (int i = 0; i < num_nodes; i++)
    {
        residual.segment<2>(i * 2) -= contact_gradient.segment<2>(i * dim);
    }
    
    // residual -= contact_gradient;
}

void VertexModel2D::addIPCHessianEntries(std::vector<Entry>& entries)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
     for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = ipc_barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance, false);
    

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    if (fake_3d)
    {
        for (auto entry : contact_entries)
        {
            if (entry.col() % 3 != 2 && entry.row() % 3 != 2)
                entries.push_back(entry);
        }
    }
    else 
        entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

    if (add_ipc_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            ipc_barrier_distance, ipc_barrier_weight, 0.5, ipc_friction_constraints
        );
        StiffnessMatrix friction_energy_hessian = ipc::compute_friction_potential_hessian(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, 1e-5
        );
        std::vector<Entry> friction_entries = entriesFromSparseMatrix(friction_energy_hessian);
        entries.insert(entries.end(), friction_entries.begin(), friction_entries.end());
    }
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
    Eigen::MatrixXd current_position, next_step_position;

    if (fake_3d)
    {
        current_position.resize(num_nodes, 3);
        next_step_position.resize(num_nodes, 3);
    }
    else
    {
        current_position.resize(num_nodes, 2);
        next_step_position.resize(num_nodes, 2);
    }   
    current_position.setZero(); next_step_position.setZero();

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        current_position.row(i).segment<2>(0) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2);
        next_step_position.row(i).segment<2>(0) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2) + du.segment<2>(i * 2);
    });

    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}