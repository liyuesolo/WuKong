#include <ipc/ipc.hpp>
#include <igl/edges.h>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/FEMSolver.h"

#include <igl/writeOBJ.h>


template <int dim>
void FEMSolver<dim>::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
        
    T current_min_dis = ipc::compute_minimum_distance(ipc_vertices, ipc_edges, ipc_faces, ipc_constraints);
    if (first_step)
        ipc_min_dis = current_min_dis;
    else
    {
        T bb_diag = (max_corner - min_corner).norm();
        ipc::update_barrier_stiffness(ipc_min_dis, current_min_dis, max_barrier_weight, barrier_weight, bb_diag);
        ipc_min_dis = current_min_dis;
    }
}

template <int dim>
void FEMSolver<dim>::computeIPCRestData()
{
    
    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    num_ipc_vtx = ipc_vertices.rows();
    
    
    ipc_faces.resize(num_surface_faces, 3);
    for (int i = 0; i < num_surface_faces; i++)
    {
        ipc_faces.row(i) = surface_indices.segment<3>(i * 3);
    }
    igl::edges(ipc_faces, ipc_edges);

    for (int i = 0; i < ipc_edges.rows(); i++)
    {
        Edge edge = ipc_edges.row(i);
        TV vi = ipc_vertices.row(edge[0]), vj = ipc_vertices.row(edge[1]);
        if ((vi - vj).norm() < barrier_distance)
            std::cout << "edge " << edge.transpose() << " has length < 1e-6 " << std::endl;
    }
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;
    
    
    T bb_diag = (max_corner - min_corner).norm();
    VectorXT dedx(num_nodes * dim), dbdx(num_nodes * dim);
    dedx.setZero(); dbdx.setZero();
    barrier_weight = 1.0;
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    if (verbose)
        std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
}
template <int dim>
T FEMSolver<dim>::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{

    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;
        
    for (int i = 0; i < num_nodes; i++)
    {
        current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
        // current_position.row(i) = undeformed.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    }
    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}
template <int dim>
void FEMSolver<dim>::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = deformed.segment<3>(i * dim);
}
template <int dim>
void FEMSolver<dim>::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
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

    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        T friction_energy = ipc::compute_friction_potential<T>(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        energy += friction_energy;
    }
}

template <int dim>
void FEMSolver<dim>::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    // std::cout << "contact force norm: " << contact_gradient.norm() << std::endl;
    residual.segment(0, num_nodes * dim) += -contact_gradient.segment(0, num_nodes * dim);

    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        VectorXT friction_energy_gradient = ipc::compute_friction_potential_gradient(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        residual.segment(0, num_nodes * dim) += -friction_energy_gradient;
    }
}

template <int dim>
void FEMSolver<dim>::addIPCHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, project_PD);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

    if (add_friction)
    {
        ipc::FrictionConstraints ipc_friction_constraints;
        ipc::construct_friction_constraint_set(
            ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints,
            barrier_distance, barrier_weight, friction_mu, ipc_friction_constraints
        );
        StiffnessMatrix friction_energy_hessian = ipc::compute_friction_potential_hessian(
            ipc_vertices, ipc_vertices_deformed, ipc_edges,
            ipc_faces, ipc_friction_constraints, epsv_times_h, project_PD
        );
        std::vector<Entry> friction_entries = entriesFromSparseMatrix(friction_energy_hessian.block(0, 0, num_nodes * dim , num_nodes * dim));
        entries.insert(entries.end(), friction_entries.begin(), friction_entries.end());
    }
}

// template class FEMSolver<2>;
template class FEMSolver<3>;