#include <ipc/ipc.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/FEMSolver.h"

// #include <igl/writeOBJ.h>

void FEMSolver::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<2>(i * 2);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
        
    T current_min_dis = ipc::compute_minimum_distance(ipc_vertices, ipc_edges, ipc_faces, ipc_constraints);
    if (first_step)
        ipc_min_dis = current_min_dis;
    else
    {
        TV min_corner, max_corner;
        computeBoundingBox(min_corner, max_corner);
        T bb_diag = (max_corner - min_corner).norm();
        ipc::update_barrier_stiffness(ipc_min_dis, current_min_dis, max_barrier_weight, barrier_weight, bb_diag);
        ipc_min_dis = current_min_dis;
    }
}

void FEMSolver::computeIPCRestData()
{
    
    ipc_vertices.resize(num_nodes, 2);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
    num_ipc_vtx = ipc_vertices.rows();
    
    std::vector<Edge> edges;
    ipc_faces.resize(num_ele, 3);
    for (int i = 0; i < num_ele; i++)
    {
        ipc_faces.row(i) = indices.segment<3>(i * 3);
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
    
    for (int i = 0; i < ipc_edges.rows(); i++)
    {
        Edge edge = ipc_edges.row(i);
        TV vi = ipc_vertices.row(edge[0]), vj = ipc_vertices.row(edge[1]);
        if ((vi - vj).norm() < barrier_distance)
            std::cout << "edge " << edge.transpose() << " has length < " << barrier_distance << std::endl;
    }
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;

    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    VectorXT dedx(num_nodes * 2), dbdx(num_nodes * 2);
    dedx.setZero(); dbdx.setZero();
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
    
}

T FEMSolver::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{

    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;
        
    for (int i = 0; i < num_nodes; i++)
    {
        current_position.row(i) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2);
        // current_position.row(i) = undeformed.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2) + du.segment<2>(i * 2);
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
        ipc_vertices.row(i) = deformed.segment<2>(i * dim);
}

void FEMSolver::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(i * 2);
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
void FEMSolver::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(i * 2);
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
void FEMSolver::addIPCHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(i * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, project_PD);

    // std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes * dim , num_nodes * dim));
    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    
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

