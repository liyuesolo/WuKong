#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <ipc/ipc.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/FEMSolver.h"
#include <unordered_set>
// #include <igl/writeOBJ.h>

void FEMSolver::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);

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
    if (use_quadratic_triangle)
    {
        std::unordered_set<int> surface_index_map;
        for (int i = 0; i < surface_indices.rows(); i++)
        {
            surface_index_map.insert(surface_indices[i]);
        }
        coarse_to_fine.resize(surface_index_map.size());
        fine_to_coarse.clear();
        int cnt = 0;
        for (const auto& value: surface_index_map)
        {
            coarse_to_fine[cnt] = value;
            fine_to_coarse[value] = cnt;
            cnt++;
        }

        num_ipc_vtx = surface_index_map.size();
        ipc_vertices.resize(num_ipc_vtx, 2);
        for (int i = 0; i < num_ipc_vtx; i++)
        {
            ipc_vertices.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2);
        }
    }
    else
    {
        ipc_vertices.resize(num_nodes, 2);
        for (int i = 0; i < num_nodes; i++)
            ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
        num_ipc_vtx = ipc_vertices.rows();
        for (int i = 0; i < num_ipc_vtx; i++)
        {
            coarse_to_fine[i] = i;
            fine_to_coarse[i] = i;
        }
        
    }
    
    std::vector<Edge> edges;
    ipc_faces.resize(num_ele, 3);

    for (int i = 0; i < num_ele; i++)
    {
        if (use_quadratic_triangle)
        {
            ipc_faces.row(i) = IV3(fine_to_coarse[surface_indices[i * 3 + 0]],
                                    fine_to_coarse[surface_indices[i * 3 + 1]],
                                    fine_to_coarse[surface_indices[i * 3 + 2]]);
        }
        else
            ipc_faces.row(i) = indices.segment<3>(i * 3);
    }
    igl::edges(ipc_faces, ipc_edges);
    // std::vector<std::vector<int>> boundary_vertices;
    // igl::boundary_loop(ipc_faces, boundary_vertices);

    // int n_bd_edge = 0;
    // for (auto loop : boundary_vertices)
    // {
    //     n_bd_edge += loop.size();
    // }
    
    // ipc_edges.resize(n_bd_edge, 2);
    // int edge_cnt = 0;
    // for (auto loop : boundary_vertices)
    // {
    //     for (int i = 0; i < loop.size(); i++)
    //         ipc_edges.row(edge_cnt++) = Edge(loop[i], loop[(i+1)%loop.size()]);
    // }
    ipc_faces.resize(0, 0);
    
    
    // for (int i = 0; i < ipc_edges.rows(); i++)
    // {
    //     Edge edge = ipc_edges.row(i);
    //     TV vi = ipc_vertices.row(edge[0]), vj = ipc_vertices.row(edge[1]);
    //     if (verbose)
    //     {
    //         if ((vi - vj).norm() < barrier_distance)
    //             std::cout << "edge " << edge.transpose() << " has length < " << barrier_distance << std::endl;
    //     }
    // }
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;
    
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    
    VectorXT dedx(num_nodes * 2), dbdx(num_nodes * 2);
    dedx.setZero(); dbdx.setZero();
    barrier_weight = 1.0;
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    // std::cout << dbdx.norm() << std::endl;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    // std::cout << dedx.norm() << std::endl;
    
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    if (verbose)
        std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
    
}

T FEMSolver::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{

    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;
        
    for (int i = 0; i < num_ipc_vtx; i++)
    {
        current_position.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2) + _u.segment<2>(coarse_to_fine[i] * 2);
        next_step_position.row(i) = undeformed.segment<2>(coarse_to_fine[i] * 2) + _u.segment<2>(coarse_to_fine[i] * 2) + du.segment<2>(coarse_to_fine[i] * 2);
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
    for (int i = 0; i < num_ipc_vtx; i++)
        ipc_vertices.row(i) = deformed.segment<2>(coarse_to_fine[i] * dim);
}

void FEMSolver::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
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
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);
    // std::cout << ipc_constraints.num_constraints() << std::endl;
    // for (int i = 0; i < ipc_constraints.vv_constraints.size(); i++)
    // {
    //     std::vector<long> idx = ipc_constraints.vv_constraints[i].vertex_indices(ipc_edges, ipc_faces);
    //     {
    //         for (int j = 0; j < idx.size(); j++)
    //         {
    //             std::cout << idx[j] << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << ipc_constraints.vv_constraints.size() << std::endl;
    // std::cout << ipc_constraints.ev_constraints.size() << std::endl;
    // std::cout << ipc_constraints.ee_constraints.size() << std::endl;
    // std::cout << ipc_constraints.fv_constraints.size() << std::endl;
    // std::cout << ipc_constraints.pv_constraints.size() << std::endl;
    // std::exit(0);

    VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    for (int i = 0; i < num_ipc_vtx; i++)
    {
        residual.segment<2>(coarse_to_fine[i] * 2) -= contact_gradient.segment<2>(i * 2);
    }
    

}
void FEMSolver::addIPCHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<2>(coarse_to_fine[i] * 2);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, project_PD);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    for (Entry& entry : contact_entries)
    {
        int node_i = std::floor(entry.row() / 2);
        int node_j = std::floor(entry.col() / 2);
        entries.push_back(Entry(coarse_to_fine[node_i] * 2 + entry.row() % 2, 
                            coarse_to_fine[node_j] * 2 + entry.col() % 2, 
                            entry.value()));
    }
    
}

