#include <ipc/ipc.hpp>
#include <igl/boundary_loop.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/edges.h>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/FEMSolver.h"

#include <igl/writeOBJ.h>

template <int dim>
void FEMSolver<dim>::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<dim>(i * dim);

    ipc::CollisionConstraints ipc_constraints;
    //ipc_constraints.set_use_convergent_formulation(true);
    ipc_constraints.build(collisionMesh, ipc_vertices_deformed, barrier_distance);
        
    T current_min_dis = ipc_constraints.compute_minimum_distance(collisionMesh, ipc_vertices_deformed);
    if (first_step)
    {
        ipc_min_dis = current_min_dis;
        
    }
    else
    {
        computeBoundingBox();
        T bb_diag = (max_corner - min_corner).norm();
        ipc::update_barrier_stiffness(ipc_min_dis, current_min_dis, max_barrier_weight, barrier_weight, bb_diag);
        ipc_min_dis = current_min_dis;
    }
}


template <int dim>
void FEMSolver<dim>::computeIPC3DRestData()
{
    assert(dim == 3);
    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    num_ipc_vtx = ipc_vertices.rows();
    
    std::vector<Edge> edges;
    ipc_faces.resize(num_surface_faces, 3);
    for (int i = 0; i < num_surface_faces; i++)
    {
        ipc_faces.row(i) = surface_indices.segment<3>(i * 3);
        // for (int j = 0; j < 3; j++)
        // {
        //     int k = (j + 1) % 3;
        //     Edge ei(ipc_faces(i, j), ipc_faces(i, k));
        //     auto find_iter = std::find_if(edges.begin(), edges.end(), 
        //         [&ei](const Edge e)->bool {return (ei[0] == e[0] && ei[1] == e[1] ) 
        //             || (ei[0] == e[1] && ei[1] == e[0]); });
        //     if (find_iter == edges.end())
        //     {
        //         edges.push_back(ei);
        //     }
        // }
    }
    // ipc_edges.resize(edges.size(), 2);
    // for (int i = 0; i < edges.size(); i++)
    //     ipc_edges.row(i) = edges[i];    
    igl::edges(ipc_faces,ipc_edges);

    std::vector<bool> is_vertex(num_nodes,false);
    for(int i=0; i<num_nodes; ++i)
    {
        if(is_surface_vertex[i] != 0)  is_vertex[i] = true;
    }
    

    // for (int i = 0; i < ipc_edges.rows(); i++)
    // {
    //     Edge edge = ipc_edges.row(i);
    //     TV vi = ipc_vertices.row(edge[0]), vj = ipc_vertices.row(edge[1]);
    //     if ((vi - vj).norm() < barrier_distance)
    //         std::cout << "edge " << edge.transpose() << " has length < 1e-6 " << std::endl;
    // }
    std::cout<<"IPC faces: "<<ipc_faces.rows()<<" IPC vertices: "<<ipc_vertices.rows()<<std::endl;
    Eigen::MatrixXd vs;
    Eigen::MatrixXi fs;
    Eigen::VectorXi svi;
    Eigen::VectorXi svj;
    igl::remove_duplicate_vertices(ipc_vertices, ipc_faces,0, vs,svi, svj, fs);
    std::cout<<"IPC faces: "<<fs.rows()<<" IPC vertices: "<<vs.rows()<<std::endl;

    igl::writeOBJ("IPC_Mesh.obj",ipc_vertices,ipc_faces);

    computeBoundingBox();
    T bb_diag = (max_corner - min_corner).norm();
    VectorXT dedx(num_nodes * 2), dbdx(num_nodes * 2);
    dedx.setZero(); dbdx.setZero();
    barrier_weight = 1.0;
    addIPC3DForceEntries(dbdx); dbdx *= -1.0;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;

    collisionMesh = ipc::CollisionMesh(is_vertex,ipc_vertices,ipc_edges,ipc_faces);
}

template <int dim>
T FEMSolver<dim>::computeCollisionFreeStepsize3D(const VectorXT& _u, const VectorXT& du)
{
    assert(dim == 3);
    Eigen::MatrixXd current_position = ipc_vertices, 
        next_step_position = ipc_vertices;

    //std::cout<<"gradient: "<<du.segment<24>(24).transpose()<<std::endl;
        
    for (int i = 0; i < num_nodes; i++)
    {
        current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<3>(i * 3)+ _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    }

    return ipc::compute_collision_free_stepsize(collisionMesh, current_position, 
            next_step_position, ipc::BroadPhaseMethod::HASH_GRID, 0, 1e-6, 1e7);
}


template <int dim>
void FEMSolver<dim>::addIPC3DEnergy(T& energy)
{
    assert(dim == 3);
    T contact_energy = 0.0;
    
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::CollisionConstraints ipc_constraints;
    //ipc_constraints.set_use_convergent_formulation(true);
    ipc_constraints.build(collisionMesh, ipc_vertices_deformed, barrier_distance);

    contact_energy = barrier_weight * ipc_constraints.compute_potential(collisionMesh, ipc_vertices_deformed, barrier_distance);
    energy += contact_energy;

    if(USE_FRICTION)
    {
        ipc::FrictionConstraints friction_constraints;
        friction_constraints.build(collisionMesh, ipc_vertices_deformed, ipc_constraints, barrier_distance, barrier_weight, friction_mu);
        double friction_energy = friction_constraints.compute_potential(collisionMesh, (x_prev-deformed)/h, epsv_times_h);
        energy += friction_energy;
    }

    
}

template <int dim>
void FEMSolver<dim>::addIPC3DForceEntries(VectorXT& residual)
{
    assert(dim == 3);
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::CollisionConstraints ipc_constraints;
    //ipc_constraints.set_use_convergent_formulation(true);
    ipc_constraints.build(collisionMesh, ipc_vertices_deformed, barrier_distance);

    VectorXT contact_gradient = barrier_weight * ipc_constraints.compute_potential_gradient(collisionMesh, ipc_vertices_deformed, barrier_distance);
    // std::cout << "contact force norm: " << contact_gradient.norm() << std::endl;
    residual.segment(0, num_nodes * dim) += -contact_gradient.segment(0, num_nodes * dim);

    if(USE_FRICTION)
    {
        ipc::FrictionConstraints friction_constraints;
        friction_constraints.build(collisionMesh, ipc_vertices_deformed, ipc_constraints, barrier_distance, barrier_weight, friction_mu);
        Eigen::VectorXd friction_potential_grad = friction_constraints.compute_potential_gradient(collisionMesh, (x_prev-deformed)/h, epsv_times_h);
        residual.segment(0, num_nodes * dim) += -friction_potential_grad.segment(0, num_nodes * dim);
    }

}


template <int dim>
void FEMSolver<dim>::addIPC3DHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    assert(dim == 3);
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::CollisionConstraints ipc_constraints;
    //ipc_constraints.set_use_convergent_formulation(true);
    ipc_constraints.build(collisionMesh, ipc_vertices_deformed, barrier_distance);

    StiffnessMatrix contact_hessian = barrier_weight * ipc_constraints.compute_potential_hessian(collisionMesh, ipc_vertices_deformed, barrier_distance, project_PD);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes * dim , num_nodes * dim));
    
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

    if(USE_FRICTION)
    {
        ipc::FrictionConstraints friction_constraints;
        friction_constraints.build(collisionMesh, ipc_vertices_deformed, ipc_constraints, barrier_distance, barrier_weight, friction_mu);
        Eigen::SparseMatrix<double> friction_potential_hess = friction_constraints.compute_potential_hessian(collisionMesh, (x_prev-deformed)/h, epsv_times_h);
        std::vector<Entry> friction_entries = entriesFromSparseMatrix(contact_hessian.block(0, 0, num_nodes * dim , num_nodes * dim));
        entries.insert(entries.end(), friction_entries.begin(), friction_entries.end());
    }
}

template <int dim>
void FEMSolver<dim>::computeIPC2DtrueRestData()
{
    // ipc_vertices.resize(num_nodes, 2);
    // for (int i = 0; i < num_nodes; i++)
    //     ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
    // num_ipc_vtx = ipc_vertices.rows();

    // ipc_faces.resize(num_ele, 3);
    
    // for (int i = 0; i < num_ele; i++)
    // {
    //     ipc_faces.row(i) = indices.segment<3>(i * 3);
    // }
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
    // ipc_faces.resize(0, 0);

    // // std::cout<<ipc_vertices<<std::endl;
    // bool self_intersect = ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces);
    // std::cout<<"IPC self_intersect: "<<self_intersect<<std::endl;

    // computeBoundingBox();
    // T bb_diag = (max_corner - min_corner).norm();
    // VectorXT dedx(num_nodes * 2), dbdx(num_nodes * 2);
    // dedx.setZero(); dbdx.setZero();
    // barrier_weight = 1.0;
    // addIPC2DtrueForceEntries(dbdx); dbdx *= -1.0;
    // computeResidual(u, dedx); dedx *= 6665-1.0; dedx -= dbdx;
    // barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    // std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
}

template <int dim>
T FEMSolver<dim>::computeCollisionFreeStepsize2Dtrue(const VectorXT& _u, const VectorXT& du)
{
    // Eigen::MatrixXd current_position, next_step_position;

    // current_position.resize(num_nodes, 2);
    // next_step_position.resize(num_nodes, 2);

    // current_position.setZero(); next_step_position.setZero();

    // tbb::parallel_for(0, num_nodes, [&](int i)
    // {
    //     current_position.row(i).segment<2>(0) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2);
    //     next_step_position.row(i).segment<2>(0) = undeformed.segment<2>(i * 2) + _u.segment<2>(i * 2) + du.segment<2>(i * 2);
    // });

    // return ipc::compute_collision_free_stepsize(current_position, 
    //         next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}

template <int dim>
void FEMSolver<dim>::addIPC2DtrueEnergy(T& energy)
{
    // Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    // }

    // ipc::Constraints ipc_constraints;
    // ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    // // tbb::enumerable_thread_specific<double> storage(0);

    // // tbb::parallel_for(
    // //     tbb::blocked_range<size_t>(size_t(0), ipc_constraints.size()),
    // //     [&](const tbb::blocked_range<size_t>& r) {
    // //         auto& local_potential = storage.local();
    // //         for (size_t i = r.begin(); i < r.end(); i++) {
    // //             local_potential +=
    // //                 ipc_constraints[i].compute_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, barrier_distance);
    // //         }
    // //     });

    // // for (size_t i = 0; i < ipc_constraints.size(); i++) {
    // //     auto local_potential =
    // //         ipc_constraints[i].compute_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, barrier_distance);
    // //     std::cout<<local_potential<<std::endl;
    // // }

    // // double potential = 0;
    // // for (const auto& local_potential : storage) {
    // //     std::cout<<"!!!!!!!!local Potential: "<<local_potential<<std::endl;
    // //     potential += local_potential;
    // // }

    
    // T energy_ipc = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    // //std::cout<<"barrier distance: "<<barrier_distance<<" IPC energy: "<<energy_ipc<<std::endl;

    // energy += energy_ipc;  
}

template <int dim>
void FEMSolver<dim>::addIPC2DtrueForceEntries(VectorXT& residual)
{
    // Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    // }
    
    // ipc::Constraints ipc_constraints;
    // ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    // VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    
    // for (int i = 0; i < num_nodes; i++)
    // {
    //     residual.segment<2>(i * 2) -= contact_gradient.segment<2>(i * dim);
    // }
}

template <int dim>
void FEMSolver<dim>::addIPC2DtrueHessianEntries(std::vector<Entry>& entries,bool project_PD)
{
    // Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    //  for (int i = 0; i < num_nodes; i++)
    // {
    //     ipc_vertices_deformed.row(i).segment<2>(0) = deformed.segment<2>(i * 2);
    // }
    
    // ipc::Constraints ipc_constraints;
    // ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    // StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, ipc_constraints, barrier_distance, false);
    

    // std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    // entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template class FEMSolver<2>;
template class FEMSolver<3>;