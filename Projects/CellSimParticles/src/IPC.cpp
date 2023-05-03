#include <ipc/ipc.hpp>
#include <ipc/ipc.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/CellSim.h"

template <int dim>
void CellSim<dim>::saveIPCMesh(bool save_edges)
{
    std::ofstream out("ipc_mesh.obj");
    for (int i = 0; i < ipc_vertices.rows(); i++)
    {
        if constexpr (dim == 2)
            out << "v " << ipc_vertices.row(i) << " 0.0" << std::endl;
        else
                out << "v " << ipc_vertices.row(i) << std::endl;
    }
    for (int i = 0; i < ipc_faces.rows(); i++)
    {
        IV obj_face = ipc_faces.row(i).transpose() + IV::Ones();
        out << "f " << obj_face.transpose() << std::endl;
    }
    out.close();
    if (save_edges)
    {
        out.open("ipc_edges.obj");
        for (int i = 0; i < ipc_vertices.rows(); i++)
        {
            if constexpr (dim == 2)
                out << "v " << ipc_vertices.row(i) << " 0.0" << std::endl;
            else
                out << "v " << ipc_vertices.row(i) << std::endl;
        }
        for (int i = 0; i < ipc_edges.rows(); i++)
            out << "l " << ipc_edges.row(i) + Edge::Ones().transpose() << std::endl;
        out.close();
    }
}

template <int dim>
void CellSim<dim>::updateBarrierInfo(bool first_step)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    for (int i = 0; i < num_nodes; i++) 
        ipc_vertices_deformed.row(i).segment<dim>(0) = deformed.segment<dim>(i * dim);

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);
        
    T current_min_dis = ipc::compute_minimum_distance(ipc_vertices, ipc_edges, ipc_faces, ipc_constraints);
    if (first_step)
        ipc_min_dis = current_min_dis;
    else
    {
        TV min_corner, max_corner;
        computeBoundingBox(min_corner, max_corner);
        T bb_diag = (max_corner - min_corner).norm();
        ipc::update_barrier_stiffness(ipc_min_dis, current_min_dis, max_barrier_weight, ipc_barrier_weight, bb_diag);
        ipc_min_dis = current_min_dis;
    }
}

template <int dim>
void CellSim<dim>::buildIPCRestData()
{
    if constexpr (dim == 2)
    {
        ipc_edges.resize(num_nodes - yolk_cell_starts, 2);
        tbb::parallel_for(yolk_cell_starts, num_nodes, [&](int i)
        {
            int j;
            if (i == num_nodes - 1)j = yolk_cell_starts;
            else j = i + 1;
            ipc_edges.row(i - yolk_cell_starts) = Edge(i, j);
        });
        int n_ipc_edges = ipc_edges.rows();

        ipc_edges.conservativeResize(n_ipc_edges + num_cells, 2);
        tbb::parallel_for(0, num_cells, [&](int i)
        {
            int j;
            if (i == num_cells - 1)j = 0;
            else j = i + 1;
            ipc_edges.row(i + n_ipc_edges) = Edge(i, j);
        });
        
        ipc_vertices.resize(num_nodes, dim);
        for (int i = 0; i < num_nodes; i++)
        {
            ipc_vertices.row(i) = undeformed.segment<2>(i * 2);
        }
    }
    std::cout << "ipc has ixn in rest state: " << ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces) << std::endl;

    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    std::cout << "BBOX diagonal " << bb_diag << std::endl;
    VectorXT dedx(num_nodes * dim), dbdx(num_nodes * dim);
    dedx.setZero(); dbdx.setZero();
    ipc_barrier_weight = 1.0;
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    std::cout << "barrier weight " <<  ipc_barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
    ipc_barrier_weight = ipc::initial_barrier_stiffness(bb_diag, ipc_barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    // if (verbose)
    std::cout << "barrier weight " <<  ipc_barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;
}

template <int dim>
void CellSim<dim>::addIPCEnergy(T& energy)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<dim>(0) = deformed.segment<dim>(i * dim);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);
    
    T energy_ipc = ipc_barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);

    energy += energy_ipc;    
}


template <int dim>
void CellSim<dim>::addIPCForceEntries(VectorXT& residual)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<dim>(0) = deformed.segment<dim>(i * dim);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);

    VectorXT contact_gradient = ipc_barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance);

    
    for (int i = 0; i < num_nodes; i++)
    {
        residual.segment<dim>(i * dim) -= contact_gradient.segment<dim>(i * dim);
    }
    
    // residual -= contact_gradient;
}

template <int dim>
void CellSim<dim>::addIPCHessianEntries(std::vector<Entry>& entries)
{
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
     for (int i = 0; i < num_nodes; i++)
    {
        ipc_vertices_deformed.row(i).segment<dim>(0) = deformed.segment<dim>(i * dim);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = ipc_barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, ipc_barrier_distance, false);
    

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian);
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());
}

template <int dim>
void CellSim<dim>::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        ipc_vertices.row(i).segment<dim>(0) = deformed.segment<dim>(i * dim);
    });
}

template <int dim>
T CellSim<dim>::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    Eigen::MatrixXd current_position, next_step_position;

    current_position.resize(num_nodes, dim);
    next_step_position.resize(num_nodes, dim);
    current_position.setZero(); next_step_position.setZero();

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        current_position.row(i).segment<dim>(0) = undeformed.segment<dim>(i * dim) + _u.segment<dim>(i * dim);
        next_step_position.row(i).segment<dim>(0) = undeformed.segment<dim>(i * dim) + _u.segment<dim>(i * dim) + du.segment<dim>(i * dim);
    });

    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}

template class CellSim<2>;
template class CellSim<3>;