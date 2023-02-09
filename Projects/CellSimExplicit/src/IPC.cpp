#include <ipc/ipc.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include "../include/CellSim.h"

void CellSim::updateBarrierInfo(bool first_step)
{
    // return;
    Eigen::MatrixXd ipc_vertices_deformed = ipc_vertices;
    
    for (int i = 0; i < num_nodes; i++) 
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);

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
        // std::cout << "barrier weight " << barrier_weight << std::endl;
    }
}

void CellSim::addIPCEnergy(T& energy)
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
void CellSim::addIPCForceEntries(VectorXT& residual)
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

void CellSim::addIPCHessianEntries(std::vector<Entry>& entries,
    bool projectPD)
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
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, projectPD);

    // StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
    //     ipc_edges, ipc_faces, ipc_constraints, barrier_distance, true);

    std::vector<Entry> contact_entries = entriesFromSparseMatrix(contact_hessian, lower_triangular);
    entries.insert(entries.end(), contact_entries.begin(), contact_entries.end());

}

void CellSim::computeIPCRestData()
{
    
    int n_ipc_face = faces.size() + yolk_cells.size();

    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    
    int face_cnt = 0;
    iterateYolkAndCellFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        if (face_vtx_list.size() == 4)
            face_cnt += 2;
        else if (face_vtx_list.size() == 5)
            face_cnt += 3;
        else if (face_vtx_list.size() == 6)
            face_cnt += 4;
        else if (face_vtx_list.size() == 7)
            face_cnt += 5;
        else if (face_vtx_list.size() == 8)
            face_cnt += 6;
        else if (face_vtx_list.size() == 9)
            face_cnt += 7;
        else
        {
            std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
        }
    });
    

    std::vector<Edge> edges_vec;
    ipc_faces.resize(face_cnt, 3);
    face_cnt = 0;
    auto appendEdges = [&](VtxList& face_vtx_list)
    {
        for (int ne = 0; ne < face_vtx_list.size(); ne++)
        {
            int ne_plus1 = (ne + 1) % face_vtx_list.size();
            edges_vec.push_back(Edge(face_vtx_list[ne], face_vtx_list[ne_plus1]));
        }
        if (face_vtx_list.size() == 4)
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[0]));
        else if (face_vtx_list.size() == 5)
        {
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));
        }
        else if (face_vtx_list.size() == 6)
        {
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));
            edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[5]));
        }
        else if (face_vtx_list.size() == 7)
        {
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[6]));
            edges_vec.push_back(Edge(face_vtx_list[6], face_vtx_list[3]));
            edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[5]));
        }
        else if (face_vtx_list.size() == 8)
        {
            edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[7]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[4]));
            edges_vec.push_back(Edge(face_vtx_list[7], face_vtx_list[4]));
            edges_vec.push_back(Edge(face_vtx_list[7], face_vtx_list[5]));
        }
        else if (face_vtx_list.size() == 9)
        {
            edges_vec.push_back(Edge(face_vtx_list[1], face_vtx_list[8]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[8]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[7]));
            edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[6]));
            edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[6]));
            edges_vec.push_back(Edge(face_vtx_list[4], face_vtx_list[6]));
        }
    };

    iterateYolkAndCellFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        
        if (face_vtx_list.size() == 4)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
            appendEdges(face_vtx_list);
        }
        else if (face_vtx_list.size() == 5)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[0]);
            appendEdges(face_vtx_list);

        }
        else if (face_vtx_list.size() == 6)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[5]);
            appendEdges(face_vtx_list);
        }
        else if (face_vtx_list.size() == 7)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[2], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[6]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[6]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[5]);
            appendEdges(face_vtx_list);
        }
        else if (face_vtx_list.size() == 8)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[0], face_vtx_list[2], face_vtx_list[1]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[4], face_vtx_list[3]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[7], face_vtx_list[2], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[2], face_vtx_list[7]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[4], face_vtx_list[7]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[5], face_vtx_list[7]);
            appendEdges(face_vtx_list);
        }
        else if (face_vtx_list.size() == 9)
        {
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[8], face_vtx_list[1], face_vtx_list[0]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[8], face_vtx_list[2], face_vtx_list[1]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[7], face_vtx_list[2], face_vtx_list[8]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[2], face_vtx_list[7]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[6]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[6]);
            ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[4], face_vtx_list[6]);
            appendEdges(face_vtx_list);
        }
        else
        {
            std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
        }
        
    });
    
    ipc_edges.resize(edges_vec.size(), 2);
    for (int i = 0; i < edges_vec.size(); i++)
        ipc_edges.row(i) = edges_vec[i];    
    
    // saveIPCData("./", 0, true);

    bool has_ixn_in_rest_shape = ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces);
    
    if (has_ixn_in_rest_shape)
        std::cout << "[ALERT] ipc mesh has self-intersection in its rest shape" << std::endl;
    // return;
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    T bb_diag = (max_corner - min_corner).norm();
    std::cout << "BBOX diagonal " << bb_diag << std::endl;
    VectorXT dedx(num_nodes * 3), dbdx(num_nodes * 3);
    dedx.setZero(); dbdx.setZero();
    barrier_weight = 1.0;
    addIPCForceEntries(dbdx); dbdx *= -1.0;
    computeResidual(u, dedx); dedx *= -1.0; dedx -= dbdx;
    barrier_weight = ipc::initial_barrier_stiffness(bb_diag, barrier_distance, 1.0, dedx, dbdx, max_barrier_weight);
    // if (verbose)
    VectorXT ipc_force_rest = VectorXT::Zero(num_nodes*3);
    addIPCForceEntries(ipc_force_rest);
    std::cout << "ipc force norm " << ipc_force_rest.norm() << std::endl;
    std::cout << "barrier weight " <<  barrier_weight << " max_barrier_weight " << max_barrier_weight << std::endl;

}

void CellSim::saveIPCData(const std::string& folder, int iter, bool save_edges)
{
    std::ofstream out(folder + "/ipc_faces_iter" + std::to_string(iter) + ".obj");
    for (int i = 0; i < ipc_vertices.rows(); i++)
    {
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
        out.open(folder + "/ipc_edges_iter" + std::to_string(iter) + ".obj");
        for (int i = 0; i < ipc_vertices.rows(); i++)
            out << "v " << ipc_vertices.row(i) << std::endl;
        for (int i = 0; i < ipc_edges.rows(); i++)
            out << "l " << ipc_edges.row(i) + Edge::Ones().transpose() << std::endl;
        out.close();
    }
}

void CellSim::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    
    // for (int i = 0; i < num_nodes; i++)
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        ipc_vertices.row(i) = deformed.segment<3>(i * 3);
    });
}

T CellSim::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    
    Eigen::MatrixXd current_position(num_nodes, 3), 
        next_step_position(num_nodes, 3);
        
    // for (int i = 0; i < num_nodes; i++)
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
        // current_position.row(i) = undeformed.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    }
    );
    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}