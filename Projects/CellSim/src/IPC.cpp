#include <ipc/ipc.hpp>
#include "../include/VertexModel.h"


void VertexModel::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    Eigen::MatrixXd ipc_vertices_deformed(n_ipc_vtx, 3);
    for (int i = 0; i < n_ipc_vtx; i++) 
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

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

    contact_energy = barrier_weight * ipc::compute_barrier_potential(ipc_vertices_deformed, 
    ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    
    
    
    energy += contact_energy;
    
}
void VertexModel::addIPCForceEntries(VectorXT& residual)
{
    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    Eigen::MatrixXd ipc_vertices_deformed(n_ipc_vtx, 3);
    for (int i = 0; i < n_ipc_vtx; i++)
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }

    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    VectorXT contact_gradient = barrier_weight * ipc::compute_barrier_potential_gradient(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance);
    
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
        residual.segment(0, n_ipc_vtx * 3) += -friction_energy_gradient;
    }

    residual.segment(0, n_ipc_vtx * 3) += -contact_gradient;
}

void VertexModel::addIPCHessianEntries(std::vector<Entry>& entries,
    bool projectPD)
{
    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    Eigen::MatrixXd ipc_vertices_deformed(n_ipc_vtx, 3);
    for (int i = 0; i < n_ipc_vtx; i++)
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
            ipc_faces, ipc_friction_constraints, epsv_times_h
        );
        std::vector<Entry> friction_entries = entriesFromSparseMatrix(friction_energy_hessian);
        entries.insert(entries.end(), friction_entries.begin(), friction_entries.end());
    }
}

void VertexModel::computeIPCRestData()
{
    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    int n_ipc_face = add_basal_faces_ipc ? lateral_face_start : basal_face_start;

    ipc_vertices.resize(n_ipc_vtx, 3);
    for (int i = 0; i < n_ipc_vtx; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    
    int face_cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int i){
        if (i < n_ipc_face)
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

    iterateFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        if (i < n_ipc_face)
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
        }
    });
    
    ipc_edges.resize(edges_vec.size(), 2);
    for (int i = 0; i < edges_vec.size(); i++)
        ipc_edges.row(i) = edges_vec[i];    
    
    // saveIPCData("./", 0, true);

    bool has_ixn_in_rest_shape = ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces);
    
    if (has_ixn_in_rest_shape)
        std::cout << "[ALERT] ipc mesh has self-intersection in its rest shape" << std::endl;
}

// void VertexModel::computeIPCRestData()
// {
//     int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
//     int n_ipc_faces = add_basal_faces_ipc ? lateral_face_start : basal_face_start;
//     ipc_vertices.resize(n_ipc_vtx, 3);
//     for (int i = 0; i < n_ipc_vtx; i++)
//         ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    
//     int face_cnt = 0;
//     iterateFaceSerial([&](VtxList& face_vtx_list, int i){
//         if (i < n_ipc_faces)
//         {
//             if (face_vtx_list.size() == 4)
//                 face_cnt += 2;
//             else if (face_vtx_list.size() == 5)
//                 face_cnt += 3;
//             else if (face_vtx_list.size() == 6)
//                 face_cnt += 4;
//             else if (face_vtx_list.size() == 7)
//                 face_cnt += 5;
//             else if (face_vtx_list.size() == 8)
//                 face_cnt += 6;
//             else if (face_vtx_list.size() == 9)
//                 face_cnt += 7;
//             else
//             {
//                 std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
//             }
//         }
//     });
    

//     std::vector<Edge> edges_vec;
//     ipc_faces.resize(face_cnt, 3);
//     face_cnt = 0;
//     auto appendEdges = [&](VtxList& face_vtx_list)
//     {
//         for (int ne = 0; ne < face_vtx_list.size(); ne++)
//         {
//             int ne_plus1 = (ne + 1) % face_vtx_list.size();
//             edges_vec.push_back(Edge(face_vtx_list[ne], face_vtx_list[ne_plus1]));
//         }
//         if (face_vtx_list.size() == 4)
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[0]));
//         else if (face_vtx_list.size() == 5)
//         {
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));
//         }
//         else if (face_vtx_list.size() == 6)
//         {
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));
//             edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[5]));
//         }
//         else if (face_vtx_list.size() == 7)
//         {
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[6]));
//             edges_vec.push_back(Edge(face_vtx_list[6], face_vtx_list[3]));
//             edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[5]));
//         }
//         else if (face_vtx_list.size() == 8)
//         {
//             edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[7]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[4]));
//             edges_vec.push_back(Edge(face_vtx_list[7], face_vtx_list[4]));
//             edges_vec.push_back(Edge(face_vtx_list[7], face_vtx_list[5]));
//         }
//         else if (face_vtx_list.size() == 9)
//         {
//             edges_vec.push_back(Edge(face_vtx_list[1], face_vtx_list[8]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[8]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[7]));
//             edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[6]));
//             edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[6]));
//             edges_vec.push_back(Edge(face_vtx_list[4], face_vtx_list[6]));
//         }
//     };

//     iterateFaceSerial([&](VtxList& face_vtx_list, int i)
//     {
//         if (i < basal_face_start)
//         {
//             if (face_vtx_list.size() == 4)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
//                 appendEdges(face_vtx_list);
//             }
//             else if (face_vtx_list.size() == 5)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[0]);
//                 appendEdges(face_vtx_list);

//             }
//             else if (face_vtx_list.size() == 6)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[5]);
//                 appendEdges(face_vtx_list);
//             }
//             else if (face_vtx_list.size() == 7)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[2], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[6]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[6]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[5]);
//                 appendEdges(face_vtx_list);
//             }
//             else if (face_vtx_list.size() == 8)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[0], face_vtx_list[2], face_vtx_list[1]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[4], face_vtx_list[3]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[7], face_vtx_list[2], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[2], face_vtx_list[7]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[4], face_vtx_list[7]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[5], face_vtx_list[7]);
//                 appendEdges(face_vtx_list);
//             }
//             else if (face_vtx_list.size() == 9)
//             {
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[8], face_vtx_list[1], face_vtx_list[0]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[8], face_vtx_list[2], face_vtx_list[1]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[7], face_vtx_list[2], face_vtx_list[8]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[6], face_vtx_list[2], face_vtx_list[7]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[6]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[6]);
//                 ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[4], face_vtx_list[6]);
//                 appendEdges(face_vtx_list);
//             }
//             else
//             {
//                 std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
//             }
//         }

//         if (add_basal_faces_ipc)
//         {
//             if (i >= basal_face_start && i < lateral_face_start)
//             {

//                 if (face_vtx_list.size() == 4)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[1], face_vtx_list[2], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[3], face_vtx_list[0]);
//                     appendEdges(face_vtx_list);
//                 }
//                 else if (face_vtx_list.size() == 5)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[1], face_vtx_list[2], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[3], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[4], face_vtx_list[0]);
//                     appendEdges(face_vtx_list);

//                 }
//                 else if (face_vtx_list.size() == 6)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[1], face_vtx_list[2], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[3], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[5], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[4], face_vtx_list[5]);
//                     appendEdges(face_vtx_list);
//                 }
//                 else if (face_vtx_list.size() == 7)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[1], face_vtx_list[2], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[6], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[3], face_vtx_list[6]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[5], face_vtx_list[6]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[4], face_vtx_list[5]);
//                     appendEdges(face_vtx_list);
//                 }
//                 else if (face_vtx_list.size() == 8)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[0], face_vtx_list[1]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[2], face_vtx_list[3]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[7], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[4], face_vtx_list[7]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[5], face_vtx_list[7]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[6], face_vtx_list[7]);
//                     appendEdges(face_vtx_list);
//                 }
//                 else if (face_vtx_list.size() == 9)
//                 {
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[1], face_vtx_list[8], face_vtx_list[0]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[8], face_vtx_list[1]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[7], face_vtx_list[8]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[6], face_vtx_list[7]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[3], face_vtx_list[6]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[4], face_vtx_list[6]);
//                     ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[5], face_vtx_list[6]);
//                     appendEdges(face_vtx_list);
//                 }
//                 else
//                 {
//                     std::cout << "Unknown polygon edges " << __FILE__ << std::endl;
//                 }
//             }
//         }
//     });
    
//     ipc_edges.resize(edges_vec.size(), 2);
//     for (int i = 0; i < edges_vec.size(); i++)
//         ipc_edges.row(i) = edges_vec[i];    
    
//     // saveIPCData("./", 0, true);

//     bool has_ixn_in_rest_shape = ipc::has_intersections(ipc_vertices, ipc_edges, ipc_faces);
    
//     if (has_ixn_in_rest_shape)
//         std::cout << "[ALERT] ipc mesh has self-intersection in its rest shape" << std::endl;
// }

void VertexModel::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;

    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    // for (int i = 0; i < n_ipc_vtx; i++)
    tbb::parallel_for(0, n_ipc_vtx, [&](int i)
    {
        ipc_vertices.row(i) = deformed.segment<3>(i * 3);
    });
}

T VertexModel::computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    int n_ipc_vtx = add_basal_faces_ipc ? num_nodes : basal_vtx_start;
    Eigen::MatrixXd current_position(n_ipc_vtx, 3), 
        next_step_position(n_ipc_vtx, 3);
        
    // for (int i = 0; i < n_ipc_vtx; i++)
    tbb::parallel_for(0, n_ipc_vtx, [&](int i)
    {
        current_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3);
        // current_position.row(i) = undeformed.segment<3>(i * 3);
        next_step_position.row(i) = undeformed.segment<3>(i * 3) + _u.segment<3>(i * 3) + du.segment<3>(i * 3);
    }
    );
    return ipc::compute_collision_free_stepsize(current_position, 
            next_step_position, ipc_edges, ipc_faces, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1e7);
}