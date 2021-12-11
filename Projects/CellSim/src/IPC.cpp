#include <ipc/ipc.hpp>
#include "../include/VertexModel.h"


void VertexModel::addIPCEnergy(T& energy)
{
    T contact_energy = 0.0;
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
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
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
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
        residual.segment(0, basal_vtx_start * 3) += -friction_energy_gradient;
    }

    residual.segment(0, basal_vtx_start * 3) += -contact_gradient;
}

void VertexModel::addIPCHessianEntries(std::vector<Entry>& entries,
    bool projectPD)
{
    Eigen::MatrixXd ipc_vertices_deformed(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
    {
        ipc_vertices_deformed.row(i) = deformed.segment<3>(i * 3);
    }
    
    ipc::Constraints ipc_constraints;
    ipc::construct_constraint_set(ipc_vertices, ipc_vertices_deformed, 
        ipc_edges, ipc_faces, barrier_distance, ipc_constraints);

    StiffnessMatrix contact_hessian = barrier_weight *  ipc::compute_barrier_potential_hessian(ipc_vertices_deformed, 
        ipc_edges, ipc_faces, ipc_constraints, barrier_distance, projectPD);

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
        std::vector<Entry> friction_entries = entriesFromSparseMatrix(contact_hessian);
        entries.insert(entries.end(), friction_entries.begin(), friction_entries.end());
    }
}

void VertexModel::computeIPCRestData()
{
    ipc_vertices.resize(basal_vtx_start, 3);
    for (int i = 0; i < basal_vtx_start; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    
    int face_cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int i){
        if (i < basal_face_start)
        {
            if (face_vtx_list.size() == 4)
                face_cnt += 2;
            else if (face_vtx_list.size() == 5)
                face_cnt += 3;
            else if (face_vtx_list.size() == 6)
                face_cnt += 4;
        }
    });
    std::vector<Edge> edges_vec;
    ipc_faces.resize(face_cnt, 3);
    face_cnt = 0;
    iterateFaceSerial([&](VtxList& face_vtx_list, int i)
    {
        if (i < basal_face_start)
        {
            if (face_vtx_list.size() == 4)
            {
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[1]));
                edges_vec.push_back(Edge(face_vtx_list[1], face_vtx_list[2]));
                edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[3]));
                edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[0]));
            }
            else if (face_vtx_list.size() == 5)
            {
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[0]);

                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[1]));
                edges_vec.push_back(Edge(face_vtx_list[1], face_vtx_list[2]));
                edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[3]));
                edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[4]));
                edges_vec.push_back(Edge(face_vtx_list[4], face_vtx_list[0]));
                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));

            }
            else if (face_vtx_list.size() == 6)
            {
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[2], face_vtx_list[1], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[3], face_vtx_list[2], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[5], face_vtx_list[3], face_vtx_list[0]);
                ipc_faces.row(face_cnt++) = IV(face_vtx_list[4], face_vtx_list[3], face_vtx_list[5]);

                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[1]));
                edges_vec.push_back(Edge(face_vtx_list[1], face_vtx_list[2]));
                edges_vec.push_back(Edge(face_vtx_list[2], face_vtx_list[3]));
                edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[4]));
                edges_vec.push_back(Edge(face_vtx_list[4], face_vtx_list[5]));
                edges_vec.push_back(Edge(face_vtx_list[5], face_vtx_list[0]));
                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[2]));
                edges_vec.push_back(Edge(face_vtx_list[0], face_vtx_list[3]));
                edges_vec.push_back(Edge(face_vtx_list[3], face_vtx_list[5]));
            }
        }
    });

    
    ipc_edges.resize(edges_vec.size(), 2);
    for (int i = 0; i < edges_vec.size(); i++)
        ipc_edges.row(i) = edges_vec[i];    
}

void VertexModel::updateIPCVertices(const VectorXT& _u)
{
    VectorXT projected = _u;
    iterateDirichletDoF([&](int offset, T target)
    {
        projected[offset] = target;
    });
    deformed = undeformed + projected;
    for (int i = 0; i < basal_vtx_start; i++)
        ipc_vertices.row(i) = deformed.segment<3>(i * 3);
}