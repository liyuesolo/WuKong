#include "../include/FEMSolver.h"
#include <fstream>

void FEMSolver::saveTetOBJ(const std::string& filename, const TetNodes& tet_vtx)
{
    
    std::ofstream out(filename);
    for (int i = 0; i < 4; i++)
        out << "v " << tet_vtx.row(i) << std::endl;
    // out << "f 2 1 3" << std::endl;
    // out << "f 1 2 4" << std::endl;
    // out << "f 2 3 4" << std::endl;
    // out << "f 3 1 4" << std::endl;
    out << "f 1 2 3" << std::endl;
    out << "f 2 1 4" << std::endl;
    out << "f 3 2 4" << std::endl;
    out << "f 1 3 4" << std::endl;
    out.close();
}

void FEMSolver::saveIPCMesh(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < ipc_vertices.rows(); i++)
        out << "v " << ipc_vertices.row(i) << std::endl;
    for (int i = 0; i < ipc_faces.rows(); i++)
        out << "f " << ipc_faces.row(i) + IV::Ones().transpose() << std::endl;
    
    out.close();
}

void FEMSolver::saveThreePointBendingData(const std::string& folder, int iter)
{
    std::ofstream out(folder + "/structure_iter_" + std::to_string(iter) + ".obj");
    for (int i = 0; i < cylinder_vtx_start; i++)
        out << "v " << deformed.segment<3>(i * dim).transpose() << std::endl;
    for (int i = 0; i < cylinder_face_start; i++)
        out << "f " << (surface_indices.segment<3>(i * 3) + IV::Ones()).transpose() << std::endl;
    out.close();
    out.open(folder + "/bc_iter_" + std::to_string(iter) + ".obj");
    for (int i = cylinder_vtx_start; i < num_nodes; i++)
        out << "v " << deformed.segment<3>(i * dim).transpose() << std::endl;
    for (int i = 0; i < cylinder_vertices.rows(); i++)
        out << "v " << cylinder_vertices.row(i) << std::endl;
    
    IV offset = IV::Ones() * cylinder_vtx_start;
    for (int i = cylinder_face_start; i < num_surface_faces; i++)
        out << "f " << (surface_indices.segment<3>(i * 3) + IV::Ones() - offset).transpose() << std::endl;
    offset = IV::Ones() * (num_nodes - cylinder_vtx_start);
    for (int i = 0; i < cylinder_faces.rows(); i++)
        out << "f " << cylinder_faces.row(i) + (IV::Ones() + offset).transpose() << std::endl;
    out.close();
}

void FEMSolver::saveToOBJ(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < num_nodes; i++)
        out << "v " << deformed.segment<3>(i * dim).transpose() << std::endl;
    for (int i = 0; i < num_surface_faces; i++)
        out << "f " << (surface_indices.segment<3>(i * 3) + IV::Ones()).transpose() << std::endl;
    out.close();
}

void FEMSolver::appendMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F, const Eigen::MatrixXd& _C)
{
    int size_v = V.rows();
    int size_f = F.rows();
    V.conservativeResize(size_v + _V.rows(), 3);
    F.conservativeResize(size_f + _F.rows(), 3);
    C.conservativeResize(size_f + _F.rows(), 3);
    MatrixXi offset(_F.rows(), 3);
    offset.setConstant(size_v);

    V.block(size_v, 0, _V.rows(), 3) = _V;
    F.block(size_f, 0, _F.rows(), 3) = _F + offset;
    C.block(size_f, 0, _F.rows(), 3) = _C;
}

void FEMSolver::appendMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F)
{
    int size_v = V.rows();
    int size_f = F.rows();
    V.conservativeResize(size_v + _V.rows(), 3);
    F.conservativeResize(size_f + _F.rows(), 3);
    MatrixXi offset(_F.rows(), 3);
    offset.setConstant(size_v);

    V.block(size_v, 0, _V.rows(), 3) = _V;
    F.block(size_f, 0, _F.rows(), 3) = _F + offset;
}

void FEMSolver::computeBBox(const Eigen::MatrixXd& V, TV& bbox_min_corner, TV& bbox_max_corner)
{
    bbox_min_corner.setConstant(1e6);
    bbox_max_corner.setConstant(-1e6);

    for (int i = 0; i < V.rows(); i++)
    {
        for (int d = 0; d < 3; d++)
        {
            bbox_max_corner[d] = std::max(bbox_max_corner[d], V(i, d));
            bbox_min_corner[d] = std::min(bbox_min_corner[d], V(i, d));
        }
    }
}