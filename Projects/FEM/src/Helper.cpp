#include "../include/FEMSolver.h"
#include <fstream>

template<int dim>
void FEMSolver<dim>::saveTetOBJ(const std::string& filename, const EleNodes& tet_vtx)
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

template<int dim>
void FEMSolver<dim>::saveIPCMesh(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < ipc_vertices.rows(); i++)
        out << "v " << ipc_vertices.row(i) << std::endl;
    for (int i = 0; i < ipc_faces.rows(); i++)
        out << "f " << ipc_faces.row(i) + IV::Ones().transpose() << std::endl;
    
    out.close();
}

template<int dim>
void FEMSolver<dim>::saveToOBJ(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < num_nodes; i++)
        out << "v " << deformed.segment<3>(i * 3).transpose() << std::endl;
    for (int i = 0; i < num_surface_faces; i++)
        out << "f " << (surface_indices.segment<dim>(i * dim) + IV::Ones()).transpose() << std::endl;
    out.close();
}

template<int dim>
void FEMSolver<dim>::computeBBox(const Eigen::MatrixXd& V, TV& bbox_min_corner, TV& bbox_max_corner)
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
template class FEMSolver<2>;
template class FEMSolver<3>;