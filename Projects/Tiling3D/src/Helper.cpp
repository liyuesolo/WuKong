#include "../include/FEMSolver.h"
#include <fstream>

void FEMSolver::saveTetOBJ(const std::string& filename, const TetNodes& tet_vtx)
{
    std::ofstream out(filename);
    for (int i = 0; i < 4; i++)
        out << "v " << tet_vtx.row(i) << std::endl;
    out << "f 2 1 3" << std::endl;
    out << "f 1 2 4" << std::endl;
    out << "f 2 3 4" << std::endl;
    out << "f 3 1 4" << std::endl;
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