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