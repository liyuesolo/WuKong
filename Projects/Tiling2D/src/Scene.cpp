#include "../include/FEMSolver.h"
#include <iomanip>
void FEMSolver::computeBoundingBox(TV& min_corner, TV& max_corner)
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < dim; d++)
        {
            max_corner[d] = std::max(max_corner[d], deformed[i * dim + d]);
            min_corner[d] = std::min(min_corner[d], deformed[i * dim + d]);
        }
    }
}

void FEMSolver::saveToOBJ(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < num_nodes; i++)
        out << std::setprecision(12) << "v " << deformed.segment<2>(i * dim).transpose() << " 0" << std::endl;
    for (int i = 0; i < num_ele; i++)
        out << "f " << (indices.segment<3>(i * 3) + IV3::Ones()).transpose() << std::endl;
    out.close();
}