#include "../include/FEMSolver.h"


void FEMSolver::initializeElementData(const Eigen::MatrixXd& TV, 
    const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT)
{
    num_nodes = TV.rows();
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = TV.row(i);
    });
    deformed = undeformed;
    u = VectorXT::Zero(num_nodes * dim);

    num_ele = TT.rows();
    indices.resize(num_ele * 4);
    tbb::parallel_for(0, num_ele, [&](int i)
    {
        indices.segment<4>(i * 4) = TT.row(i);
    });

    num_surface_faces = TF.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(TF(i, 1), TF(i, 0), TF(i, 2));
    });
}

void FEMSolver::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V.row(i) = deformed.segment<3>(i * dim);
    });

    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        C.row(i) = TV(0, 0.3, 1.0);
    });

}