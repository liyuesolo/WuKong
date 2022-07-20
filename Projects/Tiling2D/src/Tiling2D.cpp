#include "../include/Tiling2D.h"

void Tiling2D::initializeSimulationDataFromVTKFile(const std::string& filename)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    loadMeshFromVTKFile(filename, V, F);
    
    int n_vtx = V.rows(), n_ele = F.rows();
    solver.undeformed.resize(n_vtx * 2);
    solver.deformed.resize(n_vtx * 2);
    solver.u.resize(n_vtx * 2); solver.u.setZero();
    solver.f.resize(n_vtx * 2); solver.f.setZero();
    solver.indices.resize(n_ele * 3);
    tbb::parallel_for(0, n_vtx, [&](int i)
    {
        solver.undeformed.segment<2>(i * 2) = V.row(i).head<2>();
        solver.deformed.segment<2>(i * 2) = V.row(i).head<2>();
    });

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        solver.indices.segment<3>(i * 3) = F.row(i);
    });
    
}

void Tiling2D::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    int n_vtx = solver.deformed.rows() / 2;
    int n_ele = solver.indices.rows() / 3;
    V.resize(n_vtx, 3); V.setZero();
    F.resize(n_ele, 3); C.resize(n_ele, 3);
    
    tbb::parallel_for(0, n_vtx, [&](int i)
    {
        V.row(i).head<2>() = solver.deformed.segment<2>(i * 2);
    });

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        F.row(i) = solver.indices.segment<3>(i * 3);
        C.row(i) = TV3(0.0, 0.3, 1.0);
    });
}

