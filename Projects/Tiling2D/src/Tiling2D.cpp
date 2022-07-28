#include <igl/readOBJ.h>
#include "../include/Tiling2D.h"

void Tiling2D::initializeSimulationDataFromFiles(const std::string& filename, bool periodic)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    loadMeshFromVTKFile(data_folder + filename + ".vtk", V, F);
    if (periodic)
        loadPBCDataFromMSHFile(data_folder + filename + ".msh", solver.pbc_pairs);
    if (periodic)
    {
        std::ifstream translation(data_folder + filename + "Translation.txt");
        translation >> solver.t1[0] >> solver.t1[1] >> solver.t2[0] >> solver.t2[1];
        translation.close();
    }
    int n_vtx = V.rows(), n_ele = F.rows();
    solver.num_nodes = n_vtx; solver.num_ele = n_ele;
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

    for (int i = 0; i < 2; i++)
        solver.dirichlet_data[i] = 0.0;

    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);

    T scale = max_corner[0] - min_corner[0];

    solver.deformed /= scale;
    solver.undeformed /= scale;
    solver.deformed *= 0.08;
    solver.undeformed *= 0.08;
    solver.computeBoundingBox(min_corner, max_corner);

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        solver.indices.segment<3>(i * 3) = F.row(i);
    });
    
    solver.add_pbc = periodic;
    if (solver.add_pbc)
    {
        // solver.reorderPBCPairs();
        solver.addPBCPairInX();
        solver.pbc_w = 1e8;
        solver.strain_theta = M_PI / 2.0;
        solver.add_pbc_strain = false;
        solver.uniaxial_strain = 1.0;
    }
    solver.use_ipc = true;
    if (solver.use_ipc)
    {
        solver.computeIPCRestData();
        solver.add_friction = false;
        solver.barrier_distance = 1e-3;
        solver.barrier_weight = 1e6;
    }
    solver.penalty_weight = 1e8;

    
    TV min0(min_corner[0] - 1e-6, min_corner[1] - 1e-6);
    TV max0(max_corner[0] + 1e-6, min_corner[1] + 1e-6);
    // solver.addForceBox(min0, max0, TV(0, 1));
    solver.addDirichletBox(min0, max0, TV::Zero());

    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);
    // solver.addForceBox(min1, max1, TV(0, -1));
    solver.addPenaltyPairsBox(min1, max1, TV(0, -0.005));

    // Eigen::MatrixXd _V; Eigen::MatrixXi _F;
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/results/0.059000.obj", _V, _F);
    // for (int i = 0; i < _V.rows(); i++)
    // {
    //     solver.deformed.segment<2>(i*2) = _V.row(i).segment<2>(0);
    //     solver.u = solver.deformed - solver.undeformed;
    // }
    
    solver.project_block_PD = true;
}

void Tiling2D::initializeSimulationDataFromVTKFile(const std::string& filename)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    loadMeshFromVTKFile(filename, V, F);
    
    loadPBCDataFromMSHFile(data_folder + "thickshell.msh", solver.pbc_pairs);
    std::ifstream translation(data_folder + "translation.txt");
    translation >> solver.t1[0] >> solver.t1[1] >> solver.t2[0] >> solver.t2[1];
    translation.close();
    int n_vtx = V.rows(), n_ele = F.rows();
    solver.num_nodes = n_vtx; solver.num_ele = n_ele;
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

    for (int i = 0; i < 2; i++)
        solver.dirichlet_data[i] = 0.0;

    solver.deformed /= solver.deformed.maxCoeff();
    solver.undeformed /= solver.undeformed.maxCoeff();

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        solver.indices.segment<3>(i * 3) = F.row(i);
    });
    
    solver.add_pbc = true;
    if (solver.add_pbc)
    {
        solver.reorderPBCPairs();
        solver.pbc_w = 1e4;
        solver.strain_theta = M_PI / 2.0;
        solver.uniaxial_strain = 0.2;
    }
    solver.use_ipc = false;
    if (solver.use_ipc)
    {
        solver.computeIPCRestData();
        solver.add_friction = false;
        solver.barrier_distance = 1e-4;
        solver.barrier_weight = 1e3;
    }
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

void Tiling2D::generateForceDisplacementCurve(const std::string& result_folder)
{
    solver.penalty_weight = 1e8;
    T dp = 0.001;
    std::vector<T> displacements;
    std::vector<T> force_norms;
    VectorXT u_prev = solver.u;
    
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

    for (T dis = 0.005; dis < 0.06; dis += dp)
    {
        T displacement_sum = 0.0;
        solver.addPenaltyPairsBox(min1, max1, TV(0, -dis));
        solver.u = u_prev;
        solver.staticSolve();
        u_prev = solver.u;
        VectorXT interal_force(solver.num_nodes * 2);
        interal_force.setZero();
        solver.addBCPenaltyForceEntries(solver.penalty_weight, interal_force);
        displacements.push_back(dis);
        force_norms.push_back(interal_force.norm());

        solver.saveToOBJ(result_folder + std::to_string(dis) + ".obj");
        // break;
    }
    std::ofstream out(result_folder + "log.txt");
    out << "displacement in m" << std::endl;
    for (T v : displacements)
        out << v << " ";
    out << std::endl;
    out << "force in N" << std::endl;
    for (T v : force_norms)
        out << v << " ";
    out << std::endl;
    out.close();
}

void Tiling2D::tilingMeshInX(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    Eigen::MatrixXd V_tile(V.rows() * 3, 3);
    Eigen::MatrixXi F_tile(F.rows() * 3, 3);
    Eigen::MatrixXd C_tile(F.rows() * 3, 3);

    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    T dx = max_corner[0] - min_corner[0];
    int n_face = F.rows(), n_vtx = V.rows();
    V_tile.block(0, 0, n_vtx, 3) = V;
    V_tile.block(n_vtx, 0, n_vtx, 3) = V;
    V_tile.block(2 * n_vtx, 0, n_vtx, 3) = V;
    V_tile.block(0, 0, n_vtx, 1).array() -= dx;
    V_tile.block(2 * n_vtx, 0, n_vtx, 1).array() += dx;

    V = V_tile;
    Eigen::MatrixXi offset(n_face, 3);
    offset.setConstant(n_vtx);
    F_tile.block(0, 0, n_face, 3) = F;
    F_tile.block(n_face, 0, n_face, 3) = F + offset;
    F_tile.block(2 * n_face, 0, n_face, 3) = F + 2 * offset;
    F = F_tile;

    C_tile.block(0, 0, n_face, 3) = C;
    C_tile.block(n_face, 0, n_face, 3) = C;
    C_tile.block(2 * n_face, 0, n_face, 3) = C;
    C = C_tile;
}