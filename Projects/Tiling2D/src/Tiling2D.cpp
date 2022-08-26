#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <igl/jet.h>
#include "../include/Tiling2D.h"

void Tiling2D::generateSurfaceMeshFromVTKFile(const std::string& vtk_file, const std::string surface_mesh_file)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F;
}

/*
Triangle:               Triangle6:          Triangle9/10:          Triangle12/15:

v
^                                                                   2
|                                                                   | \
2                       2                    2                      9   8
|`\                     |`\                  | \                    |     \
|  `\                   |  `\                7   6                 10 (14)  7
|    `\                 5    `4              |     \                |         \
|      `\               |      `\            8  (9)  5             11 (12) (13) 6
|        `\             |        `\          |         \            |             \
0----------1 --> u      0-----3----1         0---3---4---1          0---3---4---5---1

*/
bool Tiling2D::initializeSimulationDataFromFiles(const std::string& filename, PBCType pbc_type)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F, V_quad;
    // loadMeshFromVTKFile(data_folder + filename + ".vtk", V, F);
    // loadMeshFromVTKFile(filename, V, F);
    solver.use_quadratic_triangle = true;
    if (filename.substr(filename.find_last_of(".") + 1) == "vtk")
    {
        if (solver.use_quadratic_triangle)
        {
            loadQuadraticTriangleMeshFromVTKFile(filename, V, F, V_quad);
            F.resize(V_quad.rows(), 3);
            F.col(0) = V_quad.col(0); F.col(1) = V_quad.col(1); F.col(2) = V_quad.col(2);
        }
        else
        {
            loadMeshFromVTKFile(filename, V, F);
        }
    }
    else if(filename.substr(filename.find_last_of(".") + 1) == "msh")
    {
        igl::readMSH(filename, V, F);
    }
    
    // if (periodic)
    //     loadPBCDataFromMSHFile(data_folder + filename + ".msh", solver.pbc_pairs);
    // if (periodic)
    // {
    //     std::ifstream translation(data_folder + filename + "Translation.txt");
    //     translation >> solver.t1[0] >> solver.t1[1] >> solver.t2[0] >> solver.t2[1];
    //     translation.close();
    // }
    int n_vtx = V.rows(), n_ele = F.rows();
    solver.num_nodes = n_vtx; solver.num_ele = n_ele;
    solver.undeformed.resize(n_vtx * 2);
    solver.deformed.resize(n_vtx * 2);
    solver.u.resize(n_vtx * 2); solver.u.setZero();
    solver.f.resize(n_vtx * 2); solver.f.setZero();
    solver.surface_indices.resize(n_ele * 3);
    if (solver.use_quadratic_triangle)
        solver.indices.resize(n_ele * 6);
    else
        solver.indices.resize(n_ele * 3);

    tbb::parallel_for(0, n_vtx, [&](int i)
    {
        solver.undeformed.segment<2>(i * 2) = V.row(i).head<2>();
        solver.deformed.segment<2>(i * 2) = V.row(i).head<2>();
    });

    solver.dirichlet_data.clear();

    

    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);

    T scale = max_corner[0] - min_corner[0];

    solver.deformed /= scale;
    solver.undeformed /= scale;
    solver.deformed *= 8;
    solver.undeformed *= 8; // use centimeters
    solver.computeBoundingBox(min_corner, max_corner);

    solver.E = 2.6 * 1e3;

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        solver.surface_indices.segment<3>(i * 3) = F.row(i);
        if (solver.use_quadratic_triangle)
            solver.indices.segment<6>(i * 6) = V_quad.row(i);
        else
            solver.indices.segment<3>(i * 3) = F.row(i);
    });
    
    if (pbc_type == PBC_None)
        solver.add_pbc = false;
    else
    {
        solver.add_pbc = true;
        solver.pbc_w = 1e4;
        if (pbc_type == PBC_X)
            solver.addPBCPairInX();
        else if (pbc_type == PBC_XY)
        {
            solver.addPBCPairsXY();
            solver.add_pbc_strain = true;
            // solver.strain_theta = M_PI / 2.0;
            // solver.uniaxial_strain = 0.5;
            solver.pbc_strain_w = 1e6;
            solver.prescribe_strain_tensor = true;
            solver.target_strain = TV3(0.1, 0.0, 0.);
            solver.computeMarcoBoundaryIndices();
        }
    }
    if (pbc_type == PBC_XY)
    {
        for (int i = 0; i < 2; i++)
            solver.dirichlet_data[solver.pbc_pairs[0][0][0]* 2 + i] = 0.0;
    }
    else if (pbc_type == PBC_X)
    {
        TV min0(min_corner[0] - 1e-6, min_corner[1] - 1e-6);
        TV max0(max_corner[0] + 1e-6, min_corner[1] + 1e-6);
        // solver.addForceBox(min0, max0, TV(0, 1));
        solver.addDirichletBoxY(min0, max0, TV::Zero());
        solver.addDirichletBox(min0, min0 + TV(2e-6, 2e-6), TV::Zero());

        TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
        TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);
        // solver.addForceBox(min1, max1, TV(0, -1));
        T dy = max_corner[1] - min_corner[1];
        solver.penalty_pairs.clear();
        T percent = 0.02;
        solver.addPenaltyPairsBox(min1, max1, TV(0, -percent * dy));

        solver.addPenaltyPairsBoxXY(TV(min_corner[0] - 1e-6, max_corner[1] - 1e-6), 
            TV(min_corner[0] + 1e-6, max_corner[1] + 1e-6), 
            TV(0, -percent * dy));
    }
    

    // solver.unilateral_qubic = true;
    solver.penalty_weight = 1e6;
    // solver.y_bar = max_corner[1] - 0.2 * dy;

    // Eigen::MatrixXd _V; Eigen::MatrixXi _F;
    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling2D/current_mesh.obj", _V, _F);
    // for (int i = 0; i < _V.rows(); i++)
    // {
    //     solver.deformed.segment<2>(i*2) = _V.row(i).segment<2>(0);
    //     solver.u = solver.deformed - solver.undeformed;
    // }

    T total_area = solver.computeTotalArea();
    T bbox_area = (max_corner[0] - min_corner[0]) * (max_corner[1] - min_corner[1]);
    std::cout << "Material Percentage: " << total_area / bbox_area << std::endl;
    solver.use_ipc = true;
    solver.add_friction = false;
    solver.barrier_distance = 1e-3;
    
    // if (solver.use_quadratic_triangle)
        solver.use_ipc = true;
    if (solver.use_ipc)
    {
        solver.computeIPCRestData();
        VectorXT contact_force(solver.num_nodes * 2); contact_force.setZero();
        solver.addIPCForceEntries(contact_force);
        if (contact_force.norm() > 1e-8)
        {
            // std::cout << contact_force.norm() << std::endl;
            return false;
        }
    }

    solver.project_block_PD = true;
    solver.verbose = true;
    solver.max_newton_iter = 1000;
    return true;
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
        solver.barrier_weight = 1.0;
    }
}

void Tiling2D::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C, bool show_PKstress)
{
    
    
    int n_vtx = solver.deformed.rows() / 2;
    
    int n_ele = solver.surface_indices.rows() / 3;
    V.resize(n_vtx, 3); V.setZero();
    F.resize(n_ele, 3); C.resize(n_ele, 3);
    
    tbb::parallel_for(0, n_vtx, [&](int i)
    {
        V.row(i).head<2>() = solver.deformed.segment<2>(i * 2);
    });

    tbb::parallel_for(0, n_ele, [&](int i)
    {
        F.row(i) = solver.surface_indices.segment<3>(i * 3);
        C.row(i) = TV3(0.0, 0.3, 1.0);
    });
    
    if (show_PKstress)
    {
        VectorXT PK_stress;
        solver.computeFirstPiola(PK_stress);
        Eigen::MatrixXd C_jet(n_ele, 3);
        Eigen::MatrixXd value(n_ele, 3);
        value.col(0) = PK_stress; value.col(1) = PK_stress; value.col(2) = PK_stress;
        std::cout << PK_stress.minCoeff() << " " << PK_stress.maxCoeff() << std::endl;
        igl::jet(value, PK_stress.minCoeff(), PK_stress.maxCoeff(), C_jet);
        C = C_jet;
    }
}

void Tiling2D::generateForceDisplacementCurveSingleStructure(const std::string& vtk_file, 
    const std::string& result_folder)
{
    initializeSimulationDataFromFiles(vtk_file, PBC_X);
    
    T dp = 0.02;
    solver.penalty_weight = 1e5;
    std::vector<T> displacements;
    std::vector<T> force_norms;
    VectorXT u_prev = solver.u;
    // solver.unilateral_qubic = true;
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

    T dy = max_corner[1] - min_corner[1];
    for (T dis = 0.0; dis < 0.5  + dp; dis += dp)
    {
        std::cout << "\t---------pencent " << dp << std::endl;
        std::cout << dis << std::endl;
        T displacement_sum = 0.0;
        solver.penalty_pairs.clear();
        solver.addPenaltyPairsBox(min1, max1, TV(0, -dis * dy));
        // solver.y_bar = max_corner[1] - dis * dy;
        solver.u = u_prev;
        solver.staticSolve();
        u_prev = solver.u;
        VectorXT interal_force(solver.num_nodes * 2);
        interal_force.setZero();
        solver.addBCPenaltyForceEntries(solver.penalty_weight, interal_force);
        displacements.push_back(dis * dy);
        force_norms.push_back(interal_force.norm());

        solver.saveToOBJ(result_folder + std::to_string(dis) + ".obj");
        // break;
    }
    std::ofstream out(result_folder + "log.txt");
    out << "displacement in cm" << std::endl;
    for (T v : displacements)
        out << v << " ";
    out << std::endl;
    out << "force in N" << std::endl;
    for (T v : force_norms)
        out << v << " ";
    out << std::endl;
    out.close();
}

void Tiling2D::generateForceDisplacementPolarCurve(const std::string& result_folder)
{
    std::vector<T> displacements;
    std::vector<T> force_norms;
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

    T dy = max_corner[1] - min_corner[1];
    int n_sample = 5;
    T dangle = (M_PI) / T(n_sample);
    for (T angle = 0.0; angle < M_PI; angle += dangle)
    {
        solver.strain_theta = angle;
        solver.staticSolve();
    }
}

void Tiling2D::generateForceDisplacementCurve(const std::string& result_folder)
{
    T dp = 0.02;
    solver.penalty_weight = 1e4;
    // solver.pbc_w = 1e8;
    std::vector<T> displacements;
    std::vector<T> force_norms;
    VectorXT u_prev = solver.u;
    // solver.unilateral_qubic = true;
    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
    TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);

    T dy = max_corner[1] - min_corner[1];
    for (T dis = 0.0; dis < 0.8  + dp; dis += dp)
    {
        std::cout << "\t---------pencent " << dp << std::endl;
        std::cout << dis << std::endl;
        T displacement_sum = 0.0;
        solver.penalty_pairs.clear();
        solver.addPenaltyPairsBox(min1, max1, TV(0, -dis * dy));
        solver.addPenaltyPairsBoxXY(TV(min_corner[0] - 1e-6, max_corner[1] - 1e-6), 
            TV(min_corner[0] + 1e-6, max_corner[1] + 1e-6), 
            TV(0, -dis * dy));
        // solver.y_bar = max_corner[1] - dis * dy;
        solver.u = u_prev;
        solver.staticSolve();
        u_prev = solver.u;
        VectorXT interal_force(solver.num_nodes * 2);
        interal_force.setZero();
        solver.addBCPenaltyForceEntries(solver.penalty_weight, interal_force);
        displacements.push_back(dis * dy);
        force_norms.push_back(interal_force.norm());

        solver.saveToOBJ(result_folder + std::to_string(dis) + ".obj");
        // break;
    }
    std::ofstream out(result_folder + "log.txt");
    out << "displacement in cm" << std::endl;
    for (T v : displacements)
        out << v << " ";
    out << std::endl;
    out << "force in N" << std::endl;
    for (T v : force_norms)
        out << v << " ";
    out << std::endl;
    out.close();
}

void Tiling2D::tileUnitCell(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, int n_unit)
{
    Eigen::MatrixXd V_tile(V.rows() * 4, 3);
    Eigen::MatrixXi F_tile(F.rows() * 4, 3);
    Eigen::MatrixXd C_tile(F.rows() * 4, 3);

    TV left0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][0] * 2);
    TV right0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][1] * 2);
    TV top0 = solver.deformed.segment<2>(solver.pbc_pairs[1][0][1] * 2);
    TV bottom0 = solver.deformed.segment<2>(solver.pbc_pairs[1][0][0] * 2);
    TV dx = (right0 - left0);
    TV dy = (top0 - bottom0);

    int n_face = F.rows(), n_vtx = V.rows();
    V_tile.block(0, 0, n_vtx, 3) = V;
    V_tile.block(n_vtx, 0, n_vtx, 3) = V;
    V_tile.block(2 * n_vtx, 0, n_vtx, 3) = V;
    V_tile.block(3 * n_vtx, 0, n_vtx, 3) = V;
    
    tbb::parallel_for(0, n_vtx, [&](int i){
        V_tile.row(n_vtx + i).head<2>() += dx;
        V_tile.row(3 * n_vtx + i).head<2>() += dx;
        V_tile.row(2 * n_vtx + i).head<2>() += dy;
        V_tile.row(3 * n_vtx + i).head<2>() += dy;
    });
    

    V = V_tile;
    Eigen::MatrixXi offset(n_face, 3);
    offset.setConstant(n_vtx);
    F_tile.block(0, 0, n_face, 3) = F;
    F_tile.block(n_face, 0, n_face, 3) = F + offset;
    F_tile.block(2 * n_face, 0, n_face, 3) = F + 2 * offset;
    F_tile.block(3 * n_face, 0, n_face, 3) = F + 3 * offset;
    F = F_tile;

    Eigen::MatrixXd C_unit = C;
    C_unit.col(2).setConstant(0.3); C_unit.col(1).setConstant(1.0);
    C_tile.block(0, 0, n_face, 3) = C_unit;
    C_tile.block(n_face, 0, n_face, 3) = C;
    C_tile.block(2 * n_face, 0, n_face, 3) = C;
    C_tile.block(3 * n_face, 0, n_face, 3) = C;
    C = C_tile;

}


void Tiling2D::tilingMeshInX(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    Eigen::MatrixXd V_tile(V.rows() * 3, 3);
    Eigen::MatrixXi F_tile(F.rows() * 3, 3);
    Eigen::MatrixXd C_tile(F.rows() * 3, 3);

    TV min_corner, max_corner;
    solver.computeBoundingBox(min_corner, max_corner);
    // T dx = max_corner[0] - min_corner[0];
    TV left0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][0] * 2);
    TV right0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][1] * 2);
    T dx = (right0 - left0).norm();
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
