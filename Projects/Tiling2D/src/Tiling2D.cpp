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
            TV3 e0(V.row(F(0, 1)) - V.row(F(0, 0)));
            TV3 e1(V.row(F(0, 2)) - V.row(F(0, 0)));
            if (e1.cross(e0).dot(TV3(0, 0, 1)) > 0)
            {
                F.col(0) = V_quad.col(0); F.col(1) = V_quad.col(2); F.col(2) = V_quad.col(1);
                Eigen::MatrixXi V_quad_backup = V_quad;
                V_quad.col(1) = V_quad_backup.col(2); V_quad.col(2) = V_quad_backup.col(1);
                V_quad.col(5) = V_quad_backup.col(4); V_quad.col(4) = V_quad_backup.col(5);
            }
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
    solver.deformed *= 50;
    solver.undeformed *= 50; // use milimeters
    solver.thickness = 50.0;
    solver.computeBoundingBox(min_corner, max_corner);
    std::cout << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
    // std::getchar();
    solver.E = 2.6 * 1e1;

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
            solver.strain_theta = 0.;
            solver.uniaxial_strain = 1.02667;
            solver.uniaxial_strain_ortho = 0.91;
            solver.biaxial = false;
            solver.pbc_strain_w = 1e6;
            solver.pbc_w = 1e6;
            solver.prescribe_strain_tensor = true;
            solver.target_strain = TV3(0.0270336, -0.0211052, -0.00195281);
            // solver.target_strain = TV3(0.44765, -0.0656891, 0.0956651);
            
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
    solver.barrier_distance = 1e-4;
    
    // if (solver.use_quadratic_triangle)
        solver.use_ipc = true;
    if (solver.use_ipc)
    {
        solver.computeIPCRestData();
        VectorXT contact_force(solver.num_nodes * 2); contact_force.setZero();
        solver.addIPCForceEntries(contact_force);
        if (contact_force.norm() > 1e-8)
        {
            std::cout << contact_force.norm() << std::endl;
            return false;
        }
    }

    solver.project_block_PD = false;
    solver.verbose = true;
    solver.max_newton_iter = 500;
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

void Tiling2D::sampleUniaxialStrain(const std::string& result_folder, T strain)
{
    int IH = 0;
    std::ofstream out(result_folder + "data.txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
    }
    params[0] = 0.05;

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    TV range_theta(0.0, M_PI);
    int n_sp_theta = 100;
    // for(int l = 0; l < n_sp_theta; l++)
    T dtheta = (range_theta[1] - range_theta[0]) / T(n_sp_theta);
    for (T theta = range_theta[0]; theta < range_theta[1] + dtheta; theta += dtheta)
    {
        solver.reset();
        // T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain;
        // solver.pbc_strain_w = 1e8;
        // solver.pbc_w = 1e8;
        solver.staticSolve();
        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM sigma, Cauchy_strain, Green_strain;
        solver.computeHomogenizedStressStrain(sigma, Cauchy_strain, Green_strain);
        
        TV strain_dir = TV(std::cos(theta), std::sin(theta));
        T stretch_in_d = strain_dir.dot(strain * strain_dir);
    
        // T direction_stiffness = strain_dir.dot(sigma * strain_dir) / stretch_in_d;
        T direction_stiffness = solver.computeTotalEnergy(solver.u);
        
        out << direction_stiffness << " " << theta << " " << strain << " "
            << Cauchy_strain(0, 0) << " "<< Cauchy_strain(0, 1) << " " 
            << Cauchy_strain(1, 0) << " " << Cauchy_strain(1, 1) << " "
            << Green_strain(0, 0) << " "<< Green_strain(0, 1) << " " 
            << Green_strain(1, 0) << " " << Green_strain(1, 1) << " " 
            << sigma(0, 0) << " "<< sigma(0, 1) << " " 
            << sigma(1, 0) << " "<< sigma(1, 1) << " "
                << residual.norm() << std::endl;
        solver.saveToOBJ(result_folder + "theta_" + std::to_string(theta) + ".obj");
    }

}

void Tiling2D::computeMarcoStressFromNetworkInputs(const TV3& macro_strain, int IH, 
        const VectorXT& tiling_params)
{
    std::string result_folder = "./";
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = tiling_params[j];
    }
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = true;
    solver.prescribe_strain_tensor = false;
    solver.target_strain = macro_strain;
    solver.strain_theta = 0.0;
    solver.uniaxial_strain = 1.1;
    solver.staticSolve();
    solver.saveToOBJ(result_folder + "temp.obj");
    TM sigma, epsilon;
    solver.computeHomogenizedStressStrain(sigma, epsilon);
    
    std::cout << "strain" << std::endl;
    std::cout << epsilon << std::endl;
    std::cout << "stress" << std::endl;
    std::cout << sigma << std::endl;
}

void Tiling2D::sampleDirectionWithUniaxialStrain(const std::string& result_folder,
        int n_sample, const TV& theta_range, T strain)
{
    int IH = 19;
    std::ofstream out(result_folder + "sample_theta_"+std::to_string(strain)+".txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
    }
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = true;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    auto runSim = [&](T theta, T strain, T strain_ortho)
    {
            
        bool solve_succeed = solver.staticSolve();

        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        for (int m = 0; m < params.size(); m++)
        {
            out << params[m] << " ";
        }
        out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " << psi << " " << theta << " " << strain 
            << " " << strain_ortho << " "
            << residual.norm() << std::endl;
        if (!solve_succeed)
        {
            solver.reset();
            // solver.saveToOBJ(result_folder + "_failure_theta_" + std::to_string(theta)
            //     +"_strain_" + std::to_string(strain) + "_strain_ortho_" + std::to_string(strain_ortho)+".obj");
        }
    };

    for (int i = 0; i < n_sample; i++)
    {
        solver.reset();
        T theta = theta_range[0] + ((double)i/(double)n_sample)*(theta_range[1] - theta_range[0]);
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain;
        runSim(theta, strain, 0.0);
    }
    
    out.close();
}

void Tiling2D::sampleUniAxialStrainAlongDirection(const std::string& result_folder,
        int n_sample, const TV& strain_range, T theta)
{
    int IH = 19;
    std::ofstream out(result_folder + "strain_stress.txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
    }
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = true;
    solver.prescribe_strain_tensor = false;
    T delta_strain = (strain_range[1] - strain_range[0]) / T(n_sample);

    auto runSim = [&](T theta, T strain, T strain_ortho)
    {
            
        bool solve_succeed = solver.staticSolve();

        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        for (int m = 0; m < params.size(); m++)
        {
            out << params[m] << " ";
        }
        out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " << psi << " " << theta << " " << strain 
            << " " << strain_ortho << " "
            << residual.norm() << std::endl;
        if (!solve_succeed)
        {
            solver.reset();
            solver.saveToOBJ(result_folder + "_failure_theta_" + std::to_string(theta)
                +"_strain_" + std::to_string(strain) + "_strain_ortho_" + std::to_string(strain_ortho)+".obj");
        }
    };
    solver.biaxial = false;
    for (T strain = 1.001; strain < strain_range[1]; strain += delta_strain)
    {   
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain;
        runSim(theta, strain, 0.0);
    }
    solver.reset();
    for (T strain = 1.001; strain > strain_range[0]; strain -= delta_strain)
    {    
        solver.uniaxial_strain = strain;
        runSim(theta, strain, 0.0);
    }
    out.close();
    
    
}

void Tiling2D::sampleFixedTilingParamsAlongStrain(const std::string& result_folder)
{
    int IH = 19;
    std::ofstream out(result_folder + "monotonic_test.txt");
    int n_sp_strain = 20;
    TV range_strain(0.2, 0.5);
    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    std::vector<T> params = {0.146, 0.655};
    
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    
    for (T strain = range_strain[0]; strain < range_strain[1] + delta_strain; strain += delta_strain)
    {
        solver.prescribe_strain_tensor = true;
        solver.target_strain = TV3(strain, 0.109, 0.001);
        bool solve_succeed = solver.staticSolve();
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        out << params[0] << " " << params[1] << " " << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " << psi << " " << residual.norm() << std::endl;
        if (!solve_succeed)
        {
            solver.reset();
        }
    }
    out.close();
}

void Tiling2D::sampleTilingParamsAlongStrain(const std::string& result_folder)
{
    int IH = 19;
    std::ofstream out(result_folder + "sample_tiling_vec_along_strain_compression.txt");
    int n_sp_params = 20;
    int n_sp_strain = 20;
    TV range_strain(-0.2, 0.05);
    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    TV tiling_range1(0.1, 0.2);
    TV tiling_range2(0.5, 0.8);
    
    TV init(0.17, 0.55);
    TV dir(-0.006, 0.02);
    
    T delta_T = (tiling_range1[1] - tiling_range1[0]) / T(n_sp_params);
    solver.verbose = false;
    // for (T ti = tiling_range1[0]; ti < tiling_range1[1] + delta_T; ti += delta_T)
    for (int i = 0; i < 10; i++)
    {
        std::vector<std::vector<TV2>> polygons;
        std::vector<TV2> pbc_corners; 
        Vector<T, 4> cubic_weights;
        cubic_weights << 0.25, 0, 0.75, 0;
        TV pi = init + T(i) * dir;
        // std::vector<T> params = {ti, 0.65};
        std::vector<T> params = {pi[0], pi[1]};
        fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
            cubic_weights, result_folder + "structure.txt");
        
        generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
        
        solver.pbc_translation_file = result_folder + "structure_translation.txt";
        
        bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
        if (!valid_structure)
            continue;
        
        // for (T strain = range_strain[0]; strain < range_strain[1] + delta_strain; strain += delta_strain)
        for (T strain = range_strain[1]; strain > range_strain[0] - delta_strain; strain -= delta_strain)
        {
            solver.prescribe_strain_tensor = true;
            solver.target_strain = TV3(strain, 0.06, 0.001);
            bool solve_succeed = solver.staticSolve();
            TM secondPK_stress, Green_strain;
            T psi;
            solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
            VectorXT residual(solver.num_nodes * 2); residual.setZero();
            solver.computeResidual(solver.u, residual);
            out << pi[0] << " " << pi[1] << " " << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
                << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
                << secondPK_stress(1, 0) << " " << psi << " " << residual.norm() << std::endl;
            if (!solve_succeed)
            {
                solver.reset();
            }
            // break;
        }
    }
    
    out.close();
}

void Tiling2D::generateGreenStrainSecondPKPairsServer(const std::vector<T>& params, 
    int IH, const std::string& prefix,
    const std::string& result_folder, int resume_start)
{
    std::ofstream out;
    if (resume_start == 0)
        out.open(result_folder + "data.txt");
    else
        out.open(result_folder + "data_resume.txt");

    TV range_strain(0.7, 1.5);
    TV range_strain_biaixial(0.9, 1.2);
	TV range_theta(0.0, M_PI);
    
    int n_sp_params = 10;
    int n_sp_strain = 20;
    int n_sp_strain_bi = 10;
    int n_sp_theta = 10;

    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    T delta_strain_bi = (range_strain_biaixial[1] - range_strain_biaixial[0]) / T(n_sp_strain_bi);

    auto runSim = [&](int& sim_cnt, T theta, T strain, T strain_ortho)
    {
        sim_cnt++;
        if (sim_cnt < resume_start)
            return;
        if (sim_cnt == resume_start)
            solver.reset();
            
        bool solve_succeed = solver.staticSolve();

        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        for (int m = 0; m < params.size(); m++)
        {
            out << params[m] << " ";
        }
        out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " << psi << " " << theta << " " << strain 
            << " " << strain_ortho << " "
            << residual.norm() << std::endl;
        if (!solve_succeed)
        {
            solver.reset();
            solver.saveToOBJ(result_folder + "_failure_theta_" + std::to_string(theta)
                +"_strain_" + std::to_string(strain) + "_strain_ortho_" + std::to_string(strain_ortho)+".obj");
        }
    };

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    return;
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    solver.verbose = false;
    int sim_cnt = 0;
    for(int l = 0; l < n_sp_theta; l++)
    {
        solver.biaxial = false;
        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
        // uniaxial tension
        solver.strain_theta = theta;
        solver.reset();
        for (T strain = 1.0; strain < range_strain[1]; strain += delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(sim_cnt, theta, strain, 0.0);
        }
        // uniaxial compression
        solver.reset();
        for (T strain = 1.0; strain > range_strain[0]; strain -= delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(sim_cnt, theta, strain, 0.0);
        }
        // biaxial tension
        // break;
        // solver.reset();
        solver.biaxial = true;
        for (T strain = 1.0; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.0; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 1.0; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
        }
        // solver.reset();
        for (T strain = 1.0; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.0; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 1.0; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
        }
        solver.biaxial = false;
    }
    
    out.close();
}

void Tiling2D::generateGreenStrainSecondPKPairs(const std::string& result_folder)
{
    std::ofstream out(result_folder + "training_data_IH07_new.txt");
    
    int IH = 6;
    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
        params_range[j] = TV(std::max(0.05, params[j] - 0.2), std::min(0.92, params[j] + 0.2));
    }
    int tiling_cnt = 0;
    TV range_strain(0.6, 2.0);
    TV range_strain_biaixial(0.8, 1.5);
	TV range_theta(0.0, M_PI);
    
    int n_sp_params = 10;
    int n_sp_strain = 20;
    int n_sp_strain_bi = 5;
    int n_sp_theta = 10;

    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    T delta_strain_bi = (range_strain_biaixial[1] - range_strain_biaixial[0]) / T(n_sp_strain_bi);

    auto runSim = [&](std::vector<T>& params_sp, T theta, T strain, T strain_ortho)
    {
        solver.staticSolve();

        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        for (int m = 0; m < num_params; m++)
        {
            out << params_sp[m] << " ";
        }
        out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " << psi << " " << theta << " " << strain 
            << " " << strain_ortho << " "
            << residual.norm() << std::endl;
        // solver.saveToOBJ(result_folder + "current.obj");
    };

    for (int sp = 0; sp < num_params; sp++) 
    {
        params_range[sp] = TV(std::max(0.05, params[sp] - 0.2), std::min(0.92, params[sp] + 0.2));
        for (int i = 0; i < n_sp_params; i++)
        {
            std::vector<T> params_sp = params;
            T pi = params_range[sp][0] + ((T)i/(T)n_sp_params)*(params_range[sp][1] - params_range[sp][0]);
            params_sp[sp] = pi;
            for (int sp2 = 0; sp2 < num_params; sp2++)
            {
                params_range[sp2] = TV(std::max(0.05, params[sp2] - 0.2), std::min(0.92, params[sp2] + 0.2));
                for (int j = 0; j < n_sp_params; j++)
                {
                    T pj = params_range[sp2][0] + ((T)j/(T)n_sp_params)*(params_range[sp2][1] - params_range[sp2][0]);
                    params_sp[sp2] = pj;
                    std::vector<std::vector<TV2>> polygons;
                    std::vector<TV2> pbc_corners; 
                    Vector<T, 4> cubic_weights;
                    cubic_weights << 0.25, 0, 0.75, 0;
                    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params_sp, 
                        cubic_weights, result_folder + "structure"+std::to_string(tiling_cnt)+".txt");
                    
                    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure" + std::to_string(tiling_cnt));
                    
                    solver.pbc_translation_file = result_folder + "structure"+std::to_string(tiling_cnt) +"_translation.txt";
                    initializeSimulationDataFromFiles(result_folder + "structure"+std::to_string(tiling_cnt)+".vtk", PBC_XY);
                    solver.verbose = true;
                    tiling_cnt++;
                    
                    
                    for(int l = 0; l < n_sp_theta; l++)
                    {
                        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
                        // uniaxial tension
                        solver.strain_theta = theta;
                        solver.reset();
                        for (T strain = 1.0; strain < range_strain[1]; strain += delta_strain)
                        {    
                            solver.uniaxial_strain = strain;
                            runSim(params_sp, theta, strain, 0.0);
                        }
                        // uniaxial compression
                        solver.reset();
                        for (T strain = 1.0; strain > range_strain[0]; strain -= delta_strain)
                        {    
                            solver.uniaxial_strain = strain;
                            runSim(params_sp, theta, strain, 0.0);
                        }
                        // biaxial tension
                        
                        solver.biaxial = true;
                        for (T strain = 1.0; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
                        {    
                            solver.reset();
                            solver.uniaxial_strain = strain;
                            for (T strain_ortho = 1.0; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
                            {
                                solver.uniaxial_strain_ortho = strain_ortho;
                                runSim(params_sp, theta, strain, strain_ortho);
                            }
                            solver.reset();
                            for (T strain_ortho = 1.0; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
                            {
                                solver.uniaxial_strain_ortho = strain_ortho;
                                runSim(params_sp, theta, strain, strain_ortho);
                            }
                        }
                        for (T strain = 1.0; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
                        {    
                            solver.reset();
                            solver.uniaxial_strain = strain;
                            for (T strain_ortho = 1.0; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
                            {
                                solver.uniaxial_strain_ortho = strain_ortho;
                                runSim(params_sp, theta, strain, strain_ortho);
                            }
                            solver.reset();
                            for (T strain_ortho = 1.0; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
                            {
                                solver.uniaxial_strain_ortho = strain_ortho;
                                runSim(params_sp, theta, strain, strain_ortho);
                            }
                        }
                        solver.biaxial = false;
                    }
                    break;
                }
                break;
            }
            break;
        }
        break;
    }
    
    out.close();
}

void Tiling2D::computeEnergyForSimData(const std::string& result_folder)
{
    std::ofstream out(result_folder + "training_data_with_energy.txt");
    
    int IH = 0;
    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
        params_range[j] = TV(std::max(0.05, params[j] - 0.2), std::min(0.92, params[j] + 0.2));
    }
    int tiling_cnt = 0;
    TV range_strain(0.5, 2.0);
	TV range_theta(0.0, M_PI);
    
    int n_sp_per_para = 10;
    int n_sp_strain = 100;
    int n_sp_theta = 100;
    for (int i = 0; i < num_params; i++)
    {
        for (int j = 0; j < n_sp_per_para; j++)
        {
            T pi = params_range[i][0] + ((T)j/(T)n_sp_per_para)*(params_range[i][1] - params_range[i][0]);
            std::vector<T> params_sp = params;
            // params_sp[i] = pi;
            std::vector<std::vector<TV2>> polygons;
            std::vector<TV2> pbc_corners; 
            Vector<T, 4> cubic_weights;
            cubic_weights << 0.25, 0, 0.75, 0;
            fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params_sp, 
                cubic_weights, result_folder + "structure"+std::to_string(tiling_cnt)+".txt");
            
            generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure" + std::to_string(tiling_cnt));
            
            solver.pbc_translation_file = result_folder + "structure"+std::to_string(tiling_cnt) +"_translation.txt";
            initializeSimulationDataFromFiles(result_folder + "structure"+std::to_string(tiling_cnt)+".vtk", PBC_XY);
            solver.verbose = true;
            tiling_cnt++;
            for(int k =0; k < n_sp_strain; k++)
            {
                T strain = range_strain[0] + ((double)k/(double)n_sp_strain)*(range_strain[1] - range_strain[0]);
                if (strain < 1.0)
                    continue;
                for(int l =0; l < n_sp_theta; l++)
                {
                    solver.reset();
                    T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
                    solver.strain_theta = theta;
                    solver.uniaxial_strain = strain;
                    solver.staticSolve();
                    

                    VectorXT residual(solver.num_nodes * 2); residual.setZero();
                    solver.computeResidual(solver.u, residual);
                    TM sigma, Cauchy_strain, Green_strain;
                    solver.computeHomogenizedStressStrain(sigma, Cauchy_strain, Green_strain);
                    for (int m = 0; m < num_params; m++)
                    {
                        out << params_sp[m] << " ";
                    }
                    T energy = solver.computeTotalEnergy(solver.u);
                    out << theta << " " << strain << " "
                        << Cauchy_strain(0, 0) << " "<< Cauchy_strain(0, 1) << " " 
                        << Cauchy_strain(1, 0) << " " << Cauchy_strain(1, 1) << " "
                        << Green_strain(0, 0) << " "<< Green_strain(0, 1) << " " 
                        << Green_strain(1, 0) << " " << Green_strain(1, 1) << " " 
                        << sigma(0, 0) << " "<< sigma(0, 1) << " " 
                        << sigma(1, 0) << " "<< sigma(1, 1) << " " << energy << " "
                         << residual.norm() << std::endl;
                    solver.saveToOBJ(result_folder + std::to_string(tiling_cnt-1) + "_strain_" + std::to_string(strain) + "theta_" + std::to_string(theta) + ".obj");
                }
            }
            break;
        }
        break;
    }
    out.close();
}

void Tiling2D::sampleUniaxialStrainSingleFamily(const std::string& result_folder, int IH)
{
    std::ofstream out(result_folder + "training_data.txt");
    
    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
        params_range[j] = TV(std::max(0.05, params[j] - 0.2), std::min(0.92, params[j] + 0.2));
    }
    int tiling_cnt = 0;
    // int num_data_points = 50;
    TV range_strain(0.5, 2.0);
	TV range_theta(0.0, M_PI);
    
    int n_sp_per_para = 10;
    int n_sp_strain = 50;
    int n_sp_theta = 50;
    for (int i = 0; i < num_params; i++)
    {
        for (int j = 0; j < n_sp_per_para; j++)
        {
            T pi = params_range[i][0] + ((T)j/(T)n_sp_per_para)*(params_range[i][1] - params_range[i][0]);
            std::vector<T> params_sp = params;
            params_sp[i] = pi;
            std::vector<std::vector<TV2>> polygons;
            std::vector<TV2> pbc_corners; 
            Vector<T, 4> cubic_weights;
            cubic_weights << 0.25, 0, 0.75, 0;
            fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params_sp, 
                cubic_weights, result_folder + "structure"+std::to_string(tiling_cnt)+".txt");
            
            generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure" + std::to_string(tiling_cnt));
            
            solver.pbc_translation_file = result_folder + "structure"+std::to_string(tiling_cnt) +"_translation.txt";
            initializeSimulationDataFromFiles(result_folder + "structure"+std::to_string(tiling_cnt)+".vtk", PBC_XY);
            solver.verbose = false;
            tiling_cnt++;
            for(int k =0; k < n_sp_strain; k++)
            {
                T strain = range_strain[0] + ((double)k/(double)n_sp_strain)*(range_strain[1] - range_strain[0]);

                for(int l =0; l < n_sp_theta; l++)
                {
                    solver.reset();
                    T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
                    solver.strain_theta = theta;
                    solver.uniaxial_strain = strain;
                    solver.staticSolve();
                    VectorXT residual(solver.num_nodes * 2); residual.setZero();
                    solver.computeResidual(solver.u, residual);
                    TM sigma, Cauchy_strain, Green_strain;
                    solver.computeHomogenizedStressStrain(sigma, Cauchy_strain, Green_strain);
                    for (int m = 0; m < num_params; m++)
                    {
                        out << params_sp[m] << " ";
                    }
                    
                    out << theta << " " << strain << " "
                        << Cauchy_strain(0, 0) << " "<< Cauchy_strain(0, 1) << " " 
                        << Cauchy_strain(1, 0) << " " << Cauchy_strain(1, 1) << " "
                        << Green_strain(0, 0) << " "<< Green_strain(0, 1) << " " 
                        << Green_strain(1, 0) << " " << Green_strain(1, 1) << " " 
                        << sigma(0, 0) << " "<< sigma(0, 1) << " " 
                        << sigma(1, 0) << " "<< sigma(1, 1) << " "
                         << residual.norm() << std::endl;
                    solver.saveToOBJ(result_folder + std::to_string(tiling_cnt-1) + "_strain_" + std::to_string(strain) + "theta_" + std::to_string(theta) + ".obj");
                }
            }
            break;
        }
        break;
    }
    out.close();
    //
}

void Tiling2D::sampleSingleFamily(const std::string& result_folder, 
        const TV& uniaxial_strain_range, const TV& biaxial_strain_range, 
        const TV& theta_range, int n_sp_params, int n_sp_uni, 
        int n_sp_bi, int n_sp_theta, int IH)
{
    std::ofstream out(result_folder + "training_data_p2.txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
        params_range[j] = TV(std::max(0.05, params[j] - 0.2), std::min(0.92, params[j] + 0.2));
    }
    int tiling_cnt = 0;
    for (int sp = 0; sp < num_params; sp++) 
    {
        params_range[sp] = TV(std::max(0.05, params[sp] - 0.2), std::min(0.92, params[sp] + 0.2));
        for (int i = 0; i < n_sp_params; i++)
        {
            std::vector<T> params_sp = params;
            T pi = params_range[sp][0] + ((T)i/(T)n_sp_params)*(params_range[sp][1] - params_range[sp][0]);
            params_sp[sp] = pi;
            for (int sp2 = 0; sp2 < num_params; sp2++)
            {
                params_range[sp2] = TV(std::max(0.05, params[sp2] - 0.2), std::min(0.92, params[sp2] + 0.2));
                for (int j = 0; j < n_sp_params; j++)
                {
                    T pj = params_range[sp2][0] + ((T)j/(T)n_sp_params)*(params_range[sp2][1] - params_range[sp2][0]);
                    params_sp[sp2] = pj;
                    
                    std::vector<std::vector<TV2>> polygons;
                    std::vector<TV2> pbc_corners; 
                    Vector<T, 4> cubic_weights;
                    cubic_weights << 0.25, 0, 0.75, 0;
                    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params_sp, 
                        cubic_weights, result_folder + "structure"+std::to_string(tiling_cnt)+".txt");
                    
                    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure" + std::to_string(tiling_cnt));
                    
                    solver.pbc_translation_file = result_folder + "structure"+std::to_string(tiling_cnt) +"_translation.txt";
                    initializeSimulationDataFromFiles(result_folder + "structure"+std::to_string(tiling_cnt)+".vtk", PBC_XY);
                    solver.verbose = false;
                    tiling_cnt++;

                    for(int k =0; k < n_sp_uni; k++)
                    {
                        T strain = uniaxial_strain_range[0] + ((T)k/(T)n_sp_uni)*(uniaxial_strain_range[1] - uniaxial_strain_range[0]);

                        for(int l =0; l < n_sp_theta; l++)
                        {
                            solver.reset();
                            T theta = theta_range[0] + ((T)l/(T)n_sp_theta)*(theta_range[1] - theta_range[0]);
                            solver.strain_theta = theta;
                            solver.uniaxial_strain = strain;
                            solver.staticSolve();
                            VectorXT residual(solver.num_nodes * 2); residual.setZero();
                            solver.computeResidual(solver.u, residual);
                            TM sigma, Cauchy_strain, Green_strain;
                            solver.computeHomogenizedStressStrain(sigma, Cauchy_strain, Green_strain);
                            for (int m = 0; m < num_params; m++)
                            {
                                out << params_sp[m] << " ";
                            }
                            
                            out << theta << " " << strain << " "
                                << Cauchy_strain(0, 0) << " "<< Cauchy_strain(0, 1) << " " 
                                << Cauchy_strain(1, 0) << " " << Cauchy_strain(1, 1) << " "
                                << Green_strain(0, 0) << " "<< Green_strain(0, 1) << " " 
                                << Green_strain(1, 0) << " " << Green_strain(1, 1) << " " 
                                << sigma(0, 0) << " "<< sigma(0, 1) << " " 
                                << sigma(1, 0) << " "<< sigma(1, 1) << " "
                                << residual.norm() << std::endl;
                            solver.saveToOBJ(result_folder + std::to_string(tiling_cnt-1) + "_strain_" + std::to_string(strain) + "theta_" + std::to_string(theta) + ".obj");
                        }
                    }
                }
            }
        }   
    }
    
    out.close();
}


void Tiling2D::sampleBiaxialStrainSingleFamily(const std::string& result_folder, int IH)
{
    std::ofstream out(result_folder + "training_data.txt");
    
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = new_params[j];
        params_range[j] = TV(std::max(0.05, params[j] - 0.2), std::min(0.92, params[j] + 0.2));
    }
    int tiling_cnt = 0;
    // int num_data_points = 50;
    TV range_strain(0.5, 2.0);
	TV range_theta(0.0, M_PI);
    
    int n_sp_per_para = 10;
    int n_sp_strain = 10;
    int n_sp_theta = 10;
    for (int i = 0; i < num_params; i++)
    {
        for (int j = 0; j < n_sp_per_para; j++)
        {
            T pi = params_range[i][0] + ((T)j/(T)n_sp_per_para)*(params_range[i][1] - params_range[i][0]);
            std::vector<T> params_sp = params;
            params_sp[i] = pi;
            std::vector<std::vector<TV2>> polygons;
            std::vector<TV2> pbc_corners; 
            Vector<T, 4> cubic_weights;
            cubic_weights << 0.25, 0, 0.75, 0;
            fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params_sp, 
                cubic_weights, result_folder + "structure"+std::to_string(tiling_cnt)+".txt");
            
            generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure" + std::to_string(tiling_cnt));
            
            solver.pbc_translation_file = result_folder + "structure"+std::to_string(tiling_cnt) +"_translation.txt";
            initializeSimulationDataFromFiles(result_folder + "structure"+std::to_string(tiling_cnt)+".vtk", PBC_XY);
            solver.verbose = false;
            tiling_cnt++;
            
            solver.biaxial = true;
            
            for(int k =0; k < n_sp_strain; k++)
            {
                T strain = range_strain[0] + ((double)k/(double)n_sp_strain)*(range_strain[1] - range_strain[0]);
                for(int k2 =0; k2 < n_sp_strain; k2++)
                {
                    T strain_ortho = range_strain[0] + ((double)k2/(double)n_sp_strain)*(range_strain[1] - range_strain[0]);
                    solver.uniaxial_strain_ortho = strain_ortho;
                    for(int l =0; l < n_sp_theta; l++)
                    {
                        solver.reset();
                        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
                        solver.strain_theta = theta;
                        solver.uniaxial_strain = strain;
                        solver.staticSolve();
                        VectorXT residual(solver.num_nodes * 2); residual.setZero();
                        solver.computeResidual(solver.u, residual);
                        TM sigma, Cauchy_strain, Green_strain;
                        solver.computeHomogenizedStressStrain(sigma, Cauchy_strain, Green_strain);
                        for (int m = 0; m < num_params; m++)
                        {
                            out << params_sp[m] << " ";
                        }
                        
                        out << theta << " " << strain << " " << strain_ortho << " "
                            << Cauchy_strain(0, 0) << " "<< Cauchy_strain(0, 1) << " " 
                            << Cauchy_strain(1, 0) << " " << Cauchy_strain(1, 1) << " "
                            << Green_strain(0, 0) << " "<< Green_strain(0, 1) << " " 
                            << Green_strain(1, 0) << " " << Green_strain(1, 1) << " " 
                            << sigma(0, 0) << " "<< sigma(0, 1) << " " 
                            << sigma(1, 0) << " "<< sigma(1, 1) << " "
                            << residual.norm() << std::endl;
                        solver.saveToOBJ(result_folder + std::to_string(tiling_cnt-1) + "_strain_" + std::to_string(strain) + "_strain_ortho_" + std::to_string(strain_ortho) + "_theta_" + std::to_string(theta) + ".obj");
                    }
                }
            }

            break;  
        }
        break;
    }
    
    //
}

void Tiling2D::sampleUniaxialStrainSingleStructure(const std::string& result_folder)
{
    std::ofstream out(result_folder + "training_data.txt");
    TV range_strain(0.5, 2.0);
	TV range_theta(0.0, M_PI);
    solver.verbose = false;
    int num_data_points = 60;
    for(int i=0; i<num_data_points; ++i)
    {
        T strain = range_strain[0] + ((double)i/(double)num_data_points)*(range_strain[1] - range_strain[0]);

        for(int j=0; j<num_data_points; ++j)
        {
            solver.reset();
            T theta = range_theta[0] + ((double)j/(double)num_data_points)*(range_theta[1] - range_theta[0]);
            solver.strain_theta = theta;
            solver.uniaxial_strain = strain;
            solver.staticSolve();
            VectorXT residual(solver.num_nodes * 2); residual.setZero();
            solver.computeResidual(solver.u, residual);
            TM sigma, epsilon;
            solver.computeHomogenizedStressStrain(sigma, epsilon);
            out << epsilon(0, 0) << " "<< epsilon(0, 1) << " " << epsilon(1, 0) << " " << epsilon(1, 1) << " " 
                << sigma(0, 0) << " "<< sigma(0, 1) << " "<< sigma(1, 0) << " "<< sigma(1, 1) << " " << residual.norm() << std::endl;
            solver.saveToOBJ(result_folder + "strain_" + std::to_string(strain) + "theta_" + std::to_string(theta) + ".obj");
        }
    }
    out.close();
}