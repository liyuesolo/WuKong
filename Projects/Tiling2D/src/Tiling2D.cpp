#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <igl/jet.h>
#include<cstdlib>
#include "../include/Tiling2D.h"
#include "../include/PoissonDisk.h"
#include "../include/Timer.h"

inline bool exists_test0 (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void Tiling2D::generateSurfaceMeshFromVTKFile(const std::string& vtk_file, const std::string surface_mesh_file)
{
    Eigen::MatrixXd V; Eigen::MatrixXi F;
}

/*
Triangle:               Triangle6:         

v
^                                                                   
|                                           
2                       2                   
|`\                     |`\                  
|  `\                   |  `\                
|    `\                 5    `4              
|      `\               |      `\            
|        `\             |        `\          
0----------1 --> u      0-----3----1         

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
    solver.thickness = 1.0;
    
    solver.scale = 50;
    solver.deformed *= solver.scale;
    solver.undeformed *= solver.scale; // use milimeters
    solver.thickness = solver.scale;
    solver.computeBoundingBox(min_corner, max_corner);
    if (solver.verbose)
        std::cout << "BBOX " << min_corner.transpose() << " " << max_corner.transpose() << std::endl;
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
    
    // int n_surface_triangles = n_ele;
    // VectorXT areas(n_surface_triangles);
    // tbb::parallel_for(0, n_surface_triangles, [&](int i)
    // {
    //     IV3 tri_idx = solver.surface_indices.segment<3>(i * 3);
    //     TV v0 = solver.deformed.segment<2>(tri_idx[0] * 2);
    //     TV v1 = solver.deformed.segment<2>(tri_idx[1] * 2);
    //     TV v2 = solver.deformed.segment<2>(tri_idx[2] * 2);
    //     TV e0 = v1 - v0;
    //     TV e1 = v2 - v0;
    //     areas[i] = 0.5 * TV3(e0[0], e0[1], 0.0).cross(TV3(e1[0], e1[1], 0.0))[2];
    // });
    // T area = areas.sum();

    // solver.deformed /= std::sqrt(area);
    // solver.undeformed /= std::sqrt(area);

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
            bool valid_structure = solver.addPBCPairsXY();
            if (!valid_structure)
                return false;
            solver.add_pbc_strain = true;
            solver.strain_theta = 0.4469340574;
            solver.uniaxial_strain = 1.2;
            // solver.strain_theta = 0.5 * M_PI;
            // solver.uniaxial_strain = 1.2;
            // solver.uniaxial_strain_ortho = 0.9;
            solver.biaxial = false;
            // solver.pbc_strain_w = 1e7; //IH01
            solver.pbc_strain_w = 1e6; //IH
            solver.pbc_w = 1e6;
            solver.prescribe_strain_tensor = false;
            solver.target_strain = TV3(-0.0432069, 0.0512492, 2.0* -9.5e-10);
            // solver.target_strain = TV3(0.44765, -0.0656891, 0.0956651);
            
            // solver.computeMarcoBoundaryIndices();
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
    if (pbc_type == PBC_XY)
    {
        int n_pbc_pairs = solver.pbc_pairs[0].size() + solver.pbc_pairs[1].size();
        // std::cout << solver.pbc_pairs[0].size() << " " << solver.pbc_pairs[1].size() << std::endl;
        if (solver.verbose)
            std::cout << "pbc_pairs size " << n_pbc_pairs << std::endl;
        if (n_pbc_pairs < 4)
            return false;
    }
    
    solver.penalty_weight = 1e6;

    T total_area = solver.computeTotalArea();
    T bbox_area = (max_corner[0] - min_corner[0]) * (max_corner[1] - min_corner[1]);
    
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
            std::cout << "contact force norm " << contact_force.norm() << std::endl;
            return false;
        }
    }

    solver.project_block_PD = true;
    solver.verbose = false;
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

    Eigen::MatrixXd V_tile(V.rows() * n_unit, 3);
    Eigen::MatrixXi F_tile(F.rows() * n_unit, 3);
    Eigen::MatrixXd C_tile(F.rows() * n_unit, 3);

    TV left0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][0] * 2);
    TV right0 = solver.deformed.segment<2>(solver.pbc_pairs[0][0][1] * 2);
    TV top0 = solver.deformed.segment<2>(solver.pbc_pairs[1][0][1] * 2);
    TV bottom0 = solver.deformed.segment<2>(solver.pbc_pairs[1][0][0] * 2);
    TV dx = (right0 - left0);
    TV dy = (top0 - bottom0);

    int n_unit_dir = std::sqrt(n_unit);

    int n_face = F.rows(), n_vtx = V.rows();
    for (int i = 0; i < n_unit; i++)
        V_tile.block(i * n_vtx, 0, n_vtx, 3) = V;
    
    int start = (n_unit_dir - 1) / 2;
    int cnt = 0;
    for (int left = -start; left < start + 1; left++)
    {
        for (int bottom = -start; bottom < start + 1; bottom++)
        {
            tbb::parallel_for(0, n_vtx, [&](int i){
                V_tile.row(cnt * n_vtx + i).head<2>() += T(left) * dx + T(bottom) * dy;
            });
            cnt++;
        }
    }

    // tbb::parallel_for(0, n_vtx, [&](int i){
    //     V_tile.row(n_vtx + i).head<2>() += dx;
    //     V_tile.row(3 * n_vtx + i).head<2>() += dx;
    //     V_tile.row(2 * n_vtx + i).head<2>() += dy;
    //     V_tile.row(3 * n_vtx + i).head<2>() += dy;
    //     V_tile.row(4 * n_vtx + i).head<2>() += dx;
    //     V_tile.row(4 * n_vtx + i).head<2>() -= dy;
    //     V_tile.row(5 * n_vtx + i).head<2>() -= dx;
    //     V_tile.row(5 * n_vtx + i).head<2>() -= dy;
    //     V_tile.row(6 * n_vtx + i).head<2>() -= dx;
    //     V_tile.row(7 * n_vtx + i).head<2>() -= dy;
    //     V_tile.row(8 * n_vtx + i).head<2>() -= dx;
    //     V_tile.row(8 * n_vtx + i).head<2>() += dy;
    // });
    

    V = V_tile;
    Eigen::MatrixXi offset(n_face, 3);
    offset.setConstant(n_vtx);

    for (int i = 0; i < n_unit; i++)
        F_tile.block(i * n_face, 0, n_face, 3) = F + i * offset;
    
    F = F_tile;

    Eigen::MatrixXd C_unit = C;
    C_unit.col(2).setConstant(0.3); C_unit.col(1).setConstant(1.0);
    C_tile.block(0, 0, n_face, 3) = C;
    for (int i = 1; i < n_unit; i++)
        C_tile.block(i * n_face, 0, n_face, 3) = C;
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
    std::ofstream out(result_folder + "sample_theta_"+std::to_string(strain)+"_full.txt");
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
    params[0] = 0.25; params[1] = 0.6;
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    // generateHomogenousMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = true;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    solver.pbc_strain_w = 1e6;
    solver.project_block_PD = false;
    auto runSim = [&](T theta, T strain, T strain_ortho)
    {
        TV d(std::cos(theta), std::sin(theta));
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain;
        bool solve_succeed = solver.staticSolve();
        
        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        TM cauchy_stress, cauchy_strain;
        solver.computeHomogenizationDataCauchy(cauchy_stress, cauchy_strain, psi);
        T strain_d = d.transpose() * (cauchy_strain * d);
        T stress_d = d.transpose() * (cauchy_stress * d);
        
        T stiffness = stress_d / strain_d;
        // T stiffness2 = 2.0 * psi / (strain - 1.0) / (strain - 1.0);
        // T stiffness2 = 2.0 * psi / strain_d/strain_d;
        
        for (int m = 0; m < params.size(); m++)
        {
            out << params[m] << " ";
        }
        out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            << secondPK_stress(1, 0) << " " 
            << cauchy_strain(0, 0) << " " << cauchy_strain(1, 1) << " " << cauchy_strain(0, 1) << " "
            << cauchy_stress(0, 0) << " " << cauchy_stress(1, 1) << " " << cauchy_stress(0, 1) << " "
            << stiffness << " "
            << psi << " " << theta << " " << strain 
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
        
        runSim(theta, strain, 0.0);
    }
    
    out.close();
}

void Tiling2D::generateTenPointUniaxialStrainData(const std::string& result_folder,
        int IH, T theta, const TV& strain_range, T strain_delta, const std::vector<T>& params)
{
    std::ofstream out(result_folder + "strain_stress.txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);

    int zero_strain_idx = 0;
    std::vector<T> strain_samples;
    for (T strain = strain_range[0]; strain < strain_range[1]; strain += strain_delta)
    {
        strain_samples.push_back(strain);
        if (std::abs(strain) < 1e-6)
            continue;
        zero_strain_idx++;       
    }
            
    auto runSim = [&](T theta, T strain, T strain_ortho, int idx)
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
        }
        solver.saveToOBJ(result_folder + std::to_string(idx)+".obj");
    };

    solver.verbose = false;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    for (int i = zero_strain_idx; i < strain_samples.size(); i++)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        runSim(theta, strain_samples[i], 0.0, i);
    }
    solver.reset();
    for (int i = zero_strain_idx; i > -1; i--)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        runSim(theta, strain_samples[i], 0.0, i);
    }
    out.close();
}

void Tiling2D::runSimUniAxialStrainAlongDirection(const std::string& result_folder,
        int IH, int n_sample, const TV& strain_range, T theta, const std::vector<T>& params)
{
    std::ofstream out(result_folder + "strain_stress.txt");
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = false;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    int zero_strain_idx = 0;
    std::vector<T> strain_samples;
    // T strain_delta = (strain_range[1] - strain_range[0]) / T(n_sample);
    T strain_delta = 0.01;
    for (T strain = strain_range[0]; strain < strain_range[1]; strain += strain_delta)
    {
        strain_samples.push_back(strain);
        if ((strain - 1.0) > 1e-6)
            continue;
        zero_strain_idx++;       
    }
    std::cout << "zero strain index " << zero_strain_idx << std::endl;

    auto runSim = [&](T theta, T strain, T strain_ortho, int idx)
    {
        bool solve_succeed = solver.staticSolve();
        solver.saveToOBJ(result_folder + std::to_string(idx)+"_"+std::to_string(strain_samples[idx])+".obj");
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
    };

    for (int i = zero_strain_idx; i < strain_samples.size(); i++)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        runSim(theta, strain_samples[i], 0.0, i);
    }
    solver.reset();
    for (int i = zero_strain_idx; i > -1; i--)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        runSim(theta, strain_samples[i], 0.0, i);
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
    solver.verbose = false;
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
    std::ofstream out(result_folder + "constant_energy.txt");
    int n_sp_strain = 50;
    TV range_strain(0.001, 0.2);
    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    std::vector<T> params = {0.175, 0.582};
    
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    
    for (T strain = range_strain[0]; strain < range_strain[1] + delta_strain; strain += delta_strain)
    {
        solver.prescribe_strain_tensor = true;
        solver.target_strain = TV3(strain, -0.2, 0.001);
        bool solve_succeed = solver.staticSolve();
        solver.saveToOBJ(result_folder + std::to_string(strain) + ".obj");
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

void Tiling2D::generateNHHomogenousData(const std::string& result_folder)
{
    std::ofstream out;
    out.open(result_folder + "homo_uni_bi.txt");
    
    TV range_strain(0.7, 1.5);
    TV range_strain_biaixial(0.9, 1.2);
	TV range_theta(0.0, M_PI);
    
    int n_sp_params = 10;
    int n_sp_strain = 50;
    int n_sp_strain_bi = 10;
    int n_sp_theta = 50;

    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    T delta_strain_bi = (range_strain_biaixial[1] - range_strain_biaixial[0]) / T(n_sp_strain_bi);

    auto runSim = [&](int& sim_cnt, T theta, T strain, T strain_ortho)
    {
        // std::cout << "###### theta " << theta << " #####" << std::endl;
        sim_cnt++;
        
        bool solve_succeed = solver.staticSolve();

        VectorXT residual(solver.num_nodes * 2); residual.setZero();
        solver.computeResidual(solver.u, residual);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        
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
    
    generateHomogenousMesh(polygons, pbc_corners, true, result_folder + "structure");
    
    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    solver.verbose = false;
    solver.prescribe_strain_tensor = false;
    int sim_cnt = 0;
    for(int l = 0; l < n_sp_theta; l++)
    {
        solver.biaxial = false;
        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
        // uniaxial tension
        solver.strain_theta = theta;
        solver.reset();
        for (T strain = 1.0 + delta_strain; strain < range_strain[1]; strain += delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(sim_cnt, theta, strain, 0.0);
            // break;
        }
        // uniaxial compression
        solver.reset();
        for (T strain = 1.0 - delta_strain; strain > range_strain[0]; strain -= delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(sim_cnt, theta, strain, 0.0);
            // break;
        }
        // biaxial tension
        // continue;
        // solver.reset();
        solver.biaxial = true;
        for (T strain = 1.0 + delta_strain; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.0 + delta_strain; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 1.0 - delta_strain; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
        }
        // solver.reset();
        for (T strain = 1.0 - delta_strain; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.0 + delta_strain; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 1.0 - delta_strain; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(sim_cnt, theta, strain, strain_ortho);
            }
        }
        solver.biaxial = false;
    }
    
    out.close();
}

void Tiling2D::sampleSingleStructurePoissonDisk(const std::string& result_folder, 
        const TV& uniaxial_strain_range, const TV& biaxial_strain_range, 
        const TV& theta_range, int n_sample_total, int IH)
{
    std::vector<T> params = {0.15, 0.6};
    std::ofstream out;
    out.open(result_folder + "data_poisson_disk.txt");
    PoissonDisk pd;
    Vector<T, 4> min_corner; 
    min_corner << biaxial_strain_range[0], biaxial_strain_range[0], 
                    uniaxial_strain_range[0], theta_range[0];
    Vector<T, 4> max_corner; 
    max_corner << biaxial_strain_range[1], biaxial_strain_range[1],
                    uniaxial_strain_range[1], theta_range[1];

    VectorXT samples;
    pd.sampleNDBox<4>(min_corner, max_corner, n_sample_total, samples);

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");

    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    
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
        out << std::setprecision(20) << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
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

    solver.verbose = false;
    solver.biaxial = false;

    for (int i = 0; i < n_sample_total; i++)
    {
        solver.uniaxial_strain = samples[i * 4 + 2];
        runSim(samples[i * 4 + 3], samples[i * 4 + 2], 0.0);
    }
    solver.biaxial = true;
    for (int i = 0; i < n_sample_total; i++)
    {
        solver.uniaxial_strain = samples[i * 4 + 0];
        solver.uniaxial_strain_ortho = samples[i * 4 + 1];
        runSim(samples[i * 4 + 3], samples[i * 4 + 0], samples[i * 4 + 1]);
    }
}

void Tiling2D::generateGreenStrainSecondPKPairsServerToyExample(const std::vector<T>& params,
        const std::string& result_folder)
{
    std::ofstream out;
    out.open(result_folder + "data.txt");
    TV range_strain(0.7, 1.5);
    TV range_strain_biaixial(0.9, 1.2);
	TV range_theta(0.0, M_PI);
    
    int n_sp_params = 10;
    int n_sp_strain = 50;
    int n_sp_strain_bi = 10;
    int n_sp_theta = 15;

    T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
    T delta_strain_bi = (range_strain_biaixial[1] - range_strain_biaixial[0]) / T(n_sp_strain_bi);

    auto runSim = [&](int& sim_cnt, T theta, T strain, T strain_ortho)
    {
        sim_cnt++;
            
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
        out << std::setprecision(16) << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
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

    generateToyExampleStructure(params, result_folder);
    return;
    // solver.pbc_translation_file = result_folder + "structure_translation.txt";
    
    bool valid_structure = initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    solver.verbose = false;
    int cnt = 0;
    for(int l = 0; l < n_sp_theta; l++)
    {
        solver.biaxial = false;
        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
        // uniaxial tension
        solver.strain_theta = theta;
        solver.reset();
        for (T strain = 1.001; strain < range_strain[1]; strain += delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(cnt, theta, strain, 0.0);
        }
        // uniaxial compression
        solver.reset();
        for (T strain = 0.999; strain > range_strain[0]; strain -= delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(cnt, theta, strain, 0.0);
        }
        // biaxial tension
        // break;
        // solver.reset();
        solver.biaxial = true;
        for (T strain = 1.001; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
        }
        // solver.reset();
        for (T strain = 0.999; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
        }
        solver.biaxial = false;
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
    int n_sp_strain = 50;
    int n_sp_strain_bi = 10;
    int n_sp_theta = 15;

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
        out << std::setprecision(16) << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
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
    int cnt = 0;
    for(int l = 0; l < n_sp_theta; l++)
    {
        solver.biaxial = false;
        T theta = range_theta[0] + ((double)l/(double)n_sp_theta)*(range_theta[1] - range_theta[0]);
        // uniaxial tension
        solver.strain_theta = theta;
        solver.reset();
        for (T strain = 1.001; strain < range_strain[1]; strain += delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(cnt, theta, strain, 0.0);
        }
        // uniaxial compression
        solver.reset();
        for (T strain = 0.999; strain > range_strain[0]; strain -= delta_strain)
        {    
            solver.uniaxial_strain = strain;
            runSim(cnt, theta, strain, 0.0);
        }
        // biaxial tension
        // break;
        // solver.reset();
        solver.biaxial = true;
        for (T strain = 1.001; strain < range_strain_biaixial[1]; strain += delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
        }
        // solver.reset();
        for (T strain = 0.999; strain > range_strain_biaixial[0]; strain -= delta_strain_bi)
        {   
            solver.reset(); 
            solver.uniaxial_strain = strain;
            for (T strain_ortho = 1.001; strain_ortho < range_strain_biaixial[1]; strain_ortho += delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
            }
            solver.reset();
            for (T strain_ortho = 0.999; strain_ortho > range_strain_biaixial[0]; strain_ortho -= delta_strain_bi)
            {
                solver.uniaxial_strain_ortho = strain_ortho;
                runSim(cnt, theta, strain, strain_ortho);
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

void Tiling2D::getTilingConfig(int IH, int& n_tiling_params, int& actual_IH, 
    std::vector<TV>& bounds, VectorXT& ti_default, int& unit)
{
    actual_IH = -1;
    if (IH == 21)
    {   
        n_tiling_params = 2;
        actual_IH = 19;
        bounds.push_back(TV(0.105, 0.195));
        bounds.push_back(TV(0.505, 0.795));
        ti_default = TV(0.104512, 0.65);
        unit = 5;
    }
    else if (IH == 50)
    {
        n_tiling_params = 2;
        actual_IH = 46;
        bounds.push_back(TV(0.1, 0.3));
        bounds.push_back(TV(0.25, 0.75));
        ti_default = TV(00.23076, 0.55);
        unit = 5;
    }
    else if (IH == 67)
    {
        n_tiling_params = 2;
        actual_IH = 60;
        bounds.push_back(TV(0.1, 0.3));
        bounds.push_back(TV(0.6, 1.1));
        ti_default = TV(0.2308, 0.8696);
        unit = 5;
    }
    else if (IH == 1)
    {
        n_tiling_params = 4;
        actual_IH = 0;
        bounds.push_back(TV(0.05, 0.3));
        bounds.push_back(TV(0.25, 0.75));
        bounds.push_back(TV(0.05, 0.15));
        bounds.push_back(TV(0.4, 0.8));
        ti_default = Vector<T, 4>::Zero();
        ti_default << 0.1224, 0.5, 0.1434, 0.625;
        
        unit = 10;
    }
    else if (IH == 22)
    {
        n_tiling_params = 3;
        actual_IH = 20;
        bounds.push_back(TV(0.1, 0.3));
        bounds.push_back(TV(0.3, 0.7));
        bounds.push_back(TV(0.0, 0.3));
        ti_default = TV3(0.2308, 0.5, 0.2253);
        unit = 5;
    }
    else if (IH == 28)
    {
        std::cout << "here" << std::endl;
        n_tiling_params = 2;
        actual_IH = 26;
        bounds.push_back(TV(0.005, 0.8));
        bounds.push_back(TV(0.005, 1.0));
        ti_default = TV(0.4528, 0.5);
        unit = 10;
    }
    else if (IH == 29)
    {
        n_tiling_params = 1;
        actual_IH = 27;
        bounds.push_back(TV(0.005, 1.0));
        ti_default = Vector<T, 1>(0.3669);
        unit = 10;
    }
}

void Tiling2D::minMaxCornerFromBounds(const std::vector<TV>& bounds, VectorXT& min_corner, VectorXT& max_corner)
{
    min_corner.resize(bounds.size()); max_corner.resize(bounds.size());
    for (int i = 0; i < bounds.size(); i++)
    {
        min_corner[i] = bounds[i][0]; max_corner[i] = bounds[i][1];
    }
    
}

void Tiling2D::generateStrainStressSimulationDataFromFile(const std::string& result_folder, const std::string& filename, 
     const std::string& suffix, int IH, int n_samples)
{
    std::string data_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress_"+suffix+".txt";
    std::ofstream out(data_file);
    out << std::setprecision(10);
    std::ifstream in(filename);
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);
    VectorXT samples(n_samples * n_tiling_params);
    VectorXT thetas(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_tiling_params; j++)
            in >> samples[i * n_tiling_params + j];
        in >> thetas[i];
    }

    std::cout << " generating all samples" << std::endl;

    for (int i = 0; i < n_samples; i++)
    {
        std::vector<std::vector<TV2>> polygons;
        std::vector<TV2> pbc_corners; 
        csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
        std::vector<T> params(n_tiling_params);
        for (int j = 0; j < n_tiling_params; j++)
        {
            params[j] = samples[i * n_tiling_params + j];
            // out << params[j] << " ";
        }
        Vector<T, 4> cubic_weights;
        cubic_weights << 0.25, 0., 0.75, 0.;
        std::cout << "generating tiling mesh " << i << std::endl;

        fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
            cubic_weights, "./a_structure.txt", unit);
        
        generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");
         
        solver.pbc_translation_file =  "./a_structure_translation.txt";
        initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
        if (IH == 1)
            solver.pbc_strain_w = 1e7;

        solver.max_newton_iter = 500;
        solver.project_block_PD = true;
        solver.verbose = false;
        TV strain_range = TV(0.7, 1.5);

        T theta = thetas[i];

        int zero_strain_idx = 0;
        std::vector<T> strain_samples;
        // T strain_delta = (strain_range[1] - strain_range[0]) / T(n_sample);
        T strain_delta = 0.01;
        for (T strain = strain_range[0]; strain < strain_range[1]; strain += strain_delta)
        {
            strain_samples.push_back(strain);
            if ((strain - 1.0) > strain_delta - 1e-6)
                continue;
            zero_strain_idx++;       
        }
        zero_strain_idx -= 1;

        auto runSim = [&](T _theta, T strain, T strain_ortho, int _fail_cnt) -> bool
        {
            if (_fail_cnt < 3)
            {
                // solver.reset();
                bool solve_succeed = solver.staticSolve();
                // solver.checkHessianPD(false);
                // solver.saveToOBJ(result_folder + std::to_string(idx)+"_"+std::to_string(strain_samples[idx])+".obj");
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
                    << secondPK_stress(1, 0) << " " << psi << " " << _theta << " " << strain 
                    << " " << strain_ortho << " "
                    << residual.norm() << std::endl;
                return solve_succeed;
            }
            else
            {
                for (int m = 0; m < params.size(); m++)
                {
                    out << params[m] << " ";
                }
                out << "-1" << " "<< "-1" << " " << "-1"
                    << " " << "-1" << " " << "-1" << " "
                    << "-1" << " " << "-1" << " " << _theta << " " << strain 
                    << " " << strain_ortho << " "
                    << 1e10 << std::endl;
                return false;
            }
        };
        std::cout << zero_strain_idx << " " << strain_samples[zero_strain_idx] << std::endl;

        int fail_cnt = 0;
        for (int i = zero_strain_idx; i < strain_samples.size(); i++)
        {
            // if (std::abs(strain_samples[i] - 1.2) > 1e-6)
            //     continue;
            solver.strain_theta = theta;
            solver.uniaxial_strain = strain_samples[i];
            bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt);
            if (!solve_succeed) 
                fail_cnt ++;
            else
                fail_cnt = 0;
        }
        // std::exit(0);
        solver.reset(); fail_cnt = 0;
        for (int i = zero_strain_idx-1; i > -1; i--)
        {
            solver.strain_theta = theta;
            solver.uniaxial_strain = strain_samples[i];
            bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt);
            if (!solve_succeed) 
                fail_cnt ++;
            else
                fail_cnt = 0;
        }

        // T delta_strain = (strain_range[1] - strain_range[0]) / T(50);
        // solver.strain_theta = theta;
        // solver.reset();
        // for (T strain = 1.001; strain < strain_range[1]; strain += delta_strain)
        // {    
        //     solver.uniaxial_strain = strain;
        //     runSim(theta, strain, 0.0);
        // }
        // // uniaxial compression
        // solver.reset();
        // for (T strain = 0.999; strain > strain_range[0]; strain -= delta_strain)
        // {    
        //     solver.uniaxial_strain = strain;
        //     runSim(theta, strain, 0.0);
        // }
    }
    out.close();
}

void Tiling2D::generateStrainStressSimulationData(const std::string& result_folder, int IH, int n_samples)
{
    std::srand((unsigned) time(NULL));
    std::string data_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress.txt";
    std::ofstream out(data_file);
    out << std::setprecision(10);
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);
    
    VectorXT samples(n_samples * n_tiling_params);
    VectorXT thetas(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        VectorXT sample(n_samples);
        for (int j = 0; j < n_tiling_params; j++)
        {
            sample[j] = bounds[j][0] + (float) rand() / RAND_MAX * (bounds[j][1] - bounds[j][0]);
            out << sample[j] << " ";
        }
        thetas[i] = (T) rand() / RAND_MAX * M_PI;
        out << thetas[i] << std::endl;
        samples.segment(i * n_tiling_params, n_tiling_params) = sample;
        
    }
    // VectorXT samples(2);
    // VectorXT thetas(1);
    // samples << 0.1, 0.6;
    // thetas << 0.0;
    // out << samples[0] << " " << samples[1] << " ";
    // out << thetas[0] << std::endl;
    solver.max_newton_iter = 500;

    std::cout << " generating all samples" << std::endl;

    for (int i = 0; i < n_samples; i++)
    {
        std::vector<std::vector<TV2>> polygons;
        std::vector<TV2> pbc_corners; 
        csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
        std::vector<T> params(n_tiling_params);
        for (int j = 0; j < n_tiling_params; j++)
        {
            params[j] = samples[i * n_tiling_params + j];
            // out << params[j] << " ";
        }
        Vector<T, 4> cubic_weights;
        cubic_weights << 0.25, 0., 0.75, 0.;
        std::cout << "generating tiling mesh " << i << std::endl;

        fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
            cubic_weights, "./a_structure.txt", unit);
        
        generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");
         
        solver.pbc_translation_file =  "./a_structure_translation.txt";
        bool valid_structure = initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
        if (!valid_structure)
            continue;
        if (IH == 1)
        {
            solver.pbc_strain_w = 1e7;
        }
        solver.max_newton_iter = 500;
        solver.project_block_PD = true;
        if (IH == 22 || IH == 28)
            solver.project_block_PD = false;
        solver.verbose = false;

        TV strain_range = TV(0.7, 1.5);

        T theta = thetas[i];

        int zero_strain_idx = 0;
        std::vector<T> strain_samples;
        // T strain_delta = (strain_range[1] - strain_range[0]) / T(n_sample);
        T strain_delta = 0.01;
        for (T strain = strain_range[0]; strain < strain_range[1]; strain += strain_delta)
        {
            strain_samples.push_back(strain);
            if ((strain - 1.0) > strain_delta - 1e-6)
                continue;
            zero_strain_idx++;       
        }
        zero_strain_idx -= 1;
        strain_samples[zero_strain_idx] += 1e-3;

        auto runSim = [&](T _theta, T strain, T strain_ortho, int _fail_cnt) -> bool
        {
            if (_fail_cnt < 3)
            {
                bool solve_succeed = solver.staticSolve();
                // solver.saveToOBJ(result_folder + std::to_string(idx)+"_"+std::to_string(strain_samples[idx])+".obj");
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
                    << secondPK_stress(1, 0) << " " << psi << " " << _theta << " " << strain 
                    << " " << strain_ortho << " "
                    << residual.norm() << std::endl;
                return solve_succeed;
            }
            else
            {
                for (int m = 0; m < params.size(); m++)
                {
                    out << params[m] << " ";
                }
                out << "-1" << " "<< "-1" << " " << "-1"
                    << " " << "-1" << " " << "-1" << " "
                    << "-1" << " " << "-1" << " " << _theta << " " << strain 
                    << " " << strain_ortho << " "
                    << 1e10 << std::endl;
                return false;
            }
        };

        int fail_cnt = 0;
        for (int i = zero_strain_idx; i < strain_samples.size(); i++)
        {
            solver.strain_theta = theta;
            solver.uniaxial_strain = strain_samples[i];
            bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt);
            if (!solve_succeed) 
                fail_cnt ++;
            else
                fail_cnt = 0;
        }
        solver.reset(); fail_cnt = 0;
        for (int i = zero_strain_idx-1; i > -1; i--)
        {
            solver.strain_theta = theta;
            solver.uniaxial_strain = strain_samples[i];
            bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt);
            if (!solve_succeed) 
                fail_cnt ++;
            else
                fail_cnt = 0;
        }

        // T delta_strain = (strain_range[1] - strain_range[0]) / T(50);
        // solver.strain_theta = theta;
        // solver.reset();
        // for (T strain = 1.001; strain < strain_range[1]; strain += delta_strain)
        // {    
        //     solver.uniaxial_strain = strain;
        //     runSim(theta, strain, 0.0);
        // }
        // // uniaxial compression
        // solver.reset();
        // for (T strain = 0.999; strain > strain_range[0]; strain -= delta_strain)
        // {    
        //     solver.uniaxial_strain = strain;
        //     runSim(theta, strain, 0.0);
        // }
    }
    out.close();
    
}

void Tiling2D::generatePoissonRatioDataFromParams(const std::string& result_folder, int IH)
{
    std::string log_file = result_folder + "poisson_ratio_log_IH" + std::to_string(IH) + ".txt";
    std::ifstream in(log_file);
    std::string data_file = result_folder + "IH_" + std::to_string(IH) + "_poisson_ratio_sim.txt";
    std::ofstream out(data_file);
    out << std::setprecision(10);
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;

    std::vector<T> params(n_tiling_params);
    for (int i = 0; i < n_tiling_params; i++)
    {
        in >> params[i];
    }

    fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
        cubic_weights, "./a_structure.txt", unit);
    
    generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");
        
    solver.pbc_translation_file =  "./a_structure_translation.txt";
    bool valid_structure = initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    if (IH == 1)
    {
        solver.pbc_strain_w = 1e7;
    }
    solver.max_newton_iter = 500;
    solver.project_block_PD = true;
    if (IH == 22 || IH == 28)
        solver.project_block_PD = false;
    solver.verbose = false;
    int n_samples;
    in >> n_samples;
    VectorXT stiffness_values;
    T dtheta = M_PI / T(n_samples);
    stiffness_values.resize(n_samples);
    std::vector<TV3> strains(n_samples);
    for (int i = 0; i < n_samples; i++)
        in >> strains[i][0] >> strains[i][1] >> strains[i][2];
    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "IH " << IH << " " << i << "/" << n_samples << " strain " << strains[i].transpose() << std::endl;
        TM3 elasticity_tensor;
        T theta = dtheta * T(i);
        
        solver.computeHomogenizationElasticityTensor(strains[i], elasticity_tensor);
        TV3 d_voigt = TV3(std::cos(theta) * std::cos(theta),
                            std::sin(theta) * std::sin(theta),
                            std::cos(theta) * std::sin(theta));
        TV3 n_voigt = TV3(std::sin(theta) * std::sin(theta),
                            std::cos(theta) * std::cos(theta),
                            -std::cos(theta) * std::sin(theta));
        TM3 C_inverse = elasticity_tensor.inverse();
        
        stiffness_values[i] = -d_voigt.dot(C_inverse * n_voigt) / d_voigt.dot(C_inverse * d_voigt);
    }

    for (int i = 0; i < stiffness_values.rows()-1; i++)
    {
        out << stiffness_values[i] << " ";
    }
    out << stiffness_values[stiffness_values.rows()-1];
    out.close();
}

void Tiling2D::generateStiffnessDataFromParams(const std::string& result_folder, int IH)
{
    std::string log_file = result_folder + "stiffness_log_IH" + std::to_string(IH) + ".txt";
    std::ifstream in(log_file);
    std::string data_file = result_folder + "IH_" + std::to_string(IH) + "_stiffness_sim.txt";
    std::ofstream out(data_file);
    out << std::setprecision(10);
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);

    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;

    std::vector<T> params(n_tiling_params);
    for (int i = 0; i < n_tiling_params; i++)
    {
        in >> params[i];
    }

    fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
        cubic_weights, "./a_structure.txt", unit);
    
    generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");
        
    solver.pbc_translation_file =  "./a_structure_translation.txt";
    bool valid_structure = initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    if (IH == 1)
    {
        solver.pbc_strain_w = 1e7;
    }
    solver.max_newton_iter = 500;
    solver.project_block_PD = true;
    if (IH == 22 || IH == 28)
        solver.project_block_PD = false;
    solver.verbose = false;
    int n_samples;
    in >> n_samples;
    VectorXT stiffness_values;
    T dtheta = M_PI / T(n_samples);
    stiffness_values.resize(n_samples);
    std::vector<TV3> strains(n_samples);
    for (int i = 0; i < n_samples; i++)
        in >> strains[i][0] >> strains[i][1] >> strains[i][2];
    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "IH " << IH << " " << i << "/" << n_samples << " strain " << strains[i].transpose() << std::endl;
        TM3 elasticity_tensor;
        T theta = dtheta * T(i);
        
        solver.computeHomogenizationElasticityTensor(strains[i], elasticity_tensor);
        TV3 d_voigt = TV3(std::cos(theta) * std::cos(theta),
                            std::sin(theta) * std::sin(theta),
                            std::cos(theta) * std::sin(theta));
        stiffness_values[i] = 1.0 / (d_voigt.transpose() * (elasticity_tensor.inverse() * d_voigt));
    }

    for (int i = 0; i < stiffness_values.rows()-1; i++)
    {
        out << stiffness_values[i] << " ";
    }
    out << stiffness_values[stiffness_values.rows()-1];
    out.close();
}

void Tiling2D::generateStrainStressDataFromParams(const std::string& result_folder, int IH, T theta, 
        const std::vector<T>& params, const TV& strain_range, int n_samples, bool save_result)
{
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;

    fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
        cubic_weights, "./a_structure.txt", unit);
    
    generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");

    solver.pbc_translation_file =  "./a_structure_translation.txt";
    bool valid_structure = initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    if (IH == 1)
    {
        solver.pbc_strain_w = 1e7;
    }
    else
        solver.pbc_strain_w = 1e6;
    solver.max_newton_iter = 3000;
    solver.project_block_PD = true;
    if (IH == 22 || IH == 28)
        solver.project_block_PD = false;
    solver.verbose = false;
    solver.biaxial = false;
    solver.prescribe_strain_tensor = false;

    std::vector<T> timings(n_samples);

    auto runSim = [&](T _theta, T strain, T strain_ortho, int _fail_cnt, int idx) -> bool
    {
        if (_fail_cnt < 3)
        {
            Timer timer(true);
            bool solve_succeed = solver.staticSolve();
            // solver.saveToOBJ(result_folder + std::to_string(idx)+"_"+std::to_string(strain_samples[idx])+".obj");
            VectorXT residual(solver.num_nodes * 2); residual.setZero();
            solver.computeResidual(solver.u, residual);
            TM secondPK_stress, Green_strain;
            T psi;
            solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
            std::cout << timer.elapsed_sec() << std::endl;
            timings[idx] = timer.elapsed_sec();
            std::string obj_file = result_folder + "/obj/IH_" + std::to_string(IH) + "_strain_stress_"+std::to_string(idx)+".obj";
            solver.saveToOBJ(obj_file, false);
            // for (int m = 0; m < params.size(); m++)
            // {
            //     out << params[m] << " ";
            // }
            // out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            //     << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            //     << secondPK_stress(1, 0) << " " << psi << " " << _theta << " " << strain 
            //     << " " << strain_ortho << " "
            //     << residual.norm() << std::endl;
            return solve_succeed;
        }
        else
        {
            // for (int m = 0; m < params.size(); m++)
            // {
            //     out << params[m] << " ";
            // }
            // out << "-1" << " "<< "-1" << " " << "-1"
            //     << " " << "-1" << " " << "-1" << " "
            //     << "-1" << " " << "-1" << " " << _theta << " " << strain 
            //     << " " << strain_ortho << " "
            //     << 1e10 << std::endl;
            return false;
        }
    };

    int min_strain_idx = 0;
    int cnt = 0;
    T min_strain = 1e10;
    std::vector<T> strain_samples;
    T strain_delta = (strain_range[1] - strain_range[0]) / T(n_samples);
    
    for (T strain = strain_range[0]; strain < strain_range[1]; strain += strain_delta)
    {
        strain_samples.push_back(strain);
        // std::cout << strain << std::endl;
        T value = std::abs(strain - 1.0);
        if (value < min_strain && strain > 1.0)
        {
            min_strain = value;
            min_strain_idx = cnt;
        }
        cnt++;       
    }
    
    std::cout << strain_samples[min_strain_idx] << " " << min_strain_idx << std::endl;

    int fail_cnt = 0;
    for (int i = min_strain_idx; i < strain_samples.size(); i++)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt, i);
        if (!solve_succeed) 
            fail_cnt ++;
        else
            fail_cnt = 0;
    }
    solver.reset(); fail_cnt = 0;
    for (int i = min_strain_idx-1; i > -1; i--)
    {
        solver.strain_theta = theta;
        solver.uniaxial_strain = strain_samples[i];
        bool solve_succeed = runSim(theta, strain_samples[i], 0.0, fail_cnt, i);
        if (!solve_succeed) 
            fail_cnt ++;
        else
            fail_cnt = 0;
    }
    std::ofstream out(result_folder + "/" + "IH_" + std::to_string(IH) + "_strain_stress_timing.txt");
    for (T time : timings)
        out << time << " ";
    out.close();
}

void Tiling2D::generateStrainStressDataFromParams(const std::string& result_folder, int IH, bool save_result)
{
    std::string log_file = result_folder + "uniaxial_stress_IH" + std::to_string(IH) + ".txt";
    std::ifstream in(log_file);
    std::string data_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress_sim.txt";
    
    std::ofstream out;
    if (save_result)
    {
        out.open(data_file);
        out << std::setprecision(10);
    }
    int actual_IH, n_tiling_params, unit;
    std::vector<TV> bounds; 
    VectorXT ti_default;
    getTilingConfig(IH, n_tiling_params, actual_IH, bounds, ti_default, unit);
    
    std::vector<std::vector<TV2>> polygons;
    std::vector<TV2> pbc_corners; 
    csk::IsohedralTiling a_tiling( csk::tiling_types[ actual_IH ] );
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0., 0.75, 0.;

    std::vector<T> params(n_tiling_params);
    for (int i = 0; i < n_tiling_params; i++)
    {
        in >> params[i];
        // std::cout << params[i] << std::endl;
    }

    Timer timer(true);
    fetchUnitCellFromOneFamily(actual_IH, 2, polygons, pbc_corners, params, 
        cubic_weights, "./a_structure.txt", unit);
    
    generatePeriodicMesh(polygons, pbc_corners, true, "./a_structure");

    std::cout << "mesh generation timing : " << timer.elapsed_sec() << std::endl;
        
    solver.pbc_translation_file =  "./a_structure_translation.txt";
    bool valid_structure = initializeSimulationDataFromFiles("./a_structure.vtk", PBC_XY);
    if (!valid_structure)
        return;
    if (IH == 1)
    {
        solver.pbc_strain_w = 1e7;
    }
    solver.max_newton_iter = 3000;
    solver.project_block_PD = false;
    if (IH == 22 || IH == 28)
        solver.project_block_PD = false;
    solver.verbose = false;

    int n_samples; in >> n_samples;
    std::vector<TV3> strain_samples(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        TV3 strain; in >> strain[0] >> strain[1] >> strain[2];
        strain_samples[i] = strain;
    }
    std::vector<TM> strain_sim, stress_sim;
    std::vector<T> res_norms;
    std::vector<T> timings;
    std::vector<VectorXT> deformed_states;
    auto runSim = [&](const TV3& strain, int _fail_cnt) -> bool
    {
        if (_fail_cnt < 3)
        {
            solver.target_strain = strain;
            solver.prescribe_strain_tensor = true;
            Timer timer(true);
            // solver.reset();
            bool solve_succeed = solver.staticSolve();
            deformed_states.push_back(solver.deformed);
            std::string obj_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress_temp.obj";
            solver.saveToOBJ(obj_file, false);
            // solver.saveToOBJ(result_folder + std::to_string(idx)+"_"+std::to_string(strain_samples[idx])+".obj");
            VectorXT residual(solver.num_nodes * 2); residual.setZero();
            solver.computeResidual(solver.u, residual);
            TM secondPK_stress, Green_strain;
            T psi;
            solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
            std::cout << timer.elapsed_sec() << std::endl;
            timings.push_back(timer.elapsed_sec());
            strain_sim.push_back(Green_strain);
            stress_sim.push_back(secondPK_stress);
            res_norms.push_back(residual.norm());
            // out << Green_strain(0, 0) << " "<< Green_strain(1, 1) << " " << Green_strain(1, 0)
            //     << " " << secondPK_stress(0, 0) << " " << secondPK_stress(1, 1) << " "
            //     << secondPK_stress(1, 0)  << " "
            //     << residual.norm() << std::endl;
            
            return solve_succeed;
        }
        else
        {
            // for (int m = 0; m < params.size(); m++)
            // {
            //     out << params[m] << " ";
            // }
            // out << "-1" << " "<< "-1" << " " << "-1"
            //     << " " << "-1" << " " << "-1" << " "
            //     << "-1" << " " << "-1" << " " 
            //     << 1e10 << std::endl;
            return false;
        }
    };

    int fail_cnt = 0;
    // for (auto strain : strain_samples)
    // {
    //     runSim(strain, fail_cnt);
    // }
    for (int i = 8; i < strain_samples.size(); i++)
        runSim(strain_samples[i], fail_cnt);
    solver.reset();
    for (int i = 7; i > -1; i--)
        runSim(strain_samples[i], fail_cnt);
    if (save_result)
    {
        for (int i = 0; i < 8; i++)
        {
            int idx = strain_samples.size() - i - 1;
            out << strain_sim[idx](0, 0) << " "<< strain_sim[idx](1, 1) << " " << strain_sim[idx](1, 0)
                    << " " << stress_sim[idx](0, 0) << " " << stress_sim[idx](1, 1) << " "
                    << stress_sim[idx](1, 0)  << " "
                    << res_norms[idx] << std::endl;
            solver.deformed = deformed_states[idx];
            std::string obj_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress_obj"+std::to_string(idx)+".obj";
            solver.saveToOBJ(obj_file, false);
            std::cout << timings[idx] << ", ";
        }
        for (int i = 8; i < strain_samples.size(); i++)
        {
            int idx = i - 8;
            out << strain_sim[idx](0, 0) << " "<< strain_sim[idx](1, 1) << " " << strain_sim[idx](1, 0)
                    << " " << stress_sim[idx](0, 0) << " " << stress_sim[idx](1, 1) << " "
                    << stress_sim[idx](1, 0)  << " "
                    << res_norms[idx] << std::endl;
            std::cout << timings[idx] << ", ";
            solver.deformed = deformed_states[idx];
            std::string obj_file = result_folder + "IH_" + std::to_string(IH) + "_strain_stress_obj"+std::to_string(idx)+".obj";
            solver.saveToOBJ(obj_file);
        }
        
        out.close();
    }
    else
    {
        for (int i = 0; i < 8; i++)
        {
            int idx = strain_samples.size() - i - 1;
            std::cout << timings[idx] << ", ";
        }
        for (int i = 8; i < strain_samples.size(); i++)
        {
            int idx = i - 8;
            std::cout << timings[idx] << ", ";
        }
    }
    in.close();
}