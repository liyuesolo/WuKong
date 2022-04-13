#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>


#include "../include/Simulation.h"
#include "../include/VertexModel.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/Objectives.h"
#include "../include/Misc.h"
#include "../include/app.h"
#include "../include/DataIO.h"
#include "../include/GeometryHelper.h"

using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// SensitivityAnalysis sa(simulation, obj_find_init);


int main(int argc, char** argv)
{
    
    Simulation simulation;
    ObjNucleiTracking obj(simulation);
    // ObjFindInit obj_find_init(simulation);
    SensitivityAnalysis sa(simulation, obj);
    
    // Eigen::MatrixXd VD;
    // Eigen::MatrixXi FD;
    // igl::readOBJ("/home/yueli/Downloads/drosophila.obj", VD, FD);

    // GeometryHelper::normalizePointCloud(VD);
    // Matrix<T, 3, 3> R;
    // R << 0.960277, -0.201389, 0.229468, 0.2908, 0.871897, -0.519003, -0.112462, 0.558021, 0.887263;
    // Matrix<T, 3, 3> R2 = Eigen::AngleAxis(0.20 * M_PI + 0.5 * M_PI, TV(-1.0, 0.0, 0.0)).toRotationMatrix();
    
    // for (int i = 0; i < VD.rows(); i++)
    // {
    //     VD.row(i) = (R2 * R * VD.row(i).transpose()).transpose();
    // }

    // std::cout << R2 * R << std::endl;

    // igl::writeOBJ("/home/yueli/Downloads/drosophila_normalized.obj", VD, FD);

    // std::exit(0);

    DataIO data_io;
    auto loadDrosophilaData = [&]()
    {
        // data_io.loadDataFromTxt("/home/yueli/Downloads/drosophila_data/drosophila_side_2_tracks_071621.txt");
        data_io.loadDataFromBinary("/home/yueli/Downloads/drosophila_data/drosophila_side2_time_xyz.dat", 
            "/home/yueli/Downloads/drosophila_data/drosophila_side2_ids.dat",
            "/home/yueli/Downloads/drosophila_data/drosophila_side2_scores.dat");
        data_io.trackCells();
    };

    auto registerMesh = [&]()
    {
        MatrixXT cell_trajectories;
        data_io.loadTrajectories("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat", cell_trajectories);

        VectorXT cell_centroids_all;
        simulation.cells.getAllCellCentroids(cell_centroids_all);


        VectorXT rigid_icp_result;
        VectorXT frame0 = cell_trajectories.col(0);
        GeometryHelper::registerPointCloudAToB(frame0, cell_centroids_all, rigid_icp_result);
    };
    
    
    simulation.initializeCells();
    
    simulation.cells.tet_vol_barrier_w = 1e-22;
    simulation.newton_tol = 1e-6;
    simulation.max_newton_iter = 2000;

    simulation.cells.add_perivitelline_liquid_volume = false;
    simulation.cells.Bp = 0.0;
    
    sa.max_num_iter = 2000;


    int test_case = 1;

    if (test_case == 0)
    {
        obj.loadTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_wo_inv.txt");
        obj.match_centroid = true;
        obj.add_forward_potential = true;

        obj.w_fp = 1e-3;
        obj.add_reg = false;
        obj.reg_w = 1e-5;
        sa.add_reg = true;
        sa.reg_w_H = 1e-6;
        simulation.cells.edge_weights.setConstant(0.1);
        obj.setOptimizer(SQP);

    }
    else if (test_case == 1) // match invaginated target
    {
        std::string filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_inv.txt";
        obj.loadTarget(filename, 0.1);
        obj.match_centroid = true;
        obj.add_forward_potential = true;
        obj.power = 2;
        obj.w_fp = 1e-1;
        obj.add_reg = false;
        obj.reg_w = 1e-5;
        sa.add_reg = true;
        sa.reg_w_H = 1e-6;

        // simulation.cells.edge_weights.setConstant(0.1);
        
        obj.setOptimizer(SQP);
    }
    else if (test_case == 2)
    {
        obj.setFrame(40);
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
        // obj.loadWeightedCellTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sss.txt");
        obj.loadWeightedCellTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/weighted_targets.txt");
        
        obj.match_centroid = false;
        simulation.cells.edge_weights.setConstant(0.1);
        obj.add_forward_potential = true;
        obj.w_fp = 1e-3;
        obj.add_reg = false;
        sa.add_reg = true;
        sa.reg_w_H = 1e-6;
        obj.setOptimizer(SQP);
    }
    else if (test_case == 3)
    {

    }

    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);    

    auto runSim = [&]()
    {
        SimulationApp sim_app(simulation);
        sim_app.setViewer(viewer, menu);
        simulation.verbose = true;
        simulation.save_mesh = false;
        simulation.cells.print_force_norm = true;
        
        simulation.cells.edge_weights.setConstant(0.01);
        // simulation.cells.B *= 10.0;
        // simulation.cells.By *= 10.0;
        // simulation.cells.add_perivitelline_liquid_volume = false;
        // simulation.cells.Bp = 0.0;
        // simulation.cells.bound_coeff = 1e5;  
        // simulation.cells.edge_weights.setConstant(0.05);
        // simulation.cells.checkTotalHessian();
        // VectorXT ew;
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/115/SQP_iter_12.txt", ew);
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_5.txt", ew);
        // simulation.cells.edge_weights = ew;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/115/SQP_iter_12.obj");
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_5.obj");
        // simulation.checkHessianPD(false);
        viewer.launch();
    };

    auto runSA = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        // simulation.loadDeformedState("current_mesh.obj");
        // obj.diffTestd2Odx2Scale();
        // obj.diffTestdOdxScale();
        // obj.diffTestd2Odx2();
        
        // sa.optimizeIPOPT();
        // sa.save_results = false;
        sa.save_results = true;
        // sa.resume = true;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/60/SQP_iter_1210.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/60/SQP_iter_1210.txt", sa.design_parameters);
        
        // simulation.loadDeformedState("d2odx2_check.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/45/SQP_iter_161.txt", sa.design_parameters);
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt_dense_ub25/SQP_iter_28.txt", sa.design_parameters);
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt_dense_ub25/SQP_iter_28.obj");
        // sa.checkStatesAlongGradient();
        diff_sim_app.setViewer(viewer, menu);
        viewer.launch();

    };

    auto generateNucleiGT = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        // sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_wo_inv.txt");
        sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_inv.txt");
        // sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/low_res_dense.txt");
        // simulation.staticSolve();
        // simulation.saveState("target_x.obj");
    };

    auto generateWeights = [&]()
    {
        simulation.cells.edge_weights.setConstant(0.1);
        simulation.staticSolve();
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
        // obj.computeKernelWeights();
        obj.computeCellTargetFromDatapoints();
    };

    auto visualizeData = [&]()
    {

    };

    if (argc == 1)
    {
        runSA();
        // runSim();
        // generateNucleiGT();
        // generateWeights();
    }
    else if (argc > 1)
    {
        sa.data_folder = argv[1];
        sa.saveConfig();
        runSA();
    }

    
    return 0;
}