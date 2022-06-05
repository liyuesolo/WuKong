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

    if (argc > 1)
        sa.data_folder = argv[1];

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
    std::string data_folder = "/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/";
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
    
    
    
    simulation.cells.tet_vol_barrier_w = 1e-22;
    simulation.newton_tol = 1e-6;
    simulation.max_newton_iter = 2000;
    
    simulation.cells.bound_coeff = 1e6;
    simulation.cells.add_perivitelline_liquid_volume = false;
    simulation.cells.Bp = 0.0;
    
    sa.max_num_iter = 2000;


    int test_case = 2;

    if (test_case == 0)
    {
        simulation.cells.resolution = -1;
        simulation.initializeCells();
        simulation.max_newton_iter = 300;
        std::string data_file = data_folder;
        if (simulation.cells.resolution == -1)
            data_file += "centroids_56.txt";
        
        obj.power = 2;
        
        if (obj.power == 4)
            obj.w_data *= 1e3;
        obj.loadTarget(data_file, 0.05);
        // obj.rotateTarget(0.025);
        // obj.optimizeForStableTargetSpring(0.05);
        // obj.optimizeForStableTargetDeformationGradient(0.05);
        obj.match_centroid = true;
        obj.add_forward_potential = false;
        obj.w_fp = 1e-2;
        obj.add_spatial_regularizor = true;
        if (obj.add_spatial_regularizor)
        {
            obj.w_reg_spacial = 1e-3;
            obj.buildVtxEdgeStructure();
        }
        
        obj.add_reg = false;
        obj.reg_w = 1e-5;
        sa.add_reg = true;
        sa.reg_w_H = 1e-6;
        obj.use_penalty = false;
        obj.penalty_type = Qubic;
        obj.penalty_weight = 1e6;
        
        if (obj.use_penalty)
            obj.setOptimizer(SGN);
        else
            obj.setOptimizer(SQP);
        sa.initialize();
        sa.saveConfig();
        // sa.optimizeIPOPT();
        // runSA();
        // simulation.cells.edge_weights.setConstant(10.0);
        // simulation.cells.edge_weights.array() += 2;
        // sa.checkStatesAlongGradient();
    }
    else if (test_case == 2)
    {
        simulation.cells.resolution = 1;
        simulation.initializeCells();
        simulation.max_newton_iter = 300;
        // simulation.newton_tol = 1e-9;
        obj.setFrame(40);
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
        
        std::string weights_filename = data_folder;
        if (simulation.cells.resolution == 0)
            weights_filename += "weights_124.txt";
        else if (simulation.cells.resolution == 1)
            weights_filename += "weights_463.txt";
        else if (simulation.cells.resolution == 2)
            weights_filename += "weights_1500.txt";
        obj.loadWeightedCellTarget(weights_filename);
        
        // obj.filterTrackingData3X2F();
                
        // obj.filterTrackingData3X3F();
        obj.match_centroid = false;
        // simulation.cells.edge_weights.setConstant(0.1);
        obj.add_forward_potential = false;
        
        obj.power = 2;
        obj.w_fp = 1e-2;
        if (obj.power == 4)
        {
            // obj.w_fp = 1e-4;
            // obj.w_data = 1e4;
        }
        // obj.w_data = 1e-4;
        obj.add_spatial_regularizor = true;
        if (obj.add_spatial_regularizor)
        {
            obj.w_reg_spacial = 1e-3;
            // obj.w_reg_spacial = 1.0;
            obj.buildVtxEdgeStructure();
        }
        obj.add_reg = false;
        obj.reg_w = 1e-5;
        sa.add_reg = !obj.add_reg;
        // sa.add_reg = false;
        sa.reg_w_H = 1e-6;
        obj.use_penalty = false;
        obj.penalty_type = Qubic;
        obj.penalty_weight = 1e3;
        obj.wrapper_type = 0;

        if (obj.use_penalty)
            obj.setOptimizer(SGN);
        else
            obj.setOptimizer(SQP);

        sa.initialize();
        sa.saveConfig();
        sa.optimizeIPOPT();
        int iter = 449;
        int exp_id = 606;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/x_ipopt.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/p_ipopt.txt", simulation.cells.edge_weights);
        // sa.design_parameters = simulation.cells.edge_weights;
        // sa.checkStatesAlongGradientSGN();
        // obj.diffTestGradientScale();
        
    }
    else if (test_case == 4)
    {
        simulation.cells.resolution = -1;
        simulation.cells.use_test_mesh = true;
        simulation.initializeCells();   
    }

    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    auto runSim = [&]()
    {
        SimulationApp sim_app(simulation);
        
        int iter = 42;
        int exp_id = 694;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_"+std::to_string(iter)+".obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_"+std::to_string(iter)+".txt", simulation.cells.edge_weights);
        // simulation.newton_tol = 1e-8;
        // simulation.cells.edge_weights.setConstant(0.1);
        sim_app.setViewer(viewer, menu);
        simulation.verbose = true;
        simulation.save_mesh = false;
        simulation.cells.print_force_norm = true;
        // simulation.cells.checkTotalGradientScale();
        // simulation.cells.checkTotalGradient();
        // simulation.cells.checkTotalHessianScale();
        viewer.launch();
    };

    auto runSA = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        
        // sa.optimizeLBFGS();
        // simulation.loadDeformedState("current_mesh.obj");
        // obj.diffTestd2Odx2Scale();
        
        // obj.diffTestPartialOPartialpScale();
        // obj.diffTestdOdxScale();
        // obj.diffTestd2Odx2();
        // obj.diffTestPartialOPartialp();
        // obj.diffTestPartialOPartialpScale();
        // sa.optimizeIPOPT();
        // sa.save_results = false;
        sa.save_results = true;
        // sa.resume = true;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/380/SGN_iter_65.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/380/SGN_iter_65.txt", sa.design_parameters);
        // simulation.newton_tol = 1e-8;
        
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
        // sa.initialize();
        // sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_wo_inv.txt");
        std::string data_file = data_folder;
        if (simulation.cells.resolution == -1)
            data_file += "centroids_56.txt";
        sa.generateNucleiDataSingleFrame(data_file);

        // sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/low_res_dense.txt");
        // simulation.staticSolve();
        // simulation.saveState("target_x.obj");
    };

    auto generateWeights = [&]()
    {
        obj.setFrame(0);
        simulation.cells.edge_weights.setConstant(0.01);
        simulation.verbose = true;
        simulation.save_mesh = false;
        simulation.cells.print_force_norm = false;
        simulation.staticSolve();
        // simulation.loadDeformedState("mesh_0.1.obj");
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
        std::string weights_filename = data_folder;
        if (simulation.cells.resolution == 0)
            weights_filename += "weights_124.txt";
        else if (simulation.cells.resolution == 1)
            weights_filename += "weights_463.txt";
        else if (simulation.cells.resolution == 2)
            weights_filename += "weights_1500.txt";
        obj.computeCellTargetsFromDatapoints(weights_filename);
    };

    auto visualizeData = [&]()
    {
        DataViewerApp data_viewer_app(simulation);
        data_viewer_app.setViewer(viewer, menu);
        viewer.launch();
    };

    auto runSequentialTracking = [&]()
    {

    };

    if (argc == 1)
    {
        // visualizeData();
        runSA();
        // runSim();
        // generateNucleiGT();
        // generateWeights();
    }
    else if (argc > 1)
    {
        // sa.saveConfig();
        // sa.optimizeIPOPT();
        runSA();
        // sa.runTracking(0, 40, /*load weights = */false, /*weigts_file = */"");
    }

    
    return 0; 
}