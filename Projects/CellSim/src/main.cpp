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

Simulation simulation;
ObjNucleiTracking obj(simulation);
ObjFindInit obj_find_init(simulation);
SensitivityAnalysis sa(simulation, obj);
// SensitivityAnalysis sa(simulation, obj_find_init);

int main(int argc, char** argv)
{
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
    simulation.newton_tol = 1e-8;
    simulation.max_newton_iter = 2000;
    // simulation.cells.use_ipc_contact = false;

    // obj.initializeTarget();
    // obj.loadTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test.txt");
    // obj.loadTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/low_res_dense.txt");
    obj.loadTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_wo_inv.txt");
    obj.match_centroid = true;
    sa.max_num_iter = 2000;
    // obj.setTargetObjWeights();
    
    // obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
    // obj.loadWeightedCellTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/sss.txt");
    // obj.match_centroid = false;
    
    obj.use_penalty = false;
    obj.penalty_type = Qubic;
    if (obj.penalty_type == Qubic)
        obj.barrier_distance = 0.5;
    obj.penalty_weight = 1e6;

    // obj.add_Hessian_PD_term = true;
    obj.barrier_distance = 1e-3;
    obj.barrier_weight = 1e6;


    obj.bound << 0, 50;
    // obj.setOptimizer(MMA);
    // obj.setOptimizer(GaussNewton);
    obj.setOptimizer(SQP);
    // obj.use_log_barrier = true;
    // obj.updateTarget();
    // obj.initializeTargetFromMap("idx_map.txt", 30);

    // obj_find_init.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
    // obj_find_init.updateTarget();
    
    
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
        simulation.cells.tet_vol_barrier_w = 1e-22;
        // simulation.cells.edge_weights.setConstant(0.05);
        // simulation.cells.checkTotalHessian();
        // VectorXT ew;
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/debug/7.txt", ew);
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_5.txt", ew);
        // simulation.cells.edge_weights = ew;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/debug/7.obj");
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_5.obj");

        viewer.launch();
    };

    auto runSA = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        // sa.save_results = false;
        // sa.save_results = true;
        // sa.resume = true;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt_dense_ub25/SQP_iter_27.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt_dense_ub25/SQP_iter_27.txt", sa.design_parameters);
        // sa.optimizeLBFGSB();
        // return;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/current_mesh.obj");
        // sa.setSimulationEnergyWeights();
        // sa.optimizePerEdgeWeigths();
        // diff_sim_app.runOptimization();
        // sa.initialize();
        // VectorXT ew;
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_25.txt", ew);
        // simulation.cells.edge_weights = ew;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/opt/SQP_iter_25.obj");
        // sa.objective.getDesignParameters(sa.design_parameters);
        // sa.svdOnSensitivityMatrix();
        // sa.eigenAnalysisOnSensitivityMatrix();
        // sa.dxFromdpAdjoint();
        // sa.objective.diffTestd2Odx2();
        sa.checkStatesAlongGradient();
        // sa.objective.diffTestHessianScale();
        // sa.objective.diffTestGradientScale();
        // sa.objective.diffTestGradient();
        diff_sim_app.setViewer(viewer, menu);
        viewer.launch();
    };

    auto generateNucleiGT = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        // sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_dense_test_wo_inv.txt");
        sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/low_res_dense.txt");
    };

    auto generateWeights = [&]()
    {
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
        // obj.computeKernelWeights();
        obj.computeCellTargetFromDatapoints();
    };

    if (argc == 1)
    {
        runSA();
        // runSim();
        // generateNucleiGT();
    }
    else if (argc > 1)
    {
        sa.data_folder = argv[1];
        runSA();
    }
    
    // registerMesh();
    // exit(0);

    // loadDrosophilaData();
    // runSim();
    // runSA();
    // std::cout << argc << " " << argv[1] << std::endl;
    // generateWeights();
    // generateNucleiGT();
    // obj.checkData();
    return 0;
}