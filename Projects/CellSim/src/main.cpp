#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/axis_angle_to_quat.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/png/writePNG.h>
#include <igl/trackball.h>
#include <imgui/imgui.h>

#include "../include/Misc.h"
#include "../include/Simulation.h"
#include "../include/VertexModel.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/Objectives.h"
#include "../include/app.h"
#include "../include/DataIO.h"
#include "../include/GeometryHelper.h"

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
// SensitivityAnalysis sa(simulation, obj_find_init);


int main(int argc, char** argv)
{

    // testBiharmonicBasisFunction();
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
    auto processDrosophilaData = [&]()
    {
        // data_io.loadDataFromTxt("/home/yueli/Downloads/drosophila_data/drosophila_side_2_tracks_071621.txt");
        // data_io.loadDataFromBinary("/home/yueli/Downloads/drosophila_data/drosophila_side2_time_xyz.dat", 
        //     "/home/yueli/Downloads/drosophila_data/drosophila_side2_ids.dat",
        //     "/home/yueli/Downloads/drosophila_data/drosophila_side2_scores.dat");
        data_io.loadDataFromBinary("/home/yueli/Downloads/drosophila_data/drosophila_side1_time_xyz.dat", 
            "/home/yueli/Downloads/drosophila_data/drosophila_side1_ids.dat",
            "/home/yueli/Downloads/drosophila_data/drosophila_side1_scores.dat");
        // data_io.loadDataFromTxt("/home/yueli/Downloads/drosophila_data/drosophila_side_1_tracks_071621.txt");
        // data_io.trackCells();
        // data_io.processData();
        // data_io.filterWithVelocity();
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
    
    sa.max_num_iter = 2000;


    int test_case = 2;

    if (test_case == 0)
    {
        simulation.cells.resolution = 1;
        simulation.initializeCells();
        simulation.cells.lower_triangular = true;
        simulation.max_newton_iter = 300;
        if (simulation.cells.use_cell_centroid)
            simulation.cells.tet_vol_barrier_w = 1e-12;
        else
            simulation.cells.tet_vol_barrier_w = 1e3;
        simulation.cells.add_perivitelline_liquid_volume = false;
        simulation.cells.Bp = 0.0;
        simulation.max_newton_iter = 300;
        simulation.cells.B = 1e4;
        simulation.cells.By = 1e4;
        
        simulation.cells.bound_coeff = 1e2;

        std::string data_file = data_folder;
        if (simulation.cells.resolution == -1)
            data_file += "centroids_56.txt";
        else if (simulation.cells.resolution == 1)
            data_file += "centroids_463.txt";
        else if (simulation.cells.resolution == 2)
            data_file += "centroids_1500.txt";
        
        obj.power = 2;
        
        if (obj.power == 4)
            obj.w_data *= 1e3;
        obj.loadTarget(data_file, 0.0);
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
        // obj.diffTestPartial2OPartialp2();
        // runSA();
        // obj.diffTestGradientScale();
        // simulation.cells.edge_weights.setConstant(10.0);
        VectorXT perturbance = VectorXT::Random(simulation.cells.edge_weights.rows());
        perturbance /= perturbance.maxCoeff();
        // simulation.cells.edge_weights.setConstant(5.0);
        // simulation.cells.edge_weights.array() += 0.1;
        // simulation.cells.edge_weights += 2.0 * perturbance;
        // sa.checkStatesAlongGradient();
        int exp_id = 1063;
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_29.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_29.txt", simulation.cells.edge_weights);
        // sa.design_parameters = simulation.cells.edge_weights;
        // MatrixXT H;
        // obj.hessianGN(sa.design_parameters, H, false);
        // Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // MatrixXT U = svd.matrixU();
        // VectorXT Sigma = svd.singularValues();
        // MatrixXT V = svd.matrixV();
        // std::cout << "\t[SQP] GN Hessian singular values last: " << Sigma.tail<5>().transpose() << std::endl;
        // std::cout << "\t[SQP] GN Hessian singular values first: " << Sigma.head<5>().transpose() << std::endl;
    }
    else if (test_case == 2)
    {
        simulation.cells.resolution = 2;
        simulation.initializeCells();
        simulation.cells.lower_triangular = false;
        simulation.cells.scaled_barrier = true;
        simulation.cells.edge_weights.setConstant(0.1);
        simulation.max_newton_iter = 300;
        // simulation.newton_tol = 1e-9;
        sa.max_num_iter = 400;
        if (simulation.cells.use_cell_centroid)
        {
            if (simulation.cells.scaled_barrier)
                simulation.cells.tet_vol_barrier_w = 1e-10;
            else
                simulation.cells.tet_vol_barrier_w = 1e-22;
        }
        else
            simulation.cells.tet_vol_barrier_w = 1e1;
        
        simulation.cells.add_perivitelline_liquid_volume = false;
        
        simulation.cells.Bp = 0.0;
        simulation.cells.B = 1e6;
        simulation.cells.By = 1e4;
        simulation.cells.bound_coeff = 1e4;
                
        obj.setFrame(30);
        obj.loadTargetTrajectory("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat", true);
        
        std::string weights_filename = data_folder;
        if (simulation.cells.resolution == 0)
            weights_filename += "weights_124.txt";
        else if (simulation.cells.resolution == 1)
            weights_filename += "weights_463.txt";
        else if (simulation.cells.resolution == 2)
        {
            // weights_filename += "weights_1500.txt";
            weights_filename += "weights_1500_stiff.txt";
        }

        obj.add_spatial_x = false;
        
        if (obj.add_spatial_x)
        {
            obj.w_reg_x_spacial = 1.0;
            obj.buildCentroidStructure();
        }

        obj.loadWeightedCellTarget(weights_filename, /*use_all_points = */ false);
        
        // obj.filterTrackingData3X2F();
                
        // obj.filterTrackingData3X3F();
        obj.match_centroid = false;
        // simulation.cells.edge_weights.setConstant(0.1);
        obj.add_forward_potential = false;
        
        obj.power = 2;
        obj.w_fp = 1e-2;
        
        if (obj.power == 4)
            obj.w_data *= 1e3;
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
        sa.add_reg = true;
        sa.reg_w_H = 1e-6;
        obj.use_penalty = false;
        obj.penalty_type = Qubic;
        obj.penalty_weight = 1e3;
        obj.wrapper_type = 0;

        if (obj.use_penalty)
            obj.setOptimizer(SGN);
        else
            obj.setOptimizer(SQP);

        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/842/p_ipopt.txt", simulation.cells.edge_weights);
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/842/x_ipopt.obj");
        sa.initialize(); 
        sa.saveConfig();
        // sa.optimizeIPOPT();
        // sa.optimizeLBFGSB();
        // obj.diffTestdOdxScale();
        // obj.diffTestd2Odx2Scale();
        int iter = 449;
        int exp_id = 1042;
        // simulation.loadDeformedState("current_mesh.obj");
        // simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_84.obj");
        // simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/SQP_iter_84.txt", simulation.cells.edge_weights);
        simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/1125_lbfgs_30_highres/lbfgs_iter_96.obj");
        simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/1125_lbfgs_30_highres/lbfgs_iter_96.txt", simulation.cells.edge_weights);
        sa.design_parameters = simulation.cells.edge_weights;
        // MatrixXT H_GN;
        // std::cout << "hessianGN" << std::endl;
        // obj.getDesignParameters(sa.design_parameters);
        // obj.hessianGN(sa.design_parameters, H_GN, /*simulate = */false);
        // sa.checkStatesAlongGradientSGN();
        // obj.diffTestGradientScale();
        // obj.diffTestGradient();
        // obj.diffTestGradientScale();
        
    }
    else if (test_case == 4)
    {
        simulation.cells.resolution = 3;
        simulation.initializeCells();
        simulation.cells.tet_vol_barrier_w = 1e-10;
        simulation.cells.add_perivitelline_liquid_volume = false;
        simulation.cells.Bp = 0.0;
        simulation.cells.bound_coeff = 1e8;
        simulation.cells.B = 1e6;
    }
    else if (test_case == 5)
    {
        std::string folder = "/home/yueli/Documents/ETH/WuKong/output/cells/video_data/";
        simulation.cells.resolution = 1;
        simulation.initializeCells();
        int global_cnt = 0;
        for (int i = 30; i < 40; i++)
        {
            // simulation.loadDeformedState(folder + "frame_"+std::to_string(i)+".obj");
            // VectorXT xi = simulation.deformed;
            // simulation.loadDeformedState(folder + "frame_"+std::to_string(i+1)+".obj");
            // VectorXT xj = simulation.deformed; 
            // int step = 25;
            // VectorXT dx = (xj - xi) / T(step);
            // for (int j = 0; j < step; j++)
            // {
            //     simulation.deformed = xi + T(j) * dx;
            //     simulation.saveState(folder + "sub/" + std::to_string(global_cnt)+".obj");
            //     global_cnt++;
            // }   
        }
        
    }

    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    auto runSim = [&]()
    {
        SimulationApp sim_app(simulation);
        
        int iter = 5;
        int exp_id = 1119;
        simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/lbfgs_iter_"+std::to_string(iter)+".obj");
        simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/lbfgs_iter_"+std::to_string(iter)+".txt", simulation.cells.edge_weights);
        // simulation.loadEdgeWeights("failed.txt", simulation.cells.edge_weights);
        // std::cout << simulation.cells.edge_weights.minCoeff() << " " << simulation.cells.edge_weights.maxCoeff() << std::endl;
        // simulation.loadDeformedState("failed.obj");
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
        else if (simulation.cells.resolution == 1)
            data_file += "centroids_463.txt";
        else if (simulation.cells.resolution == 2)
            data_file += "centroids_1500.txt";
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
            // weights_filename += "weights_1500.txt";
            weights_filename += "weights_1500_stiff.txt";
        else if (simulation.cells.resolution == 3)
            weights_filename += "weights_3000.txt";
        else if (simulation.cells.resolution == 4)
            weights_filename += "weights_6300.txt";
        obj.computeCellTargetsFromDatapoints(weights_filename);
    };

    auto visualizeData = [&]()
    {
        DataViewerApp data_viewer_app(simulation);
        data_viewer_app.setViewer(viewer, menu);
        viewer.launch();
    };

    auto renderScene = [&]()
    {
        // RendererApp render_app(simulation);
        // render_app.setViewer(viewer, menu);
        // viewer.launch();
        DiffSimApp diff_sim_app(simulation, sa);
        // diff_sim_app.setViewer(viewer, menu);
        diff_sim_app.show_target = false;
        diff_sim_app.show_target_current = false;
        diff_sim_app.show_edges = false;
        diff_sim_app.use_debug_color = true;

        viewer.launch_init();
        viewer.core().camera_zoom *= 0.8;
        // Eigen::Quaternionf rot_quat(Eigen::AngleAxisf(float(-M_PI /2.0), Eigen::Vector3f(1, 0, 0)));
        Eigen::Quaternionf rot_quat(Eigen::AngleAxisf(float(-M_PI /2.0), Eigen::Vector3f(1, 0, 0)));
        Eigen::Quaternionf rot_quat2(Eigen::AngleAxisf(float(-M_PI /2.0), Eigen::Vector3f(0, 1, 0)));

        // viewer.core().trackball_angle = rot_quat;
        std::string folder = "/home/yueli/Documents/ETH/WuKong/output/cells/video_data/";
        // simulation.cells.resolution = 1;
        // simulation.initializeCells();
        int global_cnt = 0;
        for (int i = 30; i < 40; i++)
        {
            simulation.loadDeformedState(folder + "frame_"+std::to_string(i)+".obj");
            VectorXT wi, wj;
            simulation.loadEdgeWeights(folder+ + "frame_"+std::to_string(i) + ".txt", wi);
            VectorXT xi = simulation.deformed;
            simulation.loadDeformedState(folder + "frame_"+std::to_string(i+1)+".obj");
            simulation.loadEdgeWeights(folder+ + "frame_"+std::to_string(i) + ".txt", wj);
            std::cout << wj.sum() / T(wj.rows()) << std::endl;
            VectorXT xj = simulation.deformed; 
            int step = 25;
            VectorXT dx = (xj - xi) / T(step);
            VectorXT dp = (wj - wi) / T(step);
            for (int j = 0; j < step; j++)
            {
                std::cout << global_cnt << std::endl;
                simulation.deformed = xi + T(j) * dx;
                // simulation.deformed = simulation.undeformed;
                diff_sim_app.edge_weights = wi + T(j) * dp;
                diff_sim_app.show_edge_weights_opt = true;
                int width = 2000, height = 2000;
                CMat R(width,height), G(width,height), B(width,height), A(width,height);
                // viewer.data().clear();
                // Eigen::MatrixXd V, C; Eigen::MatrixXi F;
                // simulation.generateMeshForRendering(V, F, C);
                // viewer.data().set_mesh(V, F);
                // viewer.data().set_colors(C);  
                viewer.core().background_color.setOnes();
                viewer.data().set_face_based(true);
                viewer.data().shininess = 1.0;
                viewer.data().point_size = 10.0;

                // viewer.data().set_mesh(V, F);     
                // viewer.data().set_colors(C);
                
                // viewer.core().align_camera_center(V);
                diff_sim_app.updateScreen(viewer);
                
                

                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                global_cnt++;
                igl::png::writePNG(R,G,B,A, folder + "sub/imgs/"+std::to_string(global_cnt)+".png");
                // std::exit(0);
            }       
            
        }
        viewer.launch_shut();
        
    };

    auto renderData = [&]()
    {
        DataViewerApp data_viewer_app(simulation);
        // data_viewer_app.connect_neighbor = true;
        data_viewer_app.setViewer(viewer, menu);
        int width = 3000, height = 2600;
        viewer.launch_init(true, false, "wukong", width, height);
        // Eigen::Quaternionf rot_quat(Eigen::AngleAxisf(float(-M_PI /2.0), Eigen::Vector3f(1, 0, 0)));
        Eigen::Quaternionf rot_quat(Eigen::AngleAxisf(float(M_PI), Eigen::Vector3f(0, 1, 0)));
        // Eigen::Quaternionf rot_quat2(Eigen::AngleAxisf(float(M_PI /8.0), Eigen::Vector3f(1, 0, 0)));
        
        viewer.core().trackball_angle = rot_quat;
        // viewer.core().camera_zoom *= 2.2;
        // viewer.core().toggle(viewer.data().show_lines);
        std::string folder = "/home/yueli/Documents/ETH/WuKong/output/cells/stats/voronoi/";
        for (int frame = 0; frame < 41; frame++)
        {
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            data_viewer_app.frame_cnt = frame;
            data_viewer_app.updateScreen(viewer);
            viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
            A.setConstant(255);
            igl::png::writePNG(R,G,B,A, folder +std::to_string(frame)+"_lateral_view2.png");
        }
        
    };

    auto runSimIPOPT = [&]()
    {
        simulation.solveIPOPT();
    };

    auto testCholmod = [&]()
    {
        int exp_id = 1017;
        simulation.loadDeformedState("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/lbfgs_iter_200.obj");
        simulation.loadEdgeWeights("/home/yueli/Documents/ETH/WuKong/output/cells/"+std::to_string(exp_id)+"/lbfgs_iter_200.txt", simulation.cells.edge_weights);
        sa.design_parameters = simulation.cells.edge_weights;
        MatrixXT H_GN;
        std::cout << "hessianGN" << std::endl;
        obj.hessianGN(sa.design_parameters, H_GN, /*simulate = */false);
    };


    if (argc == 1)
    {
        // sa.optimizeKnitro();
        // testCholmod();
        // renderData();
        // processDrosophilaData();
        visualizeData();
        // sa.optimizeKnitro();
        // runSA();
        // runSim();
        // sa.optimizeLBFGSB();
        // runSimIPOPT();
        // generateNucleiGT();
        // generateWeights();
        // renderScene();
    }
    else if (argc > 1)
    {
        // sa.saveConfig();
        // sa.optimizeIPOPT();
        
        sa.optimizeLBFGSB();
        // sa.optimizeKnitro();
        // runSA();    
        // runSim();
        // sa.runTracking(28, 45, /*load weights = */false, /*weigts_file = */"");
    }

    
    return 0; 
}