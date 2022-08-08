#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/jet.h>

#include "../include/app.h"
#include <boost/filesystem.hpp>

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 2>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    // igl::opengl::glfw::Viewer viewer;
    // igl::opengl::glfw::imgui::ImGuiMenu menu;

    // viewer.plugins.push_back(&menu);
    
    // auto renderScene = [&]()
    // {
    //     SimulationApp app(tiling);

    //     app.setViewer(viewer, menu);
    //     std::string folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingRenderingNew/";
    //     int width = 2000, height = 2000;
    //     CMat R(width,height), G(width,height), B(width,height), A(width,height);
    //     viewer.core().background_color.setOnes();
    //     viewer.data().set_face_based(true);
    //     viewer.data().shininess = 1.0;
    //     viewer.data().point_size = 10.0;
    //     viewer.core().camera_zoom *= 1.4;
    //     viewer.launch_init();
    //     for (int i = 0; i < 994; i++)
    //     {
    //         std::ifstream in("/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/"+std::to_string(i)+".vtk");
    //         if (!in.good())
    //             continue;
    //         else
    //             in.close();

                
    //         tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/"+std::to_string(i)+".vtk", true);
    //         app.updateScreen(viewer);
    //         viewer.core().align_camera_center(app.V);
    //         app.updateScreen(viewer);
    //         viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
    //         A.setConstant(255);
    //         igl::png::writePNG(R,G,B,A, folder + std::to_string(i)+".png");
    //     }
    //     viewer.launch_shut();
    // };

    // auto simApp = [&]()
    // {
    //     SimulationApp app(tiling);
    //     tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/a_structure.vtk", true);
    //     app.setViewer(viewer, menu);
    //     viewer.launch();
    // };

    // auto generateForceDisplacementCurve = [&]()
    // {
    //     tiling.solver.penalty_pairs.clear();
    //     std::string base_folder = "/home/yueli/Documents/ETH/SandwichStructure/";
    //     // std::vector<int> candidates = {7, 8, 20, 
    //     //     44, 45, 51, 66, 70,
    //     //     71, 78, 79, 80,100, 101, 
    //     //     102, 108, 130, 131, 140, 141, 142,
    //     //     144, 150, 161, 166, 167, 
    //     //     196, 224, 231, 234, 235, 
    //     //     400, 401, 417, 428, 4249, 444, 
    //     //     466, 473, 479, 516, 523, 545, 550,
    //     //     554, 587, 588, 613, 653};

    //     std::vector<int> candidates = {7, 8};

        
    //     for (int candidate : candidates)
    //     {
    //         std::cout << std::endl;
    //         std::cout << "################## STRUCTURE " << candidate << " ########################" << std::endl;
    //         tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTK/" + std::to_string(candidate)+ ".vtk", true);
    //         std::string sub_dir = base_folder + "ForceDisplacementCurve/" + std::to_string(candidate);
    //         boost::filesystem::create_directories(sub_dir);
    //         std::string result_folder = sub_dir;
    //         tiling.generateForceDisplacementCurve(result_folder + "/");
    //     }
    // };

    // auto generateFDCurveSingleStructure = [&]()
    // {
    //     std::string base_folder = "/home/yueli/Documents/ETH/SandwichStructure/";
    //     std::string vtk_file = base_folder + "TilingVTKNew/res1.vtk";
    //     tiling.generateForceDisplacementCurveSingleStructure(vtk_file, base_folder + "convergence_test/res1/");
    // };
    // generateForceDisplacementCurve();
    // simApp();
    // renderScene();    

    if (argc > 1)
    {
        
        int candidate = std::stoi(argv[1]);
        std::string input_dir = argv[2];
        int script_flag = std::stoi(argv[3]);
        std::string base_folder = input_dir + "SandwichStructure/";

        if (script_flag == 0) // run sim in parallel
        {
            std::ifstream in(input_dir + "SandwichStructure/TilingVTKNew/"+std::to_string(candidate)+".vtk");
            if (!in.good())
                return 0;
            else
                in.close();
            
            bool valid_structure = tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTKNew/" + std::to_string(candidate)+ ".vtk", true);
            if (!valid_structure)
                return 0;
            std::string result_folder = base_folder + "ForceDisplacementCurve/" + std::to_string(candidate);
            tiling.generateForceDisplacementCurve(result_folder + "/");
        }
        else if (script_flag == 1) // batch rendering
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);

            SimulationApp app(tiling);

            app.setViewer(viewer, menu);
            // std::string folder = base_folder + "SandwichStructure/ForceDisplacementCurve/";
            
            int width = 2000, height = 2000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            T dp = 0.02;
            for (T percent = 0.0; percent < 0.8 + dp; percent += dp)
            {
                std::string obj_file = base_folder + "ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".obj";
                if (!fileExist(obj_file))
                    break;
                bool load_success = igl::readOBJ(obj_file, app.V, app.F);
                
                if (percent == 0.0)
                    viewer.core().align_camera_center(app.V);
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                app.C.resize(app.F.rows(), 3);
                app.C.col(0).setZero(); app.C.col(1).setConstant(0.3); app.C.col(2).setConstant(1.0);
                viewer.data().set_colors(app.C);
                
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                
                igl::png::writePNG(R,G,B,A, base_folder + "/ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".png");
            }
            viewer.launch_shut();
        }
        else if (script_flag == 2) //batch resume
        {
            T dp = 0.02;
            tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTKNew/" + argv[1] + ".vtk", true);
            TV min_corner, max_corner;
            tiling.solver.computeBoundingBox(min_corner, max_corner);
            TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
            TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);
            T dy = max_corner[1] - min_corner[1];
            tiling.solver.penalty_weight = 1e4;
            std::vector<T> displacements;
            std::vector<T> force_norms;
            VectorXT u_prev = tiling.solver.u;
            T last_percentage = 1.0;
            for (T percent = 0.0; percent < 0.8 + dp; percent += dp)
            {
                std::string obj_file = base_folder + "ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".obj";
                if (!fileExist(obj_file))
                {
                    last_percentage = percent - dp;
                    break;
                }
                else
                {
                    Eigen::MatrixXd _V; Eigen::MatrixXi _F;
                    igl::readOBJ(obj_file, _V, _F);
                    for (int i = 0; i < _V.rows(); i++)
                    {
                        tiling.solver.deformed.segment<2>(i*2) = _V.row(i).segment<2>(0);
                        tiling.solver.u = tiling.solver.deformed - tiling.solver.undeformed;
                    }
                    u_prev = tiling.solver.u;
                    tiling.solver.computeIPCRestData();
                    tiling.solver.penalty_pairs.clear();
                    tiling.solver.addPenaltyPairsBox(min1, max1, TV(0, -percent * dy));
                    VectorXT interal_force(tiling.solver.num_nodes * 2);
                    interal_force.setZero();
                    tiling.solver.addBCPenaltyForceEntries(tiling.solver.penalty_weight, interal_force);
                    displacements.push_back(percent * dy);
                    force_norms.push_back(interal_force.norm());
                }
            }
            for (T percent = last_percentage; percent < 0.8 + dp; percent += dp)
            {
                tiling.solver.penalty_pairs.clear();
                tiling.solver.addPenaltyPairsBox(min1, max1, TV(0, -percent * dy));
                // solver.y_bar = max_corner[1] - dis * dy;
                tiling.solver.u = u_prev;
                tiling.solver.staticSolve();
                u_prev = tiling.solver.u;
                VectorXT interal_force(tiling.solver.num_nodes * 2);
                interal_force.setZero();
                tiling.solver.addBCPenaltyForceEntries(tiling.solver.penalty_weight, interal_force);
                displacements.push_back(percent * dy);
                force_norms.push_back(interal_force.norm());

                tiling.solver.saveToOBJ(base_folder + "ForceDisplacementCurve/" + argv[1] + "/" + std::to_string(percent) + ".obj");
            }
            std::ofstream out(base_folder + "ForceDisplacementCurve/" + argv[1] + "/" + "log.txt");
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
        else if (script_flag == 3) // batch generate force displacement statistcs
        {
            T dp = 0.02;
            bool valid_structure = tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTKNew/" + argv[1] + ".vtk", true);
            if (!valid_structure)
                return 0;
            std::cout << "valid structure " << argv[1] << std::endl;
            
            // tiling.solver.verbose = false;
            TV min_corner, max_corner;
            tiling.solver.computeBoundingBox(min_corner, max_corner);
            
            TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
            TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);
            T dy = max_corner[1] - min_corner[1];
            tiling.solver.penalty_weight = 1e4;
            std::vector<T> displacements;
            std::vector<T> force_norms;
            std::vector<T> residual_norms;
            for (T percent = 0.0; percent < 0.8 + dp; percent += dp)
            {
                std::string obj_file = base_folder + "ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".obj";
                Eigen::MatrixXd _V; Eigen::MatrixXi _F;
                igl::readOBJ(obj_file, _V, _F);
                
                for (int i = 0; i < _V.rows(); i++)
                {
                    tiling.solver.deformed.segment<2>(i*2) = _V.row(i).segment<2>(0);
                    tiling.solver.u = tiling.solver.deformed - tiling.solver.undeformed;
                }
                tiling.solver.updateIPCVertices(tiling.solver.u);
                tiling.solver.penalty_pairs.clear();
                tiling.solver.addPenaltyPairsBox(min1, max1, TV(0, -percent * dy));
                VectorXT interal_force(tiling.solver.num_nodes * 2);
                interal_force.setZero();
                tiling.solver.addBCPenaltyForceEntries(tiling.solver.penalty_weight, interal_force);
                VectorXT residual(tiling.solver.num_nodes * 2); residual.setZero();
                tiling.solver.computeResidual(tiling.solver.u, residual);
                displacements.push_back(percent * dy);
                force_norms.push_back(interal_force.norm());
                residual_norms.push_back(residual.norm());
                break;
            }
            
            std::ofstream out(base_folder + "ForceDisplacementCurve/" + argv[1] + "/" + "log.txt");
            out << "displacement in cm" << std::endl;
            for (T v : displacements)
                out << v << " ";
            out << std::endl;
            out << "force in N" << std::endl;
            for (T v : force_norms)
                out << v << " ";
            out << std::endl;
            out << "residual norm" << std::endl;
            for (T v : residual_norms)
                out << v << " ";
            out << std::endl;
            out.close();
        }
        else if (script_flag == 4) //batch resume
        {
            T dp = 0.02;
            bool valid_structure = tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTKNew/" + argv[1] + ".vtk", true);
            // if (!valid_structure)
            //     return 0;
            // std::cout << "valid structure " << argv[1] << std::endl;
            TV min_corner, max_corner;
            tiling.solver.computeBoundingBox(min_corner, max_corner);
            TV min1(min_corner[0] - 1e-6, max_corner[1] - 1e-6);
            TV max1(max_corner[0] + 1e-6, max_corner[1] + 1e-6);
            T dy = max_corner[1] - min_corner[1];
            tiling.solver.penalty_weight = 1e4;
            std::vector<T> displacements;
            std::vector<T> force_norms;
            std::vector<T> residual_norms;
            VectorXT u_prev = tiling.solver.u;
            T last_percentage = 1.0;
            for (T percent = 0.0; percent < 0.8 + dp; percent += dp)
            {
                std::string obj_file = base_folder + "ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".obj";
                if (!fileExist(obj_file))
                    break;
                else
                {
                    Eigen::MatrixXd _V; Eigen::MatrixXi _F;
                    igl::readOBJ(obj_file, _V, _F);
                    for (int i = 0; i < _V.rows(); i++)
                    {
                        tiling.solver.deformed.segment<2>(i*2) = _V.row(i).segment<2>(0);
                        tiling.solver.u = tiling.solver.deformed - tiling.solver.undeformed;
                    }
                    u_prev = tiling.solver.u;
                    tiling.solver.computeIPCRestData();
                    tiling.solver.penalty_pairs.clear();
                    tiling.solver.addPenaltyPairsBox(min1, max1, TV(0, -percent * dy));
                    tiling.solver.staticSolve();
                    VectorXT interal_force(tiling.solver.num_nodes * 2);
                    interal_force.setZero();
                    tiling.solver.addBCPenaltyForceEntries(tiling.solver.penalty_weight, interal_force);
                    displacements.push_back(percent * dy);
                    force_norms.push_back(interal_force.norm());
                    VectorXT residual(tiling.solver.num_nodes * 2); residual.setZero();
                    tiling.solver.computeResidual(tiling.solver.u, residual);
                    residual_norms.push_back(residual.norm());
                    std::cout << residual.norm() << std::endl;
                    tiling.solver.saveToOBJ(base_folder + "ForceDisplacementCurve/" + argv[1] + "/" + std::to_string(percent) + ".obj");
                }
            }
            
            std::ofstream out(base_folder + "ForceDisplacementCurve/" + argv[1] + "/" + "log.txt");
            out << "displacement in cm" << std::endl;
            for (T v : displacements)
                out << v << " ";
            out << std::endl;
            out << "force in N" << std::endl;
            for (T v : force_norms)
                out << v << " ";
            out << std::endl;
            out << "residual norm" << std::endl;
            for (T v : residual_norms)
                out << v << " ";
            out << std::endl;
            out.close();
        }
        else if (script_flag == 5) // batch rendering
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);

            SimulationApp app(tiling);

            app.setViewer(viewer, menu);
            // std::string folder = base_folder + "SandwichStructure/ForceDisplacementCurve/";
            
            int width = 2000, height = 2000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            T dp = 0.02;
            bool valid_structure = tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTKNew/" + argv[1] + ".vtk", true);
            
            for (T percent = 0.0; percent < 0.8 + dp; percent += dp)
            {
                std::string obj_file = base_folder + "ForceDisplacementCurve/" + argv[1] + "/" +  std::to_string(percent) + ".obj";
                if (!fileExist(obj_file))
                    break;
                bool load_success = igl::readOBJ(obj_file, app.V, app.F);
                for (int i = 0; i < app.V.rows(); i++)
                {
                    tiling.solver.deformed.segment<2>(i*2) = app.V.row(i).segment<2>(0);
                    tiling.solver.u = tiling.solver.deformed - tiling.solver.undeformed;
                }
                VectorXT PK_stress;
                tiling.solver.computeFirstPiola(PK_stress);
                Eigen::MatrixXd C_jet(tiling.solver.num_ele, 3);
                Eigen::MatrixXd value(tiling.solver.num_ele, 3);
                value.col(0) = PK_stress; value.col(1) = PK_stress; value.col(2) = PK_stress;
                std::cout << PK_stress.maxCoeff() << std::endl;
                igl::jet(value, 0, 6000, C_jet);
                app.C = C_jet;
                if (percent == 0.0)
                    viewer.core().align_camera_center(app.V);
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                viewer.data().set_colors(app.C);
                
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                
                igl::png::writePNG(R,G,B,A, base_folder + "/ForceDisplacementCurve/" + argv[1] + "/stress_" +  std::to_string(percent) + ".png");
            }
            viewer.launch_shut();
        }
    }
    else
    {
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);
        // tiling.extrudeToMesh("/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/0.txt", 
        //     "/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/0_3d.msh");
        SimulationApp app(tiling);
        tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTKNew/0.vtk", true);
        app.setViewer(viewer, menu);
        viewer.launch();
        // simApp();
        // renderScene();
        // generateFDCurveSingleStructure();
    }
    return 0;
}