#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>

#include "../include/app.h"
#include <boost/filesystem.hpp>

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;

int main(int argc, char** argv)
{
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    
    auto renderScene = [&]()
    {
        SimulationApp app(tiling);

        app.setViewer(viewer, menu);
        std::string folder = "/home/yueli/Documents/ETH/SandwichStructure/TilingRendering/";
        int width = 2000, height = 2000;
        CMat R(width,height), G(width,height), B(width,height), A(width,height);
        viewer.core().background_color.setOnes();
        viewer.data().set_face_based(true);
        viewer.data().shininess = 1.0;
        viewer.data().point_size = 10.0;
        viewer.core().camera_zoom *= 1.4;
        viewer.launch_init();
        for (int i = 160; i < 180; i++)
        {
            tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTK/"+std::to_string(i)+".vtk", true);
            app.updateScreen(viewer);
            viewer.core().align_camera_center(app.V);
            app.updateScreen(viewer);
            viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
            A.setConstant(255);
            igl::png::writePNG(R,G,B,A, folder + std::to_string(i)+".png");
        }
        viewer.launch_shut();
    };

    auto simApp = [&]()
    {
        SimulationApp app(tiling);
        tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/TilingVTK/7.vtk", true);
        app.setViewer(viewer, menu);
        viewer.launch();
    };

    auto generateForceDisplacementCurve = [&]()
    {
        tiling.solver.penalty_pairs.clear();
        std::string base_folder = "/home/yueli/Documents/ETH/SandwichStructure/";
        // std::vector<int> candidates = {7, 8, 20, 
        //     44, 45, 51, 66, 70,
        //     71, 78, 79, 80,100, 101, 
        //     102, 108, 130, 131, 140, 141, 142,
        //     144, 150, 161, 166, 167, 
        //     196, 224, 231, 234, 235, 
        //     400, 401, 417, 428, 4249, 444, 
        //     466, 473, 479, 516, 523, 545, 550,
        //     554, 587, 588, 613, 653};

        std::vector<int> candidates = {7, 8};

        
        for (int candidate : candidates)
        {
            std::cout << std::endl;
            std::cout << "################## STRUCTURE " << candidate << " ########################" << std::endl;
            tiling.initializeSimulationDataFromFiles(base_folder + "TilingVTK/" + std::to_string(candidate)+ ".vtk", true);
            std::string sub_dir = base_folder + "ForceDisplacementCurve/" + std::to_string(candidate);
            boost::filesystem::create_directories(sub_dir);
            std::string result_folder = sub_dir;
            tiling.generateForceDisplacementCurve(result_folder + "/");
        }
    };

    generateForceDisplacementCurve();
    // simApp();
    // renderScene();    


    return 0;
}