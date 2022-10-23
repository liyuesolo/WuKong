#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>

#include "../include/app.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/Objective.h"
#include "../include/HexFEMSolver.h"
#include <boost/filesystem.hpp>

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 2>;
using TV3 = Vector<T, 3>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    tbb::task_scheduler_init init(1);
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    
    if (argc > 1)
    {
        FEMSolver fem_solver;
        int IH = std::stoi(argv[1]);
        std::string result_folder = argv[2];
        int n_params = std::stoi(argv[3]);
        std::vector<T> params(n_params);
        for (int i = 0; i < n_params; i++)
        {
            params[i] = std::stod(argv[4+i]);
        }
        int resume_start = std::stoi(argv[4 + n_params]);
        tiling.generateGreenStrainSecondPKPairsServer(params, IH, "", result_folder, resume_start);
        
    }
    else
    {
        auto runSimApp = [&]()
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
                
            app.setViewer(viewer, menu);
            viewer.launch();
        };
        
        auto run3DSim = [&]()
        {
            

            HexFEMSolver hex_fem_solver;
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            Simulation3DApp app(hex_fem_solver);
            T dx = 0.1;
            hex_fem_solver.buildGrid3D(TV3::Zero(), TV3(1.0, dx, 1.0), dx);
            Vector<bool, 4> flag;
            flag << true, false, true, false;
            // hex_fem_solver.E = 0.0;
            // hex_fem_solver.nu = 0.3;
            // hex_fem_solver.updateLameParams();
            hex_fem_solver.KL_stiffness = 1e6;
            hex_fem_solver.KL_stiffness_shear = 0;
            hex_fem_solver.addCornerVtxToDirichletVertices(flag);
            // hex_fem_solver.setBCBendCorner(4.0, 0.0);
            hex_fem_solver.penaltyInPlane(0, 0.2);
            app.setViewer(viewer, menu);
            viewer.launch();
        };

        auto renderScene = [&]()
        {
            igl::opengl::glfw::Viewer viewer;
            igl::opengl::glfw::imgui::ImGuiMenu menu;

            viewer.plugins.push_back(&menu);
            SimulationApp app(tiling);
            
            app.setViewer(viewer, menu);
            std::string folder = "/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/";
            int width = 2000, height = 2000;
            CMat R(width,height), G(width,height), B(width,height), A(width,height);
            viewer.core().background_color.setOnes();
            viewer.data().set_face_based(true);
            viewer.data().shininess = 1.0;
            viewer.data().point_size = 10.0;
            viewer.core().camera_zoom *= 1.4;
            viewer.launch_init();
            int n_sp_strain = 50;
            TV range_strain(0.001, 0.2);
            T delta_strain = (range_strain[1] - range_strain[0]) / T(n_sp_strain);
            int cnt = 0;
            for (T strain = range_strain[0]; strain < range_strain[1] + delta_strain; strain += delta_strain)
            {
                igl::readOBJ(folder + std::to_string(strain) + ".obj", app.V, app.F);
                    
                viewer.data().clear();
                viewer.data().set_mesh(app.V, app.F);
                app.C.resize(app.F.rows(), 3);
                app.C.col(0).setConstant(0.0); app.C.col(1).setConstant(0.3); app.C.col(2).setConstant(1.0);
                viewer.data().set_colors(app.C);
                viewer.core().align_camera_center(app.V);
                viewer.core().draw_buffer(viewer.data(),true,R,G,B,A);
                A.setConstant(255);
                igl::png::writePNG(R,G,B,A, folder + std::to_string(cnt++)+".png");
            }
        };
        // renderScene();
        // run3DSim();
        // tiling.generateNHHomogenousData("/home/yueli/Documents/ETH/SandwichStructure/Homo/");
        tiling.sampleDirectionWithUniaxialStrain("/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/", 50, TV(0, M_PI), 1.05);
        // tiling.sampleUniAxialStrainAlongDirection("/home/yueli/Documents/ETH/SandwichStructure/Server/", 50, TV(0.8, 1.2), 0.24);
        // tiling.generatseGreenStrainSecondPKPairs("/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/");
        // fem_solver.pbc_translation_file = "/home/yueli/Documents/ETH/SandwichStructure/Server/0/structure_translation.txt";
        // tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/Server/0/structure.vtk", PBC_XY);
        // tiling.sampleFixedTilingParamsAlongStrain("/home/yueli/Documents/ETH/SandwichStructure/SampleStrain/");
        // runSimApp();
    }
    return 0;
}