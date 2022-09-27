#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/writeOBJ.h>
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
            hex_fem_solver.E = 0.0;
            hex_fem_solver.nu = 0.3;
            hex_fem_solver.updateLameParams();
            hex_fem_solver.KL_stiffness = 1e6;
            hex_fem_solver.KL_stiffness_shear = 0;
            hex_fem_solver.addCornerVtxToDirichletVertices(flag);
            // hex_fem_solver.setBCBendCorner(4.0, 0.0);
            hex_fem_solver.penaltyInPlane(0, 0.2);
            app.setViewer(viewer, menu);
            viewer.launch();
        };
        // run3DSim();
        // tiling.sampleUniAxialStrainAlongDirection("/home/yueli/Documents/ETH/SandwichStructure/Server/", 50, TV(0.8, 1.2), 0.24);
        // tiling.generatseGreenStrainSecondPKPairs("/home/yueli/Documents/ETH/SandwichStructure/TrainingData/WithEnergy/");
        // fem_solver.pbc_translation_file = "/home/yueli/Documents/ETH/SandwichStructure/Server/27/structure_translation.txt";
        // tiling.initializeSimulationDataFromFiles("/home/yueli/Documents/ETH/SandwichStructure/Server/27/structure.vtk", PBC_XY);
        runSimApp();
    }
    return 0;
}