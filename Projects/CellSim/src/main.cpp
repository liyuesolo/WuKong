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

using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Simulation simulation;
ObjNucleiTracking obj(simulation);

SensitivityAnalysis sa(simulation, obj);


int main()
{
    auto loadDrosophilaData = [&]()
    {
        DataIO data_io;
        // data_io.loadDataFromTxt("/home/yueli/Downloads/drosophila_data/drosophila_side_2_tracks_071621.txt");
        // data_io.loadDataFromBinary("/home/yueli/Downloads/drosophila_data/drosophila_side2_time_xyz.dat", 
        //     "/home/yueli/Downloads/drosophila_data/drosophila_side2_ids.dat",
        //     "/home/yueli/Downloads/drosophila_data/drosophila_side2_scores.dat");
        // data_io.trackCells();
        data_io.loadTrajectories("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/trajectories.dat");
    };

    loadDrosophilaData();
    std::exit(0);
    
    simulation.initializeCells();
    simulation.cells.tet_vol_barrier_w = 1e-22;
    // obj.initializeTarget();
    obj.loadTarget("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_test.txt");

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    

    auto runSim = [&]()
    {
        SimulationApp sim_app(simulation);
        sim_app.setViewer(viewer, menu);
        simulation.verbose = true;
        simulation.save_mesh = false;
        simulation.cells.tet_vol_barrier_w = 1e-22;
        viewer.launch();
    };

    auto runSA = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        // sa.optimizePerEdgeWeigths();
        // diff_sim_app.runOptimization();
        // sa.initialize();
        // sa.svdOnSensitivityMatrix();
        // sa.eigenAnalysisOnSensitivityMatrix();
        // sa.dxFromdpAdjoint();

        diff_sim_app.setViewer(viewer, menu);
        viewer.launch();
    };

    auto generateNucleiGT = [&]()
    {
        DiffSimApp diff_sim_app(simulation, sa);
        sa.initialize();
        sa.generateNucleiDataSingleFrame("/home/yueli/Documents/ETH/WuKong/Projects/CellSim/data/nuclei_single_frame_test.txt");
    };

    // runSim();
    runSA();
    // generateNucleiGT();

    return 0;
}