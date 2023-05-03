#include "../include/App.h"
#include "../include/NeuralElement.h"
int main(int argc, char** argv)
{
    FEMSolver<2> solver;
    SimulationApp<2> app(solver);

    auto runSimApp = [&]()
    {
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);
            
        app.setViewer(viewer, menu);
        
        viewer.launch(true, false, "WuKong viewer", 3000, 2000);
    };

    auto dataGeneration = [&]()
    {
        NeuralElement<2> neural_element(solver);
        neural_element.generateBeamSceneTrainingData("/home/yueli/Documents/ETH/WuKong/Projects/NeuralElement/python/");
    };
    runSimApp();
    // dataGeneration();
    return 0;
}