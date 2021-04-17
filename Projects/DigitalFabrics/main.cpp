#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "EoLRodSim.h"


#define T double
#define dim 2


bool USE_VIEWER = true;

EoLRodSim<T, dim> eol_sim;


Eigen::MatrixXd V;
Eigen::MatrixXi F;



bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        eol_sim.advanceOneStep();
        eol_sim.buildMeshFromRodNetwork(V, F);
    }
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    return false;
}

enum TestCase{
    FiveNodes, Bending, Stretching, Shearing, GridScene, DerivativeCheck,
    PlanePBC
};

const char* test_case_names[] = {
    "FiveNodes", "Bending", "Stretching", "Shearing", "GridScene", "DerivativeCheck",
    "PlanePBC"
};


int main()
{
    int n_test_case = sizeof(test_case_names)/sizeof(const char*);

    static TestCase test = Shearing;
    TestCase test_current = Shearing;

    auto setupScene = [&](igl::opengl::glfw::Viewer& viewer)
    {
        if(test == FiveNodes)
        {
            assert(dim == 2);
            eol_sim.build5NodeTestScene();
        }
        else if (test == GridScene)
        {
            assert(dim == 3);
            eol_sim.buildRodNetwork(2, 2);    
            // eol_sim.addBCStretchingTest();
            eol_sim.addBCShearingTest();    
        }
        else if (test == Bending)
        {
            assert(dim == 2);
            eol_sim.buildLongRodForBendingTest();
        }
        else if (test == DerivativeCheck)
        {
            eol_sim.build5NodeTestScene();
            eol_sim.runDerivativeTest();
        } 
        else if (test == Shearing)
        {
            eol_sim.buildShearingTest();
        }
        else if (test == PlanePBC)
        {

        }
        eol_sim.buildMeshFromRodNetwork(V, F);
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
    };

    igl::opengl::glfw::Viewer viewer;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // std::string display_options = "";
            // for (auto name : test_case_names)
            //     display_options += name + "\0";
            // display_options += "\0";
            // std::cout << display_options << std::endl;
            
            ImGui::Combo("TestCase", (int *)(&test_current), test_case_names, n_test_case);
            if(test != test_current)
            {
                test = test_current;
                setupScene(viewer);
            }
        }
        if (ImGui::Button("Solve", ImVec2(-1,0)))
        {
            eol_sim.advanceOneStep();
            eol_sim.buildMeshFromRodNetwork(V, F);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            eol_sim.resetScene();
            eol_sim.buildMeshFromRodNetwork(V, F);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
        }
    };
    
    viewer.data().set_face_based(true);
    setupScene(viewer);
    viewer.callback_key_down = &key_down;
    key_down(viewer,'0',0);
    viewer.launch();
    return 0;
}