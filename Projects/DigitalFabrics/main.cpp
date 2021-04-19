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
Eigen::MatrixXd C;

static bool tileUnit = false;
static bool showUnit = false;
static bool showStretching = false;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        eol_sim.advanceOneStep();
        if(tileUnit)
            eol_sim.buildPeriodicNetwork(V, F, C);
        else
            eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        if (showUnit)
            viewer.data().set_colors(C);
    }
    // viewer.data().set_face_based(true);
    return false;
}

enum TestCase{
    FiveNodes, Bending, Stretching, Shearing, GridScene,
    PlanePBC
};

const char* test_case_names[] = {
    "FiveNodes", "Bending", "Stretching", "Shearing", "GridScene",
    "PlanePBC"
};


int main()
{
    int n_test_case = sizeof(test_case_names)/sizeof(const char*);
    

    static TestCase test = FiveNodes;
    TestCase test_current = PlanePBC; // set to be a different from above or change the above one to be a random one

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
        else if (test == Shearing)
        {
            eol_sim.buildShearingTest();
        }
        else if (test == PlanePBC)
        {
            eol_sim.buildPlanePeriodicBCScene();
        }
        eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
    };

    igl::opengl::glfw::Viewer viewer;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        menu.draw_viewer_menu();
        if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
        {   
            ImGui::Combo("TestCase", (int *)(&test_current), test_case_names, n_test_case);
            if(test != test_current)
            {
                test = test_current;
                setupScene(viewer);
            }
        }
        if (ImGui::CollapsingHeader("PeriodicBC", ImGuiTreeNodeFlags_DefaultOpen))
        {   
            if (ImGui::Checkbox("tileUnit", &tileUnit))
            {
                if(tileUnit)
                    eol_sim.buildPeriodicNetwork(V, F, C);
                else
                    eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if(showUnit)
                    viewer.data().set_colors(C);
            }
            if (ImGui::Checkbox("showUnit", &showUnit))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if(showUnit)
                    viewer.data().set_colors(C);
            }

        }
        if (ImGui::CollapsingHeader("Deformation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("showStretching", &showStretching))
            {
                
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if (showStretching)
                {
                    eol_sim.getColorFromStretching(C);
                    viewer.data().set_colors(C);
                }
                
            }
            
        }
        if (ImGui::Button("Solve", ImVec2(-1,0)))
        {
            eol_sim.advanceOneStep();
            if(tileUnit)
                eol_sim.buildPeriodicNetwork(V, F, C);
            else
                eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            if(showUnit)
                viewer.data().set_colors(C);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            eol_sim.resetScene();
            if(tileUnit)
                eol_sim.buildPeriodicNetwork(V, F, C);
            else
                eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            if(showUnit)
                viewer.data().set_colors(C);
        }
    };
    
    viewer.data().set_face_based(true);
    setupScene(viewer);
    viewer.callback_key_down = &key_down;
    key_down(viewer,'0',0);
    viewer.launch();

    //================== Run Diff Test ==================
    // eol_sim.buildPlanePeriodicBCScene();
    // eol_sim.runDerivativeTest();
    return 0;
}