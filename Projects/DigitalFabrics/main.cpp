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

Eigen::MatrixXd nodes;

static bool tileUnit = false;
static bool showUnit = false;
static bool showStretching = false;
static bool show_index = false;
static bool show_original = false;
static bool per_yarn = true;

static int n_rod_per_yarn = 4;

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
        if (per_yarn)
        {
            eol_sim.getColorPerYarn(C, n_rod_per_yarn);
            viewer.data().set_colors(C);
        } 
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
            eol_sim.buildPlanePeriodicBCScene3x3();
        }
        eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        if (per_yarn)
        {
            eol_sim.getColorPerYarn(C, n_rod_per_yarn);
            viewer.data().set_colors(C);
        } 
        
    };

    igl::opengl::glfw::Viewer viewer;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        // menu.draw_viewer_menu();
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
            if (ImGui::Checkbox("TileUnit", &tileUnit))
            {
                if(tileUnit)
                    eol_sim.buildPeriodicNetwork(V, F, C);
                else
                    eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if(showUnit)
                    viewer.data().set_colors(C);
                if (per_yarn)
                {
                    eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                    C.conservativeResize(F.rows(), 3);
                    tbb::parallel_for(0, eol_sim.n_rods * 40, [&](int i){
                        for(int j = 1; j < std::floor(F.rows()/eol_sim.n_rods/40); j++)
                        {
                            C.row(j * eol_sim.n_rods * 40 + i) = C.row(i);
                        }
                    });
                    viewer.data().set_colors(C);
                } 
            }
            if (ImGui::Checkbox("ShowUnit", &showUnit))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if(showUnit)
                    viewer.data().set_colors(C);
                if (per_yarn)
                {
                    eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                    viewer.data().set_colors(C);
                } 
            }
            if (ImGui::Checkbox("ShowIndex", &show_index))
            {
                if(show_index)
                {
                    // nodes.resize(eol_sim.n_nodes,3);
                    // tbb::parallel_for(0, eol_sim.n_nodes, [&](int i)
                    // {
                    //     nodes.row(i) = eol_sim.q.col(i);
                    // });
                    // viewer.data().add_points(nodes, Eigen::RowVector3d(1,1,1));
                    for (int i = 0; i < eol_sim.n_nodes; i++)
                        viewer.data().add_label(Eigen::Vector3d(eol_sim.q(0, i)-1, 0, eol_sim.q(1, i)-1), std::to_string(i));
                    viewer.data().show_custom_labels = true;
                }
            }
            if (ImGui::Checkbox("ShowEulerianRest", &show_original))
            {
                if(show_original)
                {
                    // nodes.resize(eol_sim.n_nodes,3);
                    // tbb::parallel_for(0, eol_sim.n_nodes, [&](int i)
                    // {
                    //     nodes.row(i) = Eigen::Vector3d(eol_sim.q(0, i)-1, 0, eol_sim.q(1, i)-1);
                    // });
                    // viewer.data().add_points(nodes, Eigen::RowVector3d(1,1,1));  
                    Eigen::MatrixXd X, x;

                    eol_sim.getEulerianDisplacement(X, x);
                    for (int i = 0; i < X.rows(); i++)
                    {
                        viewer.data().add_edges(X.row(i), x.row(i), Eigen::RowVector3d(1, 1, 1));
                    }
                    viewer.data().add_points(X, Eigen::RowVector3d(1,1,1));
                    viewer.data().add_points(x, Eigen::RowVector3d(0,0,0));  
                }
            }
        }
        if (ImGui::CollapsingHeader("ColorScheme", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("ShowStretching", &showStretching))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if (showStretching)
                {
                    eol_sim.getColorFromStretching(C);
                    viewer.data().set_colors(C);
                }   
            }
            if (ImGui::Checkbox("PerYarn", &per_yarn))
            {
                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                if (per_yarn)
                {
                    eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                    viewer.data().set_colors(C);
                }   
            }   
        }
        if (ImGui::Button("Solve", ImVec2(-1,0)))
        {
            eol_sim.advanceOneStep();
            viewer.data().clear();
            if(tileUnit)
                eol_sim.buildPeriodicNetwork(V, F, C);
            else
                eol_sim.buildMeshFromRodNetwork(V, F, eol_sim.q, eol_sim.rods, eol_sim.normal);
            viewer.data().set_mesh(V, F);
            if(showUnit)
                viewer.data().set_colors(C);
            if (per_yarn)
            {
                eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                viewer.data().set_colors(C);
            }  
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
            if (per_yarn)
            {
                eol_sim.getColorPerYarn(C, n_rod_per_yarn);
                viewer.data().set_colors(C);
            }
        }
    };
    
    //================== Run GUI ==================
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;
    
    viewer.data().line_width = 5.0;
    
    
    
    setupScene(viewer);
    viewer.callback_key_down = &key_down;
    key_down(viewer,'0',0);
    viewer.launch();

    //================== Run Diff Test ==================
    // eol_sim.buildPlanePeriodicBCScene();
    // eol_sim.runDerivativeTest();
    return 0;
}