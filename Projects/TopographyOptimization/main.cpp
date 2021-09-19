#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "TopoOptSimulation.h"
#include "BoundaryCondition.h"

using namespace ZIRAN;

#define T double 
#define dim 3

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

using TV = Vector<T, dim>;
using TVStack = Matrix<T, dim, Eigen::Dynamic>;

TopographyOptimization<T, dim> topo_opt;
BoundaryCondition<T, dim, TopographyOptimization<T, dim>> bc = BoundaryCondition<T, dim, TopographyOptimization<T, dim>>(topo_opt);

auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer, const TVStack& u)
{
    viewer.data().clear();
    topo_opt.getMeshForRendering(V, F, C, u);
    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);
};

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        TVStack u = TVStack::Zero(dim, topo_opt.num_nodes);
        topo_opt.solveDisplacementField(bc, u);
        updateScreen(viewer, u);
        return true;
    }
    return false;
};

int main()
{   

    igl::opengl::glfw::Viewer viewer;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };

    auto setupScene = [&](igl::opengl::glfw::Viewer& viewer)
    {
        T dx = 0.025;
        TV min_corner = TV::Zero();
        TV max_corner = TV(1, dx, 1);
        topo_opt.initializeDesignPad(dx, min_corner, max_corner);
        topo_opt.addBox(min_corner, max_corner, 1e5, 0.3, 1, false);
        topo_opt.finalizeDesignDomain();
        topo_opt.reinitializeGrid();
        bc.addDirichletWall(TV::Zero(), TV(1, 0, 0), TV::Zero());
        auto forceLoad = [&, dx](const TV& x)
        {
            if (x[0] > 1 - dx)
                return TV(0, -0.001, 0);
            return TV(0, 0, 0);
        };

        bc.addNeumannLambda(forceLoad);
        TVStack u = TVStack::Zero(dim, topo_opt.num_nodes);
        updateScreen(viewer, u);
    };

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;
    setupScene(viewer);
    viewer.callback_key_down = &key_down;
    // viewer.data().show_lines = false;
    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    key_down(viewer,'0',0);
    viewer.launch();

    

    return 0;
}