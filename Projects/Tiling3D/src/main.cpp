#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/Tiling3D.h"

#define T double 
#define dim 3

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Tiling3D tiling;

auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();

    tiling.solver.generateMeshForRendering(V, F, C);

    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
};

int main()
{
    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            tiling.solver.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            tiling.solver.deformed = tiling.solver.undeformed;
            tiling.solver.u.setZero();
            updateScreen(viewer);
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };    

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        
        switch(key)
        {
        default: 
            return false;
        case ' ':
            tiling.solver.staticSolve();
            return true;
        }
    };

    tiling.initializeSimulationData();
    // tiling.solver.checkTotalGradientScale(true);
    // tiling.solver.checkTotalHessianScale(true);

    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.launch();

    return 0;
}