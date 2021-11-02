#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "Tiling3D.h"

#define T double 
#define dim 3

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

int main()
{
    Tiling3D tiling;

    tiling.getMeshForPrintingWithLines(V, F, C);
    
    // igl::opengl::glfw::Viewer viewer;


    // // viewer.plugins.push_back(&menu);
    

    // viewer.core().background_color.setOnes();
    // viewer.data().set_face_based(true);
    // viewer.data().shininess = 1.0;
    // viewer.data().point_size = 25.0;

    // viewer.data().set_mesh(V, F);     
    // viewer.data().set_colors(C);

    // viewer.core().align_camera_center(V);
    // viewer.core().animation_max_fps = 24.;
    // // key_down(viewer,'0',0);
    // viewer.launch();

    return 0;
}