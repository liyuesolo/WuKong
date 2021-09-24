#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "FEMSolver.h"



#define T double 
#define dim 3

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

using TV = Vector<T, dim>;
using TVStack = Matrix<T, dim, Eigen::Dynamic>;

FEMSolver<T, dim> solver;


auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    solver.generateMeshForRendering(V, F, C);
    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);
};

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        solver.staticSolve();
        // solver.computeElementDeformationGradient3D();
        updateScreen(viewer);
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


    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < solver.num_nodes; i++)
        {
            TV pos = solver.deformed.template segment<dim>(i * dim);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).template segment<dim>(0) = pos;

            Eigen::MatrixXd pxy(1, 3);
            igl::project(x3d, viewer.core().view, viewer.core().proj, viewer.core().viewport, pxy);
            if(abs(pxy.row(0)[0]-x)<20 && abs(pxy.row(0)[1]-y)<20)
            {
                std::cout << "selected " << i << std::endl;
                return true;
            }
        }
        return false;
    };

    auto setupScene = [&](igl::opengl::glfw::Viewer& viewer)
    {
        T dx = 0.05;
        TV min_corner = TV::Zero();
        TV max_corner = TV(1.0, dx, 1.0);
        solver.buildGrid3D(min_corner, max_corner, dx);
        solver.fixAxisEnd(0);
        auto forceFunc = [&, dx](const TV& pos, TV& force)->bool
        {
            // force = TV(0, -9.8 * std::pow(dx, dim), 0.0);
            force = TV(0, -9.8, 0.0) * 0.1;
            return true;

            if (pos[0]>max_corner[0]-1e-4 && pos[1] < min_corner[1] + 1e-4)
            {
                force = TV(0, -1e-4, 0);
                return true;
            }
            force = TV::Zero();
            return false;
        };

        auto displaceFunc = [&, dx](const TV& pos, TV& delta)->bool
        {
            if (pos[0]>max_corner[0]-1e-4)// && pos[1] < min_corner[1] + 1e-4)
            {
                // delta = TV(-0.02, -0.05, 0);
                delta = TV(0.1 * dx, 0, 0);
                return true;
            }
            return false;
        };
        // solver.addDirichletLambda(displaceFunc);
        solver.addNeumannLambda(forceFunc, solver.f);
        
        // solver.vol = 1.0;
        updateScreen(viewer);
    };

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;
    setupScene(viewer);
    // solver.derivativeTest();
    
    viewer.callback_key_down = &key_down;
    // viewer.data().show_lines = false;
    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    key_down(viewer,'0',0);
    viewer.launch();

    

    return 0;
}