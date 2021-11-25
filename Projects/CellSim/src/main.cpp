#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>


#include "../include/VertexModel.h"
#include "../include/Simulation.h"
#include "../include/Misc.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

using TV = Vector<double, 3>;

Simulation simulation;

static bool show_rest = false;
static bool show_membrane = false;
static bool split = false;
static int modes = 0;
double t = 0.0;


Eigen::MatrixXd evectors;
Eigen::VectorXd evalues;

auto loadEigenVectors = [&]()
{
    std::ifstream in("/home/yueli/Documents/ETH/WuKong/cell_eigen_vectors.txt");
    int row, col;
    in >> row >> col;
    evectors.resize(row, col);
    evalues.resize(col);
    double entry;
    for (int i = 0; i < col; i++)
        in >> evalues[i];
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            in >> evectors(i, j);
    in.close();
};

auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
{
    simulation.generateMeshForRendering(V, F, C, show_rest, split);

    // viewer.data_list[0].clear();
    // viewer.data_list[0].set_mesh(V, F);
    // viewer.data_list[0].set_colors(C);

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);

    if (show_membrane)
    {
        Eigen::MatrixXd bounding_surface_samples;
        Eigen::MatrixXd bounding_surface_samples_color;
        simulation.sampleBoundingSurface(bounding_surface_samples);
        bounding_surface_samples_color = bounding_surface_samples;
        for (int i = 0; i < bounding_surface_samples.rows(); i++)
            bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
};


int main()
{
    // saveOBJPrism(6);

    igl::opengl::glfw::Viewer viewer;

    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        // if (ImGui::Checkbox("ShowRest", &show_rest))
        // {
        //     updateScreen(viewer);
        // }
        if (ImGui::Checkbox("Split", &split))
        {
            updateScreen(viewer);
        }
        if (ImGui::Checkbox("ShowMembrane", &show_membrane))
        {
            updateScreen(viewer);
        }
        if (ImGui::Button("StaticSolve", ImVec2(-1,0)))
        {
            simulation.staticSolve();
            updateScreen(viewer);
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            simulation.deformed = simulation.undeformed;
            simulation.u.setZero();
            updateScreen(viewer);
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < simulation.cells.num_nodes; i++)
        {
            Vector<T, 3> pos = simulation.deformed.template segment<3>(i * 3);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).template segment<3>(0) = pos;

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

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating)
        {
            simulation.deformed = simulation.undeformed + simulation.u + evectors.col(modes) * std::sin(t);

            t += 0.1;
            viewer.data().clear();
            simulation.generateMeshForRendering(V, F, C, show_rest);
            viewer.data().set_mesh(V, F);     
            viewer.data().set_colors(C);
        }
        return false;
    };

    viewer.callback_key_pressed = 
        [&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
    {
        switch(key)
        {
        default: 
            return false;
        case ' ':
            simulation.staticSolve();
            updateScreen(viewer);
            return true;
        case '1':
            simulation.computeLinearModes();
            loadEigenVectors();
            
            for (int i = 0; i < evalues.rows(); i++)
            {
                if (evalues[i] > 1e-6)
                {
                    modes = i;
                    return true;
                }
            }
            return true;
        case '2':
            modes++;
            modes = (modes + evectors.cols()) % evectors.cols();
            std::cout << "modes " << modes << std::endl;
            return true;
        case '3': //check modes at equilirium after static solve
            loadEigenVectors();
            modes = 0;
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        }
    };

    simulation.initializeCells();
    simulation.generateMeshForRendering(V, F, C);

    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
    // viewer.core().animation_max_fps = 24.;
    

    viewer.launch();
    return 0;
}