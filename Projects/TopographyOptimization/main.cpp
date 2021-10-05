#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "TopographyOptimization.h"



#define T double 
#define dim 3

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd C;

Eigen::MatrixXd vtx_base;
Eigen::MatrixXi face_base;

using TV = Vector<T, dim>;
// FEMSolver<T, dim> solver;
ShellFEMSolver<T, dim> solver;

TopographyOptimization<T, dim, ShellFEMSolver<T, dim>> topo_opt(solver);
// TopographyOptimization<T, dim, FEMSolver<T, dim>> topo_opt(solver);

static int modes = 6;
double t = 0.0;

static bool load_baseline = false;


Eigen::MatrixXd evectors;
auto loadEigenVectors = [&]()
{
    std::ifstream in("/home/yueli/Documents/ETH/WuKong/bead_eigen_vectors.txt");
    int row, col;
    in >> row >> col;
    evectors.resize(row, col);
    double entry;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            in >> evectors(i, j);
    in.close();
};


auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
{
    viewer.data().clear();
    topo_opt.generateMeshForRendering(V, F, C);
    if (load_baseline)
    {
        int v_start = V.rows(), f_start = F.rows();
        Eigen::MatrixXi offset_ones(face_base.rows(), 3);
        offset_ones.setOnes();
        offset_ones *= v_start;

        V.conservativeResize(V.rows() + vtx_base.rows(), 3);
        F.conservativeResize(F.rows() + face_base.rows(), 3);
        C.conservativeResize(C.rows() + face_base.rows(), 3);

        V.block(v_start, 0, vtx_base.rows(), 3) = vtx_base;
        F.block(f_start, 0, face_base.rows(), 3) = face_base + offset_ones;

        C.block(f_start, 0, face_base.rows(), 3).setZero();
        C.block(f_start, 0, face_base.rows(), 1).setOnes();
    }
    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);
};

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == ' ')
    {
        topo_opt.forward();
        
        updateScreen(viewer);
        return true;
    }
    else if (key == '1')
    {
        std::cout << "1 pressed" << std::endl;
        topo_opt.solver.computeEigenMode();
        loadEigenVectors();
        return true;
    }
    else if (key == '2')
    {
        modes++;
        modes = (modes + evectors.cols()) % evectors.cols();
        std::cout << "modes " << modes << std::endl;
        return true;
    }
    else if (key == 'a')
    {
        viewer.core().is_animating = !viewer.core().is_animating;
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
        if (ImGui::CollapsingHeader("Forward", ImGuiTreeNodeFlags_DefaultOpen))
        {

        }
        if (ImGui::CollapsingHeader("Inverse", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("LoadBaseline", &load_baseline))
            {
                if (load_baseline)
                {
                    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/TopographyOptimization/Data/current_mesh.obj", vtx_base, face_base);
                }
                updateScreen(viewer);
            }
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

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating)
        {
            solver.u = evectors.col(modes) * std::sin(t);
            t += 0.1;
            viewer.data().clear();
            topo_opt.generateMeshForRendering(V, F, C);
            viewer.data().set_mesh(V, F);     
            viewer.data().set_colors(C);
        }
        return false;
    };

    auto setupScene = [&](igl::opengl::glfw::Viewer& viewer)
    {
        topo_opt.initializeScene(0);
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