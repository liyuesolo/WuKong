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

static bool enable_selection = false;
static bool show_cylinder = false;
static bool show_bc = false;
static bool incremental = false;
static bool tetgen = false;
static int modes = 0;

bool static_solve = false;

bool check_modes = false;
double t = 0.0;
using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Tiling3D tiling;

Eigen::MatrixXd evectors;
Eigen::VectorXd evalues;

int static_solve_step = 0;

auto loadEigenVectors = [&]()
{
    std::ifstream in("/home/yueli/Documents/ETH/WuKong/fem_eigen_vectors.txt");
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
    tiling.solver.generateMeshForRendering(V, F, C);
    
    if (show_cylinder)
    {
        T radius = 1.0 / tiling.solver.curvature;
        TV K1_dir(std::cos(tiling.solver.bending_direction), 0.0, -std::sin(tiling.solver.bending_direction));
        tiling.solver.appendCylinder(V, F, C, tiling.solver.center  - TV(0, radius, 0), K1_dir, radius);
    }

    if (show_bc)
    {
        int nf_current = F.rows();
        tiling.solver.updateSphere();
        tiling.solver.appendMesh(V, F, tiling.solver.sphere_vertices, tiling.solver.sphere_faces);
        int nf_sphere = tiling.solver.sphere_faces.rows();
        MatrixXd sphere_color(nf_sphere, 3);
        sphere_color.setZero();
        sphere_color.col(0).setConstant(1.0);
        C.conservativeResize(F.rows(), 3);
        C.block(nf_current, 0, nf_sphere, 3) = sphere_color;
    }

    viewer.data().clear();
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
            if (ImGui::Checkbox("SelectVertex", &enable_selection))
            {
                // updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowCylinder", &show_cylinder))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowBC", &show_bc))
            {
                updateScreen(viewer);
            } 
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("TetGen", &tetgen))
            {
                
            }
            if (ImGui::Checkbox("IncrementalLoading", &incremental))
            {
                
            }
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

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && check_modes)
        {
            tiling.solver.deformed = tiling.solver.undeformed + tiling.solver.u + evectors.col(modes) * std::sin(t);
            updateScreen(viewer);
            t += 0.1;
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = tiling.solver.staticSolveStep(static_solve_step);
            if (finished)
            {
                viewer.core().is_animating = false;
                T bending_stiffness = tiling.solver.computeBendingStiffness();
                std::cout << "bending stiffness " << bending_stiffness << std::endl;
            }
            else 
                static_solve_step++;
            updateScreen(viewer);
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
            if (incremental)
                tiling.solver.incrementalLoading();
            else
                viewer.core().is_animating = true;
                // tiling.solver.staticSolve();    
            updateScreen(viewer);
            return true;
        case '1':
            check_modes = true;
            tiling.solver.computeLinearModes();
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
        }
    };

    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (!enable_selection)
            return false;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;

        for (int i = 0; i < tiling.solver.num_nodes; i++)
        {
            Vector<T, 3> pos = tiling.solver.deformed.segment<3>(i * 3);
            Eigen::MatrixXd x3d(1, 3); x3d.setZero();
            x3d.row(0).segment<3>(0) = pos;

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

    tiling.initializeSimulationData(tetgen);
    // tiling.solver.checkTotalGradientScale(true);
    // tiling.solver.checkTotalHessianScale(true);
    // tiling.solver.runBendingHomogenization();
    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 25.0;

    viewer.core().align_camera_center(V);
    viewer.core().animation_max_fps = 24.;
    // key_down(viewer,'0',0);
    viewer.core().is_animating = false;
    viewer.launch();

    return 0;
}