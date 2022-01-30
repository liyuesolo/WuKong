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
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Simulation simulation;

static bool show_rest = false;
static bool show_current = true;
static bool show_membrane = false;
static bool split = false;
static bool split_a_bit = false;
static bool yolk_only = false;
static bool show_apical_polygon = false;
static bool show_basal_polygon = false;
static bool show_contracting_edges = true;
static bool show_outside_vtx = false;
static int modes = 0;
static bool enable_selection = false;
static bool compute_energy = false;
double t = 0.0;
int compute_energy_cnt = 0;

int static_solve_step = 0;
bool check_modes = false;

int load_obj_iter_cnt = 0;

Eigen::MatrixXd evectors;
Eigen::VectorXd evalues;

Eigen::MatrixXd bounding_surface_samples;
Eigen::MatrixXd bounding_surface_samples_color;
int sdf_test_sample_idx_offset = 0;

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
    simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);

    // viewer.data_list[0].clear();
    // viewer.data_list[0].set_mesh(V, F);
    // viewer.data_list[0].set_colors(C);

    viewer.data().clear();
    // viewer.data().set_mesh(V, F);
    // viewer.data().set_colors(C);

    if (show_contracting_edges)
    {
        // viewer.data().clear();
        simulation.cells.appendCylinderOnContractingEdges(V, F, C);
    }
        
    if (show_membrane)
    {
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
    if (show_outside_vtx)
    {
        simulation.cells.getOutsideVtx(bounding_surface_samples, 
            bounding_surface_samples_color, sdf_test_sample_idx_offset);
        viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
    }
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);
    
};


int main()
{
    // saveOBJPrism(6);

    igl::opengl::glfw::Viewer viewer;

    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("SelectVertex", &enable_selection))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowCurrent", &show_current))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowRest", &show_rest))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("SplitPrism", &split))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("SplitPrismABit", &split_a_bit))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowMembrane", &show_membrane))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("YolkOnly", &yolk_only))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ContractingEdges", &show_contracting_edges))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowOutsideVtx", &show_outside_vtx))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ComputeEnergy", &compute_energy))
            {
                updateScreen(viewer);
            }
        }
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("Dynamics", &simulation.dynamic))
            {
                if (simulation.dynamic)
                    simulation.initializeDynamicsData(1e-2, 5e-2);
            }
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
        if (!enable_selection)
            return false;
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
        if(viewer.core().is_animating && check_modes)
        {
            simulation.deformed = simulation.undeformed + simulation.u + evectors.col(modes) * std::sin(t);
            if (compute_energy)
            {
                simulation.verbose = false;
                T energy = simulation.computeTotalEnergy(simulation.u, false);
                simulation.verbose = false;
                std::cout << std::setprecision(8) << "E: " << energy << std::endl;
            }
            t += 0.1;
            compute_energy_cnt++;
            
            viewer.data().clear();
            simulation.generateMeshForRendering(V, F, C, show_current, show_rest, split, split_a_bit, yolk_only);
            viewer.data().set_mesh(V, F);     
            viewer.data().set_colors(C);
            if (show_membrane)
            {
                viewer.data().set_points(bounding_surface_samples, bounding_surface_samples_color);
            }
        }
        return false;
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = simulation.advanceOneStep(static_solve_step);
            if (finished)
            {
                viewer.core().is_animating = false;
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
        VectorXT residual(simulation.num_nodes * 3);
        residual.setZero();
        switch(key)
        {
        default: 
            return false;
        case ' ':
            // simulation.cells.Gamma = 4.0;
            // simulation.cells.gamma = 10;
            // simulation.cells.alpha = 200;
            // simulation.cells.print_force_norm = false;
            // simulation.staticSolve();
            // updateScreen(viewer);
            // simulation.reset();
            // simulation.cells.gamma = 20;
            // simulation.staticSolve();
            // updateScreen(viewer);
            // simulation.reset();
            // simulation.cells.gamma = 10;
            // simulation.staticSolve();
            // updateScreen(viewer);
            // simulation.reset();
            // simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_18.obj");
            // simulation.reset();
            // updateScreen(viewer);
            // simulation.cells.gamma = 1;
            // simulation.staticSolve();
            // updateScreen(viewer);
            // simulation.reset();
            // simulation.cells.gamma = 0.1;
            // simulation.staticSolve();
            // updateScreen(viewer);
            // simulation.reset();
            viewer.core().is_animating = true;
            return true;
        case '1':
            check_modes = true;
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
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            std::cout << simulation.computeResidual(simulation.u, residual) << std::endl;
            updateScreen(viewer);
            return true;
        case 'a':
            viewer.core().is_animating = !viewer.core().is_animating;
            return true;
        case 'n':
            load_obj_iter_cnt++;
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        case 'l':
            load_obj_iter_cnt--;
            load_obj_iter_cnt = std::max(0, load_obj_iter_cnt);
            std::cout << "state: " << load_obj_iter_cnt << std::endl;
            simulation.loadDeformedState("output/cells/cell/cell_mesh_iter_" + std::to_string(load_obj_iter_cnt) + ".obj");
            updateScreen(viewer);
            return true;
        }
    };

    simulation.initializeCells();
    simulation.dynamic = false;
    if (simulation.dynamic)
        simulation.initializeDynamicsData(1e0, 10000);

    simulation.sampleBoundingSurface(bounding_surface_samples);
    sdf_test_sample_idx_offset = bounding_surface_samples.rows();
    bounding_surface_samples_color = bounding_surface_samples;
    for (int i = 0; i < bounding_surface_samples.rows(); i++)
        bounding_surface_samples_color.row(i) = TV(0.1, 1.0, 0.1);
    updateScreen(viewer);

    // simulation.cells.loadMeshAndSaveCentroid("output/cells/cell_IPC_fix3points_1536", 0, 1536);
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