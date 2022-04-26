#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/VertexModel2D.h"
#include "../include/Objective.h"
#include "../include/SensitivityAnalysis.h"

int main(int argc, char** argv)
{
    using TV3 = Vector<T, 3>;
    using TV = Vector<T, 2>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    VertexModel2D vertex_model;
    Objective objective(vertex_model);
    SensitivityAnalysis sa(vertex_model, objective);

    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;

    bool show_current = true;
    bool show_rest = false;
    bool check_modes = false;
    bool inverse = false;
    bool forward = true;
    bool show_target = true;
    bool show_target_current = true;
    int static_solve_step = 0;

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    
    auto updateScreen = [&](igl::opengl::glfw::Viewer& viewer)
    {
        viewer.data().clear();
        vertex_model.generateMeshForRendering(V, F, C, show_current, show_rest);
        TV3 shift(0, 0, 0);
        T sphere_radius = 0.02;
        if (inverse && show_target)
        {
            TV3 color(1, 0, 0);
            std::vector<TV3> target_positions_std_vec;
            if (sa.objective.match_centroid)
                sa.objective.iterateTargets([&](int cell_idx, TV& target)
                {
                    TV3 target_3d = TV3::Zero();
                    target_3d.segment<2>(0) = target;
                    target_positions_std_vec.push_back(target_3d + shift);
                });
            
            int n_sphere = target_positions_std_vec.size();
            VectorXT target_positions(n_sphere * 3); target_positions.setZero();
            tbb::parallel_for(0, n_sphere, [&](int i){
                target_positions.segment<3>(i * 3) = target_positions_std_vec[i];
            });
            vertex_model.appendSphereToPositionVector(target_positions, sphere_radius, color, V, F, C);
        }
        if (inverse && show_target_current)
        {
            TV3 color(0, 1, 0);
            std::vector<TV3> target_positions_std_vec;
            if (sa.objective.match_centroid)
                sa.objective.iterateTargets([&](int cell_idx, TV& target)
                {
                    TV3 target_3d = TV3::Zero();
                    TV centroid;
                    vertex_model.computeCellCentroid(cell_idx, centroid);
                    target_3d.segment<2>(0) = centroid;
                    target_positions_std_vec.push_back(target_3d + shift);
                });
            
            int n_sphere = target_positions_std_vec.size();
            VectorXT target_positions(n_sphere * 3); target_positions.setZero();
            tbb::parallel_for(0, n_sphere, [&](int i){
                target_positions.segment<3>(i * 3) = target_positions_std_vec[i];
            });
            vertex_model.appendSphereToPositionVector(target_positions, sphere_radius, color, V, F, C);
        }
        if (inverse && show_target && show_target_current)
        {
            TV3 color(0, 1, 1);
            std::vector<std::pair<TV, TV>> end_points;
            if (sa.objective.match_centroid)
            {
                sa.objective.iterateTargets([&](int cell_idx, TV& target)
                {
                    TV centroid;
                    vertex_model.computeCellCentroid(cell_idx, centroid);
                    end_points.push_back(std::make_pair(centroid, target));
                });
            }
            
            vertex_model.appendCylindersToEdges(end_points, color, sphere_radius * 0.15, V, F, C);
        }
        viewer.data().set_mesh(V, F);
        viewer.data().set_colors(C); 
    };

    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("Forward", &forward))
            {
                inverse = !forward;
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("Inverse", &inverse))
            {
                forward = !inverse;
                updateScreen(viewer);
            }
        }
        if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Checkbox("ShowCurrent", &show_current))
            {
                updateScreen(viewer);
            }
            if (ImGui::Checkbox("ShowRest", &show_rest))
            {
                updateScreen(viewer);
            }
            if (inverse)
            {
                if (ImGui::Checkbox("ShowTarget", &show_target))
                {
                    updateScreen(viewer);
                }
                if (ImGui::Checkbox("ShowTargetCurrent", &show_target_current))
                {
                    updateScreen(viewer);
                }
            }
        }
        if (ImGui::Button("Reset", ImVec2(-1,0)))
        {
            vertex_model.deformed = vertex_model.undeformed;
            vertex_model.u.setZero();
            updateScreen(viewer);
        }
        if (ImGui::Button("SaveMesh", ImVec2(-1,0)))
        {
            igl::writeOBJ("current_mesh.obj", V, F);
        }
        if (ImGui::Button("SaveCentroids", ImVec2(-1,0)))
        {   
            vertex_model.saveCellCentroidsToFile("2D_test_targets.txt");
        }
        if (ImGui::Button("LoadTargets", ImVec2(-1,0)))
        {   
            objective.loadTarget("2D_test_targets.txt", 0.05);
            objective.match_centroid = true;
            objective.add_forward_potential = true;
            objective.w_fp = 1e-2;
            objective.use_penalty = true;
            objective.penalty_type = Qubic;
            objective.penalty_weight = 1e2;
            objective.optimizer = SGN;
            objective.add_reg = false;
            sa.initialize();
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
            viewer.core().is_animating = true;
            return true;
        }
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &) -> bool
    {
        if(viewer.core().is_animating && !check_modes)
        {
            bool finished = false;
            if (forward)
                finished = vertex_model.advanceOneStep(static_solve_step);
            else if (inverse)
                finished = sa.optimizeOneStep(static_solve_step, objective.optimizer);
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

    vertex_model.initializeScene();
    
    // vertex_model.checkTotalGradientScale();
    // vertex_model.checkTotalHessianScale();


    updateScreen(viewer);
    viewer.core().background_color.setOnes();
    viewer.data().set_face_based(true);
    viewer.data().shininess = 1.0;
    viewer.data().point_size = 10.0;

    viewer.data().set_mesh(V, F);     
    viewer.data().set_colors(C);

    viewer.core().align_camera_center(V);
    viewer.launch();

    return 0;
}