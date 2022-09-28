#include "../include/App.h"

void Foam2DApp::setViewer(igl::opengl::glfw::Viewer &viewer,
                          igl::opengl::glfw::imgui::ImGuiMenu &menu) {
    menu.callback_draw_viewer_menu = [&]() {
        ImGui::Checkbox("Optimize", &optimize);

        std::vector<std::string> optTypes;
        optTypes.push_back("Gradient Descent");
        optTypes.push_back("Newton's Method");
        ImGui::Combo("Optimizer", &foam.opttype, optTypes);

        std::vector<std::string> tesselationTypes;
        tesselationTypes.push_back("Voronoi");
        tesselationTypes.push_back("Sectional");
        if (ImGui::Combo("Tessellation Type", &foam.tesselation, tesselationTypes)) {
            foam.resetVertexParams();
            updateViewerData(viewer);
        }

        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Area Target", &foam.objective.area_target, 0.005f, 0.005f, "%.3f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Area Weight", &foam.objective.area_weight, 0.5f, 0.5f, "%.1f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Length Weight", &foam.objective.length_weight, 0.005f, 0.005f, "%.3f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Centroid Weight", &foam.objective.centroid_weight, 0.005f, 0.005f, "%.3f");

        if (ImGui::Checkbox("Show Dual", &show_dual)) {
            updateViewerData(viewer);
        };
    };

    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    case GLFW_KEY_SPACE:
                        optimize = !optimize;
                        return false;
                    default:
                        return false;
                }
            };

    viewer.callback_mouse_scroll =
            [&](igl::opengl::glfw::Viewer &viewer, float t) -> bool {
                return true;
            };

    viewer.callback_mouse_down =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                if (drag_idx == -1) {
                    Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                    drag_idx = foam.getClosestMovablePointThreshold(p, 0.02);
                }
                return true;
            };

    viewer.callback_mouse_up =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                drag_idx = -1;
                return true;
            };

    viewer.callback_mouse_move =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                if (drag_idx != -1) {
                    Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                    foam.moveVertex(drag_idx, p);
                    updateViewerData(viewer);
                }
                return true;
            };

    viewer.callback_pre_draw =
            [&](igl::opengl::glfw::Viewer &viewer) -> bool {
                if (optimize) {
                    foam.optimize();
                    updateViewerData(viewer);
                } else {
                    Eigen::Matrix<double, 4, 3> bb;
                    bb << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
                    viewer.core().align_camera_center(bb);
                }
                return false;
            };


    foam.generateRandomVoronoi();

    viewer.core().viewport = Eigen::Vector4f(0, 0, 1000, 1000);
    viewer.core().camera_zoom = 2.07;
    viewer.data().show_lines = 0;
    viewer.core().background_color.setOnes();
    viewer.data().point_size = 10;
    viewer.core().is_animating = true;
    viewer.data().shininess = 0;

    updateViewerData(viewer);
}

void Foam2DApp::updateViewerData(igl::opengl::glfw::Viewer &viewer) {
    Eigen::Matrix<double, -1, -1> points;
    Eigen::Matrix<double, -1, -1> nodes;
    Eigen::Matrix<int, -1, -1> lines;
    Eigen::Matrix<double, -1, -1> V;
    Eigen::Matrix<int, -1, -1> F;
    Eigen::Matrix<double, -1, -1> C;

    if (show_dual) {
        foam.getTriangulationViewerData(points, nodes, lines, V, F, C);
    } else {
        foam.getTessellationViewerData(points, nodes, lines, V, F, C);
    }
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_colors(C);

    Eigen::Matrix<double, -1, -1> points_c;
    points_c.resize(points.rows(), 3);
    points_c.setZero();
    points_c(0, 0) = 1;

    Eigen::Matrix<double, -1, -1> lines_c;
    lines_c.resize(lines.rows(), 3);
    lines_c.setZero();

    viewer.data().set_points(points, points_c);
    viewer.data().set_edges(nodes, lines, lines_c);

    Eigen::Matrix<double, 4, 3> bb;
    bb << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
    Eigen::Matrix<double, 4, 3> bb_p2;
    bb_p2 << 1, -1, 0, 1, 1, 0, -1, 1, 0, -1, -1, 0;
    Eigen::Matrix<double, 4, 3> bb_c;
    bb_c << 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0;
    viewer.data().add_edges(bb, bb_p2, bb_c);

    viewer.core().align_camera_center(bb);
}
