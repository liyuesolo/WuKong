#include "../include/App.h"
#include "../src/implot/implot.h"

void Foam2DApp::setViewer(igl::opengl::glfw::Viewer &viewer,
                          igl::opengl::glfw::imgui::ImGuiMenu &menu) {
    menu.callback_draw_viewer_menu = [&]() {
        ImGui::Checkbox("Optimize", &optimize);

        std::vector<std::string> optTypes;
        optTypes.push_back("Gradient Descent");
        optTypes.push_back("Newton's Method");
        ImGui::Combo("Optimizer", &foam.opttype, optTypes);

        ImGui::Spacing();
        ImGui::Spacing();

        std::vector<std::string> tesselationTypes;
        tesselationTypes.push_back("Voronoi");
        tesselationTypes.push_back("Sectional");
        if (ImGui::Combo("Tessellation Type", &foam.tesselation, tesselationTypes)) {
            foam.resetVertexParams();
            updateViewerData(viewer);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Objective Function Parameters");
        ImGui::Text("Area Targets");
        {
            ImGui::Indent(10.0f);
            for (int i = 0; i < foam.objective.area_targets.size(); i++) {
                ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
                if (ImGui::InputDouble((std::string("##AreaTarget") + std::to_string(i)).c_str(),
                                       &foam.objective.area_targets[i],
                                       0.005f, 0.005f, "%.3f")) {
                    updateViewerData(viewer);
                }
            }
            if (ImGui::Button("+")) {
                foam.objective.area_targets.push_back(foam.objective.area_targets[0]);
                updateViewerData(viewer);
            }
            if (foam.objective.area_targets.size() > 1) {
                ImGui::SameLine();
                if (ImGui::Button("-")) {
                    foam.objective.area_targets.pop_back();
                    updateViewerData(viewer);
                }
            }
            ImGui::Indent(-10.0f);
        }
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Area Weight", &foam.objective.area_weight, 0.5f, 0.5f, "%.1f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Length Weight", &foam.objective.length_weight, 0.005f, 0.005f, "%.3f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Centroid Weight", &foam.objective.centroid_weight, 0.005f, 0.005f, "%.3f");

        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Checkbox("Show Dual", &show_dual)) {
            updateViewerData(viewer);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Simulation Setup");

        std::vector<std::string> scenarios;
        scenarios.push_back("Boundary Cell Circle");
        scenarios.push_back("Gradient Test");
        ImGui::Combo("Scenario", &scenario, scenarios);
        if (scenario == 0) {
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Cells", &free_sites, 10, 100);
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Boundary Sites", &fixed_sites, 10, 100);
        }
        if (ImGui::Button("Generate")) {
            generateScenario();
            updateViewerData(viewer);
        }
    };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        if (!ImPlot::GetCurrentContext()) {
            auto ctx = ImPlot::CreateContext();
            ImPlot::SetCurrentContext(ctx);
        }

        updatePlotData();
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
                    selected_vertex = drag_idx;
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
                    Eigen::Matrix<double, 4, 3> camera;
                    camera << -1, -1, 0, 2, -1, 0, 2, 1, 0, -1, 1, 0;
                    viewer.core().align_camera_center(camera);
                }
                return false;
            };


    generateScenario();

    viewer.core().viewport = Eigen::Vector4f(0, 0, 1500, 1000);
    viewer.core().camera_zoom = 2.07 * 1.5;
    viewer.data().show_lines = 0;
    viewer.core().background_color.setOnes();
    viewer.data().point_size = 10;
    viewer.core().is_animating = true;
    viewer.data().shininess = 0;

    updateViewerData(viewer);
}

void Foam2DApp::generateScenario() {
    switch (scenario) {
        case 0:
            foam.initRandomSitesInCircle(free_sites, fixed_sites);
            break;
        case 1:
            foam.initBasicTestCase();
            break;
        default:
            std::cout << "Error: scenario not implemented!";
    }
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
//    points_c(0, 0) = 1; // Make the first point red.

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

    Eigen::Matrix<double, 4, 3> camera;
    camera << -1, -1, 0, 2, -1, 0, 2, 1, 0, -1, 1, 0;
    viewer.core().align_camera_center(camera);
}

void Foam2DApp::updatePlotData() {
    // Define next window position + size
    ImGui::SetNextWindowPos(ImVec2(1000, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(500, 1000), ImGuiCond_FirstUseEver);
    ImGui::Begin(
            "New Window", nullptr,
            ImGuiWindowFlags_NoSavedSettings
    );

    if (ImGui::CollapsingHeader("Cell Area to Target Ratio Histogram", ImGuiTreeNodeFlags_DefaultOpen)) {
        VectorXT areas;
        foam.getPlotAreaHistogram(areas);

        if (ImPlot::BeginPlot("##Cell Area to Target Ratio Histogram")) {
            ImPlot::SetupAxes(NULL, NULL, 0, 0);
            ImPlot::SetupAxesLimits(0, 2, 0, areas.rows(), ImPlotCond_Always);

            double halfbin = 0.025;

            for (float c = 2 * halfbin; c < 2 - 2 * halfbin; c += 2 * halfbin) {
                float r, g, b;
                float q = c;
                if (q >= 1) q = sqrt(q - 1); else q = -sqrt(1 - q);
                r = 1 - q;
                g = 1;
                b = 1 + q;

                r = fmin(fmax(r, 0), 1);
                g = fmin(fmax(g, 0), 1);
                b = fmin(fmax(b, 0), 1);

                ImPlot::SetNextFillStyle({r, g, b, 1});
                ImPlot::SetNextLineStyle({0, 0, 0, 1});
                int count = (areas.array() > c - halfbin && areas.array() < c + halfbin).count();
                VectorXT pile = c * VectorXT::Ones(count);
                ImPlot::PlotHistogram("", pile.data(), count, 1, 1.0, ImPlotRange(c - halfbin, c + halfbin));
            }
//            ImPlot::PlotHistogram("a", areas.data(), areas.rows(), 39, 1.0, ImPlotRange(0, 2));
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Current Objective Status", ImGuiTreeNodeFlags_DefaultOpen)) {
        double obj_val, gradient_norm;
        bool hessian_pd;
        foam.getPlotObjectiveStats(obj_val, gradient_norm, hessian_pd);

        ImGui::Text(("Objective Value: " + std::to_string(obj_val)).c_str());
        ImGui::Text(("Gradient Norm: " + std::to_string(gradient_norm)).c_str());
        ImGui::Text((std::string("Hessian PD: ") + (hessian_pd ? "True" : "False")).c_str());
    }

    if (ImGui::CollapsingHeader("Objective Function Landscape", ImGuiTreeNodeFlags_DefaultOpen)) {
        std::vector<std::string> objTypes;
        objTypes.push_back("Objective Value");
        objTypes.push_back("dOdx");
        objTypes.push_back("dOdy");
        ImGui::Combo("Function", &objImageType, objTypes);
        ImGui::Text(("Selected Site: " + (selected_vertex != -1 ? std::to_string(selected_vertex) : "None")).c_str());
        ImGui::DragInt("Resolution ", &objImageResolution, 1, 0, 512);
        ImGui::DragFloat("Range ", &objImageRange, 0.001, 0.001, 1);
        ImGui::Checkbox("Compute Continuously", &objImageContinuous);
        if ((ImGui::Button("Compute") || objImageContinuous) && selected_vertex != -1) {
            foam.getPlotObjectiveFunctionLandscape(selected_vertex, objImageType, objImageResolution, objImageRange,
                                                   objImage, obj_min, obj_max);
        }
        std::string legendLabel;
        switch (objImageType) {
            case 0:
                legendLabel = "Objective Value";
                break;
            case 1:
                legendLabel = "dOdx";
                break;
            case 2:
                legendLabel = "dOdy";
                break;
        }

        if (ImPlot::BeginPlot("##Objective Function Landscape", ImVec2(400, 400))) {
            ImPlot::SetupAxes(NULL, NULL, ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels,
                              ImPlotAxisFlags_NoTickMarks | ImPlotAxisFlags_NoTickLabels);

            static ImVec2 bmin(0, 0);
            static ImVec2 bmax(1, 1);
            static ImVec2 uv0(0, 0);
            static ImVec2 uv1(1, 1);
            static ImVec4 tint(1, 1, 1, 1);

            GLuint _textureHandle;
            glGenTextures(1, &_textureHandle);
            glBindTexture(GL_TEXTURE_2D, _textureHandle);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, objImageResolution, objImageResolution, 0, GL_RGB, GL_FLOAT,
                         objImage.data());
            ImPlot::PlotImage(legendLabel.c_str(), (void *) (intptr_t) _textureHandle, bmin, bmax, uv0, uv1, tint);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale("##ObjectiveLandscapeColormap", obj_max, obj_min, ImVec2(72, 400), "%g",
                              ImPlotColormapScaleFlags_Invert,
                              ImPlotColormap_Greys);
    }

    ImGui::End();
}
