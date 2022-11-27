#include "../include/App.h"
#include "../src/implot/implot.h"

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "../include/ImageMatch/Segmentation.h"

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
        tesselationTypes.push_back("Power");
        if (ImGui::Combo("Tessellation Type", &foam.tessellation, tesselationTypes)) {
            foam.resetVertexParams();
            updateViewerData(viewer);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        std::vector<std::string> dragModes;
        dragModes.push_back("Set Target");
        dragModes.push_back("Set Position");
        if (ImGui::Combo("Drag Mode", &drag_mode, dragModes)) {
            foam.energyObjective.drag_idx = -1;
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Objective Function Parameters");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        if (scenario != 3) {
            if (ImGui::InputInt("Area Targets", &numAreaTargets, 1, 1)) {
                if (numAreaTargets < 1) numAreaTargets = 1;
                if (foam.energyObjective.area_targets.rows() >= numAreaTargets) {
                    VectorXd areaTargets = foam.energyObjective.area_targets.segment(0, numAreaTargets);
                    foam.energyObjective.area_targets = areaTargets;
                } else {
                    VectorXd areaTargets(numAreaTargets);
                    areaTargets << foam.energyObjective.area_targets, foam.energyObjective.area_targets(0) *
                                                                      VectorXd::Ones(
                                                                              numAreaTargets -
                                                                              foam.energyObjective.area_targets.rows());
                    foam.energyObjective.area_targets = areaTargets;
                }
                updateViewerData(viewer);
            }
            {
                ImGui::Indent(10.0f);
                for (int i = 0; i < numAreaTargets; i++) {
                    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
                    if (ImGui::InputDouble((std::string("##AreaTarget") + std::to_string(i)).c_str(),
                                           &foam.energyObjective.area_targets(i),
                                           0.005f, 0.005f, "%.3f")) {
                        updateViewerData(viewer);
                    }
                }
                ImGui::Indent(-10.0f);
            }
        }
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Area Weight", &foam.energyObjective.area_weight, 0.01f, 0.01f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Length Weight", &foam.energyObjective.length_weight, 0.0005f, 0.0005f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Centroid Weight", &foam.energyObjective.centroid_weight, 0.01f, 0.01f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
        ImGui::InputDouble("Drag Target Weight", &foam.energyObjective.drag_target_weight, 0.01f, 0.01f, "%.4f");

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
        scenarios.push_back("Bounding Box");
        scenarios.push_back("Image Match");
        std::vector<std::string> sourceImages;
        std::string path = std::filesystem::current_path().append("../../../Projects/Foam2D/images");
        for (const auto &entry: std::filesystem::directory_iterator(path)) {
            sourceImages.push_back(proximate(entry.path(), path));
        }
        if (ImGui::Combo("Scenario", &scenario, scenarios)) {
            matchShowImage = true;
        }
        if (scenario == 0) {
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Cells", &free_sites, 10, 100);
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Boundary Sites", &fixed_sites, 10, 100);
        }
        if (scenario == 2) {
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::InputInt("Cells", &free_sites, 10, 100);
        }
        if (scenario == 3) {
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
            ImGui::Combo("Source", &matchSource, sourceImages);
            ImGui::Checkbox("Show Image", &matchShowImage);
            ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.5);
            ImGui::SliderFloat("A Slider", &matchImageW, 0.0, 1.0);
        } else {
            matchShowImage = false;
        }

        if (ImGui::Button("Generate")) {
            matchSourcePath = "../../../Projects/Foam2D/images/" + sourceImages[matchSource];
            generateScenario();
            updateViewerData(viewer);
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Dynamics");

        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
        ImGui::InputDouble("Timestep", &dynamics_dt, 0.01f, 0.01f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
        ImGui::InputDouble("Inertia", &dynamics_m, 0.001f, 0.001f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
        ImGui::InputDouble("Viscosity", &dynamics_eta, 0.001f, 0.001f, "%.4f");
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
        ImGui::InputDouble("Tolerance", &dynamics_tol, 0.000001, 0.00001f, "%.6f");
        if (ImGui::Button("Start Dynamics")) {
            dynamics = true;
            foam.dynamicsInit(dynamics_dt, dynamics_m, dynamics_eta);
        }
        if (ImGui::Button("Stop Dynamics")) {
            dynamics = false;
        }

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("Trajectory Optimization");
        if (ImGui::Button("Set Initial State")) {
            optimize = false;
            dynamics = false;
            trajOptMode = true;
            trajOpt_frame = 0;
            foam.trajectoryOptSetInit();
        }
        if (ImGui::Button("Optimize IPOPT")) {
            trajOptOptimized = true;
            foam.dynamicsInit(dynamics_dt, dynamics_m, dynamics_eta);
            foam.trajectoryOptOptimizeIPOPT(trajOpt_N);
        }
        if (ImGui::Button("Stop Optimization")) {
            foam.trajectoryOptStop();
        }
        ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.6);
        ImGui::InputInt("Steps", &trajOpt_N, 1, 10);
        if (trajOptOptimized) {
            ImGui::SliderInt("Frame", &trajOpt_frame, 0, trajOpt_N);
        }
        if (ImGui::Button("Clear")) {
            trajOptMode = false;
            trajOptOptimized = false;
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
                if (drag_idx == -1 && !trajOptOptimized) {
                    Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                    drag_idx = foam.getClosestMovablePointThreshold(p, 0.02);
                    selected_vertex = drag_idx;
                    if (drag_mode == 0) {
                        foam.energyObjective.drag_idx = drag_idx;
                        foam.energyObjective.drag_target_pos = p;
                    }
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
                if (drag_idx != -1 && !trajOptOptimized) {
                    Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                    if (drag_mode == 0) {
                        foam.energyObjective.drag_target_pos = p;
                    } else {
                        foam.moveVertex(drag_idx, p);
                    }
                    updateViewerData(viewer);
                }
                return true;
            };

    viewer.callback_pre_draw =
            [&](igl::opengl::glfw::Viewer &viewer) -> bool {
                if (trajOptMode) {
                    if (trajOptOptimized) {
                        foam.trajectoryOptGetFrame(trajOpt_frame);
                    } else {
                        foam.trajectoryOptGetFrame(0);
                    }
                    updateViewerData(viewer);
                } else if (optimize) {
                    foam.optimize(dynamics);
                    if (dynamics && foam.isConvergedDynamic(dynamics_tol)) {
                        foam.dynamicsNewStep();
                    }
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
    viewer.data(0).show_lines = 0;
    viewer.core().background_color.setOnes();
    viewer.data(0).point_size = 10;
    viewer.core().is_animating = true;
    viewer.data(0).shininess = 0;
    viewer.core().lighting_factor = 0;

    updateViewerData(viewer);
}

void Foam2DApp::generateScenario() {
    MatrixXi markersEigen;
    switch (scenario) {
        case 0:
            foam.initRandomSitesInCircle(free_sites, fixed_sites);
            break;
        case 1:
            foam.initBasicTestCase();
            break;
        case 2:
            foam.initRandomCellsInBox(free_sites);
            break;
        case 3:
            matchImage = cv::imread(matchSourcePath, cv::IMREAD_COLOR);
            imageMatchSegmentation(matchImage, matchSegmented, matchMarkers);
            cv::cv2eigen(matchMarkers, markersEigen);
            foam.initImageMatch(markersEigen);
            break;
        default:
            std::cout << "Error: scenario not implemented!";
    }
}

void Foam2DApp::updateViewerData(igl::opengl::glfw::Viewer &viewer) {
    Eigen::Matrix<double, -1, -1> points;
    Eigen::Matrix<double, -1, -1> points_c;
    Eigen::Matrix<double, -1, -1> nodes;
    Eigen::Matrix<int, -1, -1> lines;
    Eigen::Matrix<double, -1, -1> lines_c;
    Eigen::Matrix<double, -1, -1> V;
    Eigen::Matrix<int, -1, -1> F;
    Eigen::Matrix<double, -1, -1> Fc;

    if (show_dual) {
        foam.getTriangulationViewerData(points, nodes, lines, points_c, lines_c, V, F, Fc);
    } else {
        foam.getTessellationViewerData(points, nodes, lines, points_c, lines_c, V, F, Fc);
    }
    if (trajOptMode && trajOptOptimized) {
        foam.addTrajectoryOptViewerData(points, nodes, lines, points_c, lines_c, V, F, Fc);
    }

    viewer.data(0).clear();
    viewer.data(0).set_mesh(V, F);
    viewer.data(0).set_colors(Fc);

    viewer.data(0).set_points(points, points_c);
    viewer.data(0).set_edges(nodes, lines, lines_c);

//    Eigen::Matrix<double, 4, 3> bb;
//    bb << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
//    Eigen::Matrix<double, 4, 3> bb_p2;
//    bb_p2 << 1, -1, 0, 1, 1, 0, -1, 1, 0, -1, -1, 0;
//    Eigen::Matrix<double, 4, 3> bb_c;
//    bb_c << 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0;
//    viewer.data(0).add_edges(bb, bb_p2, bb_c);

    int nb = foam.boundary.rows() / 2;
    MatrixXd b1, b2, bc;
    b1.resize(nb, 3);
    b2.resize(nb, 3);
    bc.resize(nb, 3);
    for (int i = 0; i < nb; i++) {
        b1.row(i) = TV3(foam.boundary(2 * i + 0), foam.boundary(2 * i + 1), 0);
        b2.row(i) = TV3(foam.boundary(2 * ((i + 1) % nb) + 0), foam.boundary(2 * ((i + 1) % nb) + 1), 0);
        bc.row(i) = TV3(1, 0, 0);
    }
    viewer.data(0).add_edges(b1, b2, bc);

    Eigen::Matrix<double, 4, 3> camera;
    camera << -1, -1, 0, 2, -1, 0, 2, 1, 0, -1, 1, 0;
    viewer.core().align_camera_center(camera);

    if (matchShowImage) {
        displaySourceImage(viewer);
    } else {
        for (int i = 1; i < viewer.data_list.size(); i++) {
            viewer.data(i).clear();
        }
    }
}

void Foam2DApp::displaySourceImage(igl::opengl::glfw::Viewer &viewer) {
    if (viewer.data_list.size() < 2) {
        viewer.append_mesh(true);
    }

    cv::Mat combined;
    cv::addWeighted(matchImage, 1 - matchImageW, matchSegmented, matchImageW, 0, combined);

    cv::Mat bgr[3];
    cv::split(combined, bgr);
    Eigen::MatrixXf b, g, r;
    cv::cv2eigen(bgr[0], b);
    cv::cv2eigen(bgr[1], g);
    cv::cv2eigen(bgr[2], r);

    double dx = b.cols() * 0.8 / std::max(b.rows(), b.cols());
    double dy = b.rows() * 0.8 / std::max(b.rows(), b.cols());
    double eps = 1e-3;

    Eigen::Matrix<double, -1, -1> V;
    Eigen::Matrix<double, -1, -1> UV;
    Eigen::Matrix<int, -1, -1> F;

    V.resize(4, 3);
    V << -dx, -dy, eps, dx, -dy, eps, dx, dy, eps, -dx, dy, eps;

    UV.resize(4, 2);
    UV << 1, 0, 1, 1, 0, 1, 0, 0;

    F.resize(2, 3);
    F << 0, 1, 2, 2, 3, 0;

    viewer.data(1).clear();
    viewer.data(1).set_mesh(V, F);
    viewer.data(1).set_colors(Eigen::RowVector3d(1, 1, 1));
    viewer.data(1).show_lines = false;
    viewer.data(1).show_texture = true;
    viewer.data(1).shininess = 0;

    viewer.data(1).set_texture(r.cast<unsigned char>(), g.cast<unsigned char>(),
                               b.cast<unsigned char>());
    viewer.data(1).set_uv(UV);
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
        foam.getPlotObjectiveStats(dynamics, obj_val, gradient_norm, hessian_pd);

        ImGui::Text(("Objective Value: " + std::to_string(obj_val)).c_str());
        ImGui::Text(("Gradient Norm: " + std::to_string(gradient_norm)).c_str());
        ImGui::Text((std::string("Hessian PD: ") + (hessian_pd ? "True" : "False")).c_str());
    }

    if (ImGui::CollapsingHeader("Objective Function Landscape", ImGuiTreeNodeFlags_None)) {
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

    if (ImGui::CollapsingHeader("Trajectory Optimization Forces", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (trajOptOptimized) {
            VectorXT forceX, forceY;
            foam.trajectoryOptGetForces(forceX, forceY);

            if (ImPlot::BeginPlot("##Trajectory Optimization Forces")) {
                ImPlot::SetupAxes(NULL, NULL, 0, 0);
                //            ImPlot::SetupAxesLimits(0, forceX.rows(), 0, areas.rows(), ImPlotCond_Always);

                ImPlot::PlotLine("f_x", forceX.data(), forceX.rows());
                ImPlot::PlotLine("f_y", forceY.data(), forceY.rows());
                ImPlot::EndPlot();
            }
        }
    }

    ImGui::End();
}
