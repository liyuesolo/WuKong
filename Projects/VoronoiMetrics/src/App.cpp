#include "../include/App.h"
#include "../src/implot/implot.h"

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

void VoronoiMetricsApp::setViewer(igl::opengl::glfw::Viewer &viewer,
                                  igl::opengl::glfw::imgui::ImGuiMenu &menu) {
    menu.callback_draw_viewer_menu = [&]() {

    };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        if (!ImPlot::GetCurrentContext()) {
            auto ctx = ImPlot::CreateContext();
            ImPlot::SetCurrentContext(ctx);
        }
    };

    viewer.callback_key_pressed =
            [&](igl::opengl::glfw::Viewer &viewer, unsigned int key, int mods) -> bool {
                switch (key) {
                    case GLFW_KEY_SPACE:
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
            [&](igl::opengl::glfw::Viewer &viewer, int button, int modifier) -> bool {
                Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                int closest = metrics.getClosestSiteToSelect(p, 0.02);
                switch (button) {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        selectedSite = closest;
                        if (selectedSite >= 0) {
                            dragging = true;
                        }
                        break;
                    default:
                        if (closest >= 0) {
                            metrics.deleteSite(closest);
                        } else {
                            metrics.createSite(p);
                        }
                        break;
                }

                updateViewerData(viewer);
                return true;
            };

    viewer.callback_mouse_up =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                dragging = false;
                return true;
            };

    viewer.callback_mouse_move =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                if (dragging) {
                    metrics.moveSite(selectedSite, p);
                    updateViewerData(viewer);
                } else {
                    if (hoveredSite != metrics.getClosestSiteToSelect(p, 0.02)) {
                        hoveredSite = metrics.getClosestSiteToSelect(p, 0.02);
                        updateViewerData(viewer);
                    }
                }
                return true;
            };

    viewer.callback_pre_draw =
            [&](igl::opengl::glfw::Viewer &viewer) -> bool {
                Eigen::Matrix<double, 4, 3> camera;
                camera << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
                viewer.core().align_camera_center(camera);
                return false;
            };

//    viewer.core().viewport = Eigen::Vector4f(0, 0, 1500, 1000);
//    viewer.core().camera_zoom = 2.07 * 1.5;
    viewer.core().viewport = Eigen::Vector4f(0, 0, 1000, 1000);
    viewer.core().camera_zoom = 2.07 * 1.0;
    viewer.data(0).show_lines = 0;
    viewer.core().background_color.setOnes();
    viewer.data(0).point_size = 10;
    viewer.core().is_animating = true;
    viewer.data(0).shininess = 0;
    viewer.core().lighting_factor = 0;

    updateViewerData(viewer);
}

void VoronoiMetricsApp::updateViewerData(igl::opengl::glfw::Viewer &viewer) {
    {
        Eigen::Matrix<double, -1, -1> points;
        Eigen::Matrix<double, -1, -1> points_c;
        Eigen::Matrix<double, -1, -1> nodes;
        Eigen::Matrix<int, -1, -1> lines;
        Eigen::Matrix<double, -1, -1> lines_c;
        Eigen::Matrix<double, -1, -1> V;
        Eigen::Matrix<int, -1, -1> F;
        Eigen::Matrix<double, -1, -1> Fc;

        points.resize(metrics.sites.size(), 3);
        points_c = points;
        double eps = 1e-6;
        for (int i = 0; i < metrics.sites.size(); i++) {
            points.row(i) = TV3(metrics.sites[i].position.x(), metrics.sites[i].position.y(), eps);
            points_c.row(i) = TV3(0, 0, 0);
        }

        if (hoveredSite >= 0) {
            int n = metrics.sites[hoveredSite].metric.size();
            TV p = metrics.sites[hoveredSite].position;

            nodes.resize(n * 2, 3);
            lines.resize(n * 2, 2);
            lines_c.resize(n * 2, 3);
            for (int i = 0; i < n; i++) {
                double theta = metrics.sites[hoveredSite].metric[i].theta;
                double a0 = 0.1 * metrics.sites[hoveredSite].metric[i].a;
                double a1 = 10;
                nodes.row(i * 2 + 0) = TV3(p.x() + a0 * cos(theta), p.y() + a0 * sin(theta), eps);
                nodes.row(i * 2 + 1) = TV3(p.x() + a1 * cos(theta), p.y() + a1 * sin(theta), eps);
                lines.row(i * 2 + 0) = IV(i * 2 + 0, i * 2 + 1);
                lines.row(i * 2 + 1) = IV(i * 2 + 0, (i * 2 + 2) % nodes.rows());
                lines_c.row(i * 2 + 0) = TV3(0, 0, 0);
                lines_c.row(i * 2 + 1) = TV3(0, 0, 0);
            }
        }

        viewer.data(0).clear();
//        viewer.data(0).set_mesh(V, F);
//        viewer.data(0).set_colors(Fc);

        viewer.data(0).set_points(points, points_c);
        viewer.data(0).set_edges(nodes, lines, lines_c);
    }

    {
        Eigen::Matrix<double, 4, 3> camera;
        camera << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
        viewer.core().align_camera_center(camera);
    }

    {
        if (viewer.data_list.size() < 2) {
            viewer.append_mesh(true);
        }

        std::vector<cv::Vec3b> colors;
        colors.clear();
        cv::theRNG().state = 1;
        for (size_t i = 0; i < metrics.sites.size(); i++) {
            int b = cv::theRNG().uniform(0, 256);
            int g = cv::theRNG().uniform(0, 256);
            int r = cv::theRNG().uniform(0, 256);
            colors.push_back(cv::Vec3b((uchar) b, (uchar) g, (uchar) r));
        }

        int res = 200;
        cv::Mat raster(res, res, CV_8UC3, cv::Scalar(0, 0, 255));

        double dx = 1;
        double dy = 1;
        for (int i = 0; i < res; i++) {
            for (int j = 0; j < res; j++) {
                double y = 1.0 - (i * 2.0 / res);
                double x = (j * 2.0 / res) - 1.0;

                if (metrics.sites.empty()) {
                    raster.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                } else {
                    raster.at<cv::Vec3b>(i, j) = colors[metrics.getClosestSiteByMetric(TV(x, y))];
                }
            }
        }

        cv::Mat bgr[3];
        cv::split(raster, bgr);
        Eigen::MatrixXf b, g, r;
        cv::cv2eigen(bgr[0], b);
        cv::cv2eigen(bgr[1], g);
        cv::cv2eigen(bgr[2], r);

        double eps = 0;

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
}
