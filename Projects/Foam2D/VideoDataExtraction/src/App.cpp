#include <igl/triangle/triangulate.h>

#include "../include/App.h"
#include "../src/implot/implot.h"

#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>

#include "../../include/Boundary/HardwareBoundary0.h"
#include "../include/Segmentation.h"

VideoDataExtractionApp::VideoDataExtractionApp() {
    controlPoints = {{0.014,  -0.012},
                     {-0.248, 0.214},
                     {-0.096, -0.104},
                     {0.13,   0.146}};

    std::string path = std::filesystem::current_path().append("../../../../Projects/Foam2D/VideoDataExtraction/images");
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        imagePaths.push_back(proximate(entry.path(), path));
    }

    path = std::filesystem::current_path().append("../../../../Projects/Foam2D/VideoDataExtraction/videos");
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        videoPaths.push_back(proximate(entry.path(), path));
    }
}

void VideoDataExtractionApp::setViewer(igl::opengl::glfw::Viewer &viewer,
                                       igl::opengl::glfw::imgui::ImGuiMenu &menu) {

    menu.callback_draw_viewer_menu = [&]() {
        if (ImGui::Button("Load Image")) {
            loadImage();
            displayImage(viewer);
            updateViewerData(viewer);
        }
        ImGui::Combo("Image Path", &imagePathIdx, imagePaths);
        if (ImGui::Button("Load Video")) {
            loadVideoFrame();
            displayImage(viewer);
            updateViewerData(viewer);
        }
        ImGui::Combo("Video Path", &videoPathIdx, videoPaths);

        ImGui::Spacing();
        ImGui::Spacing();

        if (ImGui::Button("Process Image 1")) {
            processImage1();
            displayImage(viewer);
        }
        if (ImGui::Button("Process Image 2")) {
            processImage2();
            displayImage(viewer);
        }
    };

    // Draw additional windows
    menu.callback_draw_custom_window = [&]() {
        if (!ImPlot::GetCurrentContext()) {
            auto ctx = ImPlot::CreateContext();
            ImPlot::SetCurrentContext(ctx);
        }

//        ImGui::SetNextWindowPos(ImVec2(570, 0), ImGuiCond_FirstUseEver);
//        ImGui::SetNextWindowSize(ImVec2(430, 430), ImGuiCond_FirstUseEver);
//        ImGui::Begin(
//                "New Window", nullptr,
//                ImGuiWindowFlags_NoSavedSettings
//        );
//
//        ImGui::End();
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
                int closest = getControlPointToSelect(p, 0.02);
                dragIdx = closest;

                updateViewerData(viewer);
                return true;
            };

    viewer.callback_mouse_up =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                dragIdx = -1;
                return true;
            };

    viewer.callback_mouse_move =
            [&](igl::opengl::glfw::Viewer &viewer, int a, int b) -> bool {
                Eigen::Vector2d p((viewer.current_mouse_x - 500) / 500.0, -(viewer.current_mouse_y - 500) / 500.0);
                if (dragIdx >= 0) {
                    controlPoints[dragIdx] = p;
                    updateViewerData(viewer);
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

void VideoDataExtractionApp::updateViewerData(igl::opengl::glfw::Viewer &viewer) {
    {
        Eigen::Matrix<double, -1, -1> points;
        Eigen::Matrix<double, -1, -1> points_c;
        Eigen::Matrix<double, -1, -1> nodes;
        Eigen::Matrix<int, -1, -1> lines;
        Eigen::Matrix<double, -1, -1> lines_c;
        Eigen::Matrix<double, -1, -1> V;
        Eigen::Matrix<int, -1, -1> F;
        Eigen::Matrix<double, -1, -1> Fc;

        points.resize(controlPoints.size(), 3);
        points_c = points;
        double eps = 1e-6;
        for (int i = 0; i < controlPoints.size(); i++) {
            points.row(i) = TV3(controlPoints[i].x(), controlPoints[i].y(), eps);
            points_c.row(i) = TV3(0, 0, 0);
        }

        VectorXT boundaryPoints;
        getBoundaryOutline(boundaryPoints);
        int n_bdy = boundaryPoints.rows() / 2;
        nodes.resize(n_bdy, 3);
        lines.resize(n_bdy, 2);
        lines_c.resize(n_bdy, 3);
        for (int i = 0; i < n_bdy; i++) {
            nodes.row(i) = TV3(boundaryPoints(i * 2 + 0), boundaryPoints(i * 2 + 1), eps);
            lines.row(i) = IV(i, (i + 1) % n_bdy);
            lines_c.row(i) = TV3(1, 0, 0);
        }

        viewer.data(0).clear();
        if (V.rows() > 0) {
            viewer.data(0).set_mesh(V, F);
            viewer.data(0).set_colors(Fc);
        }

        viewer.data(0).set_points(points, points_c);
        if (lines.rows() > 0) {
            viewer.data(0).set_edges(nodes, lines, lines_c);
        }
    }

    {
        Eigen::Matrix<double, 4, 3> camera;
        camera << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
        viewer.core().align_camera_center(camera);
    }
}

int VideoDataExtractionApp::getControlPointToSelect(const TV &p, double threshold) const {
    int closest = -1;
    double dmin = 1e10;

    for (int i = 0; i < controlPoints.size(); i++) {
        double d = (p - controlPoints[i]).norm();
        if (d < dmin && d < threshold) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void VideoDataExtractionApp::getBoundaryOutline(VectorXT &boundaryPoints) const {
    TV origin = controlPoints[0];
    TV d = controlPoints[1] - origin;
    double theta = atan2(d.y(), d.x()) + 3 * M_PI_4;
    double channel_width = d.norm() / sqrt(2);
    double corner_radius = channel_width / 6;
    double scale = channel_width / 0.8;
    double piston_x = (controlPoints[2] - origin).dot(TV(cos(theta), sin(theta))) - corner_radius;
    double piston_y = (controlPoints[3] - origin).dot(TV(-sin(theta), cos(theta))) - corner_radius;

    TV p(piston_x / scale, piston_y / scale);
    HardwareBoundary0 boundary(p, {});

    int num_arc_points = 10;
    VectorXT v(boundary.v.rows() + (num_arc_points - 1) * 2);
    v.segment(0, 10) = boundary.v.segment(0, 10);
    v.segment(10 + num_arc_points * 2, boundary.v.rows() - 12) = boundary.v.segment(12, boundary.v.rows() - 12);
    for (int i = 0; i < num_arc_points; i++) {
        double theta = -M_PI_2 * (1 + i * 1.0 / num_arc_points);
        TV arcPoint = {boundary.corner_radius * (1 + cos(theta)), boundary.corner_radius * (1 + sin(theta))};
        v.segment<2>(10 + i * 2) = arcPoint;
    }

    boundaryPoints.resize(v.rows());
    for (int i = 0; i < boundaryPoints.rows() / 2; i++) {
        boundaryPoints(i * 2 + 0) =
                origin(0) + scale * (v(i * 2 + 0) * cos(theta) - v(i * 2 + 1) * sin(theta));
        boundaryPoints(i * 2 + 1) =
                origin(1) + scale * (v(i * 2 + 0) * sin(theta) + v(i * 2 + 1) * cos(theta));
    }
}

void VideoDataExtractionApp::loadImage() {
    image = cv::imread("../../../../Projects/Foam2D/VideoDataExtraction/images/" + imagePaths[imagePathIdx],
                       cv::IMREAD_COLOR);
    hasImage = true;
}

void VideoDataExtractionApp::loadVideoFrame() {
    image = cv::imread("../../../../Projects/Foam2D/VideoDataExtraction/videos/" + videoPaths[videoPathIdx],
                       cv::IMREAD_COLOR);
    hasImage = true;
}


void VideoDataExtractionApp::processImage1() {
    cv::Vec3b outsideColor(0, 0, 0);

    std::vector<std::vector<cv::Point>> fillPoly(2);

    fillPoly[1] = {{0,          0},
                   {image.cols, 0},
                   {image.cols, image.rows},
                   {0,          image.rows}};

    VectorXT boundaryPoints;
    getBoundaryOutline(boundaryPoints);
    int n_bdy = boundaryPoints.rows() / 2;
    double scale = std::max(image.size[0], image.size[1]) / 2.0;
    double x_offset = image.size[1] / 2.0;
    double y_offset = image.size[0] / 2.0;
    for (int i = 0; i < n_bdy; i++) {
        fillPoly[0].emplace_back(boundaryPoints(i * 2 + 0) * scale + x_offset,
                                 -boundaryPoints(i * 2 + 1) * scale + y_offset);
    }

    double scale_length = (controlPoints[1] - controlPoints[0]).norm() * scale;
    cv::fillPoly(image, fillPoly, outsideColor);
    cv::resize(image, image, {(int) (image.size[1] * 300 / scale_length), (int) (image.size[0] * 300 / scale_length)});
}

void VideoDataExtractionApp::processImage2() {
    cv::Mat segmented, markers;
    std::vector<cv::Vec3b> colors;
    imageMatchSegmentation(image, segmented, markers, colors);
    image = segmented;
}

void VideoDataExtractionApp::displayImage(igl::opengl::glfw::Viewer &viewer) {
    if (!hasImage) return;

    if (viewer.data_list.size() < 2) {
        viewer.append_mesh(true);
    }

    cv::Mat bgr[3];
    cv::split(image, bgr);
    Eigen::MatrixXf b, g, r;
    cv::cv2eigen(bgr[0], b);
    cv::cv2eigen(bgr[1], g);
    cv::cv2eigen(bgr[2], r);

    double dx = b.cols() * 1.0 / std::max(b.rows(), b.cols());
    double dy = b.rows() * 1.0 / std::max(b.rows(), b.cols());
    double eps = 1e-6;

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
