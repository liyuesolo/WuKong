#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <opencv4/opencv2/opencv.hpp>

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"

using TV = Vector<double, 2>;
using TV3 = Vector<double, 3>;
using TM = Matrix<double, 2, 2>;
using IV3 = Vector<int, 3>;
using IV = Vector<int, 2>;

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Vector<int, Eigen::Dynamic>;
using VectorXf = Vector<float, Eigen::Dynamic>;

class VideoDataExtractionApp {
public:
    std::vector<std::string> imagePaths;
    int imagePathIdx = 0;
    std::vector<std::string> videoPaths;
    int videoPathIdx = 0;
    int frame = 0;

    bool hasImage = false;
    cv::Mat image;
    cv::VideoCapture video;

    std::vector<TV> controlPoints;
    int dragIdx = -1;

public:

    void setViewer(igl::opengl::glfw::Viewer &viewer,
                   igl::opengl::glfw::imgui::ImGuiMenu &menu);

    void updateViewerData(igl::opengl::glfw::Viewer &viewer);

    void loadImage();

    void loadVideoFrame();

    void displayImage(igl::opengl::glfw::Viewer &viewer);

    void processImage1();

    void processImage2();

    VideoDataExtractionApp();

    ~VideoDataExtractionApp() {}

public:
    int getControlPointToSelect(const TV &p, double threshold) const;

    void getBoundaryOutline(VectorXT &boundaryPoints) const;
};
