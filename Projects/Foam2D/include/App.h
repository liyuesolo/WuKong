#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui.h>
#include <opencv4/opencv2/opencv.hpp>

#include "Foam2D.h"

class Foam2D;

class Foam2DApp {
public:
    Foam2D &foam;

    int drag_mode = 0;
    bool dragging = false;
    bool optimize = false;
    bool show_dual = false;

    int colormode = 0;

    int generate_scenario_type = 5;
    int generate_scenario_free_sites = 40;
    int generate_scenario_fixed_sites = 40;

    int numAreaTargets = 1;
    VectorXd areaTargets = 0.05 * VectorXd::Ones(1);

    VectorXf objImage;
    double obj_min;
    double obj_max;
    int objImageResolution = 64;
    float objImageRange = 0.1;
    bool objImageContinuous = false;
    int objImageType = 0;

    bool dynamics = false;
    double dynamics_tol = 1e-4;

    bool trajOptMode = false;
    bool trajOptOptimized = false;
    int trajOpt_frame = 0;

    bool matchSA = false;
    bool matchShowImage = false;
    bool matchShowPixels = false;
    float matchImageW = 0.5;
    int matchSource = 9;
    std::vector<std::string> sourcePaths;
    std::string matchSourcePath;
    cv::Mat matchImage;
    cv::Mat matchSegmented;
    cv::Mat matchMarkers;
    std::vector<cv::Vec3b> matchColors;

    bool simulatingHardware = false;
    int hardwareFrame = -1;
    int hardwareVisualizeCellIdx = 0;
    int hardwareVisualizeLap = 0;
    std::vector<VectorXT> hardwareSimulationFrames;
    int hardwareFramesPerLap = 100;

public:
    void setViewer(igl::opengl::glfw::Viewer &viewer,
                   igl::opengl::glfw::imgui::ImGuiMenu &menu);

    void generateScenario();

    void updateViewerData(igl::opengl::glfw::Viewer &viewer);

    void displaySourceImage(igl::opengl::glfw::Viewer &viewer);

    void updatePlotData();

    Foam2DApp(Foam2D &_foam) : foam(_foam) {}

    ~Foam2DApp() {}
};

#endif
