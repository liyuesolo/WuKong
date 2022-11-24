#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "Foam2D.h"

class Foam2D;

class Foam2DApp {
public:
    Foam2D &foam;

    int drag_idx = -1;
    int drag_mode = 0;
    bool optimize = false;
    bool show_dual = false;

    int scenario = 2;
    int free_sites = 40;
    int fixed_sites = 40;

    int numAreaTargets = 1;

    VectorXf objImage;
    double obj_min;
    double obj_max;
    int selected_vertex = -1;
    int objImageResolution = 64;
    float objImageRange = 0.1;
    bool objImageContinuous = false;
    int objImageType = 0;

    bool dynamics = false;
    double dynamics_dt = 0.01;
    double dynamics_m = 0.002;
    double dynamics_eta = 0.01;
    double dynamics_tol = 1e-4;

    bool trajOptMode = false;
    bool trajOptOptimized = false;
    int trajOpt_N = 50;
    int trajOpt_frame = 0;

    bool matchShowImage = false;
    float matchImageW = 0.5;
    int matchSource = 0;
    std::string matchSourcePath;

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
