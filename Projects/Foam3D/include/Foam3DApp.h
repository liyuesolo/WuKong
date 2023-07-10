#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <opencv4/opencv2/opencv.hpp>

#include "Foam3D.h"

class Foam3D;

class Foam3DApp {
public:
    Foam3D &foam;

    bool optimize = false;
    bool dynamics = false;
    int optimizer = 0;

    int colormode = 0;

    int generate_scenario_type = 0;
    int generate_scenario_num_sites = 5;

    bool slice_visible = false;
    bool slice_follow = true;
    int slice_mode = 0;
    TV3 slice_normal = TV3(0, 0, 1);
    TV3 slice_offset = TV3(0, 0, 0);

public:
    void setViewer(igl::opengl::glfw::Viewer &viewer,
                   igl::opengl::glfw::imgui::ImGuiMenu &menu);

    void generateScenario();

    void scenarioCube();

    void scenarioSphere();

    void scenarioDrosophilaLowRes();

    void scenarioDrosophilaHighRes();

    void updateViewerData(igl::opengl::glfw::Viewer &viewer);

    void updatePlotData();

    Foam3DApp(Foam3D &_foam) : foam(_foam) {}

    ~Foam3DApp() {}
};
