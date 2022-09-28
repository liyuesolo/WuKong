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
    bool optimize = false;
    bool show_dual = false;

public:
    void setViewer(igl::opengl::glfw::Viewer &viewer,
                   igl::opengl::glfw::imgui::ImGuiMenu &menu);

    void updateViewerData(igl::opengl::glfw::Viewer &viewer);

    Foam2DApp(Foam2D &_foam) : foam(_foam) {}

    ~Foam2DApp() {}
};

#endif
