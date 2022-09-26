#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "Cell.h"

class Cell2DApp 
{
public:
	CellSim cellSim;

public:
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    void updateScreen(igl::opengl::glfw::Viewer& viewer);

    Cell2DApp(CellSim& cellSim) : cellSim(cellSim) {}

private:
	bool print_step;
};

#endif
