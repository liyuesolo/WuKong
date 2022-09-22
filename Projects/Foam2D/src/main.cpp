#include "../include/App.h"

int main() {
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    Foam2D foam;
    Foam2DApp foam2d_app(foam);
    foam2d_app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}