#include "../include/App.h"
#include "../include/Foam2DInfo.h"

int main() {
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    Eigen::setNbThreads(1);

    Foam2D foam;
    Foam2DApp foam2d_app(foam);
    foam2d_app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}
