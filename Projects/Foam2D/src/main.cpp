#include "../include/App.h"
#include "../include/Foam2DInfo.h"
#include <fenv.h>

int main() {
    igl::opengl::glfw::Viewer viewer;
    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

//    feenableexcept(FE_INVALID);
    Eigen::setNbThreads(1);

    Foam2D foam;
    Foam2DApp foam2d_app(foam);
    foam2d_app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}
