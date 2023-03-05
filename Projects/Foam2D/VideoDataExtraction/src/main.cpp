#include "../include/App.h"
#include <fenv.h>

int main() {
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

//    feenableexcept(FE_INVALID);
    Eigen::setNbThreads(1);

    VideoDataExtractionApp app;
    app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}
