#pragma once

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <opencv4/opencv2/opencv.hpp>

#include "VoronoiMetrics.h"

class VoronoiMetricsApp {
public:
    VoronoiMetrics &metrics;

    int hoveredSite = -1;
    int selectedSite = -1;
    bool dragging = false;

    int raster_resolution = 100;

    int numMetricPoints = 8;
    VectorXT metricPointsX;
    VectorXT metricPointsY;

public:
    void setViewer(igl::opengl::glfw::Viewer &viewer,
                   igl::opengl::glfw::imgui::ImGuiMenu &menu);

    void updateViewerData(igl::opengl::glfw::Viewer &viewer);

    VoronoiMetricsApp(VoronoiMetrics &_metrics) : metrics(_metrics) {}

    ~VoronoiMetricsApp() {}
};
