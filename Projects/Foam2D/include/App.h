#ifndef CS2D_APP_H
#define CS2D_APP_H

#include <deque>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "CellSim.h"
#include "config.h"

class Cell2DApp 
{
public:
	cs2d::CellSim2D cellSim;

public:
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    void updateScreen(igl::opengl::glfw::Viewer& viewer);

    Cell2DApp(cs2d::CellSim2D& cellSim) : cellSim(cellSim) {

		background_color = Eigen::Vector4f(0.69f, 0.69f, 0.69f, 1.0f);
		line_color = Eigen::Vector3f(0.0f, 0.15f, 0.5f);
		cell_color = Eigen::Vector3f(0.0f,  0.3f, 1.0f);
		obstacle_color = Eigen::Vector3f(0.6f, 0.2f, 0.0f);
		for(int i = 0; i < 100; ++i) {
			plotline.push_back(std::sin(i/25.0*3.14159)+1);
		}
	}

	void initViewer(int width, int height) {
		viewer.plugins.push_back(&menu);
		setViewer(viewer, menu);
		viewer_width = width;
		viewer_height = height;
		viewer.launch(true, false, "CellSim", width, height);
		// resizable, full screen
		viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer& viewer, int w, int h) {
			viewer_width = w;
			viewer_height = h;
			return false;
		};
	}
	void static_step() {
		residual_history.push_back(cellSim.static_step());
		if (residual_history.size() > history_size) {
			residual_history.pop_front();
		}
		plotline.clear();
		for (int i = 0; i < history_size - residual_history.size(); ++i)
			plotline.push_back(0);
		for (double r: residual_history)
			plotline.push_back(r);
	}

private:
	int viewer_width, viewer_height;
	igl::opengl::glfw::Viewer viewer;
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	std::fstream src_file;
	bool animation_started = false;
	bool animation_recording = false;
	int animation_step = 0;
	bool print_step;
	int t_step = 0, s_step = 0;
	cs2d::VizConfig config;
	Eigen::Vector4f background_color;
	Eigen::Vector3f line_color, cell_color, obstacle_color;

	std::deque<cs2d::CellSim2D::PotentialValues> potential_history;
	int history_size = 40;
	int seed = 127;
	std::deque<float> residual_history;
	std::vector<float> plotline;
};

#endif
