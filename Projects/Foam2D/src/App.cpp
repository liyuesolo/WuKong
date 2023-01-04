#include <string>

#include <eigen3/Eigen/Dense>
#include <igl/triangle/triangulate.h>
#include <igl/png/render_to_png.h>

#include "../include/App.h"
#include "../include/CellSim.h"
#include "../include/SerDe.h"

using Eigen::Matrix;

// todo: keep track of size changes using 
//std::function<bool(Viewer& viewer, int w, int h)> callback_post_resize;

namespace ImGui {
#ifndef IMGUI_API
#define  IMGUI_API
#endif
	typedef int ImGuiInputTextFlags;
	IMGUI_API bool  InputText(const char* label, std::string* str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);

	struct InputTextCallback_UserData
	{
		std::string*            Str;
		ImGuiInputTextCallback  ChainCallback;
		void*                   ChainCallbackUserData;
	};

	static int InputTextCallback(ImGuiInputTextCallbackData* data)
	{
		InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
		if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
		{
			// Resize string callback
			// If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
			std::string* str = user_data->Str;
			IM_ASSERT(data->Buf == str->c_str());
			str->resize(data->BufTextLen);
			data->Buf = (char*)str->c_str();
		}
		else if (user_data->ChainCallback)
		{
			// Forward to user callback, if any
			data->UserData = user_data->ChainCallbackUserData;
			return user_data->ChainCallback(data);
		}
		return 0;
	}

	bool InputText(const char* label, std::string* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
	{
		IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
		flags |= ImGuiInputTextFlags_CallbackResize;

		InputTextCallback_UserData cb_user_data;
		cb_user_data.Str = str;
		cb_user_data.ChainCallback = callback;
		cb_user_data.ChainCallbackUserData = user_data;
		return InputText(label, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
	}

}

void Cell2DApp::setViewer(igl::opengl::glfw::Viewer& viewer,
		igl::opengl::glfw::imgui::ImGuiMenu& menu)
{

	menu.callback_draw_custom_window = [&]() {
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(180.f*menu.menu_scaling(), 500), ImGuiCond_FirstUseEver);
		ImGui::Begin("Visualization", nullptr, ImGuiWindowFlags_NoSavedSettings);

		if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::PlotLines("R", plotline.data(), plotline.size(), 0,
					nullptr, 0, FLT_MAX, ImVec2(180.f*menu.menu_scaling(), 200));
			ImGui::Text("Residual: %e", plotline.back());
			// ImGui::Text("Total: %e", plotline.back());
		}
		if (ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit4("Background color", background_color.data());
			ImGui::ColorEdit3("Membrane color", line_color.data());
			ImGui::ColorEdit3("Cell color", cell_color.data());
		}

		ImGui::End();
	};

	menu.callback_draw_viewer_menu = [&]() {
		if (animation_recording) {
			const int n_frames = 120;
			std::cout << "Writing aninmation frame " << "\n";
			std::string number = std::to_string(animation_step);
			number.insert(number.begin(), 5 - number.size(), '0');
			std::string location = config.screencaps_location + "/anim-" + number + ".png";
			igl::png::render_to_png(location, viewer_width, viewer_height);
		}
		if (ImGui::CollapsingHeader("Initialization", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputInt("RNG seed", &seed);
			if (ImGui::Button("Reinit", ImVec2(-1,0)))
			{
				cellSim = cs2d::CellSim2D(23);
				cs2d::initialize_cells(cellSim, 4, 3, 6, 4, seed);
				std::cout << "CellSim2D initialized" << std::endl;
				updateScreen(viewer);
			}
		}
		//menu.draw_text(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0,0, 1), "Hello", Eigen::Vector4f(0,0,0.04,1));
		if (ImGui::CollapsingHeader("Load/Save", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputText("Results Path", &config.res_path);
			if (ImGui::Button("Serialize", ImVec2(-1,0)))
			{
				cs2d::Serializer ser(config.res_path, cellSim);
				cellSim.record_state(ser);
				updateScreen(viewer);
			}
			ImGui::InputInt("t", &t_step);
			ImGui::SameLine();
			ImGui::InputInt("s", &s_step);
			if (ImGui::Button("Deserialize", ImVec2(-1,0)))
			{
				cs2d::Deserializer de = cs2d::Deserializer::load_file(config.res_path);
				try {
					cellSim = de.load_state(t_step, s_step);
				} catch (cs2d::NoSuchFrameException) {
					std::cout << "No such frame (t=" << t_step << ",s=" << s_step << ")!" << std::endl;
				}
				updateScreen(viewer);
				/*
				catch const cs2d::NoSuchFrameException& e{
					std::cerr << "No such frame!" << std::endl;
				}
				*/
				//cellSim.load_state(de);
			}
		}
		if (ImGui::CollapsingHeader("Console output")) {
			ImGui::Checkbox("Print Step", &print_step);
			ImGui::Checkbox("Print Hessian", &config.print_hessian);
			ImGui::Checkbox("Total Potential", &config.print_total_potential);
			ImGui::Checkbox("Verbose console", &config.verbose_console);
			ImGui::Checkbox("Individual Potentials", &config.print_separate_potential);
			if (ImGui::Button("Plot search direction", ImVec2(-1,0)))
			{
				cellSim.write_search_line("search-"+std::to_string(cellSim.t)+".csv", 200);
				std::cout << "Search Direction written" << std::endl;
			}
		}
		if (ImGui::CollapsingHeader("Visualization")) {
			ImGui::Checkbox("Show mesh", &config.show_mesh);
			ImGui::Checkbox("Show edges", &config.show_edges);
			ImGui::Checkbox("Show vertices", &config.show_vertices);
		}
		if (ImGui::CollapsingHeader("ScreenCaps", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::Button("Capture Screen", ImVec2(-1,0))) {
				std::cout << "Preparing to write output" << "\n";

				// Save it to a PNG
				std::string number = std::to_string(cellSim.t);
				number.insert(number.begin(), 5 - number.size(), '0');
				std::string location = config.screencaps_location + "/out-" + number + ".png";
				igl::png::render_to_png(location, viewer_width, viewer_height);
			}
			ImGui::Checkbox("Write Screencaps", &config.write_screencaps);
		}
		if (ImGui::CollapsingHeader("Penalties")) {
			ImGui::InputDouble("Perimeter weight", &cellSim.config.weight_perimeter);
			ImGui::InputDouble("Volume weight", &cellSim.config.weight_volume);
			ImGui::InputDouble("Collision weight", &cellSim.config.weight_collision);
			ImGui::InputDouble("Boundary C. weight", &cellSim.config.weight_boundary_collision);
			ImGui::InputDouble("Boundary Shape weight", &cellSim.config.weight_boundary_shape);
			ImGui::InputDouble("Adhesion weight", &cellSim.config.weight_adhesion);

		}
		if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::Button("Record animation", ImVec2(-1,0))) {
				animation_recording = true;
				cellSim.till_convergence();
				updateScreen(viewer);
				viewer.core().is_animating = true;
			}
		}
		if (ImGui::CollapsingHeader("Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Use Hessian", &cellSim.config.use_hessian);
			ImGui::Checkbox("Check System Matrix", &cellSim.config.check_system_matrix);
			ImGui::Checkbox("Run animation", &viewer.core().is_animating);
			if (ImGui::Button("To Convergence", ImVec2(-1,0))) {
				cellSim.till_convergence();
				updateScreen(viewer);
			}
			ImGui::InputInt("Line Search Max iter.", &cellSim.config.max_steps_line_search);
			ImGui::InputInt("Hessian Reg Max iter.", &cellSim.config.max_steps_hessian_reg);
		}
		if (ImGui::CollapsingHeader("Cell Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputDouble("Resting perimeter", &cellSim.config.perimeter_goal);
			ImGui::InputDouble("Resting volume", &cellSim.config.volume_goal);
			ImGui::InputDouble("d0", &cellSim.config.d0);
			ImGui::InputDouble("d0_adhesion", &cellSim.config.d0_adhesion);
			ImGui::InputDouble("k_adhesion", &cellSim.config.adhesion_logi_k);
		}
		if (ImGui::CollapsingHeader("Other properties", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputDouble("Boundary width", &cellSim.config.boundary_width);
			ImGui::InputDouble("Boundary height", &cellSim.config.boundary_height);
		}

	};

	viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
		if (viewer.core().is_animating) {
			updateScreen(viewer);
		}
		return false;
	};
	viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
		if (animation_recording) {
			const int n_frames = 240;
			const double fact = std::pow(2, 1.0/240); // x2 in 60 steps
			if (animation_step >= n_frames) {
				animation_recording = false;
				viewer.core().is_animating = false;
				animation_step = 0;
			}
			cellSim.config.boundary_width *= fact;
			cellSim.config.boundary_height /= fact;

			cellSim.till_convergence();
			++animation_step;
		} else if (viewer.core().is_animating) {
			if (print_step)
				std::cout << "\n"<< "Step = " << cellSim.t << "\n";
			static_step();
		}
		return false;
	};
	viewer.callback_key_pressed = 
		[&](igl::opengl::glfw::Viewer & viewer,unsigned int key,int mods)->bool
		{
			switch(key)
			{
				case 'r':
					viewer.core().is_animating = !viewer.core().is_animating;
					break;
				case ' ':
					if (print_step)
						std::cout << "\n"<< "Step = " << cellSim.t << "\n";
					for (int i = 0; i < 1; ++i)
						static_step();
					//std::cout << cellSim.vertices << std::endl;
					break;
				default: 
					return false;
			}
			updateScreen(viewer);
			if (config.write_screencaps) {

				std::cout << "Preparing to write output" << "\n";

				// Save it to a PNG
				std::string number = std::to_string(cellSim.t);
				number.insert(number.begin(), 5 - number.size(), '0');
				std::string location = config.screencaps_location + "/out-" + number + ".png";
				igl::png::render_to_png(location, 1200, 1200);
			}
			return true;

		};

	updateScreen(viewer);

	viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_NO_ROTATION);
	viewer.core().background_color = background_color;
	viewer.data().set_face_based(true);
	viewer.data().shininess = 1.0;
	viewer.data().point_size = 20.0;
	viewer.data().line_width = 4.0;

	viewer.data().show_custom_labels = true;
}

void Cell2DApp::updateScreen(igl::opengl::glfw::Viewer& viewer)
{
	viewer.data().clear();
	if (config.show_vertices) {
		auto verts = cellSim.get_vertices().transpose();
		viewer.data().add_points(verts, Eigen::RowVector3d(.7, .5, 0));
	}
	const int cell_segments = cellSim.cell_segments;
	if (config.show_edges) {
		Eigen::MatrixXd edges1(cellSim.n_vertices(), 3);
		Eigen::MatrixXd edges2(cellSim.n_vertices(), 3);
		for (int c = 0; c < cellSim.n_cells; ++c) {
			Matrix<double, Eigen::Dynamic, 3> c_verts(cell_segments, 3);
			c_verts.leftCols<2>() = cellSim.get_cell_vertices(c).transpose();
			c_verts.rightCols<1>().setZero();
			edges1.middleRows(c*cell_segments, cell_segments) = c_verts;
			edges2.middleRows(c*cell_segments, cell_segments-1) = c_verts.bottomRows(cell_segments-1);
			edges2.row(c*cell_segments+cell_segments-1) = c_verts.row(0);
		}
		viewer.data().add_edges(edges1, edges2, Eigen::RowVector3d(
					line_color(0), line_color(1), line_color(2)));
	}

	if (config.show_mesh) {
		auto [V, F] = cellSim.triangulate_all_cells();
		viewer.data().set_mesh(V, F);
		bool adhesion_colors = true;
		if (adhesion_colors) {
			const int cell_visual_verts = cellSim.cell_segments+1;
			VectorXd data_vector(cellSim.n_cells*cell_visual_verts);
			for (int i = 0; i < cellSim.n_cells; ++i) {
				data_vector.segment(i*cell_visual_verts, cell_visual_verts) = VectorXd::Constant(cell_visual_verts, cellSim.params[i]);
			}
			viewer.data().set_data(data_vector, 0, 1);
		} else {
			MatrixXd C(F.rows(), 3);
			for (int i = 0; i < F.rows(); ++i)
				C.row(i) = cell_color.cast<double>();
			viewer.data().set_colors(C);
		}
		viewer.data().show_lines = !config.show_edges;
	}

	{
		// draw boundary
		const int n_boundary = cellSim.n_boundary;
		Eigen::MatrixXd boundary1(n_boundary, 3);
		Eigen::MatrixXd boundary2(n_boundary, 3);
		boundary1.setZero();
		boundary1.leftCols<2>() = cellSim.get_boundary().transpose();
		boundary2.topRows(n_boundary-1) = boundary1.bottomRows(n_boundary-1);
		boundary2.row(n_boundary-1) = boundary1.row(0);
		viewer.data().add_edges(boundary1, boundary2, Eigen::RowVector3d(0, 0, 0));
	}
}
