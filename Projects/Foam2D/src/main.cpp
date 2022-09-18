#include "../include/App.h"
#include <cmath>
#include <random>
#define PI 3.14159265

using std::log;
using std::pow;

std::mt19937 gen;

Eigen::MatrixXd createEdgeLoop(const Eigen::MatrixXd& vertices, const std::vector<int>& vertex_idxs) {
	const size_t num_vertices = vertex_idxs.size();
	Eigen::MatrixXd verts(num_vertices, 3);
	for(size_t i = 0; i < num_vertices; i++) {
		verts.row(i) = vertices.row(i);
	}
	return verts;
}

void createCell(CellSim& cellSim, Eigen::Vector2d center, double roughness) {
	Eigen::MatrixXd cellVertices(n_segments, 3);
	cellVertices.rightCols<1>().setZero();
	std::normal_distribution<double> normal{1, 1./n_segments};
	for (int i = 0; i < n_segments; i ++) {
		cellVertices.leftCols(2).row(i) = center + normal(gen) * Eigen::Vector2d(
				std::cos((PI*2*i)/n_segments),
				std::sin((PI*2*i)/n_segments)
		);
	}

	std::vector<int> edges(n_segments);
	for (int i = 0; i < edges.size(); i++) {
		edges[i] = i;
	}
	cellSim.addCell(createEdgeLoop(cellVertices, edges));
}

int main()
{
	gen.seed(124);
	std::normal_distribution<double> normal{0, 0.2};
	std::cout << normal(gen);

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

	CellSim cellSim;


	std::cout << "CellSim initialized" << std::endl;
	createCell(cellSim, Eigen::Vector2d(-4, 1.4), 0.4);
	createCell(cellSim, Eigen::Vector2d(0, 0), 0.4);
	createCell(cellSim, Eigen::Vector2d(1, -3), 0.4);
	createCell(cellSim, Eigen::Vector2d(3, 0), 0.4);

    Cell2DApp app(cellSim);
    app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}
