#include <Eigen/Dense>
#include <random>
#include "../include/CellSim.h"
#define PI 3.14159265


using namespace cs2d;
using Eigen::Vector2d;
std::mt19937 gen;

void createCell(cs2d::CellSim2D& cs, Vector2d center, double roughness, double diameter=1) {
	Eigen::MatrixXd cellVertices(cs.cell_segments, 2);
	std::uniform_real_distribution<double> affinity(0, 1);
	std::normal_distribution<double> normal{0, 1./cs.cell_segments};
	for (int i = 0; i < cs.cell_segments; i ++) {
		cellVertices.leftCols(2).row(i) = center + diameter/2 * (1+normal(gen)*roughness) * Vector2d(
				std::cos((PI*2*i)/cs.cell_segments),
				std::sin((PI*2*i)/cs.cell_segments)
		);
	}
	cs.addCell(cellVertices.transpose(), affinity(gen));
}

void cs2d::initialize_cells(CellSim2D& cs, int cols, int rows, double width, double height, int seed) {

	gen.seed(seed);

	double r_diff = 1.0/(cols+1);
	double c_diff = 1.0/(rows+1);
	for (int i = 0; i < cols; ++i) {
		for (int j = 0; j < rows; ++j) {
			Vector2d offset(
					((i+1.0)*r_diff-0.5)*width,
					((j+1.0)*c_diff-0.5)*height);
			if (j%2 == 1)
				offset(0) += r_diff*width*0.25;
			else
				offset(0) -= r_diff*width*0.25;
			createCell(cs, offset, 0.1, 0.43);

		}
	}
}
