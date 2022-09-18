#include <cmath>
#include <cassert>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <igl/triangle/triangulate.h>
#include <igl/doublearea.h>

#ifndef CELL_S
#define CELL_S

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;
using Eigen::VectorXd;

using std::tuple;

constexpr int n_segments = 7;
using CellDerivativeVector = Eigen::Vector<double, n_segments * 2>;
using CellVertexMatrix3 = Eigen::Matrix<double, n_segments, 3>;
using CellVertexMatrix2 = Eigen::Matrix<double, n_segments, 2>;

template <typename T>
void print_dimensions(const std::string& name, const Eigen::MatrixBase<T>& matrix) {
	std::cout << name << " dims: " << matrix.rows() << " x " << matrix.cols() << std::endl;
}

class CellSim {
	public:

		using Edge = tuple<int, int>;

		struct Cell {
			int vertexOffset;
		};

		MatrixXd vertices_state;
		std::vector<Cell> cells;

		int t = 0;
		CellSim(): vertices_state(0, n_segments){ }

		void addCell(const Eigen::Matrix<double,n_segments,3>& cell_vertices);
		std::pair<MatrixXd, MatrixXi> triangulate_all_cells() const;

		Eigen::Block<const Eigen::MatrixXd, n_segments> cellVerticesC(const MatrixXd vertices, int cell_idx) const;

		Eigen::Block<Eigen::MatrixXd, n_segments> cellVertices(MatrixXd vertices, int cell_idx) const;

		Eigen::Matrix<int, n_segments, 2> cellEdgeList() const;

		std::pair<double, CellDerivativeVector> areaDerivatives(
				const Eigen::Matrix<double, n_segments, 2>& vertices
				) const;

		std::pair<double, CellDerivativeVector> perimeterDerivatives(
				const Eigen::Matrix<double, n_segments, 2>& vertices
				) const;

		std::pair<Eigen::VectorXd, Eigen::MatrixXd> volumeDerivativesAll(const MatrixXd& vertices) const;
		std::pair<Eigen::VectorXd, Eigen::MatrixXd> perimeterDerivativesAll(const MatrixXd& vertices) const;
		std::pair<double, MatrixXd> volumePotential(const MatrixXd& vertices) const;
		std::pair<double, MatrixXd> perimeterPotential(const MatrixXd& vertices) const;

		bool intervalsOverlap(const Eigen::Vector2d& a_min_max, const Eigen::Vector2d& b_min_max, double epsilon) const;
		bool compareBoundingBoxes(const MatrixXd& vertices_a, const MatrixXd& vertices_b, double epsilon) const;

		// compute the distance and derivatives for the vertices of cell A w.r.t the edges of cell B
		std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> distancesAtoB(
				const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const;

		// compute the barrier and its derivative for the vertices of cell A w.r.t the edges of cell B
		std::tuple<double, CellDerivativeVector, CellDerivativeVector> barrierAtoB(
				const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const;

		std::pair<double, MatrixXd> collisionPotential(const MatrixXd& vertices) const;
		std::pair<double, MatrixXd> potentialDerivatives(const MatrixXd& vertices) const;

		void jiggle(double delta);
		void lineSearch();
		void step();

};
#endif
