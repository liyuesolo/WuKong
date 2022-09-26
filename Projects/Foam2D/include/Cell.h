#ifndef CELL_S
#define CELL_S

#include <cmath>
#include <cassert>
#include <vector>
#include <tuple>
#include <map>
#include <algorithm>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/triangle/triangulate.h>
#include <igl/doublearea.h>

#include "config.h"
#include "constants.hpp"


using Eigen::Block;
using Eigen::MatrixBase;
using Eigen::Matrix, Eigen::MatrixXi, Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Vector, Eigen::Vector2d, Eigen::Vector3d, Eigen::VectorXd;

using std::tuple;

using CellDerivativeVector = Vector<double, N_SEGMENTS * 2>;
using CellVertexMatrix3 = Matrix<double, N_SEGMENTS, 3>;
using CellVertexMatrix2 = Matrix<double, N_SEGMENTS, 2>;

template <typename T>
void print_dimensions(const std::string& name, const MatrixBase<T>& matrix) {
	std::cout << name << " dims: " << matrix.rows() << " x " << matrix.cols() << std::endl;
}

class CellSim {
	public:

		using Edge = tuple<int, int>;

		static const Matrix<int, N_SEGMENTS, 2> cell_edges;
		struct Cell {
			int vertexOffset;
		};

		MatrixXd vertices_state;
		std::vector<Cell> cells;

		int t = 0;
		CellSim(): vertices_state(0, N_SEGMENTS){ }

		void addCell(const Matrix<double,N_SEGMENTS,3>& cell_vertices);
		std::pair<MatrixXd, MatrixXi> triangulate_all_cells() const;

		inline Block<const MatrixXd, N_SEGMENTS> cellVertices(const MatrixXd& vertices, int cell_idx) const {
			return vertices.middleRows<N_SEGMENTS>(cell_idx*N_SEGMENTS);
		}
		inline Block<MatrixXd, N_SEGMENTS> cellVertices(MatrixXd& vertices, int cell_idx) const {
			return vertices.middleRows<N_SEGMENTS>(cell_idx*N_SEGMENTS);
		}

		Matrix<int, N_SEGMENTS, 2> cellEdgeList() const;

		double volumePotential(const MatrixXd& vertices) const;
		double perimeterPotential(const MatrixXd& vertices) const;

		VectorXd volumePotentialD(const MatrixXd& vertices) const;
		VectorXd perimeterPotentialD(const MatrixXd& vertices) const;

		SparseMatrix<double> volumePotentialH(const MatrixXd& vertices) const;
		SparseMatrix<double> perimeterPotentialH(const MatrixXd& vertices) const;

		bool intervalsOverlap(const Vector2d& a_min_max, const Vector2d& b_min_max, double epsilon) const;
		bool compareBoundingBoxes(const MatrixXd& vertices_a, const MatrixXd& vertices_b, double epsilon) const;

		// compute the distance and derivatives for the vertices of cell A w.r.t the edges of cell B
		std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> distancesAtoB(
				const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const;

		// compute the barrier and its derivative for the vertices of cell A w.r.t the edges of cell B
		std::tuple<double, CellDerivativeVector, CellDerivativeVector> barrierAtoB(
				const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const;

		std::pair<double, MatrixXd> collisionPotential(const MatrixXd& vertices) const;
		double computePotential(const MatrixXd& vertices) const;
		VectorXd computeResidual(const MatrixXd& vertices) const;
		SparseMatrix<double> computeSystemMatrix(const MatrixXd& vertices) const;

		void jiggle(double delta);
		void lineSearch();
		void step();
		double perimeter_goal = 8;
		double volume_goal = 10;
		Config config;

};

#endif
