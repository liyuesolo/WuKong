#include "../include/Cell.h"
#include "../include/ad.h"
#include "../include/area.h"
#include "../include/perimeter.h"

using Eigen::VectorXi;
//Matrix<double, N_SEGMENTS, 2> buildCellMatrix(){
const Matrix<int, N_SEGMENTS, 2> CellSim::cell_edges = [](){
	Matrix<int, N_SEGMENTS, 2> edges;
	for (int i = 0; i < N_SEGMENTS; i++) {
		edges.row(i) << i, (i+1) % N_SEGMENTS;
	}
	return edges;
}();

void CellSim::addCell(const Matrix<double,N_SEGMENTS,3>& cell_vertices) {
	const int n_verts = vertices_state.rows();
	const MatrixXd oldVerts(vertices_state);
	vertices_state.resize(n_verts+N_SEGMENTS, 3);
	if (n_verts > 0)
		vertices_state.topRows(n_verts) = oldVerts;
	auto cellVerts = vertices_state.bottomRows<N_SEGMENTS>();
	cellVerts = cell_vertices;
	cells.push_back(Cell{n_verts});
}

std::pair<MatrixXd, MatrixXi> CellSim::triangulate_all_cells() const {
	MatrixXd hole_points;
	MatrixXi cell_boundary_edges(cells.size()*N_SEGMENTS, 2);

	MatrixXd V;
	MatrixXi F;
	for (size_t edge_idx = 0; edge_idx < cells.size()*N_SEGMENTS; edge_idx++) {
		cell_boundary_edges.row(edge_idx) << edge_idx, (edge_idx/N_SEGMENTS)*N_SEGMENTS + (edge_idx+1)%N_SEGMENTS;
	}

	igl::triangle::triangulate(
			vertices_state.leftCols<2>(),
			cell_boundary_edges,
			hole_points, 
			// cmd, 
			"aqQ",
			V, F);
	std::pair<MatrixXd, MatrixXi> result = std::make_pair(V, F);
	return result;

}

std::pair<double, CellDerivativeVector> CellSim::areaDerivatives(
		const Matrix<double, N_SEGMENTS, 2>& vertices
		) const {
	// computes the 2d area of a polygon defined by `vertices`
	std::vector<double> vertices_vec(3*2);

	std::vector<double> jacobian(N_SEGMENTS*2);
	Eigen::Map<Vector<double, N_SEGMENTS*2>> derivatives_vec(jacobian.data());

	double area_scalar = 0;

	Vector<double, N_SEGMENTS*2> derivatives_old;
	derivatives_old.setZero();

	MatrixXd V, areas;
	MatrixXi F;

	igl::triangle::triangulate(
			vertices,
			cell_edges,
			MatrixXd(), 
			"aqQ",  // any triangulation is fine for computing the volume
			V, F);

	for (int i = 0; i < vertices.rows(); i++) {
		if (V.row(i) != vertices.row(i)) {
			std::cout << "Vertex " << i << " changed! new: " << V.row(i) << ", original: " << vertices.row(i);
		}
	}
	igl::doublearea(V, F, areas);
	areas /= 2;
	double area = 0;
	Vector<double, N_SEGMENTS*2> derivatives;
	derivatives.setZero();

	/*
	Vector<double, N_SEGMENTS*2> derivatives_new;
	derivatives_new.setZero();
	*/
	const Vector3d unitX(1, 0, 0), unitY(0, 1, 0), unitZ(0, 0, 1);

	for (const Eigen::RowVector3i& triangle: F.rowwise()) {
		// compute the area of the face
		Vector3d a, b, c;
		a.setZero();
		b.setZero();
		c.setZero();
		a.head<2>() = V.row(triangle[1]) - V.row(triangle[0]);
		b.head<2>() = V.row(triangle[2]) - V.row(triangle[1]);
		c.head<2>() = V.row(triangle[0]) - V.row(triangle[2]);
		area += a.cross(b).dot(unitZ)/2;

		if (triangle[0] < N_SEGMENTS) {
			derivatives(triangle[0]*2 + 0) += b.cross(unitX).dot(unitZ);
			derivatives(triangle[0]*2 + 1) += b.cross(unitY).dot(unitZ);
		}
		if (triangle[1] < N_SEGMENTS) {
			derivatives(triangle[1]*2 + 0) += c.cross(unitX).dot(unitZ);
			derivatives(triangle[1]*2 + 1) += c.cross(unitY).dot(unitZ);
		}
		if (triangle[2] < N_SEGMENTS) {
			derivatives(triangle[2]*2 + 0) += a.cross(unitX).dot(unitZ);
			derivatives(triangle[2]*2 + 1) += a.cross(unitY).dot(unitZ);
		}

		/*
		const std::vector<double> vertices_vec{a[0], a[1], b[0], b[1], c[0], c[1]};
		area_scalar += triangleArea(vertices_vec);
		triangleAreaJacobian(jacobian, vertices_vec);
		derivatives_new(triangle[0]*2 + 0) += jacobian[0];
		derivatives_new(triangle[0]*2 + 1) += jacobian[1];
		derivatives_new(triangle[1]*2 + 0) += jacobian[2];
		derivatives_new(triangle[1]*2 + 1) += jacobian[3];
		derivatives_new(triangle[2]*2 + 0) += jacobian[4];
		derivatives_new(triangle[2]*2 + 1) += jacobian[5];
		*/
	}
	derivatives /= 2;
	std::cout << "Area (old): " << area
		<< ", (new): " << area_scalar << std::endl
		<< "Derivatives (old): " << derivatives.transpose() << std::endl
		//<< "Derivatives (new): " << derivatives_new.transpose()
		<< std::endl;
	return std::make_pair(area, derivatives);
}

void CellSim::jiggle(double delta) {
	vertices_state.leftCols<2>() += vertices_state.Random(vertices_state.rows(), 2)*delta;
}

void CellSim::lineSearch() {
	jiggle(0.01);
	/*
	 * 1) get derivative
	 * 2) check x axis intercept
	 * 3) repeat if solution does not improve
	 *
	const double min_step = 1e-9, start_step = 1;

	double potential;
	MatrixXd potential_d, potential_d_like_vertices(vertices_state.rows(), vertices_state.cols());
	std::tie(potential, potential_d) = potentialDerivatives(vertices_state);
	const double start_potential = potential;

	potential_d_like_vertices.setZero();
	for (size_t cell_idx = 0; cell_idx < cells.size(); cell_idx++) {
		potential_d_like_vertices.leftCols<2>().middleRows<N_SEGMENTS>(cell_idx*N_SEGMENTS) = potential_d.row(cell_idx).reshaped(2, N_SEGMENTS).transpose();
	}

	double step_size = start_step;
	//const double step_factor = potential/potential_d.sum();
	const double step_factor = 1;
	std::cout << "The potential is " << potential << " and the step factor is " << step_factor << std::endl;

	int i = 0;
	for (; step_size > min_step; step_size = std::max(min_step, step_size/2)) {
		// compute potential for given step size
		std::tie(potential, potential_d) = potentialDerivatives(vertices_state - step_size * step_factor * potential_d_like_vertices);
		if (potential < start_potential) {
			vertices_state -= step_size * step_factor * potential_d_like_vertices;
			std::cout << "Line search concluded at step " << i << std::endl;
			return;
		}
		std::cout << "Step " << i << ": \t Potential = " << potential << std::endl;
		i++;
	}
	std::cout << "Line search failed after " << i << " steps." << std::endl;
	 * */
}

double CellSim::volumePotential(const MatrixXd& vertices) const {
	double potential = 0;
	const int n_cells = cells.size();
	MatrixXd V;
	MatrixXi F;

	for (int i = 0; i < n_cells; i++) {
		igl::triangle::triangulate(
			cellVertices(vertices, i).leftCols<2>(),
			cell_edges, MatrixXd(), "aqQ", V, F);
		double volume = cellSum<3>(triangleArea, V, F);
		potential += pow(volume - volume_goal, 2);

	}
	return potential;
}

double CellSim::perimeterPotential(const MatrixXd& vertices) const {
	double potential = 0;
	const int n_cells = cells.size();
	for (int i = 0; i < n_cells; i++) {
		const double perimeter = cellSum<2>(lineLength, cellVertices(vertices, i).leftCols<2>(), cell_edges);
		potential += pow(perimeter - perimeter_goal, 2);
	}
	return potential;
}

VectorXd CellSim::volumePotentialD(const MatrixXd& vertices) const {
	const int n_cells = cells.size(), n_dof = 2*vertices.rows();
	VectorXd jacobian = VectorXd::Zero(n_dof);
	MatrixXd V;
	MatrixXi F;

	double potential = 0;

	for (int i = 0; i < n_cells; i++) {
		igl::triangle::triangulate(
			cellVertices(vertices, i).leftCols<2>(),
			cell_edges, MatrixXd(), "aqQ", V, F);

		double volume = cellSum<3>(triangleArea, V, F);
		potential += pow(volume - volume_goal, 2);

		const VectorXd area_jac = cellSumJacobian<3>(triangleAreaJacobian, V, F).head<2*N_SEGMENTS>(); 
		const VectorXd potential_j = 2 * (volume - volume_goal) * area_jac;
		jacobian.segment<N_SEGMENTS*2>(i*N_SEGMENTS*2) = potential_j;
	}
	return jacobian;
}

VectorXd CellSim::perimeterPotentialD(const MatrixXd& vertices) const {
	const int n_cells = cells.size(), n_dof = 2*vertices.rows();
	double potential = 0;
	VectorXd jacobian = VectorXd::Zero(n_dof);
	for (int i = 0; i < n_cells; i++) {
		const double perimeter = cellSum<2>(lineLength, cellVertices(vertices, i).leftCols<2>(), cell_edges);
		potential += pow(perimeter - perimeter_goal, 2);
		const VectorXd per_jac = cellSumJacobian<2>(lineLengthJacobian, cellVertices(vertices, i).leftCols<2>(), cell_edges); 
		const VectorXd potential_j = 2 * (perimeter - perimeter_goal) * per_jac;
		jacobian.segment<N_SEGMENTS*2>(i*N_SEGMENTS*2) = potential_j;
	}
	return jacobian;
}

SparseMatrix<double> CellSim::volumePotentialH(const MatrixXd& vertices) const {
	using CellHessian = Matrix<double, 2*N_SEGMENTS, 2*N_SEGMENTS>;
	const int n_cells = cells.size(), n_dof = 2*vertices.rows();
	SparseMatrix<double> hessian;
	hessian.reserve(VectorXi::Constant(n_dof, N_SEGMENTS*2));

	MatrixXd V;
	MatrixXi F;

	for (int c = 0; c < n_cells; ++c) {
		igl::triangle::triangulate(
			cellVertices(vertices, c).leftCols<2>(),
			cell_edges, MatrixXd(), "aqQ", V, F);

		const int c_off = c*N_SEGMENTS*2;
		const CellHessian cell_hessian = cellSumHessian<3>(
				triangleAreaHessian,
				V, F);
		for (int i = 0; i < 2*N_SEGMENTS; ++i) {
			for (int j = 0; j < 2*N_SEGMENTS; ++j) {
				hessian.insert(i+c_off,j+c_off) = cell_hessian(i,j);
			}
		}
	}
	hessian.makeCompressed();
	return hessian;
}

SparseMatrix<double> CellSim::perimeterPotentialH(const MatrixXd& vertices) const {
	using CellHessian = Matrix<double, 2*N_SEGMENTS, 2*N_SEGMENTS>;
	const int n_cells = cells.size(), n_dof = 2*vertices.rows();
	SparseMatrix<double> hessian;
	hessian.reserve(VectorXi::Constant(n_dof, N_SEGMENTS*2));

	for (int c = 0; c < n_cells; ++c) {
		const int c_off = c*N_SEGMENTS*2;
		const CellHessian cell_hessian = cellSumHessian<2>(
				lineLengthHessian,
				cellVertices(vertices, c).leftCols<2>(), cell_edges);
		for (int i = 0; i < 2*N_SEGMENTS; ++i) {
			for (int j = 0; j < 2*N_SEGMENTS; ++j) {
				hessian.insert(i+c_off,j+c_off) = cell_hessian(i,j);
			}
		}
	}
	hessian.makeCompressed();
	return hessian;
}

double CellSim::computePotential(const MatrixXd& vertices) const {
	const int n_dof = 2*vertices.rows();
	double potential;
	return potential;
}

VectorXd CellSim::computeResidual(const MatrixXd& vertices) const {
	const int n_dof = 2*vertices.rows();
	VectorXd r(n_dof);
	return r;
}

SparseMatrix<double> CellSim::computeSystemMatrix(const MatrixXd& vertices) const {
	const int n_dof = 2*vertices.rows();
	SparseMatrix<double> K(n_dof, n_dof);
	return K;
}

void CellSim::step() {
	// compute whole-system potential (p)
	const int n_cells = cells.size(), n_dof = 2*vertices_state.rows();
	const double p_perimeter = perimeterPotential(vertices_state);
	const double p_volume = volumePotential(vertices_state);
	const double potential = p_perimeter + p_volume;

	// compute whole-system jacobian (residual, r)
	
	const VectorXd r_perimeter = perimeterPotentialD(vertices_state);
	const VectorXd r_volume = volumePotentialD(vertices_state);
	const VectorXd jacobian = r_perimeter + r_volume;


	if (config.print_separate_potential) {
		std::cout << "Volume potential: " << p_volume << std::endl;
		std::cout << "Volume residual norm: " << r_volume.norm() << std::endl;
		std::cout << "Perimeter potential: " << p_perimeter << std::endl;
		std::cout << "Perimeter residual norm: " << r_perimeter.norm() << std::endl;
	}
	if (config.print_total_potential) {
		std::cout << "Total potential: " << potential << std::endl;
		std::cout << "Total residual norm: " << jacobian.norm() << std::endl;
	}

	auto gd_matrix = jacobian.reshaped(2, n_dof/2).transpose();
	vertices_state.leftCols<2>() -= gd_matrix * 1e-2;
	// compute whole-system hessian (system matrix, K)
	// solve K x = -r
	// take a step in direction of x
	const SparseMatrix hessian = computeSystemMatrix(vertices_state);
	++t;
}

std::pair<double, MatrixXd> CellSim::collisionPotential(const MatrixXd& vertices) const {
	const int n_cells = cells.size();
	const double d_min = 1e-1;
	double potential = 0;
	MatrixXd potential_d(n_cells, N_SEGMENTS*2);

	for (int i = 0; i < n_cells; i++) {
		for (int j = i+1; j < n_cells; j++) {
			//std::cout << "Checking cells " << i << " and " << j << " for collisions." << std::endl;
			// compute bounding box
			if (compareBoundingBoxes(cellVertices(vertices, i), cellVertices(vertices, j), d_min)) {
				std::cout << "Cells " << i << " and " << j << " have overlapping bounding boxes. Computing derivatives." << std::endl;
				double c_potential;
				CellDerivativeVector c_a_d, c_b_d;
				std::tie(c_potential, c_a_d, c_b_d) = barrierAtoB(d_min, cellVertices(vertices, i), cellVertices(vertices, j));
				std::cout << "Potential is " << c_potential << std::endl;
				potential_d.row(i) += c_a_d;
				potential_d.row(j) += c_b_d;
				//distancesAtoB(d_min, cellVerticesC(vertices, j), cellVerticesC(vertices, i));
			}
			// compute potential for each cell

		}
	}
	return std::make_pair(potential, potential_d);
	/*
	   const auto perimeter_der = perimeterDerivativesAll(vertices);
	   const VectorXd perimeterResiduals = perimeter_der.first.array() - idealPerimeter;
	   const double perimeter_potential = perimeterResiduals.array().pow(2).sum();
	   MatrixXd derivatives = perimeter_der.second;

	   for (int i = 0; i < derivatives.rows(); i++) {
	   derivatives.row(i) *= 2 * perimeterResiduals(i);
	   }
	   return std::make_pair(perimeter_potential, derivatives);
	   */
}

bool CellSim::intervalsOverlap(const Vector2d& a_min_max, const Vector2d& b_min_max, double epsilon) const {
	if (a_min_max(0) <= b_min_max(0)) {
		return (a_min_max(1) + epsilon >= b_min_max(0));
	} else {
		return intervalsOverlap(b_min_max, a_min_max, epsilon);
	}
}

bool CellSim::compareBoundingBoxes(const MatrixXd& vertices_a, const MatrixXd& vertices_b, double epsilon) const {
	const Vector3d a_min = vertices_a.colwise().minCoeff();
	const Vector3d a_max = vertices_a.colwise().maxCoeff();
	const Vector3d b_min = vertices_b.colwise().minCoeff();
	const Vector3d b_max = vertices_b.colwise().maxCoeff();
	if (intervalsOverlap(Vector2d(a_min(0), a_max(0)), Vector2d(b_min(0), b_max(0)), epsilon)) {
		if (intervalsOverlap(Vector2d(a_min(1), a_max(1)), Vector2d(b_min(1), b_max(1)), epsilon)) {
			return true;
		}
	}
	return false;
}

std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> CellSim::distancesAtoB(
		const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const {
	// each row corrsponds to a vertex of A
	std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> dist_d(N_SEGMENTS);
	for(int j = 0; j < N_SEGMENTS; j++) {
		dist_d[j] = std::make_tuple(
				d_min,
				0, Vector3d(0,0,0), 0, Vector3d(0,0,0), Vector3d(0,0,0)
				);
	}
	for(int j = 0; j < N_SEGMENTS; j++) {
		for(int i = 0; i < N_SEGMENTS; i++) {
			// i is the vertex index in A, m
			// j is the edge index in B, k, l
			const Vector3d k = vertices_b.row((i+1)%N_SEGMENTS);
			const Vector3d l = vertices_b.row(i);
			const Vector3d m = vertices_a.row(j);
			const double l_edge = (k - l).norm();

			// d = 1/(2*l_edge) * A
			// A = (m1-k1) * (l2-k2) - (m2-k2)*(l1-k1)
			const double A = std::abs((m[0]-k[0]) * (l[1]-k[1]) - (m[1]-k[1])*(l[0]-k[0]));
			const double d = 0.5 * A / l_edge;

			if (d > d_min || d > std::get<0>(dist_d[j])) continue;
			double l_edge_sq_r_h = 0.5 / (l_edge*l_edge);
			double dd_dk1 = (l_edge*(m[1]-l[1]) - (A * (k[0]-l[0])/ l_edge)) * l_edge_sq_r_h;
			double dd_dk2 = (l_edge*(l[0]-m[0]) - (A * (k[1]-l[1])/ l_edge)) * l_edge_sq_r_h;

			double dd_dl1 = (l_edge*(k[1]-m[1]) - (A * (l[0]-k[0])/ l_edge)) * l_edge_sq_r_h;
			double dd_dl2 = (l_edge*(m[0]-k[0]) - (A * (l[1]-k[1])/ l_edge)) * l_edge_sq_r_h;

			double dd_dm1 = 0.5/l_edge * (l[1]-k[1]);
			double dd_dm2 = 0.5/l_edge * (k[0]-l[0]);

			dist_d[j] = std::make_tuple(
					d,
					(i+1)%N_SEGMENTS, Vector3d(dd_dk1, dd_dk2, 0),
					i, Vector3d(dd_dl1, dd_dl2, 0),
					Vector3d(dd_dm1, dd_dm2, 0)
					);
		}
	}
	return dist_d;
}

// compute the barrier and its derivative for the vertices of cell A w.r.t the edges of cell B
std::tuple<double, CellDerivativeVector, CellDerivativeVector> CellSim::barrierAtoB(
		const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const {
	double potential = 0;
	CellDerivativeVector a_d = CellDerivativeVector::Zero(), b_d = CellDerivativeVector::Zero();
	auto results = distancesAtoB(d_min, vertices_a, vertices_b);
	for (int a_idx = 0; a_idx < N_SEGMENTS; a_idx++) {
		// compute the barrier potential incurred by vertex a_idx
		const auto [d, b1_idx, b1_d, b2_idx, b2_d, a_d_c] = results[a_idx];
		if (d>=d_min) continue;
		const double barrier_value = -std::pow(d - d_min, 2) * std::log(d/d_min);
		const double barrier_d = -(2*(d - d_min) * std::log(d/d_min) + std::pow(d - d_min, 2)/d);
		potential += barrier_value;
		a_d.segment<2>(2*a_idx) = a_d_c.head<2>();
		b_d.segment<2>(2*b1_idx) += b1_d.head<2>();
		b_d.segment<2>(2*b2_idx) += b2_d.head<2>();
	}
	return {potential, a_d, b_d};
}
