#include "../include/Cell.h"

void CellSim::addCell(const Eigen::Matrix<double,n_segments,3>& cell_vertices) {
	const int n_verts = vertices_state.rows();
	const Eigen::MatrixXd oldVerts(vertices_state);
	vertices_state.resize(n_verts+n_segments, 3);
	if (n_verts > 0)
		vertices_state.topRows(n_verts) = oldVerts;
	auto cellVerts = vertices_state.bottomRows<n_segments>();
	cellVerts = cell_vertices;
	cells.push_back(Cell{n_verts});
}


std::pair<MatrixXd, MatrixXi> CellSim::triangulate_all_cells() const {
	MatrixXd hole_points;
	MatrixXi cell_boundary_edges(cells.size()*n_segments, 2);

	MatrixXd V;
	MatrixXi F;
	for (size_t edge_idx = 0; edge_idx < cells.size()*n_segments; edge_idx++) {
		cell_boundary_edges.row(edge_idx) << edge_idx, (edge_idx/n_segments)*n_segments + (edge_idx+1)%n_segments;
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

Eigen::Block<const Eigen::MatrixXd, n_segments> CellSim::cellVerticesC(const MatrixXd vertices, int cell_idx) const {
	return vertices.middleRows<n_segments>(cell_idx*n_segments);
}

Eigen::Block<Eigen::MatrixXd, n_segments> CellSim::cellVertices(MatrixXd vertices, int cell_idx) const {
	return vertices.middleRows<n_segments>(cell_idx*n_segments);
}

Eigen::Matrix<int, n_segments, 2> CellSim::cellEdgeList() const {
	Eigen::Matrix<int, n_segments, 2> local_edges;
	for(int i = 0; i < n_segments; i++) {
		local_edges.row(i) << i, (i+1)%n_segments;
	}
	return local_edges;
}

std::pair<double, CellDerivativeVector> CellSim::areaDerivatives(
		const Eigen::Matrix<double, n_segments, 2>& vertices
		) const {
	// computes the 2d area of a polygon defined by `vertices`
	constexpr double normTol = 1e-12;

	const Eigen::MatrixXi cellEdges = cellEdgeList();

	Eigen::MatrixXd V, areas;
	Eigen::MatrixXi F;

	igl::triangle::triangulate(
			vertices,
			cellEdges,
			Eigen::MatrixXd(), 
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
	Eigen::Vector<double, n_segments*2> derivatives;
	derivatives.setZero();

	const Eigen::Vector3d unitX(1, 0, 0), unitY(0, 1, 0), unitZ(0, 0, 1);

	for (const Eigen::RowVector3i& triangle: F.rowwise()) {
		// compute the area of the face
		Eigen::Vector3d a, b, c;
		a.setZero();
		b.setZero();
		c.setZero();
		a.head<2>() = V.row(triangle[1]) - V.row(triangle[0]);
		b.head<2>() = V.row(triangle[2]) - V.row(triangle[1]);
		c.head<2>() = V.row(triangle[0]) - V.row(triangle[2]);
		area += a.cross(b).dot(unitZ)/2;

		if (triangle[0] < n_segments) {
			derivatives(triangle[0]*2 + 0) += b.cross(unitX).dot(unitZ);
			derivatives(triangle[0]*2 + 1) += b.cross(unitY).dot(unitZ);
		}
		if (triangle[1] < n_segments) {
			derivatives(triangle[1]*2 + 0) += c.cross(unitX).dot(unitZ);
			derivatives(triangle[1]*2 + 1) += c.cross(unitY).dot(unitZ);
		}
		if (triangle[2] < n_segments) {
			derivatives(triangle[2]*2 + 0) += a.cross(unitX).dot(unitZ);
			derivatives(triangle[2]*2 + 1) += a.cross(unitY).dot(unitZ);
		}
	}
	derivatives /= 2;
	return std::make_pair(area, derivatives);
}

std::pair<double, CellDerivativeVector> CellSim::perimeterDerivatives(
		const Eigen::Matrix<double, n_segments, 2>& vertices
		) const {
	// computes the perimeter of a polygon defined by `vertices`
	constexpr double normTol = 1e-12;

	Eigen::Vector<double, n_segments*2> derivatives;
	derivatives.setZero();

	double perimeter = 0;
	for (int i = 0; i < n_segments; i++) {
		const int vertA = i, vertB = (i+1)%n_segments; 
		Eigen::Vector2d edge = vertices.row(vertB) - vertices.row(vertA);
		const double edgeLength = edge.norm();
		perimeter += edgeLength;
		derivatives(vertA*2+0) -= edge(0)/edgeLength;
		derivatives(vertA*2+1) -= edge(1)/edgeLength;

		derivatives(vertB*2+0) += edge(0)/edgeLength;
		derivatives(vertB*2+1) += edge(1)/edgeLength;
	}


	return std::make_pair(perimeter, derivatives);
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
	 * */
	const double min_step = 1e-9, start_step = 1;

	double potential;
	MatrixXd potential_d, potential_d_like_vertices(vertices_state.rows(), vertices_state.cols());
	std::tie(potential, potential_d) = potentialDerivatives(vertices_state);
	const double start_potential = potential;

	potential_d_like_vertices.setZero();
	for (size_t cell_idx = 0; cell_idx < cells.size(); cell_idx++) {
		potential_d_like_vertices.leftCols<2>().middleRows<n_segments>(cell_idx*n_segments) = potential_d.row(cell_idx).reshaped(2, n_segments).transpose();
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
}


void CellSim::step() {
	t ++;
	bool use_linesearch = true;
	if (use_linesearch) {
		lineSearch();
	} else {
		const double gd_step_size = 0.04;
		//jiggle(0.01);
		double potential;
		MatrixXd derivatives;
		std::tie(potential, derivatives) = potentialDerivatives(vertices_state);
		for (size_t cell_idx = 0; cell_idx < cells.size(); cell_idx++) {
			vertices_state.leftCols<2>().middleRows<n_segments>(cell_idx*n_segments) -= 0.04*derivatives.row(cell_idx).reshaped(2, n_segments).transpose();
		}
	}
}

std::pair<double, MatrixXd> CellSim::collisionPotential(const MatrixXd& vertices) const {
	const int n_cells = cells.size();
	const double d_min = 1e-1;
	double potential = 0;
	Eigen::MatrixXd potential_d(n_cells, n_segments*2);

	for (int i = 0; i < n_cells; i++) {
		for (int j = i+1; j < n_cells; j++) {
			//std::cout << "Checking cells " << i << " and " << j << " for collisions." << std::endl;
			// compute bounding box
			if (compareBoundingBoxes(cellVerticesC(vertices, i), cellVerticesC(vertices, j), d_min)) {
				std::cout << "Cells " << i << " and " << j << " have overlapping bounding boxes. Computing derivatives." << std::endl;
				double c_potential;
				CellDerivativeVector c_a_d, c_b_d;
				std::tie(c_potential, c_a_d, c_b_d) = barrierAtoB(d_min, cellVerticesC(vertices, i), cellVerticesC(vertices, j));
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
std::pair<Eigen::VectorXd, Eigen::MatrixXd> CellSim::volumeDerivativesAll(const MatrixXd& vertices) const {
	const size_t n_cells = cells.size();
	Eigen::VectorXd volumes(n_cells);

	Eigen::MatrixXd derivatives(n_cells, n_segments*2);

	// for each cell compute the volume
	for (size_t cell_idx = 0; cell_idx < n_cells; cell_idx ++) {
		auto r = areaDerivatives(cellVertices(vertices, cell_idx).leftCols(2));
		std::cout << "Volume of cell " << cell_idx << ": " << r.first << std::endl;
		volumes(cell_idx) = r.first;
		derivatives.row(cell_idx) = r.second;
	}
	return std::make_pair(volumes, derivatives);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CellSim::perimeterDerivativesAll(const MatrixXd& vertices) const {
	const size_t n_cells = cells.size();
	Eigen::VectorXd perimeters(n_cells);

	Eigen::MatrixXd derivatives(n_cells, n_segments*2);

	// for each cell compute the perimeter
	for (size_t cell_idx = 0; cell_idx < n_cells; cell_idx ++) {
		auto r = perimeterDerivatives(cellVertices(vertices, cell_idx).leftCols(2));
		std::cout << "Perimeter of cell " << cell_idx << ": " << r.first << std::endl;
		perimeters(cell_idx) = r.first;
		derivatives.row(cell_idx) = r.second;
	}
	return std::make_pair(perimeters, derivatives);
}


std::pair<double, MatrixXd> CellSim::volumePotential(const MatrixXd& vertices) const {
	const double idealVolume = 7.5;
	const auto volume_der = volumeDerivativesAll(vertices);
	const VectorXd volumeResiduals = volume_der.first.array() - idealVolume;
	const double volumePotential = volumeResiduals.array().pow(2).sum();
	MatrixXd derivatives = volume_der.second;

	for (int i = 0; i < derivatives.rows(); i++) {
		derivatives.row(i) *= 2 * volumeResiduals(i);
	}
	return std::make_pair(volumePotential, derivatives);
}

std::pair<double, MatrixXd> CellSim::perimeterPotential(const MatrixXd& vertices) const {
	const double idealPerimeter = 10;
	const auto perimeter_der = perimeterDerivativesAll(vertices);
	const VectorXd perimeterResiduals = perimeter_der.first.array() - idealPerimeter;
	const double perimeter_potential = perimeterResiduals.array().pow(2).sum();
	MatrixXd derivatives = perimeter_der.second;

	for (int i = 0; i < derivatives.rows(); i++) {
		derivatives.row(i) *= 2 * perimeterResiduals(i);
	}
	return std::make_pair(perimeter_potential, derivatives);
}
bool CellSim::intervalsOverlap(const Eigen::Vector2d& a_min_max, const Eigen::Vector2d& b_min_max, double epsilon) const {
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
	if (intervalsOverlap(Eigen::Vector2d(a_min(0), a_max(0)), Eigen::Vector2d(b_min(0), b_max(0)), epsilon)) {
		if (intervalsOverlap(Eigen::Vector2d(a_min(1), a_max(1)), Eigen::Vector2d(b_min(1), b_max(1)), epsilon)) {
			return true;
		}
	}
	return false;
}

std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> CellSim::distancesAtoB(
		const double d_min, const CellVertexMatrix3& vertices_a, const CellVertexMatrix3& vertices_b) const {
	// each row corrsponds to a vertex of A
	std::vector<std::tuple<double, int, Vector3d, int, Vector3d, Vector3d>> dist_d(n_segments);
	for(int j = 0; j < n_segments; j++) {
		dist_d[j] = std::make_tuple(
				d_min,
				0, Vector3d(0,0,0), 0, Vector3d(0,0,0), Vector3d(0,0,0)
				);
	}
	for(int j = 0; j < n_segments; j++) {
		for(int i = 0; i < n_segments; i++) {
			// i is the vertex index in A, m
			// j is the edge index in B, k, l
			const Vector3d k = vertices_b.row((i+1)%n_segments);
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
					(i+1)%n_segments, Vector3d(dd_dk1, dd_dk2, 0),
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
	CellDerivativeVector a_d, b_d;
	a_d.setZero();
	b_d.setZero();
	auto results = distancesAtoB(d_min, vertices_a, vertices_b);
	for (int a_idx = 0; a_idx < n_segments; a_idx++) {
		// compute the barrier potential incurred by vertex a_idx
		double d;
		int b1_idx, b2_idx;
		Vector3d a_d_c, b1_d, b2_d;
		std::tie(d, b1_idx, b1_d, b2_idx, b2_d, a_d_c) = results[a_idx];

		if (d>=d_min) continue;
		const double barrier_value = -std::pow(d - d_min, 2) * std::log(d/d_min);
		const double barrier_d = -(2*(d - d_min) * std::log(d/d_min) + std::pow(d - d_min, 2)/d);
		potential += barrier_value;
		a_d.segment<2>(2*a_idx) = a_d_c.head<2>();
		b_d.segment<2>(2*b1_idx) += b1_d.head<2>();
		b_d.segment<2>(2*b2_idx) += b2_d.head<2>();
	}
	return std::make_tuple(potential, a_d, b_d);
}


std::pair<double, MatrixXd> CellSim::potentialDerivatives(const MatrixXd& vertices) const {
	// compute all the potentials
	collisionPotential(vertices);
	double volume_potential;
	MatrixXd volume_potential_d;
	std::tie(volume_potential, volume_potential_d) = volumePotential(vertices);

	double perimeter_potential;
	MatrixXd perimeter_potential_d;
	std::tie(perimeter_potential, perimeter_potential_d) = perimeterPotential(vertices);

	double collision_potential;
	MatrixXd collision_potential_d;
	std::tie(collision_potential, collision_potential_d) = collisionPotential(vertices);
	return std::make_pair(volume_potential+perimeter_potential+collision_potential, volume_potential_d+perimeter_potential_d+collision_potential_d);
}
