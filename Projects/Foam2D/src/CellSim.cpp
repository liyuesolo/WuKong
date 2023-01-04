#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include<Eigen/SparseCholesky>
#include <igl/colormap.h>

#include "../include/CellSim.h"
#include "../include/SerDe.h"
#include "../include/triangulate.h"
#include "../include/cd.h"

using namespace cs2d;


Eigen::Map<const Eigen::MatrixXd> CellSim2D::get_vertices() const {
	Eigen::Map<const Eigen::MatrixXd> vertices(cell_verts(coords_state).data(), 2, cell_verts(coords_state).size()/2);
	return vertices;
}

Eigen::Map<const Eigen::MatrixXd> CellSim2D::get_cell_vertices(int cell_idx) const {
	Eigen::Map<const Eigen::MatrixXd> vertices(cell_verts(coords_state).data()+cell_idx*cell_segments*2, 2, cell_segments);
	return vertices;
}

Eigen::Map<const Eigen::MatrixXd> CellSim2D::get_cell_vertices(const std::vector<double> coords, int cell_idx) const {
	Eigen::Map<const Eigen::MatrixXd> vertices(cell_verts(coords).data()+cell_idx*cell_segments*2, 2, cell_segments);
	return vertices;
}

void CellSim2D::addCell(const Eigen::Matrix<double, 2, Eigen::Dynamic> vertices, double affinity) {
	for(int i = 0; i < vertices.cols(); ++i)
		for (int j = 0; j < 2; ++j)
			coords_state.push_back(vertices(j, i));
	++n_cells;
	cell_verts = SliceD(8, 8+n_cells * cell_segments*2);

	params.push_back(affinity);
	cell_adhesion_affinities = SliceD(0, n_cells);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> CellSim2D::triangulate_all_cells() const {
	Eigen::MatrixXd V(n_cells*(cell_segments+1), 2);
	Eigen::MatrixXi F(n_cells*cell_segments, 3);
	for (int i = 0; i < n_cells; ++i) {
		const auto [c_verts, c_faces] = centroidTriangulate<2>(get_cell_vertices(i).transpose());
		const int v_offset = (cell_segments+1)*i;
		V.middleRows(v_offset, cell_segments+1) = c_verts;
		F.middleRows(cell_segments*i, cell_segments) = c_faces.array() + v_offset;
	}
	return {V, F};
}

void CellSim2D::record_state(Serializer& serializer) const {
	FrameInfo frame_info;
	frame_info.timestep = 0;
	frame_info.static_step = t;
	frame_info.identifier = 0;
	serializer.record_state(frame_info, config);
}

void CellSim2D::write_search_line(const std::string& filename, int n) const {
	std::ofstream o;
	o.open(filename);
	o << "G,D,Perimeter,Volume,Collision,Boundary,Total\n";
	// Recall that coords_tmp is the last effective position
	std::vector<double> coords_intermediate;
	coords_intermediate.resize(coords_state.size());
	for (int i = 0; i < n; ++i) {
		double g = ((double) i)/(n-1);
		double norm = 0;
		for (int j = 0; j < coords_state.size(); ++j) {
			coords_intermediate[j] = (1-g) * coords_tmp[j] + g * coords_state[j];
			norm += std::pow(coords_intermediate[j] - coords_tmp[j], 2);
		}
		norm = std::sqrt(norm);
		VectorXd jacobian = VectorXd::Zero(coords_state.size());
		PotentialValues potentials = compute_potential(coords_intermediate, Jacobian, &jacobian, nullptr);
		/*
		 * FIXME: compute_potentials
		auto [per_potential, p_j] = perimeter_potential( coords_intermediate, Scalar, std::vector<Tripletd>);
		auto [vol_potential, v_j] = area_potential(      coords_intermediate, Scalar, std::vector<Tripletd>);
		auto [col_potential, c_j] = collision_potential( coords_intermediate, Scalar, std::vector<Tripletd>);
		auto [bnd_potential, b_j] = boundary_potential(  coords_intermediate, Scalar, std::vector<Tripletd>);
		const double total = per_potential + vol_potential + col_potential + bnd_potential;
		*/
		o
			<< g << ","
			<< std::setprecision(16)
			<< norm << ","
			<< potentials.perimeter << ","
			<< potentials.volume << ","
			<< potentials.collision << ","
			<< potentials.boundary_collision << ","
			<< potentials.total << std::endl;
	}
	o.close();
}

void CellSim2D::check_jacobians() const {
	double h = 1e-5;
	using namespace std::placeholders;  // for _1, _2, _3...
	VectorXd jac_ad_per = VectorXd::Zero(coords_state.size());
	perimeter_potential(coords_state, Jacobian, &jac_ad_per, nullptr);
	for (auto [name, mem_fn]: {
			std::tuple{"Peri", &CellSim2D::perimeter_potential},
			{"Volm", &CellSim2D::area_potential},
			{"Coll", &CellSim2D::collision_potential},
			{"Bndy", &CellSim2D::boundary_collision_potential},
			}) {

		auto pot_fun = std::bind(mem_fn, this, _1, _2, _3, _4);
		const VectorXd jac_fd = jacobian_fd(pot_fun, h);
		VectorXd jac_ad = VectorXd::Zero(coords_state.size());
		pot_fun(coords_state, Jacobian, &jac_ad, nullptr);

		std::cout << "E " << name << ": " << std::setw(9) << (jac_fd-jac_ad).norm() << "  ";
	}
	std::cout << std::endl;
}

void CellSim2D::check_hessians() const {
	double h = 1e-3;
	using namespace std::placeholders;  // for _1, _2, _3...
	VectorXd jac_ad_per = VectorXd::Zero(coords_state.size());
	perimeter_potential(coords_state, Jacobian, &jac_ad_per, nullptr);
	for (auto [name, mem_fn]: {
			std::tuple{"Peri", &CellSim2D::perimeter_potential},
			{"Volm", &CellSim2D::area_potential},
			{"Coll", &CellSim2D::collision_potential},
			{"Bndy", &CellSim2D::boundary_collision_potential},
			}) {

		auto pot_fun = std::bind(mem_fn, this, _1, _2, _3, _4);
		const VectorXd jac_fd = jacobian_fd(pot_fun, h);
		VectorXd jac_ad = VectorXd::Zero(coords_state.size());
		pot_fun(coords_state, Jacobian, &jac_ad, nullptr);

		std::cout << "E " << name << ": " << std::setw(9) << (jac_fd-jac_ad).norm() << "  ";
	}
	std::cout << std::endl;
}

double CellSim2D::static_step() {
	const int n_dof = coords_state.size();

	VectorXd jacobian = VectorXd::Zero(n_dof);
	TripletVectorD hessian_entries;
	PotentialValues potentials = compute_potential(coords_state, Hessian, &jacobian, &hessian_entries);


	if (config.check_system_matrix) {
		check_jacobians();
	}

	int hessian_reg_steps = -1;

	VectorXd direction = jacobian;

	timer->start_timing("hessian-reg");
	if (config.use_hessian) {
		Eigen::SparseMatrix<double> A(n_dof, n_dof);
		/*
		 * FIXME: compute_potentials
		if (config.check_system_matrix) {
			Eigen::SparseMatrix<double> A_ad(n_dof, n_dof);
			A_ad.setFromTriplets(per_hessian.begin(), per_hessian.end());

			Eigen::MatrixXd A_fd = central_diff_hessian(
					[this](const std::vector<double>& x){
					auto [p, j, h] = perimeter_potential(x, Jacobian);
					return j;
				}, coords, h_fd);
			std::cout << "Perimeter Hessian Error: " << (A_fd - A_ad).norm() <<
				" (AD=" << A_ad.norm() << ", FD=" << A_fd.norm() << ")" << std::endl;
		}

		if (config.check_system_matrix) {
			{
				Eigen::SparseMatrix<double> A_ad(n_dof, n_dof);
				A_ad.setFromTriplets(col_hessian.begin(), col_hessian.end());

				Eigen::MatrixXd A_fd = central_diff_hessian(
						[this](const std::vector<double>& x){
						auto [p, j, h] = collision_potential(x, Jacobian);
						return j;
					}, coords, h_fd);
				std::cout << "Collision Hessian Error: " << (A_fd - A_ad).norm() <<
				" (AD=" << A_ad.norm() << ", FD=" << A_fd.norm() << ")" << std::endl;
			}
			{
				Eigen::SparseMatrix<double> A_ad(n_dof, n_dof);
				A_ad.setFromTriplets(col_hessian.begin(), col_hessian.end());

				Eigen::MatrixXd A_fd = central_diff_hessian(
						[this](const std::vector<double>& x){
						auto [p, j, h] = collision_potential(x, Jacobian);
						return j;
					}, coords, h_fd/2);
				std::cout << "Collision Hessian Error 2: " << (A_fd - A_ad).norm() << std::endl;
			}
		}

		if (config.check_system_matrix) {
			Eigen::SparseMatrix<double> A_ad(n_dof, n_dof);
			A_ad.setFromTriplets(vol_hessian.begin(), vol_hessian.end());

			Eigen::MatrixXd A_fd = central_diff_hessian(
					[this](const std::vector<double>& x){
					auto [p, j, h] = area_potential(x, Jacobian);
					return j;
				}, coords, h_fd);
			std::cout << "Area Hessian Error: " << (A_fd - A_ad).norm() <<
				" (AD=" << A_ad.norm() << ", FD=" << A_fd.norm() << ")" << std::endl;
		}
		*/

		A.setFromTriplets(hessian_entries.begin(), hessian_entries.end());

		Eigen::SparseMatrix<double> identity(n_dof, n_dof);
		Eigen::SparseMatrix<double> hessian_reg = A;
		identity.setIdentity();
		Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
		for (hessian_reg_steps = 0; hessian_reg_steps < config.max_steps_hessian_reg; ++hessian_reg_steps) {
			solver.compute(hessian_reg);
			if(solver.info()==Eigen::Success) {
				direction = solver.solve(jacobian);
				break;
			}
			hessian_reg = A + identity * pow(10, hessian_reg_steps-7);
		}
	}
	timer->stop_timing("hessian-reg", true, "");

	double step_size = 1;
	coords_tmp.resize(coords_state.size());
	for (int i = 0; i < n_dof; ++i)
		coords_tmp[i] = coords_state[i] - direction[i]*step_size;
	double collision_time = 1;
	timer->start_timing("cell-collision");
	auto collisions = find_collisions(coords_state, coords_tmp);
	for (int i = 0; i < n_cells; ++i) for (int j = i+1; j < n_cells; ++j) {
			const double ij_collision_time = timeOfCollision(coords_state, coords_tmp, i, j);
			const double ji_collision_time = timeOfCollision(coords_state, coords_tmp, j, i);
			double collision_time_min =  std::min(ij_collision_time, ji_collision_time);

			if (collision_time_min < collision_time)
				collision_time = collision_time_min;
	}
	timer->stop_timing("cell-collision", true, "");

	timer->start_timing("boundary-collision");
	for (int i = 0; i < n_cells; ++i) {
		timer->start_timing("boundary-collision-time");
		const double bnd_collision_time = timeOfBoundaryCollision(coords_state, coords_tmp, i);
		timer->stop_timing("boundary-collision-time", false, "");
		if (bnd_collision_time < collision_time)
			collision_time = bnd_collision_time;
	}
	timer->stop_timing("boundary-collision", true, "");
	if (collision_time < 1) {
		step_size *= collision_time * 0.8;
		for (int i = 0; i < n_dof; ++i)
			coords_tmp[i] = coords_state[i] - direction[i]*step_size;
	}
	// line search
	PotentialValues new_potentials;
	int line_step;
	timer->start_timing("line-search");
	for (line_step = 0; line_step < config.max_steps_line_search; ++line_step) {
		new_potentials = compute_potential(coords_tmp, Scalar, nullptr, nullptr);
		if (new_potentials.total < potentials.total)
			break;
		step_size /= 2;
		for (int i = 0; i < n_dof; ++i)
			coords_tmp[i] = coords_state[i] - direction[i]*step_size;
	}
	timer->stop_timing("line-search", true, "");

	const double residual = jacobian.norm();
	std::cout << "P Coll: " << std::setw(9) << new_potentials.collision << "  ";
	std::cout << "P Bndy: " << std::setw(9) << new_potentials.boundary_collision << "  ";
	std::cout << "P Peri: " << std::setw(9) << new_potentials.perimeter << "  ";
	std::cout << "P Volm: " << std::setw(9) << new_potentials.volume << "  ";
	std::cout << "P Totl: " << std::setw(9) << new_potentials.total << "  ";
	std::cout << "Residual: " << std::setw(9) << residual << std::endl;

	std::cout << "T Coll: " << std::setw(9) << collision_time << "  ";
	if (hessian_reg_steps < config.max_steps_hessian_reg)
		std::cout << "#HeReg: " << std::setw(9) <<  hessian_reg_steps << "  ";
	else
		std::cout << "#HeReg: " << std::setw(9) <<  "FAIL" << "  ";
	if (line_step < config.max_steps_line_search)
		std::cout << "#LinSc: " << std::setw(9) <<  line_step << "  " << std::endl;
	else
		std::cout << "#LinSc: " << std::setw(9) <<  "FAIL" << "  " << std::endl;
	std::swap(coords_state, coords_tmp);
	Eigen::Map<Eigen::MatrixXd> verts(coords_state.data(), 2, n_dof/2);
	++t;
	return residual;
}

vector<pair<int, int>> CellSim2D::find_collisions(const vector<double>& coords0, const vector<double>& coords1) const {
	// construct the bb vectors
	vector<pair<double, double>> bbs_x, bbs_y;
	for (int i = 0; i < n_cells; ++i) {
		// move bb to the right by d0
		const int cell_offset = 2 * i * cell_segments;
		double x_min = cell_verts(coords0).data()[cell_offset], x_max = cell_verts(coords0).data()[cell_offset];
		double y_min = cell_verts(coords0).data()[cell_offset+1], y_max = cell_verts(coords0).data()[cell_offset+1];
		for (int i = 0; i < cell_segments; ++i) {
			double x = cell_verts(coords0).data()[cell_offset + 2*i];
			double y = cell_verts(coords0).data()[cell_offset + 2*i+1];
			x_min = std::min(x_min, x);
			x_max = std::max(x_max, x);
			y_min = std::min(y_min, y);
			y_max = std::max(y_max, y);
		}
		for (int i = 0; i < cell_segments; ++i) {
			double x = cell_verts(coords1).data()[cell_offset + 2*i];
			double y = cell_verts(coords1).data()[cell_offset + 2*i+1];
			x_min = std::min(x_min, x);
			x_max = std::max(x_max, x);
			y_min = std::min(y_min, y);
			y_max = std::max(y_max, y);
		}
		bbs_x.emplace_back(x_min, x_max);
		bbs_y.emplace_back(y_min, y_max);
	}
	return bb_collisions(bbs_x, bbs_y);
}

vector<pair<int, int>> CellSim2D::find_collisions(const vector<double>& coords, double d0) const {
	// construct the bb vectors
	vector<pair<double, double>> bbs_x, bbs_y;
	for (int i = 0; i < n_cells; ++i) {
		// move bb to the right by d0
		const int cell_offset = 2 * i * cell_segments;
		double x_min = coords[cell_offset], x_max = coords[cell_offset];
		double y_min = coords[cell_offset+1], y_max = coords[cell_offset+1];
		for (int i = 0; i < cell_segments; ++i) {
			double x = coords[cell_offset + 2*i];
			double y = coords[cell_offset + 2*i+1];
			x_min = std::min(x_min, x);
			x_max = std::max(x_max, x);
			y_min = std::min(y_min, y);
			y_max = std::max(y_max, y);
		}
		bbs_x.emplace_back(x_min, x_max+d0);
		bbs_y.emplace_back(y_min, y_max+d0);
	}
	return bb_collisions(bbs_x, bbs_y);
}

/* =================  OLD CODE ================== */

double CellSim2D::timeOfCollision(const vector<double> verts_old, const vector<double> verts_new, int cell_i, int cell_j) const {
	const MatrixXd v_i_old = get_cell_vertices(verts_old, cell_i).transpose();
	const MatrixXd v_j_old = get_cell_vertices(verts_old, cell_j).transpose();

	const MatrixXd v_i_new = get_cell_vertices(verts_new, cell_i).transpose();
	const MatrixXd v_j_new = get_cell_vertices(verts_new, cell_j).transpose();

	double t_coll_min = 1;
	for (int i = 0; i < cell_segments; ++i) {
		const int i_next = (i+1)%cell_segments;
		const Vector2d a0 = v_i_old.row(i), a1 = v_i_new.row(i);
		const Vector2d b0 = v_i_old.row(i_next), b1 = v_i_new.row(i_next);
		const double t_coll = collisionTime(a0, b0, a1, b1, v_j_old, v_j_new);
		const double t_coll0 = collisionTimeConservative(a0, b0, a1, b1, v_j_old, v_j_new, 0);
		const double t_coll_cons = collisionTimeConservative(a0, b0, a1, b1, v_j_old, v_j_new, config.d0/8);
		const double t_coll_cons2 = collisionTimeConservative2(a0, b0, a1, b1, v_j_old, v_j_new, config.d0/8);
		if (t_coll_cons2 > t_coll)
			std::cout << "OOPS!" << std::endl;
		if (t_coll != 1 || t_coll0 != 1 || t_coll_cons != 1 || t_coll_cons2 != 1)
			std::cout << "TColl=" << t_coll << ", TColl0=" << t_coll0 << ", TCollC=" << t_coll_cons << ", TCollC2=" << t_coll_cons2 << std::endl;
		if(t_coll < t_coll_min) {
			t_coll_min = t_coll;
		}
	}
	return t_coll_min;
}

double CellSim2D::timeOfBoundaryCollision(const vector<double>& verts_old, const vector<double>&  verts_new, int cell_i) const {
	const MatrixXd v_i_old = get_boundary(verts_old).transpose();
	const MatrixXd v_j_old = get_cell_vertices(verts_old, cell_i).transpose();

	const MatrixXd v_i_new = get_boundary(verts_new).transpose();
	const MatrixXd v_j_new = get_cell_vertices(verts_new, cell_i).transpose();

	double t_coll_min = 1;
	for (int i = 0; i < n_boundary; ++i) {
		const int i_next = (i+1)%n_boundary;
		const Vector2d a0 = v_i_old.row(i), a1 = v_i_new.row(i);
		const Vector2d b0 = v_i_old.row(i_next), b1 = v_i_new.row(i_next);
		const double t_coll = collisionTime(a0, b0, a1, b1, v_j_old, v_j_new);
		const double t_coll0 = collisionTimeConservative(a0, b0, a1, b1, v_j_old, v_j_new, 0);
		const double t_coll_cons = collisionTimeConservative(a0, b0, a1, b1, v_j_old, v_j_new, config.d0/8);
		const double t_coll_cons2 = collisionTimeConservative2(a0, b0, a1, b1, v_j_old, v_j_new, config.d0/8);
		if (t_coll != 1 || t_coll0 != 1 || t_coll_cons != 1 || t_coll_cons2 != 1)
			std::cout << "TColl=" << t_coll << ", TColl0=" << t_coll0 << ", TCollC=" << t_coll_cons << ", TCollC2=" << t_coll_cons2 << std::endl;
		if(t_coll < t_coll_min) {
			t_coll_min = t_coll;
		}
	}
	return t_coll_min;
}
