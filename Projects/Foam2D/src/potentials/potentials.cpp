#include <cassert>
#include <iostream>
#include <array>

#include "../../include/CellSim.h"
#include "../../include/potentials.h"

using namespace cs2d;

inline void check_pointers(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) {
	const long ndof = coords.size();
	if (mode >= Jacobian) {
		assert(jacobian != nullptr);
		assert(jacobian->size() == ndof);
		assert((hessian != nullptr) == (mode == Hessian));
	} else {
		assert (jacobian == nullptr);
	}
}

/* ================  Collision  ====================== */
void CellSim2D::add_boundary_collision_potential(const std::vector<double>& coords, int cell_v, ComputeOrder mode, double& potential, VectorXd* jacobian, std::vector<Tripletd>* hessian) const {
	const double * c_ptr = cell_verts(coords).data();
	const int cells_offset = cell_verts.idx_begin;
	const int boundary_offset = boundary_verts.idx_begin;
	const int cell_offset_v = 2 * cell_v * cell_segments;
	const double * cell_ptr_v = c_ptr + cell_offset_v;
	const double * boundary_ptr = boundary_verts(coords).data();

	for (int i = 0; i < n_boundary; ++i) {
		for (int j = 0; j < cell_segments; ++j) {
			// cell-level dofs
			const int a = 2*i, b = 2*((i+1)%n_boundary);
			const int p = 2*j;

			const std::array<int, 6> dofs {
				cell_offset_v + p, cell_offset_v + p + 1,
				a, a + 1,
				b, b + 1,
			};

			double ve_potential;
			const pot::collision::VertexEdgeMode vem = pot::collision::vertex_edge_potential<0>(cell_ptr_v + p, boundary_ptr+a, boundary_ptr+b, config.d0, &ve_potential);
			if (vem == pot::collision::Null) continue;
			ve_potential *= config.weight_boundary_collision;
			potential += ve_potential;

			if (mode == Scalar)
				continue;

			std::array<double, 6> ve_jacobian;
			pot::collision::vertex_edge_potential<1>(cell_ptr_v + p, boundary_ptr+a, boundary_ptr+b, config.d0, ve_jacobian.data());
			for (int k = 0; k < 2; ++k)
				(*jacobian)[cells_offset+dofs[k]] += config.weight_boundary_collision * ve_jacobian[k];
			for (int k = 2; k < 6; ++k)
				(*jacobian)[boundary_offset+dofs[k]] += config.weight_boundary_collision * ve_jacobian[k];
			if (mode == Jacobian)
				continue;

			std::array<double, 36> ve_hessian;
			pot::collision::vertex_edge_potential<2>(cell_ptr_v + p, boundary_ptr+a, boundary_ptr+b, config.d0, ve_hessian.data());

			for (int k = 0; k < 6; ++k) {
				for (int l = 0; l < 6; ++l)
					hessian->push_back(Tripletd(
								(k<2 ? cells_offset : boundary_offset) + dofs[k],
								(l<2 ? cells_offset : boundary_offset) + dofs[l],
								config.weight_boundary_collision * ve_hessian[l*6+k]));
			}
		}
	}
}

double CellSim2D::boundary_collision_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	// compute potential, derivative and hessian for the perimeter potential
	// iterate over cells, since this is a cell-level potential

	double potential = 0;
	check_pointers(coords, mode, jacobian, hessian);

	for (int i = 0; i < n_cells; ++i) {
			// add_collision_potential(coords, i, i, mode, potential, jacobian, hessian_entries);
		add_boundary_collision_potential(coords, i, mode, potential, jacobian, hessian);
	}

	return potential;
}

void CellSim2D::add_collision_potential(const std::vector<double>& coords, int cell_offset_e, int cell_offset_v, ComputeOrder mode, double& potential, VectorXd* jacobian, TripletVectorD* hessian) const {
	const double * c_ptr = cell_verts(coords).data();
	const double * cell_ptr_e = c_ptr + cell_offset_e, * cell_ptr_v = c_ptr + cell_offset_v;
	const int cells_offset = cell_verts.idx_begin;

	for (int i = 0; i < cell_segments; ++i) {
		for (int j = 0; j < cell_segments; ++j) {
			// cell-level dofs
			const int a = 2*i, b = 2*((i+1)%cell_segments);
			const int p = 2*j;

			const std::array<int, 6> dofs {
				cell_offset_v + p, cell_offset_v + p + 1,
				cell_offset_e + a, cell_offset_e + a + 1,
				cell_offset_e + b, cell_offset_e + b + 1
			};

			double ve_potential;
			const pot::collision::VertexEdgeMode vem = pot::collision::vertex_edge_potential<0>(cell_ptr_v + p, cell_ptr_e+a, cell_ptr_e+b, config.d0, &ve_potential);
			if (vem == pot::collision::Null) continue;
			ve_potential *= config.weight_collision;
			potential += ve_potential;

			if (mode == Scalar)
				continue;

			std::array<double, 6> ve_jacobian;
			pot::collision::vertex_edge_potential<1>(cell_ptr_v + p, cell_ptr_e+a, cell_ptr_e+b, config.d0, ve_jacobian.data());
			for (int k = 0; k < 6; ++k)
				(*jacobian)[cells_offset+dofs[k]] += config.weight_collision * ve_jacobian[k];

			if (mode == Jacobian)
				continue;

			std::array<double, 36> ve_hessian;
			pot::collision::vertex_edge_potential<2>(cell_ptr_v + p, cell_ptr_e+a, cell_ptr_e+b, config.d0, ve_hessian.data());
			for (int k = 0; k < 6; ++k) {
				for (int l = 0; l < 6; ++l) {
					hessian->push_back(Tripletd(dofs[k]+cells_offset, dofs[l]+cells_offset, config.weight_collision * ve_hessian[l*6+k]));
				}
			}
		}
	}
}

double CellSim2D::collision_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	// compute potential, derivative and hessian for the perimeter potential
	// iterate over cells, since this is a cell-level potential

	double potential = 0;
	check_pointers(coords, mode, jacobian, hessian);

	auto collisions = find_collisions(coords, config.d0);
	for (auto [cell_e, cell_v]: collisions) {
			// add_collision_potential(coords, i, i, mode, potential, jacobian, hessian_entries);
		const int cell_offset_e = 2 * cell_e * cell_segments;
		const int cell_offset_v = 2 * cell_v * cell_segments;
		add_collision_potential(coords, cell_offset_e, cell_offset_v, mode, potential, jacobian, hessian);
		add_collision_potential(coords, cell_offset_v, cell_offset_e, mode, potential, jacobian, hessian);
	}

	return potential;
}

/* ================  Cell Area / Perimeter  ============ */
double CellSim2D::perimeter_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	// compute potential, derivative and hessian for the perimeter potential
	// iterate over cells, since this is a cell-level potential
	double potential = 0;
	check_pointers(coords, mode, jacobian, hessian);

	const double * c_ptr = cell_verts(coords).data();
	const int cells_offset = cell_verts.idx_begin;
	for (int cell = 0; cell < n_cells; ++cell) {
		const int cell_offset = 2 * cell * cell_segments;
		const double * cell_ptr = c_ptr + cell_offset;
		for (int i = 0; i < cell_segments; ++i) {
			// for now use a per-edge potential
			// cell-level dofs
			const int a = 2*i, b = 2*((i+1)%cell_segments);
			double edge_length;
			pot::line_length<0>(cell_ptr+a, cell_ptr+b, &edge_length);
			potential += config.weight_perimeter * pen::square<0>(edge_length, config.perimeter_goal);

			if (mode == Scalar)
				continue;
			const double penalty_deriv = config.weight_perimeter * pen::square<1>(edge_length, config.perimeter_goal);
			// jacobian
			std::array<double, 4> edge_jacobian;
			pot::line_length<1>(cell_ptr+a, cell_ptr+b, edge_jacobian.data());

			const std::array<int, 4> dofs {
				cell_offset+a,
				cell_offset+a+1,
				cell_offset+b,
				cell_offset+b+1
			};
			for (int k = 0; k < 4; ++k)
				(*jacobian)[cells_offset+dofs[k]] += penalty_deriv * edge_jacobian[k];

			if (mode == Jacobian)
				continue;

			const double penalty_hessian = config.weight_perimeter * pen::square<2>(edge_length, config.perimeter_goal);
			std::array<double, 16> edge_hessian;
			pot::line_length<2>(cell_ptr+a, cell_ptr+b, edge_hessian.data());
			for (int k = 0; k < 4; ++k)
				for (int l = 0; l < 4; ++l) {
					edge_hessian[k*4+l] *= penalty_deriv;
					edge_hessian[k*4+l] += edge_jacobian[l]*edge_jacobian[k] * penalty_hessian;
				}
			for (int k = 0; k < 4; ++k)
				for (int l = 0; l < 4; ++l)
					hessian->push_back(Tripletd(dofs[k]+cells_offset, dofs[l]+cells_offset, edge_hessian[l*4+k]));
		}
	}
	return potential;
}

double CellSim2D::area_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	// compute potential, derivative and hessian for the perimeter potential
	// iterate over cells, since this is a cell-level potential
	double potential = 0;
	check_pointers(coords, mode, jacobian, hessian);

	const double * c_ptr = cell_verts(coords).data();
	const int cells_offset = cell_verts.idx_begin;
	for (int cell = 0; cell < n_cells; ++cell) {
		const int cell_offset = 2 * cell * cell_segments;
		const double * cell_ptr = c_ptr + cell_offset;
		std::array<double,2> centroid {0, 0};
		for (int i = 0; i < cell_segments; ++i) {
			centroid[0] += c_ptr[cell_offset + 2*i]/cell_segments;
			centroid[1] += c_ptr[cell_offset + 2*i+1]/cell_segments;
		}

		double cell_area = 0;

		for (int i = 0; i < cell_segments; ++i) {
			// cell-level dofs
			const int a = 2*i, b = 2*((i+1)%cell_segments);
			double tri_area;
			pot::triangle_area_signed<0>(cell_ptr+a, cell_ptr+b, centroid.data(), &tri_area);
			if (tri_area < 0)
				std::cout << "NEGATIVE AREA! A=" << tri_area;
			cell_area += tri_area;
		}
		const double cell_potential = config.weight_volume * pen::square<0>(cell_area, config.volume_goal);
		potential += cell_potential;

		if (mode == Scalar)
			continue;

		const double cell_penalty_deriv = config.weight_volume * pen::square<1>(cell_area, config.volume_goal);

		Eigen::VectorXd cell_area_jacobian = VectorXd::Zero(2*cell_segments);
		Eigen::MatrixXd cell_hessian(2*cell_segments, 2*cell_segments);
		cell_hessian.setZero();
		for (int i = 0; i < cell_segments; ++i) {
			const int a = 2*i, b = 2*((i+1)%cell_segments);
			// jacobian
			std::array<double, 6> tri_jacobian;
			pot::triangle_area_signed<1>(cell_ptr+a, cell_ptr+b, centroid.data(), tri_jacobian.data());
			const std::array<int, 4> dofs {
				cell_offset + a  ,
				cell_offset + a+1,
				cell_offset + b  ,
				cell_offset + b+1
			};

			for (int k = 0; k < 4; ++k) {
				(*jacobian)[cells_offset+dofs[k]] += cell_penalty_deriv * tri_jacobian[k];
				cell_area_jacobian[dofs[k]-cell_offset] += tri_jacobian[k];
			}

			if (mode <= Jacobian)
				continue;

			std::array<double, 36> tri_hessian;
			pot::triangle_area_signed<2>(cell_ptr+a, cell_ptr+b, centroid.data(), tri_hessian.data());
			for (int k = 0; k < 4; ++k) {
				for (int l = 0; l < 4; ++l) {
					cell_hessian(dofs[k]-cell_offset, dofs[l]-cell_offset) += tri_hessian[l*6+k];
				}
			}
		}

		if (mode <= Jacobian)
			continue;

		const double cell_penalty_2 = config.weight_volume * pen::square<2>(cell_area, config.volume_goal);
		cell_hessian *= cell_penalty_deriv;
		cell_hessian += cell_area_jacobian * cell_area_jacobian.transpose() * cell_penalty_2;

		// distribute cell hessian
		for (int k = 0; k < cell_segments*2; ++k) {
			for (int l = 0; l < cell_segments*2; ++l) {
				hessian->push_back(Tripletd(cell_offset+cells_offset + k, cell_offset+cells_offset + l, cell_hessian(k, l)));
			}
		}
	}

	return potential;
}

/* ================  Boundary Shape ==================== */
double CellSim2D::boundary_shape_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	check_pointers(coords, mode, jacobian, hessian);

	const double* boundary_ptr = boundary_verts(coords).data();

	double potential = 0;
	for (int i = 0; i < 4; ++i) {
		const int a = 2*i;
		const int s1 = i % 3 < 1? -1 : 1, s2 = i / 2 ? 1 : -1;
		const std::array<double, 2> expected_position = {s1 * config.boundary_width/2, s2 * config.boundary_height/2};

		double offset;
		pot::line_length<0>(boundary_ptr+a, expected_position.data(), &offset);
		//const double length_goal = i%2 ? config.boundary_height : config.boundary_width;
		const int boundary_offset = boundary_verts.idx_begin;
		potential += config.weight_boundary_shape * pen::square<0>(offset, 0);

		if (mode == Scalar || offset == 0)
			continue;
		const double penalty_deriv = config.weight_boundary_shape * pen::square<1>(offset, 0);
		// jacobian
		std::array<double, 4> edge_jacobian;
		pot::line_length<1>(boundary_ptr+a, expected_position.data(), edge_jacobian.data());

		const std::array<int, 2> dofs { a, a+1};
		for (int k = 0; k < 2; ++k)
			(*jacobian)[boundary_offset+dofs[k]] += penalty_deriv * edge_jacobian[k];

		if (mode == Jacobian)
			continue;

		const double penalty_hessian = config.weight_boundary_shape * pen::square<2>(offset, 0);
		std::array<double, 16> edge_hessian;
		pot::line_length<2>(boundary_ptr+a, expected_position.data(), edge_hessian.data());
		for (int k = 0; k < 2; ++k)
			for (int l = 0; l < 2; ++l) {
				edge_hessian[k*4+l] *= penalty_deriv;
				edge_hessian[k*4+l] += edge_jacobian[l]*edge_jacobian[k] * penalty_hessian;
			}
		for (int k = 0; k < 2; ++k)
			for (int l = 0; l < 2; ++l)
				hessian->push_back(Tripletd(boundary_offset+dofs[k], boundary_offset+dofs[l], edge_hessian[l*4+k]));
	}
	return potential;
}

/* ================  Adhesion  ====================== */
void CellSim2D::add_adhesion_potential(const std::vector<double>& coords, int cell_offset_i, int cell_offset_j, double affinity_i, double affinity_j, ComputeOrder mode, double& potential, VectorXd* jacobian, TripletVectorD* hessian) const {
	const double * c_ptr = cell_verts(coords).data();
	const double * cell_ptr_i = c_ptr + cell_offset_i, * cell_ptr_j = c_ptr + cell_offset_j;
	const int cells_offset = cell_verts.idx_begin;

	const double affinity_factor = 1.0/(1.0+64*pow(affinity_i-affinity_j, 2))-0.5;
	for (int i = 0; i < cell_segments; ++i) {
		for (int j = 0; j < cell_segments; ++j) {
			// cell-level dofs
			const int a = 2*i, b = 2*((i+1)%cell_segments);
			const int d = 2*j, c = 2*((j+1)%cell_segments);

			const std::array<int, 8> dofs {
				cell_offset_i + a, cell_offset_i + a + 1,
				cell_offset_i + b, cell_offset_i + b + 1,
				cell_offset_j + c, cell_offset_j + c + 1,
				cell_offset_j + d, cell_offset_j + d + 1
			};

			double adh_potential;
			bool adhesion_valid = pot::adhesion::active(
					cell_ptr_i + a, cell_ptr_i + b,
					cell_ptr_j + c, cell_ptr_j + d,
					config.d0_adhesion);


			if (!adhesion_valid) continue;
			pot::adhesion::potential<0>(
					cell_ptr_i + a, cell_ptr_i + b,
					cell_ptr_j + c, cell_ptr_j + d,
					config.d0_adhesion, config.adhesion_logi_k, &adh_potential);
			potential += adh_potential * config.weight_adhesion * affinity_factor;

			if (mode == Scalar)
				continue;

			std::array<double, 8> ve_jacobian;
			pot::adhesion::potential<1>(
					cell_ptr_i + a, cell_ptr_i + b,
					cell_ptr_j + c, cell_ptr_j + d,
					config.d0_adhesion, config.adhesion_logi_k, ve_jacobian.data());
			for (int k = 0; k < 8; ++k)
				(*jacobian)[cells_offset+dofs[k]] += config.weight_adhesion * affinity_factor * ve_jacobian[k];

			if (mode == Jacobian)
				continue;

			std::array<double, 64> ve_hessian;
			pot::adhesion::potential<2>(
					cell_ptr_i + a, cell_ptr_i + b,
					cell_ptr_j + c, cell_ptr_j + d,
					config.d0_adhesion, config.adhesion_logi_k, ve_hessian.data());

			for (int k = 0; k < 8; ++k) {
				for (int l = 0; l < 8; ++l) {
					hessian->push_back(Tripletd(dofs[k]+cells_offset, dofs[l]+cells_offset, config.weight_adhesion * affinity_factor * ve_hessian[l*8+k]));
				}
			}
		}
	}
}

double CellSim2D::adhesion_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	// compute potential, derivative and hessian for the perimeter potential
	// iterate over cells, since this is a cell-level potential

	double potential = 0;
	check_pointers(coords, mode, jacobian, hessian);

	auto collisions = find_collisions(coords, config.d0_adhesion);
	for (auto [i, j]: collisions) {
			// add_adhesion_potential(coords, i, i, mode, potential, jacobian, hessian_entries);
		const int cell_offset_i = 2 * i * cell_segments;
		const int cell_offset_j = 2 * j * cell_segments;
		add_adhesion_potential(coords, cell_offset_i, cell_offset_j, params[i], params[j], mode, potential, jacobian, hessian);
		add_adhesion_potential(coords, cell_offset_j, cell_offset_i, params[j], params[i], mode, potential, jacobian, hessian);
	}

	return potential;
}

/* ================  Total Potential  ====================== */
CellSim2D::PotentialValues CellSim2D::compute_potential(const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const {
	PotentialValues pot;
	pot.perimeter 	= perimeter_potential(coords, mode, jacobian, hessian);
	pot.volume 		= area_potential(coords, mode, jacobian, hessian);
	timer->start_timing("collision-potential");
	pot.collision 	= collision_potential(coords, mode, jacobian, hessian);
	timer->stop_timing("collision-potential", false, "");
	pot.boundary_shape 	= boundary_shape_potential(coords, mode, jacobian, hessian);
	pot.boundary_collision 	= boundary_collision_potential(coords, mode, jacobian, hessian);
	pot.adhesion 	= adhesion_potential(coords, mode, jacobian, hessian);
	pot.total = pot.perimeter + pot.volume + pot.collision + pot.boundary_collision + pot.adhesion;
	return pot;
}
