#ifndef CS2D_SIM_H
#define CS2D_SIM_H

/*
 * CellSim2D Simulation code
 * ISO C++ 17 
 * Approach:
 * - C++ in the backend (where possible), Eigen wrappers in the frontend
 * - Index-based logic
 * - Avoid dynamic memory allocations in low-level code
 * Dependencies:
 * - Eigen 3
 * - nlohmann JSON
 */

#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "config.h"
#include "finite_difference.h"
#include "tsc_x86.h"
#include "vectorview.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Tripletd = Eigen::Triplet<double>;


/*
 * Vertex i component j is: i*n_vertices + j
 */

namespace cs2d {
	using std::tuple, std::vector, std::pair;
	using TripletVectorD = vector<Eigen::Triplet<double>>;
	using SliceD = Slice<double>;

	class Potential {
		static double value(const vector<double>& coords);
		static void jacobian(const vector<double>& coords);
		static void hessian(const vector<double>& coords);
	};

	enum ComputeOrder {Scalar = 0, Jacobian = 1, Hessian = 2};
	class Serializer;
	class Deserializer;

	class CellSim2D {
		public:
			struct PotentialValues {
				double perimeter;
				double volume;
				double collision;
				double boundary_collision;
				double boundary_shape;
				double adhesion;
				double total;
			};

			struct Cell {
				Eigen::Map<Eigen::MatrixXd> vertices;
				Eigen::MatrixXi cells;
			};
			Eigen::Map<const Eigen::MatrixXd> get_vertices() const;
			Eigen::Map<const Eigen::MatrixXd> get_cell_vertices(int cell_idx) const;
			Eigen::Map<const Eigen::MatrixXd> get_cell_vertices(const vector<double> coords, int cell_idx) const;

			int cell_segments;
			int n_cells;
			int n_boundary;

			SliceD cell_verts;
			SliceD boundary_verts;

			SliceD cell_adhesion_affinities;

			CellSim2D(int cell_segments):
				cell_segments{cell_segments}, n_cells{0}, n_boundary{4}, cell_verts(8,8), boundary_verts(0, 8), cell_adhesion_affinities(0, 0),
				timer{&(tsc::TSCTimer::get_timer("timings.json"))}, coords_state(8), coords_tmp() {
				cell_verts = SliceD(0, n_cells * cell_segments*2);
				std::array<double, 8> boundary = {-3, -2, 3, -2, 3,  2, -3,  2};
				for (int i = 0; i < 8; ++i)
					boundary_verts(coords_state).data()[i] = boundary[i];
				timer->deactivate();
			}

			CellSim2D(const SimConfig& config, const std::vector<double>& params, const std::vector<double>& coords, int cell_segments, int n_boundary, int n_cells):
				cell_segments{cell_segments}, n_cells{n_cells}, n_boundary{n_boundary},
				cell_verts(0, 0), boundary_verts(0, 0), cell_adhesion_affinities(0, 0),
				config{config},
				timer{&(tsc::TSCTimer::get_timer("timings.json"))},
				coords_state(coords), coords_tmp(), params(params) {
					std::cout << "constructor" << std::endl;

				const int size_verts = n_cells*cell_segments*2, size_boundary = 2*n_boundary;
				cell_adhesion_affinities = SliceD(0, n_cells);
				boundary_verts = SliceD(0, size_boundary);
				cell_verts = SliceD(size_boundary, size_boundary + size_verts);

				timer->deactivate();
			}


			[[deprecated]] int n_vertices() const {
				return cell_verts(coords_state).size()/2;
			}

			void addCell(const Eigen::Matrix<double, 2, Eigen::Dynamic> vertices, double affinity = 0.5);
			// This class is responsible for:
			// * keeping track of cells
			// * computing potentials, derivatives, hessians
			// * providing common representations of cells
			pair<Eigen::MatrixXd, Eigen::MatrixXi> triangulate_all_cells() const;

			PotentialValues compute_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			// individiual potentials
			double area_potential(		const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			double adhesion_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			double perimeter_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			double collision_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			double boundary_collision_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;
			double boundary_shape_potential(	const vector<double>& coords, ComputeOrder mode, VectorXd* jacobian, TripletVectorD* hessian) const;

			void check_jacobians() const;
			void check_hessians() const;

			using PotentialFunction = std::function<double(const vector<double>&, ComputeOrder, VectorXd*, TripletVectorD*)>;
			VectorXd jacobian_fd(PotentialFunction potential_fun, double h) const {
				Eigen::VectorXd fd = central_diff(
						[potential_fun](const vector<double>& x){
						return potential_fun(x, Scalar, nullptr, nullptr);
						}, coords_state, h
						);
				return fd;
			}

			MatrixXd hessian_fd(PotentialFunction potential_fun, double h) const {
				const int n_dof = coords_state.size();
				const Eigen::MatrixXd fd = central_diff_hessian(
						[potential_fun,n_dof](const vector<double>& x){
						Eigen::VectorXd jacobian = VectorXd::Zero(n_dof);
						potential_fun(x, Jacobian, &jacobian, nullptr);
						return jacobian;
						}, coords_state, h
						);
				return fd;
			}

			int t = 0;
			void write_search_line(const std::string& filename, int n = 11) const;
			double timeOfCollision(const vector<double> verts_old, const vector<double> verts_new, int cell_i, int cell_j) const;
			double timeOfBoundaryCollision(const vector<double>& verts_old, const vector<double>&  verts_new, int cell_i) const;

			double static_step();
			inline bool till_convergence(double tol=1e-9, int it_max=1000) {
				residuals.clear();
				for (int i = 0; i < it_max; ++i) {
					double residual = static_step();
					residuals.push_back(residual);
					if (residual < tol) {
						return true;
					}
				}
				return false;
			}
			Eigen::Map<const Eigen::MatrixXd> get_boundary() const {
				return Eigen::Map<const Eigen::MatrixXd>(boundary_verts(coords_state).data(), 2, boundary_verts(coords_state).size()/2);
			}
			Eigen::Map<const Eigen::MatrixXd> get_boundary(const vector<double>& coords) const {
				return Eigen::Map<const Eigen::MatrixXd>(boundary_verts(coords).data(), 2, boundary_verts(coords).size()/2);
			}

			//void store_state(const std::string& path, int timestep, int static_step) const;
			void record_state(Serializer& serializer) const;
			SimConfig config;
			tsc::TSCTimer* timer = nullptr;
		private:
			// verts is a 2xN_CELLS*N_VERTS_PER_CELL matrix in column major format

			// given bounding boxes(min_x, max_x), (min_y, max_y), find collisions
			vector<pair<int, int>> find_collisions(const vector<double>& coords, double d0) const;
			vector<pair<int, int>> find_collisions(const vector<double>& coords0, const vector<double>& coords1) const;
			void add_adhesion_potential(
					const vector<double>& coords, int cell_offset_e, int cell_offset_v, double affinity_i, double affinity_j,
					ComputeOrder mode, double& potential, VectorXd* jacobian, TripletVectorD* hessian) const;
			void add_collision_potential(
					const vector<double>& coords, int cell_offset_e, int cell_offset_v, ComputeOrder mode,
					double& potential, VectorXd* jacobian, TripletVectorD* hessian) const;
			void add_boundary_collision_potential(
					const vector<double>& coords, int cell_e, ComputeOrder mode,
					double& potential, VectorXd* jacobian, TripletVectorD* hessian) const;
			friend class Serializer;
			friend class Deserializer;
			// coords are degrees of freedom of the simulation
			vector<double> coords_state;
			mutable vector<double> coords_tmp;
			// params are all values that may change during a simulation, that are not coords
		public:
			vector<double> params;
			vector<double> residuals;
	};

	void initialize_cells(CellSim2D& cs, int cols, int rows, double width, double height, int seed = 127);
	vector<pair<int, int>> bb_collisions(const vector<pair<double, double>>& bbs_x, const vector<pair<double, double>>& bbs_y);

}

#endif
