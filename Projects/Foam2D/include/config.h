#ifndef CS2D_CFG_H
#define CS2D_CFG_H
#include <string>
#include<nlohmann/json.hpp>


namespace cs2d {
	struct ExperimentConfig {
		// Checking
		bool check_system_matrix = false;
	};

	struct SimConfig {
		// Checking
		bool check_system_matrix = false;

		// Solver
		bool use_hessian = true;
		int max_steps_line_search = 4;
		int max_steps_hessian_reg = 20;

		// Potential parameters
		double d0 = 1e-2;
		double d0_adhesion = 4e-1;
		double d0_obst = 0;
		double volume_goal = 3;
		double perimeter_goal = 0;
		double adhesion_logi_k = 3;

		// Weights
		double weight_volume = 1;
		double weight_perimeter = 3;
		double weight_collision = 1e4;
		double weight_boundary_collision = 1e4;
		double weight_boundary_shape = 1e4;
		double weight_adhesion = 2e2;

		// Boundary
		double boundary_width = 6;
		double boundary_height = 4;
	};
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SimConfig,
			// Checking
			check_system_matrix,
			// Solver
			use_hessian,
			use_hessian,
			max_steps_line_search,
			max_steps_hessian_reg,
			// Potential parameters
			d0,
			volume_goal,
			perimeter_goal,
			// Weights
			weight_volume,
			weight_perimeter,
			weight_collision,
			weight_boundary_collision
	)

	struct VizConfig {
		// console printing
		bool print_total_potential = true;
		bool print_separate_potential = true;
		bool print_hessian = false;

		bool verbose_console = false;
		bool show_mesh = true;
		bool show_vertices = false;
		bool show_edges = true;

		bool write_screencaps = false;
		std::string screencaps_location = "/tmp/png/";
		std::string res_path = "/tmp/source_file.dat";
	};
	NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VizConfig,
		// console printing
		print_total_potential,
		print_separate_potential,
		print_hessian,
		verbose_console,
		show_mesh,
		show_vertices,
		show_edges,
		write_screencaps,
		screencaps_location
	)
}


#endif
