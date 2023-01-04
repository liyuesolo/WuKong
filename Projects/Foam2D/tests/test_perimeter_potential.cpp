#include <cmath>
#include <cstdlib>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "test_utils.h"
#include "../include/finite_difference.h"
#include "../include/potentials.h"
#include "../include/CellSim.h"

#define TOL 1e-5

using namespace cs2d;
using namespace cs2d::pot;
using Eigen::Vector2d, Eigen::Vector4d, Eigen::VectorXd, Eigen::Vector, Eigen::Matrix2d;

TEST(PerimeterPotentialTest, TriplePotential) {
	CellSim2D cs(3);
	std::vector<double> coords{0, 0, 3, 0, 0, 4};
	std::cout << "Cell count " << cs.n_cells();
	cs.config.perimeter_goal = 4;
	double potential = std::get<0>(cs.compute_perimeter_potential(coords, Scalar));
	EXPECT_FLOAT_EQ(potential, 2);
}

TEST(PerimeterPotentialTest, Jacobian) {
	using CellCoordVec = Eigen::Vector<double, 6>;
	using CellCoordMat = Eigen::Matrix<double, 2, 3>;

	CellSim2D cs(3);
	cs.config.perimeter_goal = 4;

	std::vector<double> x_std_vec(6);
	Eigen::Map<CellCoordMat> x_mat(x_std_vec.data());
	x_mat << 0, 0, 3, 0, 0, 4;
	const double h = 1e-3;
	auto pot_fun = [&cs](CellCoordVec v){
		std::vector<double> coords(6);
		Eigen::Map<CellCoordVec> x_vec(coords.data());
		x_vec = v;
		double potential = std::get<0>(cs.compute_perimeter_potential(coords, Scalar));
		return potential;
	};
	CellCoordVec jac_ad = std::get<1>(cs.compute_perimeter_potential(x_std_vec, Jacobian));
	const CellCoordVec jac_fd = central_diff<6>(pot_fun, x_mat.reshaped(), h);

	EXPECT_NEAR((jac_fd - jac_ad).norm(), 0, TOL) <<
		"Jacobian error (norm=" << (jac_fd - jac_ad).norm() << ")for x=\n" << jac_fd.transpose() << " !=\n" << jac_ad.transpose() << std::endl;
}


