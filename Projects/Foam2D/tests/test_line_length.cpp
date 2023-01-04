#include <cmath>
#include <cstdlib>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "test_utils.h"
#include "../include/finite_difference.h"
#include "../include/potentials.h"

#define TOL 1e-5

using namespace cs2d::pot;
using Eigen::Vector2d, Eigen::Vector4d, Eigen::VectorXd, Eigen::Vector, Eigen::Matrix2d;

TEST(LineLengthTest, BasicLength) {
	Matrix2d x;
	x << 0, 0, 1, 0;
	double len;
	line_length<0>(x.data(), x.data()+2, &len);
	EXPECT_NEAR(len, 1.0, TOL);
}

TEST(LineLengthTest, RotationInvariant) {
	Matrix2d x;
	x << 0, 0, 1, 0;
	double len;
	for (Matrix2d r: rotations(10)) {
		Matrix2d x_rot = r * x;
		line_length<0>(x_rot.data(), x_rot.data()+2, &len);
		EXPECT_NEAR(len, 1.0, TOL) << "Rotation invariance test failed for rotation " << r;
	}
}

TEST(LineLengthTest, TranslationInvariant) {
	Matrix2d x;
	x << 0, 0, 1, 0;
	double len;
	for (Vector2d t: translations(10)) {
		Matrix2d x_trans = x.colwise()+t;
		line_length<0>(x_trans.data(), x_trans.data()+2, &len);
		EXPECT_NEAR(len, 1.0, TOL) << "Translation invariance test failed for translation " << t;
	}
}

TEST(LineLengthTest, Jacobian) {
	Matrix2d x;
	x << 0, 0, 1, 0;
	const double h = 1e-3;
	auto pot_fun = [](Vector4d v){
		double len;
		line_length<0>(v.data(), v.data()+2, &len);
		return len;
	};
	for (auto rot: rotations(5)) {
		Matrix2d x_rot = rot * x;
		for (auto trans: translations(6)) {
			Matrix2d x_rot_trans = x_rot.colwise()+trans; 
			const Vector4d jac_fd = central_diff<4>(pot_fun, x_rot_trans.reshaped(), h);
			const Vector4d jac_fd2 = central_diff<4>(pot_fun, x_rot_trans.reshaped(), h/2);
			Vector4d jac_ad;
			line_length<1>(x_rot_trans.data(), x_rot_trans.data()+2, jac_ad.data());
			//std::cout << "Error scales with " << (jac_fd - jac_ad).norm() / (jac_fd2 - jac_ad).norm() << std::endl;
			EXPECT_NEAR((jac_fd - jac_ad).norm(), 0, TOL) <<
				"Jacobian error (norm=" << (jac_fd - jac_ad).norm() << ")for x=\n" << x_rot_trans << ": \n" << jac_fd.transpose() << " !=\n" << jac_ad.transpose() << std::endl;
		}
	}
}
