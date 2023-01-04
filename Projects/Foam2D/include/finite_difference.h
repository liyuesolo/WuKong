#ifndef CS2D_FID_H
#define CS2D_FID_H

#include <cmath>
#include <cstdlib>
#include <functional>
#include <vector>

#include <eigen3/Eigen/Dense>

using Eigen::Vector2d, Eigen::VectorXd, Eigen::Vector, Eigen::Matrix2d, Eigen::MatrixXd;

// central finite difference computation of jacobian, index-by-index
template <int dim>
Eigen::Vector<double, dim> central_diff(
		const std::function<double(VectorXd)> fun,
		const Eigen::Vector<double, dim> x0, double h) {
	using Vec = Eigen::Vector<double, dim>;
	Vec jacobian = Vec::Zero();
	for (int i = 0; i < dim; ++i) {
		const Vec x_n = x0 - Vec::Unit(i)*h, x_p = x0 + Vec::Unit(i)*h;
		const double y_n = fun(x_n);
		const double y_p = fun(x_p);
		jacobian[i] = 0.5 * (y_p - y_n)/h;
	}
	return jacobian;
}

// central finite difference computation of jacobian, index-by-index
inline Eigen::VectorXd central_diff(
		const std::function<double(const std::vector<double>&)> fun,
		const std::vector<double>& x0, double h) {
	const int dim = x0.size();
	VectorXd jacobian = VectorXd::Zero(dim);
	for (int i = 0; i < dim; ++i) {
		std::vector<double> x_test = x0;
		x_test[i] -= h;
		const double y_n = fun(x_test);
		x_test[i] = x0[i] + h;
		const double y_p = fun(x_test);
		jacobian[i] = 0.5 * (y_p - y_n)/h;
	}
	return jacobian;
}

// central finite difference computation of jacobian, index-by-index
inline Eigen::MatrixXd central_diff_hessian(
		const std::function<VectorXd(const std::vector<double>&)> fun,
		const std::vector<double>& x0, double h) {
	const int dim = x0.size();
	MatrixXd hessian = MatrixXd::Zero(dim, dim);
	for (int i = 0; i < dim; ++i) {
		hessian.row(i) = central_diff(
			[&fun,i](const std::vector<double>& x){
				return fun(x)[i];
			}, x0, h
		);
	}
	return hessian;
}

#endif
