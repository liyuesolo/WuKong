#include "AreaEnergy.h"



void computeSignedTriangleArea(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, double& energy){
	double _i_var[9];
	_i_var[0] = (vj(0,0))-(center(0,0));
	_i_var[1] = (vi(1,0))-(center(1,0));
	_i_var[2] = (vj(1,0))-(center(1,0));
	_i_var[3] = (vi(0,0))-(center(0,0));
	_i_var[4] = (_i_var[1])*(_i_var[0]);
	_i_var[5] = (_i_var[3])*(_i_var[2]);
	_i_var[6] = (_i_var[5])-(_i_var[4]);
	_i_var[7] = 0.5;
	_i_var[8] = (_i_var[7])*(_i_var[6]);
	energy = _i_var[8];
}
void computeSignedTriangleAreaGradient(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 1>& energygradient){
	double _i_var[10];
	_i_var[0] = (vj(1,0))-(center(1,0));
	_i_var[1] = 0.5;
	_i_var[2] = (vj(0,0))-(center(0,0));
	_i_var[3] = -0.5;
	_i_var[4] = (vi(1,0))-(center(1,0));
	_i_var[5] = (vi(0,0))-(center(0,0));
	_i_var[6] = (_i_var[1])*(_i_var[0]);
	_i_var[7] = (_i_var[3])*(_i_var[2]);
	_i_var[8] = (_i_var[3])*(_i_var[4]);
	_i_var[9] = (_i_var[1])*(_i_var[5]);
	energygradient(0,0) = _i_var[6];
	energygradient(1,0) = _i_var[7];
	energygradient(2,0) = _i_var[8];
	energygradient(3,0) = _i_var[9];
}
void computeSignedTriangleAreaHessian(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 4>& energyhessian){
	double _i_var[3];
	_i_var[0] = 0;
	_i_var[1] = 0.5;
	_i_var[2] = -0.5;
	energyhessian(0,0) = _i_var[0];
	energyhessian(1,0) = _i_var[0];
	energyhessian(2,0) = _i_var[0];
	energyhessian(3,0) = _i_var[1];
	energyhessian(0,1) = _i_var[0];
	energyhessian(1,1) = _i_var[0];
	energyhessian(2,1) = _i_var[2];
	energyhessian(3,1) = _i_var[0];
	energyhessian(0,2) = _i_var[0];
	energyhessian(1,2) = _i_var[2];
	energyhessian(2,2) = _i_var[0];
	energyhessian(3,2) = _i_var[0];
	energyhessian(0,3) = _i_var[1];
	energyhessian(1,3) = _i_var[0];
	energyhessian(2,3) = _i_var[0];
	energyhessian(3,3) = _i_var[0];
}
