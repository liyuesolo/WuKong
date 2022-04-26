#include "../../include/autodiff/EdgeEnergy.h"

void computeEdgeSquaredNorm2D(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, double& energy){
	double _i_var[7];
	_i_var[0] = (r1(1,0))-(r0(1,0));
	_i_var[1] = (r1(0,0))-(r0(0,0));
	_i_var[2] = (_i_var[0])*(_i_var[0]);
	_i_var[3] = (_i_var[1])*(_i_var[1]);
	_i_var[4] = (_i_var[3])+(_i_var[2]);
	_i_var[5] = 0.5;
	_i_var[6] = (_i_var[5])*(_i_var[4]);
	energy = _i_var[6];
}
void computeEdgeSquaredNorm2DGradient(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, Eigen::Matrix<double, 4, 1>& energygradient){
	double _i_var[11];
	_i_var[0] = (r1(0,0))-(r0(0,0));
	_i_var[1] = 0.5;
	_i_var[2] = (r1(1,0))-(r0(1,0));
	_i_var[3] = (_i_var[1])*(_i_var[0]);
	_i_var[4] = 2;
	_i_var[5] = (_i_var[1])*(_i_var[2]);
	_i_var[6] = -1;
	_i_var[7] = (_i_var[4])*(_i_var[3]);
	_i_var[8] = (_i_var[4])*(_i_var[5]);
	_i_var[9] = (_i_var[7])*(_i_var[6]);
	_i_var[10] = (_i_var[8])*(_i_var[6]);
	energygradient(0,0) = _i_var[9];
	energygradient(1,0) = _i_var[10];
	energygradient(2,0) = _i_var[7];
	energygradient(3,0) = _i_var[8];
}
void computeEdgeSquaredNorm2DHessian(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, Eigen::Matrix<double, 4, 4>& energyhessian){
	double _i_var[3];
	_i_var[0] = 1;
	_i_var[1] = 0;
	_i_var[2] = -1;
	energyhessian(0,0) = _i_var[0];
	energyhessian(1,0) = _i_var[1];
	energyhessian(2,0) = _i_var[2];
	energyhessian(3,0) = _i_var[1];
	energyhessian(0,1) = _i_var[1];
	energyhessian(1,1) = _i_var[0];
	energyhessian(2,1) = _i_var[1];
	energyhessian(3,1) = _i_var[2];
	energyhessian(0,2) = _i_var[2];
	energyhessian(1,2) = _i_var[1];
	energyhessian(2,2) = _i_var[0];
	energyhessian(3,2) = _i_var[1];
	energyhessian(0,3) = _i_var[1];
	energyhessian(1,3) = _i_var[2];
	energyhessian(2,3) = _i_var[1];
	energyhessian(3,3) = _i_var[0];
}