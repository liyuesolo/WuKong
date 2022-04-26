#include "../../include/autodiff/MembraneEnergy.h"

void computeMembraneQubicPenalty(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, double& energy){
	double _i_var[10];
	_i_var[0] = (vi(1,0))-(center(1,0));
	_i_var[1] = (vi(0,0))-(center(0,0));
	_i_var[2] = (_i_var[0])*(_i_var[0]);
	_i_var[3] = (_i_var[1])*(_i_var[1]);
	_i_var[4] = (_i_var[3])+(_i_var[2]);
	_i_var[5] = std::sqrt(_i_var[4]);
	_i_var[6] = 3;
	_i_var[7] = (_i_var[5])-(radius);
	_i_var[8] = std::pow(_i_var[7],_i_var[6]);
	_i_var[9] = (stiffness)*(_i_var[8]);
	energy = _i_var[9];
}
void computeMembraneQubicPenaltyGradient(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 2, 1>& energygradient){
	double _i_var[20];
	_i_var[0] = (vi(1,0))-(center(1,0));
	_i_var[1] = (vi(0,0))-(center(0,0));
	_i_var[2] = (_i_var[0])*(_i_var[0]);
	_i_var[3] = (_i_var[1])*(_i_var[1]);
	_i_var[4] = (_i_var[3])+(_i_var[2]);
	_i_var[5] = std::sqrt(_i_var[4]);
	_i_var[6] = (_i_var[5])-(radius);
	_i_var[7] = 2;
	_i_var[8] = (_i_var[6])*(_i_var[6]);
	_i_var[9] = 3;
	_i_var[10] = (_i_var[7])*(_i_var[5]);
	_i_var[11] = 1;
	_i_var[12] = (_i_var[9])*(_i_var[8]);
	_i_var[13] = (_i_var[11])/(_i_var[10]);
	_i_var[14] = (stiffness)*(_i_var[12]);
	_i_var[15] = (_i_var[14])*(_i_var[13]);
	_i_var[16] = (_i_var[15])*(_i_var[1]);
	_i_var[17] = (_i_var[15])*(_i_var[0]);
	_i_var[18] = (_i_var[7])*(_i_var[16]);
	_i_var[19] = (_i_var[7])*(_i_var[17]);
	energygradient(0,0) = _i_var[18];
	energygradient(1,0) = _i_var[19];
}
void computeMembraneQubicPenaltyHessian(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 2, 2>& energyhessian){
	double _i_var[39];
	_i_var[0] = (vi(1,0))-(center(1,0));
	_i_var[1] = (vi(0,0))-(center(0,0));
	_i_var[2] = (_i_var[0])*(_i_var[0]);
	_i_var[3] = (_i_var[1])*(_i_var[1]);
	_i_var[4] = (_i_var[3])+(_i_var[2]);
	_i_var[5] = std::sqrt(_i_var[4]);
	_i_var[6] = 2;
	_i_var[7] = (_i_var[6])*(_i_var[5]);
	_i_var[8] = (_i_var[7])*(_i_var[7]);
	_i_var[9] = 1;
	_i_var[10] = (_i_var[9])/(_i_var[8]);
	_i_var[11] = (_i_var[5])-(radius);
	_i_var[12] = -(_i_var[10]);
	_i_var[13] = (_i_var[11])*(_i_var[11]);
	_i_var[14] = 3;
	_i_var[15] = (_i_var[6])*(_i_var[11]);
	_i_var[16] = (_i_var[9])/(_i_var[7]);
	_i_var[17] = (_i_var[12])*(_i_var[6]);
	_i_var[18] = (_i_var[14])*(_i_var[13]);
	_i_var[19] = (_i_var[14])*(_i_var[15]);
	_i_var[20] = (_i_var[17])*(_i_var[16]);
	_i_var[21] = (stiffness)*(_i_var[18]);
	_i_var[22] = (stiffness)*(_i_var[19]);
	_i_var[23] = (_i_var[16])*(_i_var[16]);
	_i_var[24] = (_i_var[21])*(_i_var[20]);
	_i_var[25] = (_i_var[23])*(_i_var[22]);
	_i_var[26] = (_i_var[6])*(_i_var[1]);
	_i_var[27] = (_i_var[6])*(_i_var[0]);
	_i_var[28] = (_i_var[21])*(_i_var[16]);
	_i_var[29] = (_i_var[25])+(_i_var[24]);
	_i_var[30] = (_i_var[26])*(_i_var[26]);
	_i_var[31] = (_i_var[27])*(_i_var[27]);
	_i_var[32] = (_i_var[28])*(_i_var[6]);
	_i_var[33] = (_i_var[30])*(_i_var[29]);
	_i_var[34] = (_i_var[26])*(_i_var[29]);
	_i_var[35] = (_i_var[31])*(_i_var[29]);
	_i_var[36] = (_i_var[33])+(_i_var[32]);
	_i_var[37] = (_i_var[27])*(_i_var[34]);
	_i_var[38] = (_i_var[35])+(_i_var[32]);
	energyhessian(0,0) = _i_var[36];
	energyhessian(1,0) = _i_var[37];
	energyhessian(0,1) = _i_var[37];
	energyhessian(1,1) = _i_var[38];
}