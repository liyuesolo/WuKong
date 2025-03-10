#ifndef MEMBRANE_ENERGY_H
#define MEMBRANE_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void sphereBoundEnergy(double stiffness, double Rk, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, double& energy){
	double _i_var[13];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = -4;
	_i_var[10] = (_i_var[8])-(Rk);
	_i_var[11] = std::pow(_i_var[10],_i_var[9]);
	_i_var[12] = (stiffness)*(_i_var[11]);
	energy = _i_var[12];
}
void sphereBoundEnergyGradient(double stiffness, double Rk, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, Eigen::Matrix<double, 3, 1>& energygradient){
	double _i_var[26];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = -5;
	_i_var[10] = (_i_var[8])-(Rk);
	_i_var[11] = 2;
	_i_var[12] = std::pow(_i_var[10],_i_var[9]);
	_i_var[13] = -4;
	_i_var[14] = (_i_var[11])*(_i_var[8]);
	_i_var[15] = 1;
	_i_var[16] = (_i_var[13])*(_i_var[12]);
	_i_var[17] = (_i_var[15])/(_i_var[14]);
	_i_var[18] = (stiffness)*(_i_var[16]);
	_i_var[19] = (_i_var[18])*(_i_var[17]);
	_i_var[20] = (_i_var[19])*(_i_var[1]);
	_i_var[21] = (_i_var[19])*(_i_var[0]);
	_i_var[22] = (_i_var[19])*(_i_var[2]);
	_i_var[23] = (_i_var[11])*(_i_var[20]);
	_i_var[24] = (_i_var[11])*(_i_var[21]);
	_i_var[25] = (_i_var[11])*(_i_var[22]);
	energygradient(0,0) = _i_var[23];
	energygradient(1,0) = _i_var[24];
	energygradient(2,0) = _i_var[25];
}
void sphereBoundEnergyHessian(double stiffness, double Rk, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, Eigen::Matrix<double, 3, 3>& energyhessian){
	double _i_var[52];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = 2;
	_i_var[10] = (_i_var[9])*(_i_var[8]);
	_i_var[11] = (_i_var[10])*(_i_var[10]);
	_i_var[12] = 1;
	_i_var[13] = -6;
	_i_var[14] = (_i_var[8])-(Rk);
	_i_var[15] = (_i_var[12])/(_i_var[11]);
	_i_var[16] = -5;
	_i_var[17] = std::pow(_i_var[14],_i_var[13]);
	_i_var[18] = -(_i_var[15]);
	_i_var[19] = std::pow(_i_var[14],_i_var[16]);
	_i_var[20] = -4;
	_i_var[21] = (_i_var[16])*(_i_var[17]);
	_i_var[22] = (_i_var[12])/(_i_var[10]);
	_i_var[23] = (_i_var[18])*(_i_var[9]);
	_i_var[24] = (_i_var[20])*(_i_var[19]);
	_i_var[25] = (_i_var[20])*(_i_var[21]);
	_i_var[26] = (_i_var[23])*(_i_var[22]);
	_i_var[27] = (stiffness)*(_i_var[24]);
	_i_var[28] = (stiffness)*(_i_var[25]);
	_i_var[29] = (_i_var[22])*(_i_var[22]);
	_i_var[30] = (_i_var[27])*(_i_var[26]);
	_i_var[31] = (_i_var[29])*(_i_var[28]);
	_i_var[32] = (_i_var[9])*(_i_var[1]);
	_i_var[33] = (_i_var[9])*(_i_var[0]);
	_i_var[34] = (_i_var[9])*(_i_var[2]);
	_i_var[35] = (_i_var[27])*(_i_var[22]);
	_i_var[36] = (_i_var[31])+(_i_var[30]);
	_i_var[37] = (_i_var[32])*(_i_var[32]);
	_i_var[38] = (_i_var[33])*(_i_var[33]);
	_i_var[39] = (_i_var[34])*(_i_var[34]);
	_i_var[40] = (_i_var[35])*(_i_var[9]);
	_i_var[41] = (_i_var[37])*(_i_var[36]);
	_i_var[42] = (_i_var[32])*(_i_var[36]);
	_i_var[43] = (_i_var[34])*(_i_var[36]);
	_i_var[44] = (_i_var[38])*(_i_var[36]);
	_i_var[45] = (_i_var[39])*(_i_var[36]);
	_i_var[46] = (_i_var[41])+(_i_var[40]);
	_i_var[47] = (_i_var[33])*(_i_var[42]);
	_i_var[48] = (_i_var[32])*(_i_var[43]);
	_i_var[49] = (_i_var[44])+(_i_var[40]);
	_i_var[50] = (_i_var[33])*(_i_var[43]);
	_i_var[51] = (_i_var[45])+(_i_var[40]);
	energyhessian(0,0) = _i_var[46];
	energyhessian(1,0) = _i_var[47];
	energyhessian(2,0) = _i_var[48];
	energyhessian(0,1) = _i_var[47];
	energyhessian(1,1) = _i_var[49];
	energyhessian(2,1) = _i_var[50];
	energyhessian(0,2) = _i_var[48];
	energyhessian(1,2) = _i_var[50];
	energyhessian(2,2) = _i_var[51];
}


void computeRadiusPenalty(double stiffness, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, double& energy){
	double _i_var[14];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = 0.5;
	_i_var[10] = (Rc)-(_i_var[8]);
	_i_var[11] = (_i_var[9])*(stiffness);
	_i_var[12] = (_i_var[11])*(_i_var[10]);
	_i_var[13] = (_i_var[12])*(_i_var[10]);
	energy = _i_var[13];
}
void computeRadiusPenaltyGradient(double stiffness, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, Eigen::Matrix<double, 3, 1>& energygradient){
	double _i_var[28];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = 0.5;
	_i_var[9] = std::sqrt(_i_var[7]);
	_i_var[10] = (_i_var[8])*(stiffness);
	_i_var[11] = (Rc)-(_i_var[9]);
	_i_var[12] = 2;
	_i_var[13] = (_i_var[11])*(_i_var[10]);
	_i_var[14] = (_i_var[10])*(_i_var[11]);
	_i_var[15] = (_i_var[12])*(_i_var[9]);
	_i_var[16] = 1;
	_i_var[17] = -1;
	_i_var[18] = (_i_var[14])+(_i_var[13]);
	_i_var[19] = (_i_var[16])/(_i_var[15]);
	_i_var[20] = (_i_var[18])*(_i_var[17]);
	_i_var[21] = (_i_var[20])*(_i_var[19]);
	_i_var[22] = (_i_var[21])*(_i_var[1]);
	_i_var[23] = (_i_var[21])*(_i_var[0]);
	_i_var[24] = (_i_var[21])*(_i_var[2]);
	_i_var[25] = (_i_var[12])*(_i_var[22]);
	_i_var[26] = (_i_var[12])*(_i_var[23]);
	_i_var[27] = (_i_var[12])*(_i_var[24]);
	energygradient(0,0) = _i_var[25];
	energygradient(1,0) = _i_var[26];
	energygradient(2,0) = _i_var[27];
}
void computeRadiusPenaltyHessian(double stiffness, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, Eigen::Matrix<double, 3, 3>& energyhessian){
	double _i_var[50];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = 2;
	_i_var[10] = (_i_var[9])*(_i_var[8]);
	_i_var[11] = (_i_var[10])*(_i_var[10]);
	_i_var[12] = 1;
	_i_var[13] = 0.5;
	_i_var[14] = (_i_var[12])/(_i_var[11]);
	_i_var[15] = (_i_var[13])*(stiffness);
	_i_var[16] = (Rc)-(_i_var[8]);
	_i_var[17] = -(_i_var[14]);
	_i_var[18] = (_i_var[16])*(_i_var[15]);
	_i_var[19] = (_i_var[15])*(_i_var[16]);
	_i_var[20] = (_i_var[12])/(_i_var[10]);
	_i_var[21] = (_i_var[17])*(_i_var[9]);
	_i_var[22] = -1;
	_i_var[23] = (_i_var[19])+(_i_var[18]);
	_i_var[24] = (_i_var[21])*(_i_var[20]);
	_i_var[25] = (_i_var[23])*(_i_var[22]);
	_i_var[26] = (_i_var[9])*(_i_var[15]);
	_i_var[27] = (_i_var[20])*(_i_var[20]);
	_i_var[28] = (_i_var[25])*(_i_var[24]);
	_i_var[29] = (_i_var[27])*(_i_var[26]);
	_i_var[30] = (_i_var[9])*(_i_var[1]);
	_i_var[31] = (_i_var[9])*(_i_var[0]);
	_i_var[32] = (_i_var[9])*(_i_var[2]);
	_i_var[33] = (_i_var[25])*(_i_var[20]);
	_i_var[34] = (_i_var[29])+(_i_var[28]);
	_i_var[35] = (_i_var[30])*(_i_var[30]);
	_i_var[36] = (_i_var[31])*(_i_var[31]);
	_i_var[37] = (_i_var[32])*(_i_var[32]);
	_i_var[38] = (_i_var[33])*(_i_var[9]);
	_i_var[39] = (_i_var[35])*(_i_var[34]);
	_i_var[40] = (_i_var[30])*(_i_var[34]);
	_i_var[41] = (_i_var[32])*(_i_var[34]);
	_i_var[42] = (_i_var[36])*(_i_var[34]);
	_i_var[43] = (_i_var[37])*(_i_var[34]);
	_i_var[44] = (_i_var[39])+(_i_var[38]);
	_i_var[45] = (_i_var[31])*(_i_var[40]);
	_i_var[46] = (_i_var[30])*(_i_var[41]);
	_i_var[47] = (_i_var[42])+(_i_var[38]);
	_i_var[48] = (_i_var[31])*(_i_var[41]);
	_i_var[49] = (_i_var[43])+(_i_var[38]);
	energyhessian(0,0) = _i_var[44];
	energyhessian(1,0) = _i_var[45];
	energyhessian(2,0) = _i_var[46];
	energyhessian(0,1) = _i_var[45];
	energyhessian(1,1) = _i_var[47];
	energyhessian(2,1) = _i_var[48];
	energyhessian(0,2) = _i_var[46];
	energyhessian(1,2) = _i_var[48];
	energyhessian(2,2) = _i_var[49];
}



void computeRadiusBarrier(double stiffness, double dhat, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, 
	double& energy){
	double _i_var[17];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = (Rc)-(_i_var[8]);
	_i_var[10] = (_i_var[9])-(dhat);
	_i_var[11] = (_i_var[9])/(dhat);
	_i_var[12] = (_i_var[10])*(_i_var[10]);
	_i_var[13] = -(stiffness);
	_i_var[14] = std::log(_i_var[11]);
	_i_var[15] = (_i_var[13])*(_i_var[12]);
	_i_var[16] = (_i_var[15])*(_i_var[14]);
	energy = _i_var[16];
}
void computeRadiusBarrierGradient(double stiffness, double dhat, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, 
	Eigen::Matrix<double, 3, 1>& energygradient){
	double _i_var[37];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = (Rc)-(_i_var[8]);
	_i_var[10] = (_i_var[9])/(dhat);
	_i_var[11] = (_i_var[9])-(dhat);
	_i_var[12] = -(stiffness);
	_i_var[13] = std::log(_i_var[10]);
	_i_var[14] = 1;
	_i_var[15] = (_i_var[11])*(_i_var[11]);
	_i_var[16] = (_i_var[13])*(_i_var[12]);
	_i_var[17] = (_i_var[14])/(_i_var[10]);
	_i_var[18] = (_i_var[12])*(_i_var[15]);
	_i_var[19] = (_i_var[16])*(_i_var[11]);
	_i_var[20] = 2;
	_i_var[21] = (_i_var[14])/(dhat);
	_i_var[22] = (_i_var[18])*(_i_var[17]);
	_i_var[23] = (_i_var[20])*(_i_var[19]);
	_i_var[24] = (_i_var[22])*(_i_var[21]);
	_i_var[25] = (_i_var[20])*(_i_var[8]);
	_i_var[26] = -1;
	_i_var[27] = (_i_var[24])+(_i_var[23]);
	_i_var[28] = (_i_var[14])/(_i_var[25]);
	_i_var[29] = (_i_var[27])*(_i_var[26]);
	_i_var[30] = (_i_var[29])*(_i_var[28]);
	_i_var[31] = (_i_var[30])*(_i_var[1]);
	_i_var[32] = (_i_var[30])*(_i_var[0]);
	_i_var[33] = (_i_var[30])*(_i_var[2]);
	_i_var[34] = (_i_var[20])*(_i_var[31]);
	_i_var[35] = (_i_var[20])*(_i_var[32]);
	_i_var[36] = (_i_var[20])*(_i_var[33]);
	energygradient(0,0) = _i_var[34];
	energygradient(1,0) = _i_var[35];
	energygradient(2,0) = _i_var[36];
}
void computeRadiusBarrierHessian(double stiffness, double dhat, double Rc, const Eigen::Matrix<double,3,1> & vertex, const Eigen::Matrix<double,3,1> & center, 
	Eigen::Matrix<double, 3, 3>& energyhessian){
	double _i_var[71];
	_i_var[0] = (vertex(1,0))-(center(1,0));
	_i_var[1] = (vertex(0,0))-(center(0,0));
	_i_var[2] = (vertex(2,0))-(center(2,0));
	_i_var[3] = (_i_var[0])*(_i_var[0]);
	_i_var[4] = (_i_var[1])*(_i_var[1]);
	_i_var[5] = (_i_var[2])*(_i_var[2]);
	_i_var[6] = (_i_var[4])+(_i_var[3]);
	_i_var[7] = (_i_var[6])+(_i_var[5]);
	_i_var[8] = std::sqrt(_i_var[7]);
	_i_var[9] = (Rc)-(_i_var[8]);
	_i_var[10] = (_i_var[9])/(dhat);
	_i_var[11] = 1;
	_i_var[12] = 2;
	_i_var[13] = (_i_var[9])-(dhat);
	_i_var[14] = -(stiffness);
	_i_var[15] = (_i_var[11])/(_i_var[10]);
	_i_var[16] = (_i_var[10])*(_i_var[10]);
	_i_var[17] = (_i_var[12])*(_i_var[8]);
	_i_var[18] = (_i_var[13])*(_i_var[13]);
	_i_var[19] = (_i_var[15])*(_i_var[14]);
	_i_var[20] = (_i_var[12])*(_i_var[13]);
	_i_var[21] = (_i_var[11])/(_i_var[16]);
	_i_var[22] = (_i_var[17])*(_i_var[17]);
	_i_var[23] = std::log(_i_var[10]);
	_i_var[24] = (_i_var[14])*(_i_var[18]);
	_i_var[25] = (_i_var[20])*(_i_var[19]);
	_i_var[26] = (_i_var[11])/(dhat);
	_i_var[27] = -(_i_var[21]);
	_i_var[28] = (_i_var[11])/(_i_var[22]);
	_i_var[29] = (_i_var[23])*(_i_var[14]);
	_i_var[30] = (_i_var[24])*(_i_var[15]);
	_i_var[31] = (_i_var[26])*(_i_var[25]);
	_i_var[32] = (_i_var[24])*(_i_var[27]);
	_i_var[33] = (_i_var[26])*(_i_var[26]);
	_i_var[34] = -(_i_var[28]);
	_i_var[35] = (_i_var[29])*(_i_var[20]);
	_i_var[36] = (_i_var[30])*(_i_var[26]);
	_i_var[37] = (_i_var[12])*(_i_var[31]);
	_i_var[38] = (_i_var[33])*(_i_var[32]);
	_i_var[39] = (_i_var[11])/(_i_var[17]);
	_i_var[40] = (_i_var[34])*(_i_var[12]);
	_i_var[41] = -1;
	_i_var[42] = (_i_var[36])+(_i_var[35]);
	_i_var[43] = (_i_var[29])*(_i_var[12]);
	_i_var[44] = (_i_var[38])+(_i_var[37]);
	_i_var[45] = (_i_var[40])*(_i_var[39]);
	_i_var[46] = (_i_var[42])*(_i_var[41]);
	_i_var[47] = (_i_var[44])+(_i_var[43]);
	_i_var[48] = (_i_var[39])*(_i_var[39]);
	_i_var[49] = (_i_var[46])*(_i_var[45]);
	_i_var[50] = (_i_var[48])*(_i_var[47]);
	_i_var[51] = (_i_var[12])*(_i_var[1]);
	_i_var[52] = (_i_var[12])*(_i_var[0]);
	_i_var[53] = (_i_var[12])*(_i_var[2]);
	_i_var[54] = (_i_var[46])*(_i_var[39]);
	_i_var[55] = (_i_var[50])+(_i_var[49]);
	_i_var[56] = (_i_var[51])*(_i_var[51]);
	_i_var[57] = (_i_var[52])*(_i_var[52]);
	_i_var[58] = (_i_var[53])*(_i_var[53]);
	_i_var[59] = (_i_var[54])*(_i_var[12]);
	_i_var[60] = (_i_var[56])*(_i_var[55]);
	_i_var[61] = (_i_var[51])*(_i_var[55]);
	_i_var[62] = (_i_var[53])*(_i_var[55]);
	_i_var[63] = (_i_var[57])*(_i_var[55]);
	_i_var[64] = (_i_var[58])*(_i_var[55]);
	_i_var[65] = (_i_var[60])+(_i_var[59]);
	_i_var[66] = (_i_var[52])*(_i_var[61]);
	_i_var[67] = (_i_var[51])*(_i_var[62]);
	_i_var[68] = (_i_var[63])+(_i_var[59]);
	_i_var[69] = (_i_var[52])*(_i_var[62]);
	_i_var[70] = (_i_var[64])+(_i_var[59]);
	energyhessian(0,0) = _i_var[65];
	energyhessian(1,0) = _i_var[66];
	energyhessian(2,0) = _i_var[67];
	energyhessian(0,1) = _i_var[66];
	energyhessian(1,1) = _i_var[68];
	energyhessian(2,1) = _i_var[69];
	energyhessian(0,2) = _i_var[67];
	energyhessian(1,2) = _i_var[69];
	energyhessian(2,2) = _i_var[70];
}


#endif