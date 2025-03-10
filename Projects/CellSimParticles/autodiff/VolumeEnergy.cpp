#include "VolumeEnergy.h"
void computeTetVolume(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, double& energy){
	double _i_var[25];
	_i_var[0] = (v2(2,0))-(v0(2,0));
	_i_var[1] = (v1(0,0))-(v0(0,0));
	_i_var[2] = (v2(0,0))-(v0(0,0));
	_i_var[3] = (v1(2,0))-(v0(2,0));
	_i_var[4] = (v2(1,0))-(v0(1,0));
	_i_var[5] = (v1(1,0))-(v0(1,0));
	_i_var[6] = (_i_var[1])*(_i_var[0]);
	_i_var[7] = (_i_var[3])*(_i_var[2]);
	_i_var[8] = (_i_var[3])*(_i_var[4]);
	_i_var[9] = (_i_var[5])*(_i_var[0]);
	_i_var[10] = (_i_var[5])*(_i_var[2]);
	_i_var[11] = (_i_var[1])*(_i_var[4]);
	_i_var[12] = (v3(1,0))-(v0(1,0));
	_i_var[13] = (_i_var[7])-(_i_var[6]);
	_i_var[14] = (v3(0,0))-(v0(0,0));
	_i_var[15] = (_i_var[9])-(_i_var[8]);
	_i_var[16] = (v3(2,0))-(v0(2,0));
	_i_var[17] = (_i_var[11])-(_i_var[10]);
	_i_var[18] = (_i_var[13])*(_i_var[12]);
	_i_var[19] = (_i_var[15])*(_i_var[14]);
	_i_var[20] = (_i_var[17])*(_i_var[16]);
	_i_var[21] = (_i_var[19])+(_i_var[18]);
	_i_var[22] = (_i_var[21])+(_i_var[20]);
	_i_var[23] = 0.16666666666666666;
	_i_var[24] = (_i_var[23])*(_i_var[22]);
	energy = _i_var[24];
}
void computeTetVolumeGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 1>& energygradient){
	double _i_var[62];
	_i_var[0] = (v3(2,0))-(v0(2,0));
	_i_var[1] = 0.16666666666666666;
	_i_var[2] = (v3(0,0))-(v0(0,0));
	_i_var[3] = (v3(1,0))-(v0(1,0));
	_i_var[4] = -1;
	_i_var[5] = (_i_var[1])*(_i_var[0]);
	_i_var[6] = (v2(1,0))-(v0(1,0));
	_i_var[7] = (v1(2,0))-(v0(2,0));
	_i_var[8] = (v2(2,0))-(v0(2,0));
	_i_var[9] = (v1(1,0))-(v0(1,0));
	_i_var[10] = (v1(0,0))-(v0(0,0));
	_i_var[11] = (v2(0,0))-(v0(0,0));
	_i_var[12] = (_i_var[1])*(_i_var[2]);
	_i_var[13] = (_i_var[1])*(_i_var[3]);
	_i_var[14] = (_i_var[5])*(_i_var[4]);
	_i_var[15] = (_i_var[7])*(_i_var[6]);
	_i_var[16] = (_i_var[9])*(_i_var[8]);
	_i_var[17] = (_i_var[10])*(_i_var[8]);
	_i_var[18] = (_i_var[7])*(_i_var[11]);
	_i_var[19] = (_i_var[12])*(_i_var[4]);
	_i_var[20] = (_i_var[9])*(_i_var[11]);
	_i_var[21] = (_i_var[10])*(_i_var[6]);
	_i_var[22] = (_i_var[13])*(_i_var[4]);
	_i_var[23] = (_i_var[13])*(_i_var[7]);
	_i_var[24] = (_i_var[14])*(_i_var[9]);
	_i_var[25] = (_i_var[16])-(_i_var[15]);
	_i_var[26] = (_i_var[12])*(_i_var[8]);
	_i_var[27] = (_i_var[14])*(_i_var[11]);
	_i_var[28] = (_i_var[18])-(_i_var[17]);
	_i_var[29] = (_i_var[13])*(_i_var[11]);
	_i_var[30] = (_i_var[19])*(_i_var[6]);
	_i_var[31] = (_i_var[21])-(_i_var[20]);
	_i_var[32] = (_i_var[22])*(_i_var[8]);
	_i_var[33] = (_i_var[5])*(_i_var[6]);
	_i_var[34] = (_i_var[24])+(_i_var[23]);
	_i_var[35] = (_i_var[1])*(_i_var[25]);
	_i_var[36] = (_i_var[19])*(_i_var[7]);
	_i_var[37] = (_i_var[5])*(_i_var[10]);
	_i_var[38] = (_i_var[27])+(_i_var[26]);
	_i_var[39] = (_i_var[1])*(_i_var[28]);
	_i_var[40] = (_i_var[22])*(_i_var[10]);
	_i_var[41] = (_i_var[12])*(_i_var[9]);
	_i_var[42] = (_i_var[30])+(_i_var[29]);
	_i_var[43] = (_i_var[1])*(_i_var[31]);
	_i_var[44] = (_i_var[33])+(_i_var[32]);
	_i_var[45] = (_i_var[34])*(_i_var[4]);
	_i_var[46] = (_i_var[35])*(_i_var[4]);
	_i_var[47] = (_i_var[37])+(_i_var[36]);
	_i_var[48] = (_i_var[38])*(_i_var[4]);
	_i_var[49] = (_i_var[39])*(_i_var[4]);
	_i_var[50] = (_i_var[41])+(_i_var[40]);
	_i_var[51] = (_i_var[42])*(_i_var[4]);
	_i_var[52] = (_i_var[43])*(_i_var[4]);
	_i_var[53] = (_i_var[44])*(_i_var[4]);
	_i_var[54] = (_i_var[46])+(_i_var[45]);
	_i_var[55] = (_i_var[47])*(_i_var[4]);
	_i_var[56] = (_i_var[49])+(_i_var[48]);
	_i_var[57] = (_i_var[50])*(_i_var[4]);
	_i_var[58] = (_i_var[52])+(_i_var[51]);
	_i_var[59] = (_i_var[54])+(_i_var[53]);
	_i_var[60] = (_i_var[56])+(_i_var[55]);
	_i_var[61] = (_i_var[58])+(_i_var[57]);
	energygradient(0,0) = _i_var[59];
	energygradient(1,0) = _i_var[60];
	energygradient(2,0) = _i_var[61];
	energygradient(3,0) = _i_var[44];
	energygradient(4,0) = _i_var[38];
	energygradient(5,0) = _i_var[42];
	energygradient(6,0) = _i_var[34];
	energygradient(7,0) = _i_var[47];
	energygradient(8,0) = _i_var[50];
}
void computeTetVolumeHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 9>& energyhessian){
	double _i_var[70];
	_i_var[0] = (v3(2,0))-(v0(2,0));
	_i_var[1] = 0.16666666666666666;
	_i_var[2] = (v3(1,0))-(v0(1,0));
	_i_var[3] = (v3(0,0))-(v0(0,0));
	_i_var[4] = -1;
	_i_var[5] = (_i_var[1])*(_i_var[0]);
	_i_var[6] = (_i_var[1])*(_i_var[2]);
	_i_var[7] = (v1(1,0))-(v0(1,0));
	_i_var[8] = (_i_var[1])*(_i_var[3]);
	_i_var[9] = (_i_var[5])*(_i_var[4]);
	_i_var[10] = -0.16666666666666666;
	_i_var[11] = (v1(2,0))-(v0(2,0));
	_i_var[12] = (v2(2,0))-(v0(2,0));
	_i_var[13] = (_i_var[4])*(_i_var[6]);
	_i_var[14] = (_i_var[7])*(_i_var[1]);
	_i_var[15] = (v2(1,0))-(v0(1,0));
	_i_var[16] = (_i_var[8])*(_i_var[4]);
	_i_var[17] = (v2(0,0))-(v0(0,0));
	_i_var[18] = (v1(0,0))-(v0(0,0));
	_i_var[19] = (_i_var[4])*(_i_var[9]);
	_i_var[20] = (_i_var[11])*(_i_var[10]);
	_i_var[21] = (_i_var[11])*(_i_var[1]);
	_i_var[22] = (_i_var[12])*(_i_var[10]);
	_i_var[23] = (_i_var[6])*(_i_var[4]);
	_i_var[24] = (_i_var[14])+(_i_var[13]);
	_i_var[25] = (_i_var[15])*(_i_var[1]);
	_i_var[26] = (_i_var[4])*(_i_var[16]);
	_i_var[27] = (_i_var[17])*(_i_var[10]);
	_i_var[28] = (_i_var[18])*(_i_var[10]);
	_i_var[29] = (_i_var[17])*(_i_var[1]);
	_i_var[30] = (_i_var[4])*(_i_var[5]);
	_i_var[31] = (_i_var[12])*(_i_var[1]);
	_i_var[32] = (_i_var[20])+(_i_var[19]);
	_i_var[33] = (_i_var[4])*(_i_var[21]);
	_i_var[34] = (_i_var[4])*(_i_var[22]);
	_i_var[35] = (_i_var[4])*(_i_var[23]);
	_i_var[36] = (_i_var[7])*(_i_var[10]);
	_i_var[37] = (_i_var[15])*(_i_var[10]);
	_i_var[38] = (_i_var[4])*(_i_var[24]);
	_i_var[39] = (_i_var[4])*(_i_var[25]);
	_i_var[40] = (_i_var[4])*(_i_var[8]);
	_i_var[41] = (_i_var[18])*(_i_var[1]);
	_i_var[42] = (_i_var[27])+(_i_var[26]);
	_i_var[43] = (_i_var[4])*(_i_var[28]);
	_i_var[44] = (_i_var[4])*(_i_var[29]);
	_i_var[45] = (_i_var[31])+(_i_var[30]);
	_i_var[46] = (_i_var[4])*(_i_var[32]);
	_i_var[47] = (_i_var[34])+(_i_var[33]);
	_i_var[48] = (_i_var[36])+(_i_var[35]);
	_i_var[49] = (_i_var[4])*(_i_var[37]);
	_i_var[50] = (_i_var[39])+(_i_var[38]);
	_i_var[51] = (_i_var[41])+(_i_var[40]);
	_i_var[52] = (_i_var[4])*(_i_var[42]);
	_i_var[53] = (_i_var[44])+(_i_var[43]);
	_i_var[54] = (_i_var[4])*(_i_var[45]);
	_i_var[55] = (_i_var[47])+(_i_var[46]);
	_i_var[56] = (_i_var[4])*(_i_var[48]);
	_i_var[57] = (_i_var[50])+(_i_var[49]);
	_i_var[58] = (_i_var[4])*(_i_var[51]);
	_i_var[59] = (_i_var[53])+(_i_var[52]);
	_i_var[60] = 0;
	_i_var[61] = (_i_var[55])+(_i_var[54]);
	_i_var[62] = (_i_var[57])+(_i_var[56]);
	_i_var[63] = (_i_var[22])+(_i_var[19]);
	_i_var[64] = (_i_var[25])+(_i_var[13]);
	_i_var[65] = (_i_var[21])+(_i_var[30]);
	_i_var[66] = (_i_var[59])+(_i_var[58]);
	_i_var[67] = (_i_var[37])+(_i_var[35]);
	_i_var[68] = (_i_var[29])+(_i_var[40]);
	_i_var[69] = (_i_var[28])+(_i_var[26]);
	energyhessian(0,0) = _i_var[60];
	energyhessian(1,0) = _i_var[61];
	energyhessian(2,0) = _i_var[62];
	energyhessian(3,0) = _i_var[60];
	energyhessian(4,0) = _i_var[63];
	energyhessian(5,0) = _i_var[64];
	energyhessian(6,0) = _i_var[60];
	energyhessian(7,0) = _i_var[65];
	energyhessian(8,0) = _i_var[48];
	energyhessian(0,1) = _i_var[61];
	energyhessian(1,1) = _i_var[60];
	energyhessian(2,1) = _i_var[66];
	energyhessian(3,1) = _i_var[45];
	energyhessian(4,1) = _i_var[60];
	energyhessian(5,1) = _i_var[42];
	energyhessian(6,1) = _i_var[32];
	energyhessian(7,1) = _i_var[60];
	energyhessian(8,1) = _i_var[51];
	energyhessian(0,2) = _i_var[62];
	energyhessian(1,2) = _i_var[66];
	energyhessian(2,2) = _i_var[60];
	energyhessian(3,2) = _i_var[67];
	energyhessian(4,2) = _i_var[68];
	energyhessian(5,2) = _i_var[60];
	energyhessian(6,2) = _i_var[24];
	energyhessian(7,2) = _i_var[69];
	energyhessian(8,2) = _i_var[60];
	energyhessian(0,3) = _i_var[60];
	energyhessian(1,3) = _i_var[45];
	energyhessian(2,3) = _i_var[67];
	energyhessian(3,3) = _i_var[60];
	energyhessian(4,3) = _i_var[60];
	energyhessian(5,3) = _i_var[60];
	energyhessian(6,3) = _i_var[60];
	energyhessian(7,3) = _i_var[5];
	energyhessian(8,3) = _i_var[23];
	energyhessian(0,4) = _i_var[63];
	energyhessian(1,4) = _i_var[60];
	energyhessian(2,4) = _i_var[68];
	energyhessian(3,4) = _i_var[60];
	energyhessian(4,4) = _i_var[60];
	energyhessian(5,4) = _i_var[60];
	energyhessian(6,4) = _i_var[9];
	energyhessian(7,4) = _i_var[60];
	energyhessian(8,4) = _i_var[8];
	energyhessian(0,5) = _i_var[64];
	energyhessian(1,5) = _i_var[42];
	energyhessian(2,5) = _i_var[60];
	energyhessian(3,5) = _i_var[60];
	energyhessian(4,5) = _i_var[60];
	energyhessian(5,5) = _i_var[60];
	energyhessian(6,5) = _i_var[6];
	energyhessian(7,5) = _i_var[16];
	energyhessian(8,5) = _i_var[60];
	energyhessian(0,6) = _i_var[60];
	energyhessian(1,6) = _i_var[32];
	energyhessian(2,6) = _i_var[24];
	energyhessian(3,6) = _i_var[60];
	energyhessian(4,6) = _i_var[9];
	energyhessian(5,6) = _i_var[6];
	energyhessian(6,6) = _i_var[60];
	energyhessian(7,6) = _i_var[60];
	energyhessian(8,6) = _i_var[60];
	energyhessian(0,7) = _i_var[65];
	energyhessian(1,7) = _i_var[60];
	energyhessian(2,7) = _i_var[69];
	energyhessian(3,7) = _i_var[5];
	energyhessian(4,7) = _i_var[60];
	energyhessian(5,7) = _i_var[16];
	energyhessian(6,7) = _i_var[60];
	energyhessian(7,7) = _i_var[60];
	energyhessian(8,7) = _i_var[60];
	energyhessian(0,8) = _i_var[48];
	energyhessian(1,8) = _i_var[51];
	energyhessian(2,8) = _i_var[60];
	energyhessian(3,8) = _i_var[23];
	energyhessian(4,8) = _i_var[8];
	energyhessian(5,8) = _i_var[60];
	energyhessian(6,8) = _i_var[60];
	energyhessian(7,8) = _i_var[60];
	energyhessian(8,8) = _i_var[60];
}