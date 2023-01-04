#include "../../include/potentials.h"

template <>
void cs2d::pot::line_length<1>(const double* a, const double* b, double* y) noexcept {
	// Offsets 0: a, 2: b
	std::array<double, 3> v;  // temporary variables
	v[0] = b[0] - a[0];
	v[1] = b[1] - a[1];
	v[2] = (1 / sqrt(v[0] * v[0] + v[1] * v[1])) / 2.;
	y[2] = v[2] * v[0] + v[2] * v[0];
	y[0] = 0 - y[2];
	y[3] = v[2] * v[1] + v[2] * v[1];
	y[1] = 0 - y[3];
	auto* k = y, *k2 = y+4;
}

template <>
void cs2d::pot::line_length<2>(const double* a, const double* b, double* y) noexcept {
	// Offsets 0: a, 2: b
	std::array<double, 8> v;  // temporary variables
	v[0] = b[0] - a[0];
	v[1] = b[1] - a[1];
	v[2] = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[3] = 1 / v[2];
	v[4] = v[3] / 2.;
	v[3] = ((0 - v[3] * ((-1 * v[0] + v[0] * -1) / 2.) / v[2]) * v[3]) / 2.;
	y[8] = v[4] * -1 + v[4] * -1 + v[3] * v[0] + v[3] * v[0];
	y[0] = 0 - y[8];
	v[4] = 1 / v[2];
	v[5] = ((0 - v[4] * ((-1 * v[1] + v[1] * -1) / 2.) / v[2]) * v[4]) / 2.;
	y[9] = v[5] * v[0] + v[5] * v[0];
	y[1] = 0 - y[9];
	v[6] = 1 / v[2];
	v[7] = v[6] / 2.;
	v[6] = ((0 - v[6] * ((v[0] + v[0]) / 2.) / v[2]) * v[6]) / 2.;
	y[10] = v[7] + v[7] + v[6] * v[0] + v[6] * v[0];
	y[2] = 0 - y[10];
	v[7] = 1 / v[2];
	v[2] = ((0 - v[7] * ((v[1] + v[1]) / 2.) / v[2]) * v[7]) / 2.;
	y[11] = v[2] * v[0] + v[2] * v[0];
	y[3] = 0 - y[11];
	y[12] = v[3] * v[1] + v[3] * v[1];
	y[4] = 0 - y[12];
	v[4] = v[4] / 2.;
	y[13] = v[4] * -1 + v[4] * -1 + v[5] * v[1] + v[5] * v[1];
	y[5] = 0 - y[13];
	y[14] = v[6] * v[1] + v[6] * v[1];
	y[6] = 0 - y[14];
	v[7] = v[7] / 2.;
	y[15] = v[7] + v[7] + v[2] * v[1] + v[2] * v[1];
	y[7] = 0 - y[15];
}
