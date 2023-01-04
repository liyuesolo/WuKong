#include "../../include/potentials.h"

template <>
void cs2d::pot::triangle_area_signed<1>(const double *a, const double *b, const double *c, double* y) noexcept {
	// Offsets 0: a, 2: b, 4: c
	y[4] = -1 * (b[1] - a[1]);
	y[2] = c[1] - a[1];
	y[0] = 0 - y[4] - y[2];
	y[5] = b[0] - a[0];
	y[3] = -1 * (c[0] - a[0]);
	y[1] = 0 - y[5] - y[3];
}

template <>
void cs2d::pot::triangle_area_signed<2>(const double *, const double *, const double *, double* y) noexcept {
	// Offsets 0: a, 2: b, 4: c
	// dependent variables without operations
	y[0] = 0;
	y[1] = 0;
	y[2] = 0;
	y[3] = 1;
	y[4] = 0;
	y[5] = -1;
	y[6] = 0;
	y[7] = 0;
	y[8] = -1;
	y[9] = 0;
	y[10] = 1;
	y[11] = 0;
	y[12] = 0;
	y[13] = -1;
	y[14] = 0;
	y[15] = 0;
	y[16] = 0;
	y[17] = 1;
	y[18] = 1;
	y[19] = 0;
	y[20] = 0;
	y[21] = 0;
	y[22] = -1;
	y[23] = 0;
	y[24] = 0;
	y[25] = 1;
	y[26] = 0;
	y[27] = -1;
	y[28] = 0;
	y[29] = 0;
	y[30] = -1;
	y[31] = 0;
	y[32] = 1;
	y[33] = 0;
	y[34] = 0;
	y[35] = 0;
}

