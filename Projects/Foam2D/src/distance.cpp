#include "../include/distance.h"

using std::sqrt, std::pow, std::abs, std::log;


VertexEdgeMode vertexEdgeMode(const std::array<double, 7> x) {
	const double d0 = x[6];
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	const double kld = sqrt(pow(lx - kx, 2) + pow(ly-ky, 2));
	const double a = kx*(my-ly) + ky*(lx-mx) + mx*ly - my*lx;
	const double dist_perp = abs(a / kld);
	if (dist_perp >= d0)
		return Null;
	if ((mx - kx)*(lx-kx) + (my-ky)*(ly-ky) < 0) {
		if (sqrt(pow(mx-kx, 2) + pow(my-ky, 2)) >= d0)
			return Null;
		else
			return K;
	}
	if ((mx - lx)*(kx-lx) + (my-ly)*(ky-ly) < 0) {
		if (sqrt(pow(mx-lx, 2) + pow(my-ly, 2)) >= d0)
			return Null;
		else
			return L;
	}
	return Edge;
}

double vertexToEdgeDistance(const std::array<double,7> x) {
	// vertices is a vector of vertices stored as 2d coordinates
	// the distance computed is from the first vertex to the line spanning the
	// other vertices
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	const double kld = sqrt(pow(lx - kx, 2) + pow(ly-ky, 2));
	//const double a = kx*(my-ly) + ky*(lx-mx) + mx*ly - my*lx;
	const double a = kx*(my-ly) + ky*(lx-mx) + mx*ly - my*lx;
	const double dist = abs(a / kld);
	return dist;
}

double vertexToLineDistancePotential(const std::array<double,7> x) {
	// vertices is a vector of vertices stored as 2d coordinates
	// the distance computed is from the first vertex to the line spanning the
	// other vertices
	const double d0 = x[6];
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	const double kld = sqrt(pow(lx - kx, 2) + pow(ly-ky, 2));
	const double a = kx*(my-ly) + ky*(lx-mx) + mx*ly - my*lx;
	const double dist = abs(a / kld);

	return -pow(dist-d0, 2) * log(dist/d0);
}

double vertexToKDistancePotential(const std::array<double,7> x) {
	// vertices is a vector of vertices stored as 2d coordinates
	// the distance computed is from the first vertex to the line spanning the
	// other vertices
	const double d0 = x[6];
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	const double kmd = sqrt(pow(mx - kx, 2) + pow(my-ky, 2));
	return -pow(kmd-d0, 2) * log(kmd/d0);
}

double vertexToLDistancePotential(const std::array<double,7> x) {
	// vertices is a vector of vertices stored as 2d coordinates
	// the distance computed is from the first vertex to the line spanning the
	// other vertices
	const double d0 = x[6];
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	const double kld = sqrt(pow(mx - lx, 2) + pow(my-ly, 2));
	return -pow(kld-d0, 2) * log(kld/d0);
}

double vertexToEdgeDistancePotential(const std::array<double,7> x) {
	switch (vertexEdgeMode(x)) {
		case Edge:
			return vertexToLineDistancePotential(x);
		case K:
			return vertexToKDistancePotential(x);
		case L:
			return vertexToLDistancePotential(x);
		case Null:
			return 0;
	}
}

void vertexToLineDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x) {
	double * const y = jacobian.data();
	double v[10];  // temporary variables
	v[0] = x[1] - x[5];
	v[1] = x[4] - x[0];
	v[2] = x[4] - x[2];
	v[3] = x[5] - x[3];
	v[4] = sqrt(v[2] * v[2] + v[3] * v[3]);
	v[5] = (x[2] * v[0] + x[3] * v[1] + x[0] * x[5] - x[1] * x[4]) / v[4];
	v[6] = fabs(v[5]);
	v[7] = v[6] / x[6];
	v[8] = 0 - log(v[7]);
	v[6] = v[6] - x[6];
	v[6] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[8] * v[6] + v[8] * v[6] + (0 - v[6] * v[6]) * 1 / v[7] * 1 / x[6]) * 1 / v[4];
	v[8] = v[6] * x[3];
	y[0] = 0 - v[8] + v[6] * x[5];
	v[7] = v[6] * x[2];
	v[9] = 0 - v[6];
	y[1] = v[7] + v[9] * x[4];
	v[5] = ((0 - v[6] * v[5]) * 1 / v[4]) / 2.;
	v[2] = v[5] * v[2] + v[5] * v[2];
	y[2] = v[6] * v[0] - v[2];
	v[5] = v[5] * v[3] + v[5] * v[3];
	y[3] = v[6] * v[1] - v[5];
	y[4] = v[8] + v[9] * x[1] + v[2];
	y[5] = 0 - v[7] + v[6] * x[0] + v[5];
}

void vertexToKDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x) {
	double * const y = jacobian.data();
	double v[6];  // temporary variables
	v[0] = x[0] - x[2];
	v[1] = x[1] - x[3];
	v[2] = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[3] = v[2] / x[6];
	v[4] = 0 - log(v[3]);
	v[5] = v[2] - x[6];
	v[5] = ((v[4] * v[5] + v[4] * v[5] + (0 - v[5] * v[5]) * 1 / v[3] * 1 / x[6]) * 1 / v[2]) / 2.;
	y[0] = v[5] * v[0] + v[5] * v[0];
	y[1] = v[5] * v[1] + v[5] * v[1];
	y[2] = 0 - y[0];
	y[3] = 0 - y[1];
	// dependent variables without operations
	y[4] = 0;
	y[5] = 0;
	y[6] = 0;
}

void vertexToLDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x) {
	double * const y = jacobian.data();
	double v[6];  // temporary variables
	v[0] = x[0] - x[2];
	v[1] = x[1] - x[3];
	v[2] = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[3] = v[2] / x[6];
	v[4] = 0 - log(v[3]);
	v[5] = v[2] - x[6];
	v[5] = ((v[4] * v[5] + v[4] * v[5] + (0 - v[5] * v[5]) * 1 / v[3] * 1 / x[6]) * 1 / v[2]) / 2.;
	y[0] = v[5] * v[0] + v[5] * v[0];
	y[1] = v[5] * v[1] + v[5] * v[1];
	y[2] = 0 - y[0];
	y[3] = 0 - y[1];
	// dependent variables without operations
	y[4] = 0;
	y[5] = 0;
	y[6] = 0;
}

void vertexToEdgeDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x) {
	for (double& j: jacobian)
		j = 0;
	switch (vertexEdgeMode(x)) {
		case Edge:
			vertexToLineDistancePotentialJacobian(jacobian, x);
			break;
		case K:
			vertexToKDistancePotentialJacobian(jacobian, x);
			break;
		case L:
			vertexToLDistancePotentialJacobian(jacobian, x);
			break;
		case Null:
			break;
	}
}

bool vertexToEdgeDistanceValid(const std::array<double,7>& x) {
	const double d0 = x[6];
	const double mx = x[0], my = x[1], kx = x[2], ky = x[3], lx = x[4], ly = x[5];
	// now check if it's in the range
	if ((mx - kx)*(lx-kx) + (my-ky)*(ly-ky) < 0)
		return false;
	if ((mx - lx)*(kx-lx) + (my-ly)*(ky-ly) < 0)
		return false;

	if (vertexToEdgeDistance(x) > d0)
		return false;
	return true;
}

void vertexToLineDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x);
void vertexToLDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x);
void vertexToKDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x);

void vertexToEdgeDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x) {
	for (double& h: hessian)
		h = 0;
	switch (vertexEdgeMode(x)) {
		case Edge:
			vertexToLineDistancePotentialHessian(hessian, x);
			break;
		case K:
			vertexToKDistancePotentialHessian(hessian, x);
			break;
		case L:
			vertexToLDistancePotentialHessian(hessian, x);
			break;
		case Null:
			break;
	}
}

void vertexToLineDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x) {
	std::array<double, 7*7> hessian_temp;
	double * const y = hessian_temp.data();
	double v[21];  // temporary variables

	v[0] = x[1] - x[5];
	v[1] = x[4] - x[0];
	v[2] = x[4] - x[2];
	v[3] = x[5] - x[3];
	v[4] = sqrt(v[2] * v[2] + v[3] * v[3]);
	v[5] = (x[2] * v[0] + x[3] * v[1] + x[0] * x[5] - x[1] * x[4]) / v[4];
	v[6] = fabs(v[5]);
	v[7] = v[6] / x[6];
	v[8] = log(v[7]);
	v[9] = 0 - v[8];
	v[10] = (x[3] * -1 + x[5]) / v[4];
	v[11] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[10];
	v[12] = (v[11] / x[6]) / v[7];
	v[13] = 0 - v[12];
	v[6] = v[6] - x[6];
	v[14] = 0 - v[6] * v[6];
	v[15] = 1 / v[7];
	v[16] = v[14] * v[15];
	v[17] = 1 / x[6];
	v[18] = 1 / v[4];
	v[15] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[9] * v[11] + v[9] * v[11] + v[13] * v[6] + v[13] * v[6] + (0 - v[16] * v[12] + (-(v[11] * v[6] + v[6] * v[11])) * v[15]) * v[17]) * v[18];
	v[13] = v[15] * x[3];
	y[0] = 0 - v[13] + v[15] * x[5];
	v[12] = v[15] * x[2];
	v[11] = 0 - v[15];
	y[1] = v[12] + v[11] * x[4];
	v[18] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[9] * v[6] + v[9] * v[6] + v[16] * v[17]) * v[18];
	v[10] = ((0 - v[18] * v[10] - v[15] * v[5]) * 1 / v[4]) / 2.;
	v[17] = v[10] * v[2] + v[10] * v[2];
	y[2] = v[15] * v[0] - v[17];
	v[10] = v[10] * v[3] + v[10] * v[3];
	y[3] = v[18] * -1 + v[15] * v[1] - v[10];
	y[4] = v[13] + v[11] * x[1] + v[17];
	y[5] = 0 - v[12] + v[18] + v[15] * x[0] + v[10];
	v[10] = 0 - v[8];
	v[18] = (x[2] - x[4]) / v[4];
	v[12] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[18];
	v[15] = (v[12] / x[6]) / v[7];
	v[17] = 0 - v[15];
	v[11] = 1 / v[7];
	v[13] = v[14] * v[11];
	v[16] = 1 / x[6];
	v[9] = 1 / v[4];
	v[11] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[10] * v[12] + v[10] * v[12] + v[17] * v[6] + v[17] * v[6] + (0 - v[13] * v[15] + (-(v[12] * v[6] + v[6] * v[12])) * v[11]) * v[16]) * v[9];
	v[17] = v[11] * x[2];
	v[15] = 0 - v[11];
	y[8] = v[17] + v[15] * x[4];
	v[9] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[10] * v[6] + v[10] * v[6] + v[13] * v[16]) * v[9];
	v[18] = ((0 - v[9] * v[18] - v[11] * v[5]) * 1 / v[4]) / 2.;
	v[16] = v[18] * v[2] + v[18] * v[2];
	y[9] = v[9] + v[11] * v[0] - v[16];
	v[18] = v[18] * v[3] + v[18] * v[3];
	y[10] = v[11] * v[1] - v[18];
	y[11] = v[11] * x[3] + 0 - v[9] + v[15] * x[1] + v[16];
	y[12] = 0 - v[17] + v[11] * x[0] + v[18];
	v[18] = 0 - v[8];
	v[17] = ((-1 * v[2] + v[2] * -1) / 2.) / v[4];
	v[11] = (v[0] - v[5] * v[17]) / v[4];
	v[16] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[11];
	v[9] = (v[16] / x[6]) / v[7];
	v[15] = 0 - v[9];
	v[13] = 1 / v[7];
	v[10] = v[14] * v[13];
	v[12] = 1 / x[6];
	v[19] = 1 / v[4];
	v[20] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[18] * v[6] + v[18] * v[6] + v[10] * v[12]) * v[19];
	v[19] = (((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[18] * v[16] + v[18] * v[16] + v[15] * v[6] + v[15] * v[6] + (0 - v[10] * v[9] + (-(v[16] * v[6] + v[6] * v[16])) * v[13]) * v[12]) - v[20] * v[17]) * v[19];
	v[12] = 1 / v[4];
	v[10] = (0 - v[20] * v[5]) * v[12];
	v[13] = v[10] / 2.;
	v[10] = ((0 - v[20] * v[11] - v[19] * v[5] - v[10] * v[17]) * v[12]) / 2.;
	v[13] = v[13] * -1 + v[13] * -1 + v[10] * v[2] + v[10] * v[2];
	y[16] = v[19] * v[0] - v[13];
	v[10] = v[10] * v[3] + v[10] * v[3];
	y[17] = v[19] * v[1] - v[10];
	y[18] = v[19] * x[3] + (0 - v[19]) * x[1] + v[13];
	y[19] = 0 - (v[20] + v[19] * x[2]) + v[19] * x[0] + v[10];
	v[10] = 0 - v[8];
	v[19] = ((-1 * v[3] + v[3] * -1) / 2.) / v[4];
	v[20] = (v[1] - v[5] * v[19]) / v[4];
	v[13] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[20];
	v[0] = (v[13] / x[6]) / v[7];
	v[12] = 0 - v[0];
	v[11] = 1 / v[7];
	v[17] = v[14] * v[11];
	v[15] = 1 / x[6];
	v[9] = 1 / v[4];
	v[16] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[10] * v[6] + v[10] * v[6] + v[17] * v[15]) * v[9];
	v[9] = (((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[10] * v[13] + v[10] * v[13] + v[12] * v[6] + v[12] * v[6] + (0 - v[17] * v[0] + (-(v[13] * v[6] + v[6] * v[13])) * v[11]) * v[15]) - v[16] * v[19]) * v[9];
	v[15] = 1 / v[4];
	v[17] = (0 - v[16] * v[5]) * v[15];
	v[11] = v[17] / 2.;
	v[17] = ((0 - v[16] * v[20] - v[9] * v[5] - v[17] * v[19]) * v[15]) / 2.;
	v[11] = v[11] * -1 + v[11] * -1 + v[17] * v[3] + v[17] * v[3];
	y[24] = v[9] * v[1] - v[11];
	y[25] = v[16] + v[9] * x[3] + (0 - v[9]) * x[1] + v[17] * v[2] + v[17] * v[2];
	y[26] = 0 - v[9] * x[2] + v[9] * x[0] + v[11];
	v[11] = 0 - v[8];
	v[9] = ((v[2] + v[2]) / 2.) / v[4];
	v[17] = (x[3] - x[1] - v[5] * v[9]) / v[4];
	v[16] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[17];
	v[1] = (v[16] / x[6]) / v[7];
	v[15] = 0 - v[1];
	v[20] = 1 / v[7];
	v[19] = v[14] * v[20];
	v[12] = 1 / x[6];
	v[0] = 1 / v[4];
	v[13] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[11] * v[6] + v[11] * v[6] + v[19] * v[12]) * v[0];
	v[0] = (((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[11] * v[16] + v[11] * v[16] + v[15] * v[6] + v[15] * v[6] + (0 - v[19] * v[1] + (-(v[16] * v[6] + v[6] * v[16])) * v[20]) * v[12]) - v[13] * v[9]) * v[0];
	v[12] = 1 / v[4];
	v[19] = (0 - v[13] * v[5]) * v[12];
	v[20] = v[19] / 2.;
	v[19] = ((0 - v[13] * v[17] - v[0] * v[5] - v[19] * v[9]) * v[12]) / 2.;
	y[32] = v[0] * x[3] + (0 - v[0]) * x[1] + v[20] + v[20] + v[19] * v[2] + v[19] * v[2];
	y[33] = 0 - v[0] * x[2] + v[0] * x[0] + v[19] * v[3] + v[19] * v[3];
	v[8] = 0 - v[8];
	v[19] = ((v[3] + v[3]) / 2.) / v[4];
	v[0] = (x[2] * -1 + x[0] - v[5] * v[19]) / v[4];
	v[20] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * v[0];
	v[2] = (v[20] / x[6]) / v[7];
	v[12] = 0 - v[2];
	v[7] = 1 / v[7];
	v[14] = v[14] * v[7];
	v[13] = 1 / x[6];
	v[17] = 1 / v[4];
	v[9] = ((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[8] * v[6] + v[8] * v[6] + v[14] * v[13]) * v[17];
	v[17] = (((v[5] > 0?1:(v[5] < 0?-1:0))) * (v[8] * v[20] + v[8] * v[20] + v[12] * v[6] + v[12] * v[6] + (0 - v[14] * v[2] + (-(v[20] * v[6] + v[6] * v[20])) * v[7]) * v[13]) - v[9] * v[19]) * v[17];
	v[4] = 1 / v[4];
	v[13] = (0 - v[9] * v[5]) * v[4];
	v[14] = v[13] / 2.;
	v[13] = ((0 - v[9] * v[0] - v[17] * v[5] - v[13] * v[19]) * v[4]) / 2.;
	y[40] = 0 - v[17] * x[2] + v[17] * x[0] + v[14] + v[14] + v[13] * v[3] + v[13] * v[3];
	// variable duplicates: 15
	y[7] = y[1];
	y[14] = y[2];
	y[15] = y[9];
	y[21] = y[3];
	y[22] = y[10];
	y[23] = y[17];
	y[28] = y[4];
	y[29] = y[11];
	y[30] = y[18];
	y[31] = y[25];
	y[35] = y[5];
	y[36] = y[12];
	y[37] = y[19];
	y[38] = y[26];
	y[39] = y[33];


	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian[i*6+j] = hessian_temp[i*7+j];
		}
	}
}

void vertexToLDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x) {
	std::array<double, 7*7> hessian_temp;
	double * const y = hessian_temp.data();
	double v[21];  // temporary variables
	v[0] = x[0] - x[2];
	v[1] = x[1] - x[3];
	v[2] = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[3] = v[2] / x[6];
	v[4] = log(v[3]);
	v[5] = 0 - v[4];
	v[6] = v[2] - x[6];
	v[7] = 0 - v[6] * v[6];
	v[8] = 1 / v[3];
	v[9] = v[7] * v[8];
	v[10] = 1 / x[6];
	v[11] = 1 / v[2];
	v[12] = (v[5] * v[6] + v[5] * v[6] + v[9] * v[10]) * v[11];
	v[13] = v[12] / 2.;
	v[14] = ((v[0] + v[0]) / 2.) / v[2];
	v[15] = (v[14] / x[6]) / v[3];
	v[16] = 0 - v[15];
	v[16] = ((v[5] * v[14] + v[5] * v[14] + v[16] * v[6] + v[16] * v[6] + (0 - v[9] * v[15] + (-(v[14] * v[6] + v[6] * v[14])) * v[8]) * v[10] - v[12] * v[14]) * v[11]) / 2.;
	y[0] = v[13] + v[13] + v[16] * v[0] + v[16] * v[0];
	y[1] = v[16] * v[1] + v[16] * v[1];
	y[2] = 0 - y[0];
	y[3] = 0 - y[1];
	v[16] = 0 - v[4];
	v[13] = 1 / v[3];
	v[15] = v[7] * v[13];
	v[14] = 1 / x[6];
	v[12] = 1 / v[2];
	v[11] = (v[16] * v[6] + v[16] * v[6] + v[15] * v[14]) * v[12];
	v[10] = v[11] / 2.;
	v[9] = ((v[1] + v[1]) / 2.) / v[2];
	v[8] = (v[9] / x[6]) / v[3];
	v[5] = 0 - v[8];
	v[5] = ((v[16] * v[9] + v[16] * v[9] + v[5] * v[6] + v[5] * v[6] + (0 - v[15] * v[8] + (-(v[9] * v[6] + v[6] * v[9])) * v[13]) * v[14] - v[11] * v[9]) * v[12]) / 2.;
	y[8] = v[10] + v[10] + v[5] * v[1] + v[5] * v[1];
	y[9] = 0 - (v[5] * v[0] + v[5] * v[0]);
	y[10] = 0 - y[8];
	v[5] = 0 - v[4];
	v[10] = 1 / v[3];
	v[8] = v[7] * v[10];
	v[9] = 1 / x[6];
	v[11] = 1 / v[2];
	v[12] = (v[5] * v[6] + v[5] * v[6] + v[8] * v[9]) * v[11];
	v[14] = v[12] / 2.;
	v[15] = ((-1 * v[0] + v[0] * -1) / 2.) / v[2];
	v[13] = (v[15] / x[6]) / v[3];
	v[16] = 0 - v[13];
	v[16] = ((v[5] * v[15] + v[5] * v[15] + v[16] * v[6] + v[16] * v[6] + (0 - v[8] * v[13] + (-(v[15] * v[6] + v[6] * v[15])) * v[10]) * v[9] - v[12] * v[15]) * v[11]) / 2.;
	y[16] = 0 - (v[14] * -1 + v[14] * -1 + v[16] * v[0] + v[16] * v[0]);
	y[17] = 0 - (v[16] * v[1] + v[16] * v[1]);
	v[4] = 0 - v[4];
	v[16] = 1 / v[3];
	v[7] = v[7] * v[16];
	v[14] = 1 / x[6];
	v[0] = 1 / v[2];
	v[13] = (v[4] * v[6] + v[4] * v[6] + v[7] * v[14]) * v[0];
	v[15] = v[13] / 2.;
	v[2] = ((-1 * v[1] + v[1] * -1) / 2.) / v[2];
	v[3] = (v[2] / x[6]) / v[3];
	v[12] = 0 - v[3];
	v[12] = ((v[4] * v[2] + v[4] * v[2] + v[12] * v[6] + v[12] * v[6] + (0 - v[7] * v[3] + (-(v[2] * v[6] + v[6] * v[2])) * v[16]) * v[14] - v[13] * v[2]) * v[0]) / 2.;
	y[24] = 0 - (v[15] * -1 + v[15] * -1 + v[12] * v[1] + v[12] * v[1]);
	// variable duplicates: 6
	y[7] = y[1];
	y[14] = y[2];
	y[15] = y[9];
	y[21] = y[3];
	y[22] = y[10];
	y[23] = y[17];
	// dependent variables without operations

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian[i*6+j] = hessian_temp[i*7+j];
		}
	}
}

void vertexToKDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x) {
	std::array<double, 7*7> hessian_temp;
	double * const y = hessian_temp.data();
	double v[21];  // temporary variables
	v[0] = x[0] - x[2];
	v[1] = x[1] - x[3];
	v[2] = sqrt(v[0] * v[0] + v[1] * v[1]);
	v[3] = v[2] / x[6];
	v[4] = log(v[3]);
	v[5] = 0 - v[4];
	v[6] = v[2] - x[6];
	v[7] = 0 - v[6] * v[6];
	v[8] = 1 / v[3];
	v[9] = v[7] * v[8];
	v[10] = 1 / x[6];
	v[11] = 1 / v[2];
	v[12] = (v[5] * v[6] + v[5] * v[6] + v[9] * v[10]) * v[11];
	v[13] = v[12] / 2.;
	v[14] = ((v[0] + v[0]) / 2.) / v[2];
	v[15] = (v[14] / x[6]) / v[3];
	v[16] = 0 - v[15];
	v[16] = ((v[5] * v[14] + v[5] * v[14] + v[16] * v[6] + v[16] * v[6] + (0 - v[9] * v[15] + (-(v[14] * v[6] + v[6] * v[14])) * v[8]) * v[10] - v[12] * v[14]) * v[11]) / 2.;
	y[0] = v[13] + v[13] + v[16] * v[0] + v[16] * v[0];
	y[1] = v[16] * v[1] + v[16] * v[1];
	y[2] = 0 - y[0];
	y[3] = 0 - y[1];
	v[16] = 0 - v[4];
	v[13] = 1 / v[3];
	v[15] = v[7] * v[13];
	v[14] = 1 / x[6];
	v[12] = 1 / v[2];
	v[11] = (v[16] * v[6] + v[16] * v[6] + v[15] * v[14]) * v[12];
	v[10] = v[11] / 2.;
	v[9] = ((v[1] + v[1]) / 2.) / v[2];
	v[8] = (v[9] / x[6]) / v[3];
	v[5] = 0 - v[8];
	v[5] = ((v[16] * v[9] + v[16] * v[9] + v[5] * v[6] + v[5] * v[6] + (0 - v[15] * v[8] + (-(v[9] * v[6] + v[6] * v[9])) * v[13]) * v[14] - v[11] * v[9]) * v[12]) / 2.;
	y[8] = v[10] + v[10] + v[5] * v[1] + v[5] * v[1];
	y[9] = 0 - (v[5] * v[0] + v[5] * v[0]);
	y[10] = 0 - y[8];
	v[5] = 0 - v[4];
	v[10] = 1 / v[3];
	v[8] = v[7] * v[10];
	v[9] = 1 / x[6];
	v[11] = 1 / v[2];
	v[12] = (v[5] * v[6] + v[5] * v[6] + v[8] * v[9]) * v[11];
	v[14] = v[12] / 2.;
	v[15] = ((-1 * v[0] + v[0] * -1) / 2.) / v[2];
	v[13] = (v[15] / x[6]) / v[3];
	v[16] = 0 - v[13];
	v[16] = ((v[5] * v[15] + v[5] * v[15] + v[16] * v[6] + v[16] * v[6] + (0 - v[8] * v[13] + (-(v[15] * v[6] + v[6] * v[15])) * v[10]) * v[9] - v[12] * v[15]) * v[11]) / 2.;
	y[16] = 0 - (v[14] * -1 + v[14] * -1 + v[16] * v[0] + v[16] * v[0]);
	y[17] = 0 - (v[16] * v[1] + v[16] * v[1]);
	v[4] = 0 - v[4];
	v[16] = 1 / v[3];
	v[7] = v[7] * v[16];
	v[14] = 1 / x[6];
	v[0] = 1 / v[2];
	v[13] = (v[4] * v[6] + v[4] * v[6] + v[7] * v[14]) * v[0];
	v[15] = v[13] / 2.;
	v[2] = ((-1 * v[1] + v[1] * -1) / 2.) / v[2];
	v[3] = (v[2] / x[6]) / v[3];
	v[12] = 0 - v[3];
	v[12] = ((v[4] * v[2] + v[4] * v[2] + v[12] * v[6] + v[12] * v[6] + (0 - v[7] * v[3] + (-(v[2] * v[6] + v[6] * v[2])) * v[16]) * v[14] - v[13] * v[2]) * v[0]) / 2.;
	y[24] = 0 - (v[15] * -1 + v[15] * -1 + v[12] * v[1] + v[12] * v[1]);
	// variable duplicates: 6
	y[7] = y[1];
	y[14] = y[2];
	y[15] = y[9];
	y[21] = y[3];
	y[22] = y[10];
	y[23] = y[17];
	// dependent variables without operations
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian[i*6+j] = hessian_temp[i*7+j];
		}
	}
}
