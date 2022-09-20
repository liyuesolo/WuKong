#include <vector>
#include <cmath>

#include "../include/constants.hpp"

using std::abs;


template <typename T>
// vertices are stored as x0, y0, x1, ..., y2
T triangleArea(const std::vector<T>& x) {
	// vertices is a vector of vertices stored as 2d coordinates
	T a_h = x[4] - x[2], a_v = x[5] - x[3];
	T b_h = x[4] - x[0], b_v = x[5] - x[1];
	T c_h = x[2] - x[0], c_v = x[3] - x[1];

	T area = abs(a_h * b_v - b_h * a_v);
	return area;
}

template <typename T>
void triangleAreaJacobian(std::vector<T>& jacobian, const std::vector<T>& vertices) {
	double v[6];
	std::vector<double>& y = jacobian;
	std::vector<double>& x = vertices;
	v[0] = x[4] - x[2];
	v[1] = x[5] - x[1];
	v[2] = x[4] - x[0];
	v[3] = x[5] - x[3];
	v[4] = v[0] * v[1] - v[2] * v[3];
	v[4] = (v[4] > 0?1:(v[4] < 0?-1:0));
	v[5] = 0 - v[4];
	v[3] = v[5] * v[3];
	y[0] = 0 - v[3];
	v[0] = v[4] * v[0];
	y[1] = 0 - v[0];
	v[4] = v[4] * v[1];
	y[2] = 0 - v[4];
	v[5] = v[5] * v[2];
	y[3] = 0 - v[5];
	y[4] = v[3] + v[4];
	y[5] = v[0] + v[5];

}

template <typename T>
void triangleAreaHessian(std::vector<T>& hessian, const std::vector<T>& vertices) {
	double v[4];
	std::vector<double>& y = hessian;
	std::vector<double>& x = vertices;
	v[0] = (x[4] - x[2]) * (x[5] - x[1]) - (x[4] - x[0]) * (x[5] - x[3]);
	y[27] = (0 - ((v[0] > 0?1:(v[0] < 0?-1:0)))) * -1;
	y[3] = 0 - y[27];
	v[1] = (v[0] > 0?1:(v[0] < 0?-1:0));
	v[2] = 0 - v[1];
	y[5] = 0 - v[2];
	y[32] = ((v[0] > 0?1:(v[0] < 0?-1:0))) * -1;
	y[8] = 0 - y[32];
	v[3] = (v[0] > 0?1:(v[0] < 0?-1:0));
	y[10] = 0 - v[3];
	y[25] = ((v[0] > 0?1:(v[0] < 0?-1:0))) * -1;
	y[13] = 0 - y[25];
	y[17] = 0 - v[1];
	y[30] = (0 - ((v[0] > 0?1:(v[0] < 0?-1:0)))) * -1;
	y[18] = 0 - y[30];
	v[0] = 0 - v[3];
	y[22] = 0 - v[0];
	y[29] = v[2] + v[1];
	y[34] = v[3] + v[0];
	// dependent variables without operations
	y[0] = 0;
	y[1] = 0;
	y[2] = 0;
	y[4] = 0;
	y[6] = 0;
	y[7] = 0;
	y[9] = 0;
	y[11] = 0;
	y[12] = 0;
	y[14] = 0;
	y[15] = 0;
	y[16] = 0;
	y[19] = 0;
	y[20] = 0;
	y[21] = 0;
	y[23] = 0;
	y[24] = 0;
	y[26] = 0;
	y[28] = 0;
	y[31] = 0;
	y[33] = 0;
	y[35] = 0;

}
