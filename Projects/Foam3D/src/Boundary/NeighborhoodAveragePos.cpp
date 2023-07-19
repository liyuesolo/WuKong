#include "../../include/Boundary/NeighborhoodAveragePos.h"

// @formatter:off
void NeighborhoodAveragePos::getValue(NeighborhoodValue &value) const {
    int n = value.v.size();
    value.value = 0;

    for (int i = 0; i < n; i++) {
        value.value += value.v[i](coord) / n;
    }
}

void NeighborhoodAveragePos::getGradient(NeighborhoodValue &value) const {
    int n = value.v.size();
    int nvars = 3 * n + 3;
    value.gradient = VectorXT::Zero(nvars);

    for (int i = 0; i < n; i++) {
        value.gradient(i * 3 + coord) = 1.0 / n;
    }
}

void NeighborhoodAveragePos::getHessian(NeighborhoodValue &value) const {
    int n = value.v.size();
    int nvars = 3 * n + 3;
    value.hessian = MatrixXT::Zero(nvars, nvars);
}
