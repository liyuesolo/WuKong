#include "../../include/Boundary/NeighborhoodLaplacian.h"

// @formatter:off
void NeighborhoodLaplacian::getValue(NeighborhoodValue &value) const {
    value.value = 0;
    NeighborhoodValue averagePosValues[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        averagePosFunc[i].getValue(averagePosValues[i]);
        value.value += 0.5 * pow(value.c(i) - averagePosValues[i].value, 2.0);
    }
}

void NeighborhoodLaplacian::getGradient(NeighborhoodValue &value) const {
    int n = value.v.size();
    int nvars = 3 * n + 3;
    value.gradient = VectorXT::Zero(nvars);

    NeighborhoodValue averagePosValues[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        averagePosFunc[i].getValue(averagePosValues[i]);
        averagePosFunc[i].getGradient(averagePosValues[i]);
        VectorXT siteGrad = VectorXT::Zero(nvars);
        siteGrad(siteGrad.rows() - 3 + i) = 1;
        value.gradient += (value.c(i) - averagePosValues[i].value) * (siteGrad - averagePosValues[i].gradient);
    }
}

void NeighborhoodLaplacian::getHessian(NeighborhoodValue &value) const {
    int n = value.v.size();
    int nvars = 3 * n + 3;
    value.hessian = MatrixXT::Zero(nvars, nvars);

    NeighborhoodValue averagePosValues[3] = {value, value, value};
    for (int i = 0; i < 3; i++) {
        averagePosFunc[i].getValue(averagePosValues[i]);
        averagePosFunc[i].getGradient(averagePosValues[i]);
        // averagePos hessian is zero
        VectorXT siteGrad = VectorXT::Zero(nvars);
        siteGrad(siteGrad.rows() - 3 + i) = 1;
        value.hessian += (siteGrad - averagePosValues[i].gradient) * (siteGrad - averagePosValues[i].gradient).transpose();
    }
}
