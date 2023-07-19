#include "Projects/Foam3D/include/Boundary/BoundaryEnergySphericalBarrier.h"
#include "../../include/Energy/PerTriangleVolume.h"

void BoundaryEnergySphericalBarrier::getValue(Boundary *boundary, double &value) const {
    value = 0;
    for (int i = 0; i < boundary->v.size(); i++) {
        TV3 v = boundary->v[i].pos;
        double rsq = (v - center).squaredNorm();
        double RSQ = pow(radius, 2.0);
        value += eps * pow(RSQ - rsq, exponent);
    }
}

void BoundaryEnergySphericalBarrier::getGradient(Boundary *boundary, VectorXT &gradient) const {
    gradient = VectorXT::Zero(boundary->v.size() * 3);
    for (int i = 0; i < boundary->v.size(); i++) {
        TV3 v = boundary->v[i].pos;
        double rsq = (v - center).squaredNorm();
        double RSQ = pow(radius, 2.0);
        double aaa = -eps * exponent * pow(RSQ - rsq, exponent - 1);
        for (int j = 0; j < 3; j++) {
            gradient(i * 3 + j) = aaa * 2 * (v(j) - center(j));
        }
    }
}

void BoundaryEnergySphericalBarrier::getHessian(Boundary *boundary, MatrixXT &hessian) const {
    hessian = MatrixXT::Zero(boundary->v.size() * 3, boundary->v.size() * 3);

    for (int i = 0; i < boundary->v.size(); i++) {
        TV3 v = boundary->v[i].pos;
        double rsq = (v - center).squaredNorm();
        double RSQ = pow(radius, 2.0);
        for (int j = 0; j < 3; j++) {
            hessian(i * 3 + j, i * 3 + j) +=
                    -eps * exponent * pow(RSQ - rsq, exponent - 1) * 2;
            for (int k = 0; k < 3; k++) {
                hessian(i * 3 + j, i * 3 + k) +=
                        eps * exponent * (exponent - 1) * pow(RSQ - rsq, exponent - 2) * 4 * (v(j) - center(j)) *
                        (v(k) - center(k));
            }
        }
    }
}
