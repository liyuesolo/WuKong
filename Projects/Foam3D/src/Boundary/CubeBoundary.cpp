#include "../../include/Boundary/CubeBoundary.h"

CubeBoundary::CubeBoundary(double r, const VectorXi &free_) : Boundary(VectorXT::Constant(1, r), free_) {
    MatrixXi btri(12, 3);
    btri << 0, 2, 1,
            2, 3, 1,
            0, 1, 4,
            1, 5, 4,
            0, 4, 2,
            4, 6, 2,
            4, 5, 6,
            5, 7, 6,
            1, 3, 5,
            3, 7, 5,
            2, 6, 3,
            6, 7, 3;

    initialize(8, btri);
}

bool CubeBoundary::checkValid() {
    return p(0) > 0;
}

void CubeBoundary::computeVertices() {
    double a = p(0);

    for (int i = 0; i < 8; i++) {
        double sgnx = (i % 2 < 1 ? -1 : 1);
        double sgny = (i % 4 < 2 ? -1 : 1);
        double sgnz = (i % 8 < 4 ? -1 : 1);

        v[i].pos = {pow(a, 1.0 / 3.0) * sgnx, pow(a, 1.0 / 3.0) * sgny, pow(a, 1.0 / 3.0) * sgnz};
        addGradientEntry(i, 0, 0, 1.0 / 3.0 * pow(a, -2.0 / 3.0) * sgnx);
        addGradientEntry(i, 1, 0, 1.0 / 3.0 * pow(a, -2.0 / 3.0) * sgny);
        addGradientEntry(i, 2, 0, 1.0 / 3.0 * pow(a, -2.0 / 3.0) * sgnz);
        addHessianEntry(i, 0, 0, 0, -2.0 / 9.0 * pow(a, -5.0 / 3.0) * sgnx);
        addHessianEntry(i, 1, 0, 0, -2.0 / 9.0 * pow(a, -5.0 / 3.0) * sgny);
        addHessianEntry(i, 2, 0, 0, -2.0 / 9.0 * pow(a, -5.0 / 3.0) * sgnz);
    }
}

double CubeBoundary::computeEnergy() {
    double a = p(0);
    return (a - 2) * (a - 2);
}

void CubeBoundary::computeEnergyGradient(VectorXT &gradient) {
    double a = p(0);

    Eigen::SparseVector<double> grad_sp(nfree);
    addEnergyGradientEntry(grad_sp, 0, 2 * (a - 2));

    gradient = grad_sp;
}

void CubeBoundary::computeEnergyHessian(MatrixXT &hessian) {
    double a = p(0);

    Eigen::SparseMatrix<double> hess_sp(nfree, nfree);
    addEnergyHessianEntry(hess_sp, 0, 0, 2);

    hessian = hess_sp;
}

