#include "../../include/ImageMatch/CellFunctionImageMatch.h"
#include <iostream>

void CellFunctionImageMatch::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                          const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    double xp = 0, yp = 0;
    for (int i = 0; i < n_pix; i++) {
        xp += cellInfo->border_pix(i * 2 + 0);
        yp += cellInfo->border_pix(i * 2 + 1);
    }
    xp /= n_pix;
    yp /= n_pix;

    value += pow(site(0) - xp,2.0) + pow(site(1) - yp, 2.0);
}

void CellFunctionImageMatch::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                             VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    double xp = 0, yp = 0;
    for (int i = 0; i < n_pix; i++) {
        xp += cellInfo->border_pix(i * 2 + 0);
        yp += cellInfo->border_pix(i * 2 + 1);
    }
    xp /= n_pix;
    yp /= n_pix;

    gradient_c(0) += 2 * (site(0) - xp);
    gradient_c(1) += 2 * (site(1) - yp);
}

void CellFunctionImageMatch::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
                                            const CellInfo *cellInfo) const {
    hessian(0, 0) += 2;
    hessian(1, 1) += 2;
}
