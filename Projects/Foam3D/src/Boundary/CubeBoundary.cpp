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

        v[i].pos = {a * sgnx, a * sgny, a * sgnz};
        setGradientEntry(i, 0, 0, sgnx);
        setGradientEntry(i, 1, 0, sgny);
        setGradientEntry(i, 2, 0, sgnz);
    }
}

