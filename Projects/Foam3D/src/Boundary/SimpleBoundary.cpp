#include "../../include/Boundary/SimpleBoundary.h"

SimpleBoundary::SimpleBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_) : Boundary(
        Eigen::Map<VectorXT>(MatrixXT(v_.transpose()).data(), v_.size()), free_) {
    initialize(v_.rows(), f_);
}

bool SimpleBoundary::checkValid() {
    // TODO: Check no self intersections if we're gonna use this boundary type.
    return true;
}

void SimpleBoundary::computeVertices() {
    for (int i = 0; i < v.size(); i++) {
        v[i].pos = p.segment<3>(i * 3);

        for (int j = 0; j < 3; j++) {
            setGradientEntry(i, j, i * 3 + j, 1.0);
        }
    }
}

