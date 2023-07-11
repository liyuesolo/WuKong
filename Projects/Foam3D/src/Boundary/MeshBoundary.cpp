#include <igl/fast_find_self_intersections.h>
#include "../../include/Boundary/MeshBoundary.h"

MeshBoundary::MeshBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_) : Boundary(
        Eigen::Map<VectorXT>(MatrixXT(v_.transpose()).data(), v_.size()), free_) {
    initialize(v_.rows(), f_);
}

bool MeshBoundary::checkValid() {
    MatrixXT V(v.size(), 3);
    for (int i = 0; i < v.size(); i++) {
        V.row(i) = v[i].pos;
    }
    MatrixXi F(f.size(), 3);
    for (int i = 0; i < f.size(); i++) {
        F.row(i) = f[i].vertices;
    }
    MatrixXi I;

    return !igl::fast_find_self_intersections(V, F, I);
}

void MeshBoundary::computeVertices() {
    for (int i = 0; i < v.size(); i++) {
        v[i].pos = p.segment<3>(i * 3);

        for (int j = 0; j < 3; j++) {
            addGradientEntry(i, j, i * 3 + j, 1.0);
        }
    }
}

