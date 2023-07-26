#include <igl/loop.h>
#include "../../include/Boundary/SubdivisionMeshBoundary.h"

SubdivisionMeshBoundary::SubdivisionMeshBoundary(MatrixXT &v_, MatrixXi &f_, const VectorXi &free_,
                                                 int nSubdivision) : MeshBoundary(v_, f_, free_), nSub(nSubdivision) {
    MatrixXi f2;
    igl::loop(v_.rows(), f_, S, f2);
    v_ = S * v_;
    f_ = f2;
    for (int i = 0; i < nSub - 1; i++) {
        Eigen::SparseMatrix<double> S2;
        igl::loop(v_.rows(), f_, S2, f2);
        v_ = S2 * v_;
        S = S2 * S;
        f_ = f2;
    }
    initialize(v_.rows(), f_);
}

void SubdivisionMeshBoundary::computeVertices() {
    MatrixXT vp = Eigen::Map<MatrixXT>(p.data(), 3, np / 3).transpose();
    MatrixXT vs = S * vp;

    for (int i = 0; i < v.size(); i++) {
        v[i].pos = vs.row(i);
    }

    for (int k = 0; k < S.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            for (int j = 0; j < 3; j++) {
                addGradientEntry(it.row(), j, it.col() * 3 + j, it.value());
            }
        }
    }
}

