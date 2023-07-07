#include "../../include/Boundary/Boundary.h"

Boundary::Boundary(const VectorXT &p_, const VectorXi &free_) {
    p = p_;
    free_idx = free_;

    np = p.rows();
    nfree = free_idx.rows();

    free_map = -1 * VectorXi::Ones(np);
    for (int i = 0; i < nfree; i++) {
        free_map(free_idx(i)) = i;
    }
}

void Boundary::initialize(int nv, const MatrixXi &f_) {
    v.resize(nv);
    for (BoundaryVertex &bv: v) {
        bv.grad = MatrixXT::Zero(3, nfree);
        for (int j = 0; j < 3; j++) {
            bv.hess[j] = MatrixXT::Zero(nfree, nfree);
        }
    }

    int nf = f_.rows();
    f.resize(nf);
    for (int i = 0; i < nf; i++) {
        f[i].vertices = f_.row(i);
    }
}

void Boundary::compute(const VectorXT &p_free) {
    for (int i = 0; i < nfree; i++) {
        p(free_idx(i)) = p_free(i);
    }

    computeVertices();
}

VectorXT Boundary::get_p_free() {
    VectorXT ret(nfree);
    for (int i = 0; i < nfree; i++) {
        ret(i) = p(free_idx(i));
    }
    return ret;
}
