#include "../../include/TrajectoryOpt/TrajectoryOptNLP.h"

#define IDX_C(k, i) (((k - 1) * n_free + i) * dims)
#define IDX_U(k) ((N * n_free * dims) + k * 2)
#define NX_C (N * n_free * dims)
#define NX_U (N * 2)
#define NX (N * n_free * dims + N * 2)
#define NC (n_free * dims)

double TrajectoryOptNLP::eval_f(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;
    TV final_pos = x.segment<2>(IDX_C(N, agent));

    double target_f = (final_pos - target_pos).squaredNorm();
    double input_f = x.segment(IDX_U(0), NX_U).squaredNorm();
    return target_weight * target_f + input_weight * input_f;
}

VectorXd TrajectoryOptNLP::eval_grad_f(const Eigen::VectorXd &x) const {
    VectorXd grad_f = VectorXd::Zero(x.rows());

    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;
    TV final_pos = x.segment<2>(IDX_C(N, agent));

    grad_f.segment<2>(IDX_C(N, agent)) = target_weight * (final_pos - target_pos);
    grad_f.segment(IDX_U(0), NX_U) = input_weight * x.segment(IDX_U(0), NX_U);

    return grad_f;
}

VectorXd TrajectoryOptNLP::eval_g(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0 - v0 * dynamics->h;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    VectorXd force_int = VectorXd::Zero(NX_C);
    VectorXd force_ext = VectorXd::Zero(NX_C);
    for (int k = 0; k < N; k++) {
        force_int.segment(IDX_C(k, 0), NC) = -1.0 * energy->get_dOdc(c_curr.segment(IDX_C(k, 0), NC));
        force_ext.segment<2>(IDX_C(k, agent)) = x.segment<2>(IDX_U(k));
    }

    VectorXd Ma =
            dynamics->M.replicate(N, 1).asDiagonal()
            * (c_curr - 2 * c_prev + c_2prev) / (dynamics->h * dynamics->h);
    VectorXd G = Ma - (force_int + force_ext);

    return G;
}

Eigen::SparseMatrix<double> TrajectoryOptNLP::eval_jac_g(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0 - v0 * dynamics->h;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    Eigen::SparseMatrix<double> jac(NX_C, NX);

    // dG{k}/dc{k} Energy hessians
    for (int k = 0; k < N; k++) {
        Eigen::SparseMatrix<double> d2Edc2 = energy->get_d2Odc2(c_curr.segment(IDX_C(k, 0), NC));
        for (int i = 0; i < d2Edc2.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(d2Edc2, i); it; ++it) {
                jac.coeffRef(it.row() + NC * k, it.col() + NC * k) += it.value();
            }
        }
    }
    // Diagonals
    Eigen::VectorXd Mhh = dynamics->M / (dynamics->h * dynamics->h);
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < NC; i++) {
            int idx = k * NC + i;
            jac.coeffRef(idx, idx) += Mhh(i); //dG{k}/dc{k}
            if (k > 0) jac.coeffRef(idx, idx - NC) += -2 * Mhh(i); //dG{k}/dc{k-1}
            if (k > 1) jac.coeffRef(idx, idx - 2 * NC) += Mhh(i); //dG{k}/dc{k-2}
        }
    }
    // Control input terms
    for (int k = 0; k < N; k++) {
        jac.coeffRef(IDX_C(k, agent), IDX_U(k)) += -1;
        jac.coeffRef(IDX_C(k, agent) + 1, IDX_U(k) + 1) += -1;
    }

    return jac;
}
