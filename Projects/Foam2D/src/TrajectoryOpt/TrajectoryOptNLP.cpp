#include "../../include/TrajectoryOpt/TrajectoryOptNLP.h"

#define IDX_C(k, i) (((k) * n_free + (i)) * dims)
#define IDX_U(k) ((N * n_free * dims) + (k) * 2)
#define NX_C (N * n_free * dims)
#define NX_U (N * 2)
#define NX (N * n_free * dims + N * 2)
#define NC (n_free * dims)

void TrajectoryOptNLP::check_gradients(const Eigen::VectorXd &x) const {
    double eps = 1e-4;

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    double nx = info->trajOpt_N * (info->n_free * dims + 2);
    double ng = info->trajOpt_N * info->n_free * dims;

    VectorXT g = eval_g(x);
    Eigen::SparseMatrix<double> jac = eval_jac_g_sparsematrix(x);
    for (int i = 0; i < nx; i++) {
        VectorXd xp = x;
        xp(i) += eps;
        VectorXT gp = eval_g(xp);
        xp(i) += eps;
        VectorXT gp2 = eval_g(xp);

        for (int j = 0; j < ng; j++) {
            std::cout << "jac_g[" << j << ", " << i << "] g value " << g(j) << " jac value " << (gp(j) - g(j)) / eps
                      << " "
                      << jac.coeff(j, i)
                      << " ratio "
                      << (gp2(j) - g(j) - 2 * eps * jac.coeff(j, i)) << " "
                      << (gp(j) - g(j) - eps * jac.coeff(j, i)) << " "
                      << (gp2(j) - g(j) - 2 * eps * jac.coeff(j, i)) / (gp(j) - g(j) - eps * jac.coeff(j, i))
                      << std::endl;
        }
    }

    double f = eval_f(x);
    VectorXd grad_f = eval_grad_f(x);
    for (int i = 0; i < nx; i++) {
        VectorXd xp = x;
        xp(i) += eps;
        double fp = eval_f(xp);
        xp(i) += eps;
        double fp2 = eval_f(xp);

        std::cout << "f[" << i << "] " << f << " " << fp << " " << fp2 << " " << (fp - f) / eps << " " << grad_f(i)
                  << " " << (fp - f - eps * grad_f(i)) << " " << (fp2 - f - 2 * eps * grad_f(i)) << " "
                  << (fp2 - f - 2 * eps * grad_f(i)) / (fp - f - eps * grad_f(i))
                  << std::endl;
    }
}

double TrajectoryOptNLP::eval_f(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->trajOpt_N;

    TV final_pos = x.segment<2>(IDX_C(N - 1, info->selected));
    TV final_pos2 = x.segment<2>(IDX_C(N - 2, info->selected));

    double target_f = (final_pos - info->selected_target_pos).squaredNorm();
    double velocity_f = (final_pos - final_pos2).squaredNorm() / (info->dynamics_dt * info->dynamics_dt);
    double input_f = x.segment(IDX_U(0), NX_U).squaredNorm();
    return info->trajOpt_target_weight * target_f + info->trajOpt_velocity_weight * velocity_f +
           info->trajOpt_input_weight * input_f;
}

VectorXd TrajectoryOptNLP::eval_grad_f(const Eigen::VectorXd &x) const {
    VectorXd grad_f = VectorXd::Zero(x.rows());

    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->trajOpt_N;

    TV final_pos = x.segment<2>(IDX_C(N - 1, info->selected));
    TV final_pos2 = x.segment<2>(IDX_C(N - 2, info->selected));

    grad_f.segment<2>(IDX_C(N - 1, info->selected)) =
            2 * info->trajOpt_target_weight * (final_pos - info->selected_target_pos) +
            2 * info->trajOpt_velocity_weight * (final_pos - final_pos2) / (info->dynamics_dt * info->dynamics_dt);
    grad_f.segment<2>(IDX_C(N - 2, info->selected)) =
            -2 * info->trajOpt_velocity_weight * (final_pos - final_pos2) / (info->dynamics_dt * info->dynamics_dt);
    grad_f.segment(IDX_U(0), NX_U) = info->trajOpt_input_weight * 2 * x.segment(IDX_U(0), NX_U);

    return grad_f;
}

VectorXd TrajectoryOptNLP::eval_g(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->trajOpt_N;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0 - v0 * info->dynamics_dt;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    VectorXd force_int = VectorXd::Zero(NX_C);
    VectorXd force_ext = VectorXd::Zero(NX_C);
    for (int k = 0; k < N; k++) {
        force_int.segment(IDX_C(k, 0), NC) = -1.0 * energy->get_dOdc(c_curr.segment(IDX_C(k, 0), NC));
        force_ext.segment<2>(IDX_C(k, info->selected)) = x.segment<2>(IDX_U(k));
    }
    force_int -= info->dynamics_eta * (c_curr - c_prev) / info->dynamics_dt;

    VectorXd Ma = info->dynamics_m * (c_curr - 2 * c_prev + c_2prev) / (info->dynamics_dt * info->dynamics_dt);
    VectorXd G = Ma - (force_int + force_ext);

    return G;
}

Eigen::SparseMatrix<double> TrajectoryOptNLP::eval_jac_g_sparsematrix(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->trajOpt_N;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0 - v0 * info->dynamics_dt;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    Eigen::SparseMatrix<double> jac(NX_C, NX);

    // dG{k}/dc{k} Energy hessians
    for (int k = 0; k < N; k++) {
        Eigen::SparseMatrix<double> d2Edc2 = energy->get_d2Odc2(c_curr.segment(IDX_C(k, 0), NC));
//        std::cout << "NNZ: " << d2Edc2.nonZeros() << std::endl;
        for (int i = 0; i < d2Edc2.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(d2Edc2, i); it; ++it) {
                jac.coeffRef(it.row() + NC * k, it.col() + NC * k) += it.value();
            }
        }
    }
    // Diagonals
    double m_hh = info->dynamics_m / (info->dynamics_dt * info->dynamics_dt);
    double eta_h = info->dynamics_eta / info->dynamics_dt;
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < NC; i++) {
            int idx = k * NC + i;
            jac.coeffRef(idx, idx) += m_hh + eta_h; //dG{k}/dc{k}
            if (k > 0) jac.coeffRef(idx, idx - NC) += -2 * m_hh - eta_h; //dG{k}/dc{k-1}
            if (k > 1) jac.coeffRef(idx, idx - 2 * NC) += m_hh; //dG{k}/dc{k-2}
        }
    }
    // Control input terms
    for (int k = 0; k < N; k++) {
        jac.coeffRef(IDX_C(k, info->selected), IDX_U(k)) += -1;
        jac.coeffRef(IDX_C(k, info->selected) + 1, IDX_U(k) + 1) += -1;
    }

    return jac;
}
