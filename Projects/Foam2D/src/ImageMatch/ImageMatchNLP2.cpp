#include "../../include/ImageMatch/ImageMatchNLP2.h"

#define IDX_C(k, i) (((k) * n_free + (i)) * dims)
#define IDX_A(i) ((N * n_free * dims) + (i))
#define NX_C (N * n_free * dims)
#define NX_A (n_free)
#define NX (N * n_free * dims + n_free)
#define NC (n_free * dims)

void ImageMatchNLP2::check_gradients(const Eigen::VectorXd &x) const {
    double eps = 1e-6;

    VectorXT g = eval_g(x);
    Eigen::SparseMatrix<double> jac = eval_jac_g_sparsematrix(x);

    int nx = x.rows();
    int ng = g.rows();

    for (int i = 0; i < nx; i++) {
        VectorXd xp = x;
        xp(i) += eps;
        VectorXT gp = eval_g(xp);
        xp(i) += eps;
        VectorXT gp2 = eval_g(xp);

        for (int j = 0; j < ng; j++) {
            std::cout << "g[" << j << ", " << i << "] " << g(j) << " " << (gp(j) - g(j)) / eps << " " << jac.coeff(j, i)
                      << " " << (gp2(j) - g(j) - 2 * eps * jac.coeff(j, i)) / (gp(j) - g(j) - eps * jac.coeff(j, i))
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

double ImageMatchNLP2::eval_f(const Eigen::VectorXd &x) const {
//    return 0;

    int dims = info->getTessellation()->getNumVertexParams() + 2;
    VectorXd c_free = x.segment((info->imageMatch_N - 1) * info->n_free * dims, info->n_free * dims);

    return objective->evaluate(c_free);
}

VectorXd ImageMatchNLP2::eval_grad_f(const Eigen::VectorXd &x) const {
//    return VectorXd::Zero(x.rows());

    int dims = info->getTessellation()->getNumVertexParams() + 2;
    VectorXd c_free = x.segment((info->imageMatch_N - 1) * info->n_free * dims, info->n_free * dims);

    VectorXd grad_f = VectorXd::Zero(x.rows());
    grad_f.segment((info->imageMatch_N - 1) * info->n_free * dims, info->n_free * dims) = objective->get_dOdc(c_free);
    return grad_f;
}

VectorXd ImageMatchNLP2::eval_g(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->imageMatch_N;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    VectorXd force_int = VectorXd::Zero(NX_C);
    for (int k = 0; k < N; k++) {
        VectorXT xk(NC + n_free);
        xk << c_curr.segment(IDX_C(k, 0), NC), x.segment(IDX_A(0), n_free);
        force_int.segment(IDX_C(k, 0), NC) = -1.0 * energy->get_dOdx(xk).segment(0, NC);
    }

    VectorXd force_visc = -info->dynamics_eta * (c_curr - c_prev);
    VectorXd Ma = info->dynamics_m * (c_curr - 2 * c_prev + c_2prev);
    for (int k = 0; k < N; k++) {
        double dt = (info->imageMatch_h0 * pow(info->imageMatch_hgrowth, k));
        force_visc.segment(IDX_C(k, 0), NC) /= dt;
        Ma.segment(IDX_C(k, 0), NC) /= (dt * dt);
    }
    VectorXd G = Ma - (force_int + force_visc);

    return G;
}

Eigen::SparseMatrix<double> ImageMatchNLP2::eval_jac_g_sparsematrix(const Eigen::VectorXd &x) const {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    int n_free = info->n_free;
    int N = info->imageMatch_N;

    VectorXd c_curr = x.segment(0, NX_C);
    VectorXd c_prev = VectorXd::Zero(NX_C);
    c_prev.segment(0, NC) = c0;
    c_prev.segment(NC, (N - 1) * NC) = c_curr.segment(0, (N - 1) * NC);
    VectorXd c_2prev = VectorXd::Zero(NX_C);
    c_2prev.segment(0, NC) = c0;
    c_2prev.segment(NC, (N - 1) * NC) = c_prev.segment(0, (N - 1) * NC);

    Eigen::SparseMatrix<double> jac(NX_C, NX);

    // dG{k}/dc{k} Energy hessians
    for (int k = 0; k < N; k++) {
        VectorXT xk(NC + n_free);
        xk << c_curr.segment(IDX_C(k, 0), NC), x.segment(IDX_A(0), n_free);
        Eigen::SparseMatrix<double> d2Edx2 = energy->get_d2Odx2(xk);
//        std::cout << "NNZ: " << d2Edc2.nonZeros() << std::endl;
        for (int i = 0; i < d2Edx2.outerSize(); ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(d2Edx2, i); it; ++it) {
                if (it.row() >= NC) {continue;}
                if (it.col() >= NC) {jac.coeffRef(it.row() + NC * k, NX_C + (it.col() - NC)) += it.value();}
                else {jac.coeffRef(it.row() + NC * k, it.col() + NC * k) += it.value();}
            }
        }
    }
    // Diagonals
    for (int k = 0; k < N; k++) {
        double dt = (info->imageMatch_h0 * pow(info->imageMatch_hgrowth, k));
        double m_hh = info->dynamics_m / (dt * dt);
        double eta_h = info->dynamics_eta / dt;
        for (int i = 0; i < NC; i++) {
            int idx = k * NC + i;
            jac.coeffRef(idx, idx) += m_hh + eta_h; //dG{k}/dc{k}
            if (k > 0) jac.coeffRef(idx, idx - NC) += -2 * m_hh - eta_h; //dG{k}/dc{k-1}
            if (k > 1) jac.coeffRef(idx, idx - 2 * NC) += m_hh; //dG{k}/dc{k-2}
        }
    }

    return jac;
}
