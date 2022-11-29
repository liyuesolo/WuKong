#include "../../include/ImageMatch/ImageMatchNLP.h"

double ImageMatchNLP::eval_f(const Eigen::VectorXd &x) const {
//    int dims = energy->tessellation->getNumVertexParams() + 2;
//    int n_free = energy->n_free;
//    VectorXd c_free = x.segment(0, n_free * dims);
//
//    return objective->evaluate(c_free);

    return 0;
}

VectorXd ImageMatchNLP::eval_grad_f(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;
//    VectorXd c_free = x.segment(0, n_free * dims);
//
//    VectorXd grad_f = objective->get_dOdc(c_free);
//    return grad_f;
//
    return VectorXd::Zero(n_free * dims);
}

VectorXd ImageMatchNLP::eval_g(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;
    VectorXd c_free = x.segment(0, n_free * dims);

    VectorXd g = energy->get_dOdc(c_free);
    return g;
}

Eigen::SparseMatrix<double> ImageMatchNLP::eval_jac_g_sparsematrix(const Eigen::VectorXd &x) const {
    int dims = energy->tessellation->getNumVertexParams() + 2;
    int n_free = energy->n_free;
    VectorXd c_free = x.segment(0, n_free * dims);

    Eigen::SparseMatrix<double> jac;
    energy->getHessian(c_free, jac);

    return jac;
}
