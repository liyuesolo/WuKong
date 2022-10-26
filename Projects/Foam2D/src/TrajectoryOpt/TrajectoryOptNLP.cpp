#include "../../include/TrajectoryOpt/TrajectoryOptNLP.h"

double TrajectoryOptNLP::get_f(const Eigen::VectorXd &x) const {
    return 0;
}

VectorXd TrajectoryOptNLP::get_grad_f(const Eigen::VectorXd &x) const {
    return VectorXd::Zero(N);
}

VectorXd TrajectoryOptNLP::get_g(const Eigen::VectorXd &x) const {
    return VectorXd::Zero(N);
}

Eigen::SparseMatrix<double> TrajectoryOptNLP::get_jac_g(const Eigen::VectorXd &x) const {
    return Eigen::SparseMatrix<double>(1, 1);
}
