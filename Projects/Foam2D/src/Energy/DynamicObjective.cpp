#include "../../include/Energy/DynamicObjective.h"

void DynamicObjective::newStep(const Eigen::VectorXd &y) {
    v_prev = (y - y_prev) / info->dynamics_dt;
    y_prev = y;
}

VectorXd DynamicObjective::get_a(const VectorXd &y) const {
    return (y - y_prev) / (info->dynamics_dt * info->dynamics_dt) - v_prev / info->dynamics_dt;
}

void DynamicObjective::check_gradients(const VectorXd &y) const {
    double eps = 1e-4;

    VectorXd x = y;

    double f = evaluate(x);
    VectorXd grad = getGradient(x);
    for (int i = 0; i < x.rows(); i++) {
        VectorXd xp = x;
        xp(i) += eps;
        double fp = evaluate(xp);
        xp(i) += eps;
        double fp2 = evaluate(xp);

        std::cout << "f[" << i << "] " << f << " " << fp << " " << fp2 << " " << (fp - f) / eps << " " << grad(i)
                  << " " << (fp - f - eps * grad(i)) << " " << (fp2 - f - 2 * eps * grad(i)) << " "
                  << (fp2 - f - 2 * eps * grad(i)) / (fp - f - eps * grad(i))
                  << std::endl;
    }

    Eigen::SparseMatrix<double> hessian;
    getHessian(x, hessian);
    MatrixXT hess = hessian;
    for (int i = 0; i < x.rows(); i++) {
        VectorXd xp = x;
        xp(i) = x(i) + eps;
        VectorXd gradp = getGradient(xp);
        xp(i) = x(i) + 2 * eps;
        VectorXd gradp2 = getGradient(xp);
        xp(i) = x(i) - eps;
        VectorXd gradm = getGradient(xp);
        xp(i) = x(i) - 2 * eps;
        VectorXd gradm2 = getGradient(xp);

        for (int j = 0; j < grad.rows(); j++) {
            std::cout << "check hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hess(j, i)
                      << std::endl;
            double a = (gradp[j] - gradm[j]) / (2 * eps) - hess(j, i);
            double b = (gradp2[j] - gradm2[j]) / (4 * eps) - hess(j, i);
            std::cout << a << " " << b << " " << b / a << std::endl;
        }
    }
}

double DynamicObjective::evaluate(const VectorXd &y) const {
    double O = energyObjective->evaluate(y);

    VectorXd a = get_a(y);
    O += .5 * pow(info->dynamics_dt, 2) * a.transpose() * info->dynamics_m * a;
    O += (1.0 / info->dynamics_dt) *
         (0.5 * y.transpose() * info->dynamics_eta * y -
          y.transpose() * info->dynamics_eta * y_prev).value();

    return O;
}

void DynamicObjective::addGradientTo(const VectorXd &y, VectorXd &grad) const {
    energyObjective->addGradientTo(y, grad);

    grad += info->dynamics_m * get_a(y);
    grad += info->dynamics_eta * (y - y_prev) / info->dynamics_dt;
}

void DynamicObjective::getHessian(const VectorXd &y, SparseMatrixd &hessian) const {
    energyObjective->getHessian(y, hessian);

    for (int i = 0; i < y.size(); ++i) {
        hessian.coeffRef(i, i) += info->dynamics_m / pow(info->dynamics_dt, 2);
        hessian.coeffRef(i, i) += info->dynamics_eta / info->dynamics_dt;
    }
}

