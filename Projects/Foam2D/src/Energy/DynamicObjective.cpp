#include "../../include/Energy/DynamicObjective.h"

void DynamicObjective::newStep(const Eigen::VectorXd &c_free) {
    v_prev = (c_free - c_prev) / info->dynamics_dt;
    c_prev = c_free;
}

VectorXd DynamicObjective::get_a(const VectorXd &c_free) const {
    return (c_free - c_prev) / (info->dynamics_dt * info->dynamics_dt) - v_prev / info->dynamics_dt;
}

void DynamicObjective::check_gradients(const VectorXd &c_free) const {
    double eps = 1e-4;

    VectorXd x = c_free;

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

double DynamicObjective::evaluate(const VectorXd &c_free) const {
    double O = energyObjective->evaluate(c_free);

    VectorXd a = get_a(c_free);
    O += .5 * pow(info->dynamics_dt, 2) * a.transpose() * info->dynamics_m * a;
    O += (1.0 / info->dynamics_dt) *
         (0.5 * c_free.transpose() * info->dynamics_eta * c_free -
          c_free.transpose() * info->dynamics_eta * c_prev).value();

    return O;
}

void DynamicObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    energyObjective->addGradientTo(c_free, grad);

    grad += info->dynamics_m * get_a(c_free);
    grad += info->dynamics_eta * (c_free - c_prev) / info->dynamics_dt;
}

VectorXd DynamicObjective::getGradient(const VectorXd &c_free) const {
    VectorXT grad = VectorXT::Zero(c_free.rows());
    addGradientTo(c_free, grad);
    return grad;
}

void DynamicObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    energyObjective->getHessian(c_free, hessian);

    for (int i = 0; i < c_free.size(); ++i) {
        hessian.coeffRef(i, i) += info->dynamics_m / pow(info->dynamics_dt, 2);
        hessian.coeffRef(i, i) += info->dynamics_eta / info->dynamics_dt;
    }
}

