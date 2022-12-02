#include "../../include/Energy/DynamicObjective.h"

void DynamicObjective::newStep(const Eigen::VectorXd &c_free) {
    v_prev = (c_free - c_prev) / info->dynamics_dt;
    c_prev = c_free;
}

VectorXd DynamicObjective::get_a(const VectorXd &c_free) const {
    return (c_free - c_prev) / (info->dynamics_dt * info->dynamics_dt) - v_prev / info->dynamics_dt;
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

void DynamicObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    energyObjective->getHessian(c_free, hessian);

    for (int i = 0; i < c_free.size(); ++i) {
        hessian.coeffRef(i, i) += info->dynamics_m / pow(info->dynamics_dt, 2);
        hessian.coeffRef(i, i) += info->dynamics_eta / info->dynamics_dt;
    }
}

