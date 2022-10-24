#include "../../include/Objective/DynamicObjective.h"

void DynamicObjective::init(const Eigen::VectorXd &c_free, double dt, double m, EnergyObjective *energy) {
    energyObjective = energy;
    c_prev = c_free;
    v_prev = VectorXd::Zero(c_free.rows());;
    M = m * VectorXd::Ones(c_free.rows());
    h = dt;
}

void DynamicObjective::newStep(const Eigen::VectorXd &c_free) {
    v_prev = (c_free - c_prev) / h;
    c_prev = c_free;
}

double DynamicObjective::evaluate(const VectorXd &c_free) const {
    double O = energyObjective->evaluate(c_free);

    VectorXd a = get_a(c_free);
    O += .5 * pow(h, 2) * a.transpose() * M.asDiagonal() * a;

    return O;
}

void DynamicObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    energyObjective->addGradientTo(c_free, grad);

    grad += M.asDiagonal() * get_a(c_free);
}

void DynamicObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    energyObjective->getHessian(c_free, hessian);

    for (int i = 0; i < c_free.size(); ++i) {
        hessian.coeffRef(i, i) += M[i] / pow(h, 2);
    }
}

