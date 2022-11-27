#include "../../include/Energy/DynamicObjective.h"

void DynamicObjective::init(const Eigen::VectorXd &c_free, double dt, double m, double mu, EnergyObjective *energy) {
    energyObjective = energy;
    c_prev = c_free;
    v_prev = VectorXd::Zero(c_free.rows());;

    VectorXd P_cell = VectorXd::Zero(2 + energyObjective->tessellation->getNumVertexParams());
    P_cell(0) = 1;
    P_cell(1) = 1;
//    VectorXd P = P_cell.replicate(energyObjective->n_free, 1);
    VectorXd P = VectorXd::Ones(c_free.rows());

    M = m * P;
    H = mu * P;

    h = dt;
}

void DynamicObjective::newStep(const Eigen::VectorXd &c_free) {
    v_prev = (c_free - c_prev) / h;
    c_prev = c_free;
}

VectorXd DynamicObjective::get_a(const VectorXd &c_free) const {
    return (c_free - c_prev) / (h * h) - v_prev / h;
}

double DynamicObjective::evaluate(const VectorXd &c_free) const {
    double O = energyObjective->evaluate(c_free);

    VectorXd a = get_a(c_free);
    O += .5 * pow(h, 2) * a.transpose() * M.asDiagonal() * a;
    O += (1.0 / h) *
         (0.5 * c_free.transpose() * H.asDiagonal() * c_free - c_free.transpose() * H.asDiagonal() * c_prev).value();

    return O;
}

void DynamicObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    energyObjective->addGradientTo(c_free, grad);

    grad += M.asDiagonal() * get_a(c_free);
    grad += H.asDiagonal() * (c_free - c_prev) / h;
}

void DynamicObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    energyObjective->getHessian(c_free, hessian);

    for (int i = 0; i < c_free.size(); ++i) {
        hessian.coeffRef(i, i) += M[i] / pow(h, 2);
        hessian.coeffRef(i, i) += H[i] / h;
    }
}

