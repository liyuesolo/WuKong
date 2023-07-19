#include "../../include/Energy/DynamicObjective.h"

void DynamicObjective::minimize(GradientDescentLineSearch *minimizer, VectorXd &y, bool optimizeWeights_) const {
    energyObjective->optimizeWeights = optimizeWeights_;
    energyObjective->optDims = energyObjective->optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!energyObjective->optimizeWeights) {
        VectorXd verts;
        energyObjective->tessellation->separateVerticesParams(
                y.head(y.rows() - energyObjective->tessellation->boundary->nfree), verts, energyObjective->paramsSave);
        yTemp.resize(verts.rows() + energyObjective->tessellation->boundary->nfree);
        yTemp << verts, y.tail(energyObjective->tessellation->boundary->nfree);
    }

    minimizer->minimize(this, yTemp);

    if (energyObjective->optimizeWeights) {
        y = yTemp;
    } else {
        VectorXT c = energyObjective->tessellation->combineVerticesParams(
                yTemp.head(yTemp.rows() - energyObjective->tessellation->boundary->nfree),
                energyObjective->paramsSave);
        y.resize(c.rows() + energyObjective->tessellation->boundary->nfree);
        y << c, yTemp.tail(energyObjective->tessellation->boundary->nfree);
    }
}

void DynamicObjective::newStep(const Eigen::VectorXd &y, bool optimizeWeights_) {
    VectorXd yTemp = y;
    if (!optimizeWeights_) {
        VectorXd verts;
        energyObjective->tessellation->separateVerticesParams(
                y.head(y.rows() - energyObjective->tessellation->boundary->nfree), verts, energyObjective->paramsSave);
        yTemp.resize(verts.rows() + energyObjective->tessellation->boundary->nfree);
        yTemp << verts, y.tail(energyObjective->tessellation->boundary->nfree);
    }

    if (!initialized) {
        v_prev = VectorXT::Zero(yTemp.rows());
    } else {
        v_prev = (yTemp - y_prev) / dt;
    }
    y_prev = yTemp;
}

VectorXd DynamicObjective::get_a(const VectorXd &y) const {
    return (y - y_prev) / (dt * dt) - v_prev / dt;
}

void DynamicObjective::check_gradients(const VectorXd &y, bool optimizeWeights_) const {
    energyObjective->optimizeWeights = optimizeWeights_;
    energyObjective->optDims = energyObjective->optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!energyObjective->optimizeWeights) {
        VectorXd verts;
        energyObjective->tessellation->separateVerticesParams(
                y.head(y.rows() - energyObjective->tessellation->boundary->nfree), verts, energyObjective->paramsSave);
        yTemp.resize(verts.rows() + energyObjective->tessellation->boundary->nfree);
        yTemp << verts, y.tail(energyObjective->tessellation->boundary->nfree);
    }

    double eps = 1e-6;

    VectorXd x = yTemp;

    double f = evaluate(x);
    VectorXd grad = 0 * x;
    addGradientTo(x, grad);
    for (int i = 0; i < x.rows(); i++) {
        VectorXd xp = x;
        xp(i) += eps;
        double fp = evaluate(xp);
        xp(i) += eps;
        double fp2 = evaluate(xp);

        std::cout << "f[" << i << "] " << f << " " << fp << " " << (fp - f) / eps << " "
                  << (fp2 - f) / (2 * eps) << " " << grad(i)
                  << " " << (fp - f - eps * grad(i)) << " " << (fp2 - f - 2 * eps * grad(i)) << " "
                  << (fp2 - f - 2 * eps * grad(i)) / (fp - f - eps * grad(i))
                  << std::endl;
    }

    SparseMatrixd hess_sp;
    getHessian(x, hess_sp);
    MatrixXT hess = hess_sp;
    MatrixXT hessFD = 0 * hess;
    for (int i = 0; i < x.rows(); i++) {
        VectorXd gradp = 0 * x, gradp2 = 0 * x, gradm = 0 * x, gradm2 = 0 * x;

        VectorXd xp = x;
        xp(i) = x(i) + eps;
        addGradientTo(xp, gradp);
        xp(i) = x(i) + 2 * eps;
        addGradientTo(xp, gradp2);
        xp(i) = x(i) - eps;
        addGradientTo(xp, gradm);
        xp(i) = x(i) - 2 * eps;
        addGradientTo(xp, gradm2);

        for (int j = 0; j < grad.rows(); j++) {
//            if (fabs(hess(j, i)) < 1e-10 ||
//                fabs((gradp[j] - grad[j]) / eps - hess(j, i)) < 1e-2 * fabs(hess(j, i)))
//                continue;
            hessFD(j, i) = (gradp[j] - grad[j]) / eps;
            double a = (gradp[j] - gradm[j]) - (2 * eps) * hess(j, i);
            double b = (gradp2[j] - gradm2[j]) - (4 * eps) * hess(j, i);
            std::cout << "check hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hess(j, i)
                      << " " << a << " " << b << " " << b / a
                      << std::endl;
        }
    }

    std::cout << "Hessian error norm: " << (hessFD - hess).norm();
}

double DynamicObjective::evaluate(const VectorXd &y) const {
    double O = energyObjective->evaluate(y);
    if (!energyObjective->tessellation->isValid) {
        return 1e10;
    }

    VectorXd a = get_a(y);
    O += .5 * pow(dt, 2) * a.transpose() * m * a;
    O += (1.0 / dt) *
         (0.5 * y.transpose() * eta * y -
          y.transpose() * eta * y_prev).value();

//    std::cout << "dynamics objective value: " << O << std::endl;

    return O;
}

void DynamicObjective::addGradientTo(const VectorXd &y, VectorXd &grad) const {
    energyObjective->addGradientTo(y, grad);
    if (!energyObjective->tessellation->isValid) {
        return;
    }

    grad += m * get_a(y);
    grad += eta * (y - y_prev) / dt;

//    std::cout << "dynamics gradient norm: " << grad.norm() << std::endl;
}

void DynamicObjective::getHessian(const VectorXd &y, SparseMatrixd &hessian) const {
    energyObjective->getHessian(y, hessian);
    if (!energyObjective->tessellation->isValid) {
        return;
    }

    for (int i = 0; i < y.size(); ++i) {
        hessian.coeffRef(i, i) += m / pow(dt, 2);
        hessian.coeffRef(i, i) += eta / dt;
    }
}

