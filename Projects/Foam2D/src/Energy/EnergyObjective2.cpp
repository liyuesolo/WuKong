#include "../../include/Energy/EnergyObjective2.h"
#include "../../include/Energy/CellFunctionEnergy.h"

static void printVectorXT(std::string name, const VectorXT &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

static void printVectorXi(std::string name, const VectorXi &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

void EnergyObjective2::check_gradients(const VectorXd &c_free) const {
    double eps = 1e-6;

    VectorXd x = c_free;

    double f = evaluate(x);
    VectorXd grad = get_dOdc(x);
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

    MatrixXT hess = get_d2Odc2(x);
    for (int i = 0; i < x.rows(); i++) {
        VectorXd xp = x;
        xp(i) += eps;
        VectorXd gradp = get_dOdc(xp);
        xp(i) += eps;
        VectorXd gradp2 = get_dOdc(xp);

        for (int j = 0; j < grad.rows(); j++) {
            std::cout << "check hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hess(j,i) << std::endl;
            double a = (gradp[j] - grad[j]) / eps - hess(j, i);
            double b = (gradp2[j] - grad[j]) / (2 * eps) - hess(j, i);
            std::cout << a << " " << b << " " << b / a << std::endl;
        }
    }
}

double EnergyObjective2::evaluate(const VectorXd &c_free) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    if (!info->getTessellation()->isValid) {
        return 1e10;
    }

    CellFunctionEnergy energy(info);

    double O = 0;
    info->getTessellation()->addFunctionValue(energy, O);

    return O;
}

void EnergyObjective2::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd EnergyObjective2::get_dOdc(const VectorXd &c_free) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    VectorXT gradient = VectorXT::Zero(c_free.rows());
    if (!info->getTessellation()->isValid) {
        return gradient;
    }

    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionGradient(energy, gradient);

    return gradient;
}

void EnergyObjective2::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> EnergyObjective2::get_d2Odc2(const VectorXd &c_free) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    MatrixXT hessian = MatrixXT::Zero(c_free.rows(), c_free.rows());
    if (!info->getTessellation()->isValid) {
        return hessian.sparseView();
    }

    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionHessian(energy, hessian);

    return hessian.sparseView();
}

