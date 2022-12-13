#include "../../include/Energy/EnergyObjective.h"
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

void EnergyObjective::check_gradients(const VectorXd &c_free) const {
    double eps = 1e-4;

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
        xp(i) = x(i) + eps;
        VectorXd gradp = get_dOdc(xp);
        xp(i) = x(i) + 2 * eps;
        VectorXd gradp2 = get_dOdc(xp);
        xp(i) = x(i) - eps;
        VectorXd gradm = get_dOdc(xp);
        xp(i) = x(i) - 2 * eps;
        VectorXd gradm2 = get_dOdc(xp);

        for (int j = 0; j < grad.rows(); j++) {
            std::cout << "check hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hess(j, i)
                      << std::endl;
            double a = (gradp[j] - gradm[j]) / (2 * eps) - hess(j, i);
            double b = (gradp2[j] - gradm2[j]) / (4 * eps) - hess(j, i);
            std::cout << a << " " << b << " " << b / a << std::endl;
        }
    }
}

void EnergyObjective::preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    cellInfos.resize(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        cellInfos[i].target_area = info->energy_area_targets(i);
        cellInfos[i].agent = false;
    }
    if (info->selected >= 0) {
        cellInfos[info->selected].agent = true;
        cellInfos[info->selected].target_position = info->selected_target_pos;
    }
}

double EnergyObjective::evaluate(const VectorXd &c_free) const {
    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    if (!info->getTessellation()->isValid) {
        return 1e10;
    }

    CellFunctionEnergy energy(info);

    double O = 0;
    info->getTessellation()->addFunctionValue(energy, O, cellInfos);

    return O;
}

void EnergyObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd EnergyObjective::get_dOdc(const VectorXd &c_free) const {
    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    VectorXT gradient = VectorXT::Zero(c_free.rows());
    if (!info->getTessellation()->isValid) {
        return gradient;
    }

    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionGradient(energy, gradient, cellInfos);

    return gradient;
}

void EnergyObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &c_free) const {
    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    MatrixXT hessian = MatrixXT::Zero(c_free.rows(), c_free.rows());
    if (!info->getTessellation()->isValid) {
        return hessian.sparseView();
    }

    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionHessian(energy, hessian, cellInfos);

    return hessian.sparseView();
}
