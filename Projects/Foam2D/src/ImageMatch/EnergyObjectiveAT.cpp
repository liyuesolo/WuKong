#include "../../include/ImageMatch/EnergyObjectiveAT.h"
#include "../../include/Energy/CellFunctionEnergy.h"
#include "../../include/Energy/CellFunctionArea.h"

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

void EnergyObjectiveAT::preProcess(const VectorXd &x, std::vector<CellInfo> &cellInfos) const {
    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c_free = x.segment(0, dims * info->n_free);

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    cellInfos.resize(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        cellInfos[i].target_area = 1.0 / x(c_free.rows() + i);
        cellInfos[i].agent = false;
    }
}

double EnergyObjectiveAT::evaluate(const VectorXd &x) const {
    std::vector<CellInfo> cellInfos;
    preProcess(x, cellInfos);

    if (!info->getTessellation()->isValid) {
        return 1e10;
    }

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c_free = x.segment(0, dims * info->n_free);

    CellFunctionEnergy energy(info);

    double O = 0;
    info->getTessellation()->addFunctionValue(energy, O, cellInfos);

    return O;
}

void EnergyObjectiveAT::addGradientTo(const VectorXd &x, VectorXd &grad) const {
    grad += get_dOdx(x);
}

VectorXd EnergyObjectiveAT::get_dOdx(const VectorXd &x) const {
    std::vector<CellInfo> cellInfos;
    preProcess(x, cellInfos);

    VectorXT gradient = VectorXT::Zero(x.rows());
    if (!info->getTessellation()->isValid) {
        return gradient;
    }

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c_free = x.segment(0, dims * info->n_free);

    VectorXT gradient_c = VectorXT::Zero(c_free.rows());
    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionGradient(energy, gradient_c, cellInfos);

    VectorXT gradient_tau = VectorXT::Zero(info->n_free);
    CellFunctionArea areaFunction;
    for (int cell = 0; cell < info->n_free; cell++) {
        double area = 0;
        info->getTessellation()->addSingleCellFunctionValue(cell, areaFunction, area, &cellInfos[cell]);

        double tau = x(c_free.rows() + cell);
        gradient_tau(cell) += info->energy_area_weight * 2 * area * (area * tau - 1);
    }

    gradient << gradient_c, gradient_tau;

    return gradient;
}

void EnergyObjectiveAT::getHessian(const VectorXd &x, SparseMatrixd &hessian) const {
    hessian = get_d2Odx2(x);
}

Eigen::SparseMatrix<double> EnergyObjectiveAT::get_d2Odx2(const VectorXd &x) const {
    std::vector<CellInfo> cellInfos;
    preProcess(x, cellInfos);

    MatrixXT hessian = MatrixXT::Zero(x.rows(), x.rows());
    if (!info->getTessellation()->isValid) {
        return hessian.sparseView();
    }

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c_free = x.segment(0, dims * info->n_free);

    MatrixXT hessian_c = MatrixXT::Zero(c_free.rows(), c_free.rows());
    CellFunctionEnergy energy(info);
    info->getTessellation()->addFunctionHessian(energy, hessian_c, cellInfos);

    VectorXT hessian_tau_tau_diag = VectorXT::Zero(info->n_free);
    MatrixXT hessian_tau_x = MatrixXT::Zero(info->n_free, c_free.rows());
    CellFunctionArea areaFunction;
    for (int cell = 0; cell < info->n_free; cell++) {
        double area = 0;
        info->getTessellation()->addSingleCellFunctionValue(cell, areaFunction, area, &cellInfos[cell]);

        VectorXT area_gradient = VectorXT::Zero(c_free.rows());
        info->getTessellation()->addSingleCellFunctionGradient(cell, areaFunction, area_gradient, &cellInfos[cell]);

        double tau = x(c_free.rows() + cell);
        hessian_tau_tau_diag(cell) += info->energy_area_weight * 2 * area * area;
        hessian_tau_x.row(cell) += info->energy_area_weight * (4 * area * tau - 2) * area_gradient;
    }

    hessian.block(0, 0, c_free.rows(), c_free.rows()) = hessian_c;
    hessian.block(c_free.rows(), c_free.rows(), info->n_free, info->n_free) = hessian_tau_tau_diag.asDiagonal();
    hessian.block(c_free.rows(), 0, info->n_free, c_free.rows()) = hessian_tau_x;
    hessian.block(0, c_free.rows(), c_free.rows(), info->n_free) = hessian_tau_x.transpose();

    return hessian.sparseView();
}


