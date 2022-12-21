#include "../../include/ImageMatch/ImageMatchSAObjective.h"
#include "../../include/ImageMatch/CellFunctionImageMatchAreaScaled.h"
#include "../../src/optLib/NewtonFunctionMinimizer.h"

typedef CellFunctionImageMatchAreaScaled TypedefImageMatchFunction;

static void printVectorXT(std::string name, const VectorXT &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

void ImageMatchSAObjective::preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    cellInfos.resize(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        cellInfos[i].border_pix = pix[i];
    }
}

double ImageMatchSAObjective::evaluate(const VectorXd &tau) const {
    VectorXT c_free;
    bool success = getC(tau, c_free);

    if (!success) {
        std::cout << "Minimum not found in getC. Returning large objective value." << std::endl;
//        printVectorXT("tau", tau);
        return 1e10;
    }

    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    if (!info->getTessellation()->isValid) {
//        std::cout << "Invalid tessellation in getC. Returning large objective value." << std::endl;
        return 1e10;
    }

    TypedefImageMatchFunction imageMatchFunction;

    double O = 0;
    info->getTessellation()->addFunctionValue(imageMatchFunction, O, cellInfos);

    std::cout << "Change " << (c_free-c0).norm() << " Objective " << O << std::endl;

    return O;
}

void ImageMatchSAObjective::addGradientTo(const VectorXd &tau, VectorXd &grad) const {
    grad += get_dOdtau(tau);
}

// Not a real gradient - don't check with FD. This is the sensitivity analysis line search direction.
VectorXd ImageMatchSAObjective::get_dOdtau(const VectorXd &tau) const {
    VectorXT c_free;
    bool success = getC(tau, c_free);

    if (!success) {
        return VectorXT::Zero(tau.rows());
    }

    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    VectorXT x(c0.rows() + tau.rows());
    x << c_free, tau;
    SparseMatrixd energyHessianAT;
    energyObjectiveAT->getHessian(x, energyHessianAT);

    SparseMatrixd dGdc = energyHessianAT.block(0, 0, c0.rows(), c0.rows());
    SparseMatrixd dGdtau = energyHessianAT.block(0, c0.rows(), c0.rows(), tau.rows());
    SparseMatrixd diag = VectorXT::Ones(c0.rows()).asDiagonal().toDenseMatrix().sparseView();
    double stab = 1e-6;
    for (int i = 0; i < 10; i++) {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver(dGdc + (stab * diag));
        bool hessian_pd = solver.info() != Eigen::ComputationInfo::NumericalIssue;
        std::cout << "Hessian pd " << hessian_pd << std::endl;
        if (hessian_pd) break;
    }
    Eigen::SparseLU<SparseMatrixd> solver;
    solver.compute(dGdc + (stab * diag));
    SparseMatrixd dcdtau = solver.solve(-dGdtau);

    VectorXT dLdc = VectorXT::Zero(c_free.rows());
    if (!info->getTessellation()->isValid) {
        return VectorXT::Zero(tau.rows());
    }

    TypedefImageMatchFunction imageMatchFunction;
    info->getTessellation()->addFunctionGradient(imageMatchFunction, dLdc, cellInfos);

    VectorXd gradient = dLdc.transpose() * dcdtau;
    std::cout << "Gradient norm " << gradient.norm()  << " " << dcdtau.norm() << " " << dLdc.norm() << " " << solver.info() << std::endl;

    return gradient / fmax(gradient.norm(), 1.0);
}

bool ImageMatchSAObjective::getC(const VectorXd &tau, VectorXd &c) const {
    NewtonFunctionMinimizer newton(50, 1e-6, 15);

    EnergyObjective energyObjective;
    Foam2DInfo tempInfo(*info);
    tempInfo.energy_area_targets = tau;
    energyObjective.info = &tempInfo;

    c = c0;
    newton.minimize(&energyObjective, c);

    bool success = energyObjective.getGradient(c).norm() < 1e-6 && energyObjective.info->getTessellation()->isValid;
    if (!success) {
        std::cout << "Valid " << energyObjective.info->getTessellation()->isValid << " Grad " << energyObjective.getGradient(c).norm() << std::endl;
    }

    return success;
}
