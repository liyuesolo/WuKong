#include "../../include/ImageMatch/ImageMatchSAObjectiveParallel.h"
#include "../../include/ImageMatch/CellFunctionImageMatch2AreaScaled.h"
#include "../../src/optLib/NewtonFunctionMinimizer.h"
#include "../../include/Energy/EnergyObjective.h"
#include "../../include/Tessellation/Power.h"

typedef CellFunctionImageMatch2AreaScaled TypedefImageMatchFunction;

static void printVectorXT(std::string name, const VectorXT &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

bool ImageMatchSAObjectiveParallel::preProcess(const VectorXd &tau, Tessellation *tessellation, std::vector<CellInfo> &cellInfos, VectorXd &c_free, bool need_get_c = true) const {
    c_free = c0;
    if (need_get_c && !getC(tau, tessellation, c_free)) {
        return false;
    }

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);
    tessellation->tessellate(vertices, params, info->boundary, info->n_free);

    cellInfos.resize(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        cellInfos[i].border_pix = pix[i];
    }

    return true;
}

double ImageMatchSAObjectiveParallel::evaluate(const VectorXd &tau) const {
    Power tessellation;
    std::vector<CellInfo> cellInfos;
    VectorXT c_free;
    bool success = preProcess(tau, &tessellation, cellInfos, c_free);

    if (!success) {
        std::cout << "Minimum not found in getC. Returning large objective value." << std::endl;
//        printVectorXT("tau", tau);
        return 1e10;
    }

    TypedefImageMatchFunction imageMatchFunction;

    double O = 0;
    tessellation.addFunctionValue(imageMatchFunction, O, cellInfos);

    std::cout << "Change " << (c_free-c0).norm() << " Objective " << O << std::endl;
    sols.insert(std::pair<double, VectorXT>(O, c_free));

    return O;
}

void ImageMatchSAObjectiveParallel::addGradientTo(const VectorXd &tau, VectorXd &grad) const {
    grad += get_dOdtau(tau);
}

// Not a real gradient - don't check with FD. This is the sensitivity analysis line search direction.
VectorXd ImageMatchSAObjectiveParallel::get_dOdtau(const VectorXd &tau) const {
    Power tessellation;
    std::vector<CellInfo> cellInfos;
    VectorXT c_free;
    bool success = preProcess(tau, &tessellation, cellInfos, c_free, false);

    if (!success) {
        std::cout << "Failed to evaluate image match gradient. Something went wrong here. :(" << std::endl;
        return VectorXT::Zero(tau.rows());
    }

    VectorXT x(c0.rows() + tau.rows());
    x << c_free, tau;
    EnergyObjectiveAT energyObjectiveAT;
    Foam2DInfo tempInfo(*info);
    tempInfo.tessellation = 1;
    tempInfo.tessellations[1] = &tessellation;
    energyObjectiveAT.info = &tempInfo;
    SparseMatrixd energyHessianAT;
    energyObjectiveAT.getHessian(x, energyHessianAT);

    SparseMatrixd dGdc = energyHessianAT.block(0, 0, c0.rows(), c0.rows());
    SparseMatrixd dGdtau = energyHessianAT.block(0, c0.rows(), c0.rows(), tau.rows());
    SparseMatrixd diag = VectorXT::Ones(c0.rows()).asDiagonal().toDenseMatrix().sparseView();
    double stab = 1e-6;
    Eigen::SparseLU<SparseMatrixd> solver;
    solver.compute(dGdc + (stab * diag));
    SparseMatrixd dcdtau = solver.solve(-dGdtau);

    VectorXT dLdc = VectorXT::Zero(c_free.rows());
    TypedefImageMatchFunction imageMatchFunction;
    tessellation.addFunctionGradient(imageMatchFunction, dLdc, cellInfos);

    VectorXd gradient = dLdc.transpose() * dcdtau;
    std::cout << "Gradient norm " << gradient.norm()  << " " << dcdtau.norm() << " " << dLdc.norm() << " " << solver.info() << std::endl;

    return gradient / fmax(gradient.norm() * 100.0, 1.0);
}

void ImageMatchSAObjectiveParallel::getHessian(const VectorXd &tau, SparseMatrixd &hessian) const {
    hessian = get_d2Odtau2(tau);
}

// TODO: This approximation doesn't work very well...
Eigen::SparseMatrix<double> ImageMatchSAObjectiveParallel::get_d2Odtau2(const VectorXd &tau) const {
    Power tessellation;
    std::vector<CellInfo> cellInfos;
    VectorXT c_free;
    bool success = preProcess(tau, &tessellation, cellInfos, c_free, false);

    if (!success) {
        std::cout << "Failed to evaluate image match hessian. Something went wrong here. :(" << std::endl;
        return {};
    }

    VectorXT x(c0.rows() + tau.rows());
    x << c_free, tau;
    EnergyObjectiveAT energyObjectiveAT;
    Foam2DInfo tempInfo(*info);
    tempInfo.tessellation = 1;
    tempInfo.tessellations[1] = &tessellation;
    energyObjectiveAT.info = &tempInfo;
    SparseMatrixd energyHessianAT;
    energyObjectiveAT.getHessian(x, energyHessianAT);

    SparseMatrixd dGdc = energyHessianAT.block(0, 0, c0.rows(), c0.rows());
    SparseMatrixd dGdtau = energyHessianAT.block(0, c0.rows(), c0.rows(), tau.rows());
    SparseMatrixd diag = VectorXT::Ones(c0.rows()).asDiagonal().toDenseMatrix().sparseView();
    double stab = 1e-6;
    Eigen::SparseLU<SparseMatrixd> solver;
    solver.compute(dGdc + (stab * diag));
    SparseMatrixd dcdtau = solver.solve(-dGdtau);

    MatrixXT d2Ldc2 = MatrixXT::Zero(c_free.rows(), c_free.rows());
    TypedefImageMatchFunction imageMatchFunction;
    tessellation.addFunctionHessian(imageMatchFunction, d2Ldc2, cellInfos);

    return (dcdtau.transpose() * d2Ldc2 * dcdtau).sparseView();
}


bool ImageMatchSAObjectiveParallel::getC(const VectorXd &tau, Tessellation *tessellation, VectorXT &c) const {
    NewtonFunctionMinimizer newton(50, 1e-6, 15);

    EnergyObjective energyObjective;
    Foam2DInfo tempInfo(*info);
    tempInfo.energy_area_targets = tau;
    tempInfo.tessellation = 1;
    tempInfo.tessellations[1] = tessellation;
    energyObjective.info = &tempInfo;

    c = c0;
    newton.minimize(&energyObjective, c);

    bool success = energyObjective.getGradient(c).norm() < 1e-6 && energyObjective.info->getTessellation()->isValid;
//    if (!success) {
//        std::cout << "Valid " << energyObjective.info->getTessellation()->isValid << " Grad " << energyObjective.getGradient(c).norm() << std::endl;
//    }

    return success;
}
