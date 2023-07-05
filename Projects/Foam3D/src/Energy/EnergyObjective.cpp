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

void EnergyObjective::minimize(GradientDescentLineSearch *minimizer, VectorXd &y, bool optimizeWeights_) {
    optimizeWeights = optimizeWeights_;
    optDims = optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!optimizeWeights) {
        tessellation->separateVerticesParams(y, yTemp, paramsSave);
    }

    minimizer->minimize(this, yTemp);

    if (optimizeWeights) {
        y = yTemp;
    } else {
        y = tessellation->combineVerticesParams(yTemp, paramsSave);
    }
}

void EnergyObjective::check_gradients(const VectorXd &y, bool optimizeWeights_) {
    optimizeWeights = optimizeWeights_;
    optDims = optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!optimizeWeights) {
        tessellation->separateVerticesParams(y, yTemp, paramsSave);
    }

    double eps = 1e-6;

    VectorXd x = yTemp;

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
//            if (fabs(hess(j, i)) < 1e-10 ||
//                fabs((gradp[j] - grad[j]) / eps - hess(j, i)) < 1e-2 * fabs(hess(j, i)))
//                continue;
            double a = (gradp[j] - gradm[j]) - (2 * eps) * hess(j, i);
            double b = (gradp2[j] - gradm2[j]) - (4 * eps) * hess(j, i);
            std::cout << "check hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hess(j, i)
                      << " " << a << " " << b << " " << b / a
                      << std::endl;
        }
    }
}

void EnergyObjective::preProcess(const VectorXd &y) const {
    VectorXT yTemp = y;
    if (!optimizeWeights) {
        yTemp = tessellation->combineVerticesParams(y, paramsSave);
    }

    double infp = 10;
    VectorXd infbox(8 * 4);
    infbox << -infp, -infp, -infp, 0,
            -infp, -infp, infp, 0,
            -infp, infp, -infp, 0,
            -infp, infp, infp, 0,
            infp, -infp, -infp, 0,
            infp, -infp, infp, 0,
            infp, infp, -infp, 0,
            infp, infp, infp, 0;
    VectorXd y_with_infbox(yTemp.rows() + infbox.rows());
    y_with_infbox << yTemp, infbox;

    VectorXT vertices, params;
    tessellation->separateVerticesParams(y_with_infbox, vertices, params);
    tessellation->tessellate(vertices, params);
}

double EnergyObjective::evaluate(const VectorXd &y) const {
    preProcess(y);

    double value = 0;
    if (!tessellation->isValid) {
        std::cout << "eval invalid" << std::endl;
        return 1e10;
    }

    for (Cell cell: tessellation->cells) {
        CellValue cellValue(cell);
        energyFunction.getValue(tessellation, cellValue);
        value += cellValue.value;
    }

    std::cout << "eval " << value << std::endl;
    return value;
}

void EnergyObjective::addGradientTo(const VectorXd &y, VectorXd &grad) const {
    grad += get_dOdc(y);
}

struct CoolStruct {
    int gen_c_idx;
    int cell_node_idx;
    int nodepos_start_idx;

    CoolStruct(int i0, int i1, int i2) {
        gen_c_idx = i0;
        cell_node_idx = i1;
        nodepos_start_idx = i2;
    }
};

VectorXd EnergyObjective::get_dOdc(const VectorXd &y) const {
    preProcess(y);

    VectorXT gradient = VectorXT::Zero(y.rows());
    if (!tessellation->isValid) {
        std::cout << "grad invalid" << std::endl;
        return gradient;
    }

    for (Cell cell: tessellation->cells) {
        CellValue cellValue(cell);
        energyFunction.getGradient(tessellation, cellValue);
        for (auto n: cell.nodeIndices) {
            Node node = n.first;
            int nodeIdx = n.second;
            NodePosition nodePos = tessellation->nodes[node];

            std::vector<CoolStruct> coolStructs;
            switch (node.type) {
                case STANDARD:
                    for (int i = 0; i < 4; i++) {
                        coolStructs.emplace_back(node.gen[i], nodeIdx, i * 4);
                    }
                    break;
                case B_FACE:
                    for (int i = 0; i < 3; i++) {
                        coolStructs.emplace_back(node.gen[i + 1], nodeIdx, i * 4);
                    }
                    break;
                case B_EDGE:
                    for (int i = 0; i < 2; i++) {
                        coolStructs.emplace_back(node.gen[i + 2], nodeIdx, i * 4);
                    }
                    break;
                default:
                    break;
            }

            for (CoolStruct cs: coolStructs) {
                gradient.segment(cs.gen_c_idx * optDims, optDims) +=
                        cellValue.gradient.segment<3>(cs.cell_node_idx * 3).transpose() *
                        nodePos.grad.block(0, cs.nodepos_start_idx, 3, optDims); // dxdc^T * dFdx
            }
        }
        gradient.segment(cell.cellIndex * optDims, optDims) += cellValue.gradient.segment(cellValue.gradient.rows() - 4,
                                                                                          optDims); // dFdc
    }

    std::cout << "gradient norm " << gradient.norm() << std::endl;
    return gradient;
}

void EnergyObjective::getHessian(const VectorXd &y, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(y);
}

Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &y) const {
    preProcess(y);

    MatrixXT hessian = MatrixXT::Zero(y.rows(), y.rows());
    if (!tessellation->isValid) {
        return hessian.sparseView();
    }

    for (Cell cell: tessellation->cells) {
        CellValue cellValue(cell);
        energyFunction.getGradient(tessellation, cellValue);
        energyFunction.getHessian(tessellation, cellValue);
        int numNodes = cell.nodeIndices.size();

        for (auto n0: cell.nodeIndices) {
            Node node0 = n0.first;
            int nodeIdx0 = n0.second;
            NodePosition nodePos0 = tessellation->nodes[node0];

            std::vector<CoolStruct> coolStructs0;
            switch (node0.type) {
                case STANDARD:
                    for (int i = 0; i < 4; i++) {
                        coolStructs0.emplace_back(node0.gen[i], nodeIdx0, i * 4);
                    }
                    break;
                case B_FACE:
                    for (int i = 0; i < 3; i++) {
                        coolStructs0.emplace_back(node0.gen[i + 1], nodeIdx0, i * 4);
                    }
                    break;
                case B_EDGE:
                    for (int i = 0; i < 2; i++) {
                        coolStructs0.emplace_back(node0.gen[i + 2], nodeIdx0, i * 4);
                    }
                    break;
                default:
                    break;
            }

            for (CoolStruct cs0: coolStructs0) {
                MatrixXT temp = cellValue.hessian.block(numNodes * 3, cs0.cell_node_idx * 3, optDims, 3) *
                                nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, optDims); // d2Fdcdx * dxdc

                hessian.block(cs0.gen_c_idx * optDims, cell.cellIndex * optDims, optDims, optDims) += temp.transpose();
                hessian.block(cell.cellIndex * optDims, cs0.gen_c_idx * optDims, optDims, optDims) += temp;

                for (CoolStruct cs1: coolStructs0) {
                    for (int i = 0; i < 3; i++) {
                        hessian.block(cs0.gen_c_idx * optDims, cs1.gen_c_idx * optDims, optDims, optDims) +=
                                nodePos0.hess[i].block(cs0.nodepos_start_idx, cs1.nodepos_start_idx, optDims, optDims) *
                                cellValue.gradient(cs0.cell_node_idx * 3 + i); // d2xdc2 * dFdx
                    }
                }
            }

            for (auto n1: cell.nodeIndices) {
                Node node1 = n1.first;
                int nodeIdx1 = n1.second;
                NodePosition nodePos1 = tessellation->nodes[node1];

                std::vector<CoolStruct> coolStructs1;
                switch (node1.type) {
                    case STANDARD:
                        for (int i = 0; i < 4; i++) {
                            coolStructs1.emplace_back(node1.gen[i], nodeIdx1, i * 4);
                        }
                        break;
                    case B_FACE:
                        for (int i = 0; i < 3; i++) {
                            coolStructs1.emplace_back(node1.gen[i + 1], nodeIdx1, i * 4);
                        }
                        break;
                    case B_EDGE:
                        for (int i = 0; i < 2; i++) {
                            coolStructs1.emplace_back(node1.gen[i + 2], nodeIdx1, i * 4);
                        }
                        break;
                    default:
                        break;
                }

                for (CoolStruct cs0: coolStructs0) {
                    for (CoolStruct cs1: coolStructs1) {
                        hessian.block(cs0.gen_c_idx * optDims, cs1.gen_c_idx * optDims, optDims, optDims) +=
                                nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, optDims).transpose() *
                                cellValue.hessian.block<3, 3>(cs0.cell_node_idx * 3, cs1.cell_node_idx * 3) *
                                nodePos1.grad.block(0, cs1.nodepos_start_idx, 3, optDims); // dxdc^T * d2Fdx2 * dxdc
                    }
                }
            }
        }
        hessian.block(cell.cellIndex * optDims, cell.cellIndex * optDims, optDims, optDims) +=
                cellValue.hessian.block(cellValue.hessian.rows() - 4, cellValue.hessian.cols() - 4, optDims,
                                        optDims); // d2Fdc2
    }

    return hessian.sparseView();
}
