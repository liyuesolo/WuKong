#include "../../include/Energy/EnergyObjective.h"
#include "../../include/Energy/CellFunctionEnergy.h"
#include <chrono>
#include <tbb/tbb.h>

#define PRINT_INTERMEDIATE_TIMES true
#define PRINT_TOTAL_TIME true

static void
printTime(std::chrono::high_resolution_clock::time_point tstart, std::string description = "", bool final = false) {
    if (PRINT_INTERMEDIATE_TIMES || (final && PRINT_TOTAL_TIME)) {
        const auto tcurr = std::chrono::high_resolution_clock::now();
        std::cout << description << "Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(tcurr - tstart).count() * 1.0e-6
                  << std::endl;
    }
}


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

void EnergyObjective::minimize(GradientDescentLineSearch *minimizer, VectorXd &y, bool optimizeWeights_) const {
    optimizeWeights = optimizeWeights_;
    optDims = optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!optimizeWeights) {
        VectorXd verts;
        tessellation->separateVerticesParams(y.head(y.rows() - tessellation->boundary->nfree), verts, paramsSave);
        yTemp.resize(verts.rows() + tessellation->boundary->nfree);
        yTemp << verts, y.tail(tessellation->boundary->nfree);
    }

    minimizer->minimize(this, yTemp);

    if (optimizeWeights) {
        y = yTemp;
    } else {
        VectorXT c = tessellation->combineVerticesParams(yTemp.head(yTemp.rows() - tessellation->boundary->nfree),
                                                         paramsSave);
        y.resize(c.rows() + tessellation->boundary->nfree);
        y << c, yTemp.tail(tessellation->boundary->nfree);
    }
}

void EnergyObjective::check_gradients(const VectorXd &y, bool optimizeWeights_) const {
    optimizeWeights = optimizeWeights_;
    optDims = optimizeWeights ? 4 : 3;
    VectorXd yTemp = y;
    if (!optimizeWeights) {
        VectorXd verts;
        tessellation->separateVerticesParams(y.head(y.rows() - tessellation->boundary->nfree), verts, paramsSave);
        yTemp.resize(verts.rows() + tessellation->boundary->nfree);
        yTemp << verts, y.tail(tessellation->boundary->nfree);
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

        std::cout << "f[" << i << "] " << f << " " << fp << " " << (fp - f) / eps << " "
                  << (fp2 - f) / (2 * eps) << " " << grad(i)
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
    VectorXT c = y.head(y.rows() - tessellation->boundary->nfree);
    if (!optimizeWeights) {
        c = tessellation->combineVerticesParams(y.head(y.rows() - tessellation->boundary->nfree), paramsSave);
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
    VectorXd c_with_infbox(c.rows() + infbox.rows());
    c_with_infbox << c, infbox;

    VectorXT vertices, params;
    tessellation->separateVerticesParams(c_with_infbox, vertices, params);
    tessellation->tessellate(vertices, params, y.tail(tessellation->boundary->nfree));
}

double EnergyObjective::evaluate(const VectorXd &y) const {
    preProcess(y);

    double value = 0;
    if (!tessellation->isValid) {
//        std::cout << "eval invalid" << std::endl;
        return 1e10;
    }

    for (Cell cell: tessellation->cells) {
        CellValue cellValue(cell);
        energyFunction.getValue(tessellation, cellValue);
        value += cellValue.value;
    }
    value += tessellation->boundary->computeEnergy();

//    std::cout << "energy value: " << value << std::endl;
    return value;
}

void EnergyObjective::addGradientTo(const VectorXd &y, VectorXd &grad) const {
    grad += get_dOdc(y);
}

struct CoolStruct {
    bool gen_is_boundary;
    int gen_idx;
    int cell_node_idx;
    int nodepos_start_idx;

    CoolStruct(bool b, int i0, int i1, int i2) {
        gen_is_boundary = b;
        gen_idx = i0;
        cell_node_idx = i1;
        nodepos_start_idx = i2;
    }
};

VectorXd EnergyObjective::get_dOdc(const VectorXd &y) const {
    preProcess(y);

    const auto tstart = std::chrono::high_resolution_clock::now();

    VectorXT gradient = VectorXT::Zero(y.rows());
    if (!tessellation->isValid) {
        std::cout << "grad invalid" << std::endl;
        return gradient;
    }

    int nc = tessellation->cells.size() * optDims;
    int nx = tessellation->nodes.size() * 3;
    int nv = tessellation->boundary->v.size() * 3;
    int np = tessellation->boundary->nfree;

    VectorXT dFdx = VectorXT::Zero(nx);
    VectorXT dFdc = VectorXT::Zero(nc);
    VectorXT dFdp = VectorXT::Zero(np);

//    for (Cell cell: tessellation->cells) {
//        CellValue cellValue(cell);
//        energyFunction.getGradient(tessellation, cellValue);
//
//        for (auto n: cell.nodeIndices) {
//            Node node = n.first;
//            int nodeIdxInCell = n.second;
//            NodePosition nodePos = tessellation->nodes[node];
//
//            dFdx.segment<3>(nodePos.ix * 3) += cellValue.gradient.segment<3>(nodeIdxInCell * 3);
//        }
//        dFdc.segment(cell.cellIndex * optDims, optDims) += cellValue.gradient.segment(cell.nodeIndices.size() * 3,
//                                                                                      optDims);
//    }

    tbb::concurrent_vector<double> dFdxBuilder(nx);
    std::fill(dFdxBuilder.begin(), dFdxBuilder.end(), 0);

    tbb::parallel_for_each(tessellation->cells.begin(), tessellation->cells.end(), [&](Cell cell) {
        CellValue cellValue(cell);
        energyFunction.getGradient(tessellation, cellValue);

        for (auto n: cell.nodeIndices) {
            Node node = n.first;
            int nodeIdxInCell = n.second;
            NodePosition nodePos = tessellation->nodes[node];

            dFdxBuilder[nodePos.ix * 3 + 0] += cellValue.gradient(nodeIdxInCell * 3 + 0);
            dFdxBuilder[nodePos.ix * 3 + 1] += cellValue.gradient(nodeIdxInCell * 3 + 1);
            dFdxBuilder[nodePos.ix * 3 + 2] += cellValue.gradient(nodeIdxInCell * 3 + 2);
        }
        dFdc.segment(cell.cellIndex * optDims, optDims) += cellValue.gradient.segment(cell.nodeIndices.size() * 3,
                                                                                      optDims); // TODO: Should be threadsafe?
    });
    dFdx = Eigen::Map<VectorXT>(&dFdxBuilder[0], nx);
    dFdp = tessellation->boundary->computeEnergyGradient();

    gradient.segment(0, nc) = dFdx.transpose() * tessellation->dxdc + dFdc.transpose();
    gradient.tail(np) = dFdx.transpose() * tessellation->dxdv * tessellation->dvdp + dFdp.transpose();

    printTime(tstart, "Energy gradient ", false);
    return gradient;
}

void EnergyObjective::getHessian(const VectorXd &y, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(y);
}

Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &y) const {
    preProcess(y);

    const auto tstart = std::chrono::high_resolution_clock::now();

    MatrixXT hessian = MatrixXT::Zero(y.rows(), y.rows());
    if (!tessellation->isValid) {
        return hessian.sparseView();
    }

    int nc = tessellation->cells.size() * optDims;
    int nx = tessellation->nodes.size() * 3;
    int nv = tessellation->boundary->v.size() * 3;
    int np = tessellation->boundary->nfree;

//    VectorXT dFdx = VectorXT::Zero(nx);
//    MatrixXT d2Fdc2 = MatrixXT::Zero(nc, nc);
//    MatrixXT d2Fdcdx = MatrixXT::Zero(nc, nx);
//    MatrixXT d2Fdx2 = MatrixXT::Zero(nx, nx);
//    MatrixXT d2Fdp2 = MatrixXT::Zero(np, np);

//    printTime(tstart, "Hessian initialization ");

//    for (Cell cell: tessellation->cells) {
//        CellValue cellValue(cell);
//        energyFunction.getGradient(tessellation, cellValue);
//        energyFunction.getHessian(tessellation, cellValue);
//
//        for (auto n0: cell.nodeIndices) {
//            Node node0 = n0.first;
//            int nodeIdxInCell0 = n0.second;
//            NodePosition nodePos0 = tessellation->nodes[node0];
//
//            dFdx.segment<3>(nodePos0.ix * 3) += cellValue.gradient.segment<3>(nodeIdxInCell0 * 3);
//            d2Fdcdx.block(cell.cellIndex * optDims, nodePos0.ix * 3, optDims, 3) += cellValue.hessian.block(
//                    cell.nodeIndices.size() * 3, nodeIdxInCell0 * 3, optDims, 3);
//
//            for (auto n1: cell.nodeIndices) {
//                Node node1 = n1.first;
//                int nodeIdxInCell1 = n1.second;
//                NodePosition nodePos1 = tessellation->nodes[node1];
//
//                d2Fdx2.block(nodePos0.ix * 3, nodePos1.ix * 3, 3, 3) += cellValue.hessian.block(nodeIdxInCell0 * 3,
//                                                                                                nodeIdxInCell1 * 3, 3,
//                                                                                                3);
//            }
//        }
//
//        d2Fdc2.block(cell.cellIndex * optDims, cell.cellIndex * optDims, optDims, optDims) += cellValue.hessian.block(
//                cell.nodeIndices.size() * 3, cell.nodeIndices.size() * 3,
//                optDims, optDims);
//    }

    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsdFdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdc2;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdcdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdx2;

    printTime(tstart, "Hessian initialization ");

    tbb::parallel_for_each(tessellation->cells.begin(), tessellation->cells.end(), [&](Cell cell) {
        CellValue cellValue(cell);
        energyFunction.getGradient(tessellation, cellValue);
        energyFunction.getHessian(tessellation, cellValue);

        for (auto n0: cell.nodeIndices) {
            Node node0 = n0.first;
            int nodeIdxInCell0 = n0.second;
            NodePosition nodePos0 = tessellation->nodes[node0];

            for (int ii = 0; ii < 3; ii++) {
                tripletsdFdx.emplace_back(nodePos0.ix * 3 + ii, 0, cellValue.gradient(nodeIdxInCell0 * 3 + ii));
            }
            for (int ii = 0; ii < optDims; ii++) {
                for (int jj = 0; jj < 3; jj++) {
                    tripletsd2Fdcdx.emplace_back(cell.cellIndex * optDims + ii, nodePos0.ix * 3 + jj, cellValue.hessian(
                            cell.nodeIndices.size() * 3 + ii, nodeIdxInCell0 * 3 + jj));
                }
            }

            for (auto n1: cell.nodeIndices) {
                Node node1 = n1.first;
                int nodeIdxInCell1 = n1.second;
                NodePosition nodePos1 = tessellation->nodes[node1];
                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        tripletsd2Fdx2.emplace_back(nodePos0.ix * 3 + ii, nodePos1.ix * 3 + jj,
                                                    cellValue.hessian(nodeIdxInCell0 * 3 + ii,
                                                                      nodeIdxInCell1 * 3 + jj));
                    }
                }
            }
        }

        for (int ii = 0; ii < optDims; ii++) {
            for (int jj = 0; jj < optDims; jj++) {
                tripletsd2Fdc2.emplace_back(cell.cellIndex * optDims + ii, cell.cellIndex * optDims + jj,
                                            cellValue.hessian(
                                                    cell.nodeIndices.size() * 3 + ii,
                                                    cell.nodeIndices.size() * 3 + jj));
            }
        }
    });

//    for (Cell cell: tessellation->cells) {
//        CellValue cellValue(cell);
//        energyFunction.getGradient(tessellation, cellValue);
//        energyFunction.getHessian(tessellation, cellValue);
//
//        for (auto n0: cell.nodeIndices) {
//            Node node0 = n0.first;
//            int nodeIdxInCell0 = n0.second;
//            NodePosition nodePos0 = tessellation->nodes[node0];
//
//            for (int ii = 0; ii < 3; ii++) {
//                tripletsdFdx.emplace_back(nodePos0.ix * 3 + ii, 0, cellValue.gradient(nodeIdxInCell0 * 3 + ii));
//            }
//            for (int ii = 0; ii < optDims; ii++) {
//                for (int jj = 0; jj < 3; jj++) {
//                    tripletsd2Fdcdx.emplace_back(cell.cellIndex * optDims + ii, nodePos0.ix * 3 + jj, cellValue.hessian(
//                            cell.nodeIndices.size() * 3 + ii, nodeIdxInCell0 * 3 + jj));
//                }
//            }
//
//            for (auto n1: cell.nodeIndices) {
//                Node node1 = n1.first;
//                int nodeIdxInCell1 = n1.second;
//                NodePosition nodePos1 = tessellation->nodes[node1];
//                for (int ii = 0; ii < 3; ii++) {
//                    for (int jj = 0; jj < 3; jj++) {
//                        tripletsd2Fdx2.emplace_back(nodePos0.ix * 3 + ii, nodePos1.ix * 3 + jj,
//                                                    cellValue.hessian(nodeIdxInCell0 * 3 + ii,
//                                                                      nodeIdxInCell1 * 3 + jj));
//                    }
//                }
//            }
//        }
//
//        for (int ii = 0; ii < optDims; ii++) {
//            for (int jj = 0; jj < optDims; jj++) {
//                tripletsd2Fdc2.emplace_back(cell.cellIndex * optDims + ii, cell.cellIndex * optDims + jj,
//                                            cellValue.hessian(
//                                                    cell.nodeIndices.size() * 3 + ii,
//                                                    cell.nodeIndices.size() * 3 + jj));
//            }
//        }
//    }
    printTime(tstart, "Hessian iterate over cells ");

    Eigen::SparseMatrix<double> dFdx(nx, 1);
    Eigen::SparseMatrix<double> d2Fdc2(nc, nc);
    Eigen::SparseMatrix<double> d2Fdcdx(nc, nx);
    Eigen::SparseMatrix<double> d2Fdx2(nx, nx);
//    dFdx.setFromTriplets(tripletsdFdx.begin(), tripletsdFdx.end());
//    d2Fdc2.setFromTriplets(tripletsd2Fdc2.begin(), tripletsd2Fdc2.end());
//    d2Fdcdx.setFromTriplets(tripletsd2Fdcdx.begin(), tripletsd2Fdcdx.end());
//    d2Fdx2.setFromTriplets(tripletsd2Fdx2.begin(), tripletsd2Fdx2.end());

    {
        tbb::task_group g;
        g.run([&] {
            dFdx.setFromTriplets(tripletsdFdx.begin(), tripletsdFdx.end());
        });
        g.run([&] {
            d2Fdc2.setFromTriplets(tripletsd2Fdc2.begin(), tripletsd2Fdc2.end());
        });
        g.run([&] {
            d2Fdcdx.setFromTriplets(tripletsd2Fdcdx.begin(), tripletsd2Fdcdx.end());
        });
        g.run([&] {
            d2Fdx2.setFromTriplets(tripletsd2Fdx2.begin(), tripletsd2Fdx2.end());
        });
        g.wait();
    }

    printTime(tstart, "Hessian build from triplets ");

    MatrixXT d2Fdp2 = tessellation->boundary->computeEnergyHessian();

    printTime(tstart, "Hessian boundary ");

    MatrixXT sum_dFdx_d2xdc2 = MatrixXT::Zero(nc, nc);
    MatrixXT sum_dFdx_d2xdcdv = MatrixXT::Zero(nc, nv);
    MatrixXT sum_dFdx_d2xdv2 = MatrixXT::Zero(nv, nv);
    for (int i = 0; i < nx; i++) {
        sum_dFdx_d2xdc2 += dFdx.coeff(i, 0) * tessellation->d2xdc2[i];
        sum_dFdx_d2xdcdv += dFdx.coeff(i, 0) * tessellation->d2xdcdv[i];
        sum_dFdx_d2xdv2 += dFdx.coeff(i, 0) * tessellation->d2xdv2[i];
    }
    MatrixXT sum_dFdx_dxdv_d2vdp2 = MatrixXT::Zero(np, np);
    VectorXT temp_dFdx_dxdv = dFdx.transpose() * tessellation->dxdv;
    for (int i = 0; i < nv; i++) {
        sum_dFdx_dxdv_d2vdp2 += temp_dFdx_dxdv(i) * tessellation->d2vdp2[i];
    }

//    MatrixXT sum_dFdx_d2xdc2 = MatrixXT::Zero(nc, nc);
//    MatrixXT sum_dFdx_d2xdcdv = MatrixXT::Zero(nc, nv);
//    MatrixXT sum_dFdx_d2xdv2 = MatrixXT::Zero(nv, nv);
//    MatrixXT sum_dFdx_dxdv_d2vdp2 = MatrixXT::Zero(np, np);
//    {
//        tbb::task_group g;
//        g.run([&] {
//            for (int i = 0; i < nx; i++) {
//                sum_dFdx_d2xdc2 += dFdx.coeff(i, 0) * tessellation->d2xdc2[i];
//            }
//        });
//        g.run([&] {
//            for (int i = 0; i < nx; i++) {
//                sum_dFdx_d2xdcdv += dFdx.coeff(i, 0) * tessellation->d2xdcdv[i];
//            }
//        });
//        g.run([&] {
//            for (int i = 0; i < nx; i++) {
//                sum_dFdx_d2xdv2 += dFdx.coeff(i, 0) * tessellation->d2xdv2[i];
//            }
//        });
//        g.run([&] {
//            VectorXT temp_dFdx_dxdv = dFdx.transpose() * tessellation->dxdv;
//            for (int i = 0; i < nv; i++) {
//                sum_dFdx_dxdv_d2vdp2 += temp_dFdx_dxdv(i) * tessellation->d2vdp2[i];
//            }
//        });
//        g.wait();
//    }

    printTime(tstart, "Hessian compute sums ");

    Eigen::SparseMatrix<double> dxdc = tessellation->dxdc;
    Eigen::SparseMatrix<double> dxdcT = dxdc.transpose();
    MatrixXT d2Fdxdc = d2Fdcdx.transpose();
    Eigen::SparseMatrix<double> dvdp = tessellation->dvdp;
    Eigen::SparseMatrix<double> dvdpT = dvdp.transpose();
    Eigen::SparseMatrix<double> dxdv_dvdp = tessellation->dxdv * dvdp;
    Eigen::SparseMatrix<double> dxdv_dvdpT = dxdv_dvdp.transpose();

//    MatrixXT dxdcT_d2Fdx2 = dxdcT * d2Fdx2;
//    MatrixXT d2Fdx2_dxdv_dvdp = d2Fdx2 * dxdv_dvdp;

    printTime(tstart, "Hessian precompute quantities ");

//    MatrixXT D2FDC2 = dxdcT * d2Fdx2 * dxdc +
//                      dxdcT * d2Fdxdc +
//                      d2Fdcdx * dxdc +
//                      sum_dFdx_d2xdc2 +
//                      d2Fdc2;
//    MatrixXT D2FDCDP = dxdcT * d2Fdx2_dxdv_dvdp +
//                       d2Fdcdx * dxdv_dvdp +
//                       sum_dFdx_d2xdcdv * dvdp;
//    MatrixXT D2FDP2 = dxdv_dvdpT * d2Fdx2_dxdv_dvdp +
//                      dvdpT * sum_dFdx_d2xdv2 * dvdp +
//                      sum_dFdx_dxdv_d2vdp2 +
//                      d2Fdp2;

    MatrixXT D2FDC2, D2FDCDP, D2FDP2;
    {
        tbb::task_group g;
        g.run([&] {
            D2FDC2 = dxdcT * d2Fdx2 * dxdc +
                     dxdcT * d2Fdxdc +
                     d2Fdcdx * dxdc +
                     sum_dFdx_d2xdc2 +
                     d2Fdc2;
        });
        g.run([&] {
            D2FDCDP = dxdcT * d2Fdx2 * dxdv_dvdp +
                      d2Fdcdx * dxdv_dvdp +
                      sum_dFdx_d2xdcdv * dvdp;
        });
        g.run([&] {
            D2FDP2 = dxdv_dvdpT * d2Fdx2 * dxdv_dvdp +
                     dvdpT * sum_dFdx_d2xdv2 * dvdp +
                     sum_dFdx_dxdv_d2vdp2 +
                     d2Fdp2;
        });
        g.wait();
    }

    hessian.block(0, 0, nc, nc) = D2FDC2;
    hessian.block(nc, 0, np, nc) = D2FDCDP.transpose();
    hessian.block(0, nc, nc, np) = D2FDCDP;
    hessian.block(nc, nc, np, np) = D2FDP2;

    printTime(tstart, "Energy hessian ", true);

    return hessian.sparseView();
}

//Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &y) const {
//    preProcess(y);
//
//    MatrixXT hessian = MatrixXT::Zero(y.rows(), y.rows());
//    if (!tessellation->isValid) {
//        return hessian.sparseView();
//    }
//
//    int nc = tessellation->cells.size() * optDims;
//    int nx = tessellation->nodes.size() * 3;
//    int nv = tessellation->boundary->v.size() * 3;
//    int np = tessellation->boundary->nfree;
//
//    VectorXT dFdx = VectorXT::Zero(nx);
//    MatrixXT d2Fdc2 = MatrixXT::Zero(nc, nc);
//    MatrixXT d2Fdcdx = MatrixXT::Zero(nc, nx);
//    MatrixXT d2Fdx2 = MatrixXT::Zero(nx, nx);
//    MatrixXT d2Fdp2 = MatrixXT::Zero(np, np);
//
//    int bbb = 0;
//    for (Cell cell: tessellation->cells) {
//        std::cout << "hessian cell " << bbb++ << std::endl;
//        CellValue cellValue(cell);
//        energyFunction.getGradient(tessellation, cellValue);
//        energyFunction.getHessian(tessellation, cellValue);
//        int numNodes = cell.nodeIndices.size();
//
//        for (auto n0: cell.nodeIndices) {
//            Node node0 = n0.first;
//            int nodeIdx0 = n0.second;
//            NodePosition nodePos0 = tessellation->nodes[node0];
//
//            std::vector<CoolStruct> coolStructs0;
//            switch (node0.type) {
//                case STANDARD:
//                    for (int i = 0; i < 4; i++) {
//                        coolStructs0.emplace_back(false, node0.gen[i], nodeIdx0, i * 4);
//                    }
//                    break;
//                case B_FACE:
//                    for (int i = 0; i < 3; i++) {
//                        coolStructs0.emplace_back(false, node0.gen[i + 1], nodeIdx0, i * 4);
//                    }
//                    for (int i = 0; i < 3; i++) {
//                        coolStructs0.emplace_back(true, tessellation->boundary->f[node0.gen[0]].vertices(i), nodeIdx0,
//                                                  3 * 4 + i * 3);
//                    }
//                    break;
//                case B_EDGE:
//                    for (int i = 0; i < 2; i++) {
//                        coolStructs0.emplace_back(false, node0.gen[i + 2], nodeIdx0, i * 4);
//                    }
//                    for (int i = 0; i < 2; i++) {
//                        coolStructs0.emplace_back(true, node0.gen[i], nodeIdx0, 2 * 4 + i * 3);
//                    }
//                    break;
//                case B_VERTEX:
//                    coolStructs0.emplace_back(true, node0.gen[0], nodeIdx0, 0);
//                default:
//                    break;
//            }
//
//            for (CoolStruct cs0: coolStructs0) {
//                if (cs0.gen_is_boundary) {
//                    MatrixXT d2Fdcdx_dxdv_dvdp =
//                            cellValue.hessian.block(numNodes * 3, cs0.cell_node_idx * 3, optDims, 3) *
//                            nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, 3) *
//                            tessellation->boundary->v[cs0.gen_idx].grad; // d2Fdcdx * dxdv * dvdp
//
//                    hessian.block(cell.cellIndex * optDims, hessian.rows() - tessellation->boundary->nfree, optDims,
//                                  tessellation->boundary->nfree) += d2Fdcdx_dxdv_dvdp;
//                    hessian.block(hessian.rows() - tessellation->boundary->nfree, cell.cellIndex * optDims,
//                                  tessellation->boundary->nfree, optDims) += d2Fdcdx_dxdv_dvdp.transpose();
//
//                    for (int i = 0; i < 3; i++) {
//                        for (int j = 0; j < 3; j++) {
//                            hessian.block(hessian.rows() - tessellation->boundary->nfree,
//                                          hessian.rows() - tessellation->boundary->nfree, tessellation->boundary->nfree,
//                                          tessellation->boundary->nfree) +=
//                                    cellValue.gradient(cs0.cell_node_idx * 3 + i) *
//                                    nodePos0.grad(i, cs0.nodepos_start_idx + j) *
//                                    tessellation->boundary->v[cs0.gen_idx].hess[j]; // dFdx * dxdv * d2vdp2
//                        }
//                    }
//                } else {
//                    MatrixXT d2Fdcdx_dxdc = cellValue.hessian.block(numNodes * 3, cs0.cell_node_idx * 3, optDims, 3) *
//                                            nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, optDims); // d2Fdcdx * dxdc
//
//                    hessian.block(cell.cellIndex * optDims, cs0.gen_idx * optDims, optDims, optDims) += d2Fdcdx_dxdc;
//                    hessian.block(cs0.gen_idx * optDims, cell.cellIndex * optDims, optDims,
//                                  optDims) += d2Fdcdx_dxdc.transpose();
//                }
//
//                for (CoolStruct cs1: coolStructs0) {
//                    for (int i = 0; i < 3; i++) {
//                        if (cs0.gen_is_boundary && cs1.gen_is_boundary) {
//                            hessian.block(hessian.rows() - tessellation->boundary->nfree,
//                                          hessian.rows() - tessellation->boundary->nfree, tessellation->boundary->nfree,
//                                          tessellation->boundary->nfree) +=
//                                    tessellation->boundary->v[cs0.gen_idx].grad.transpose() *
//                                    (cellValue.gradient(cs0.cell_node_idx * 3 + i) *
//                                     nodePos0.hess[i].block(cs0.nodepos_start_idx, cs1.nodepos_start_idx, 3,
//                                                            3)).sparseView() *
//                                    tessellation->boundary->v[cs1.gen_idx].grad; // dvdp^T * dFdx * d2xdv2 * dvdp
//                        } else if (cs0.gen_is_boundary) {
//                            hessian.block(hessian.rows() - tessellation->boundary->nfree, cs1.gen_idx * optDims,
//                                          tessellation->boundary->nfree, optDims) +=
//                                    tessellation->boundary->v[cs0.gen_idx].grad.transpose() *
//                                    cellValue.gradient(cs0.cell_node_idx * 3 + i) *
//                                    nodePos0.hess[i].block(cs0.nodepos_start_idx, cs1.nodepos_start_idx, 3,
//                                                           optDims); // dvdp^T * dFdx * d2xdvdc
//                        } else if (cs1.gen_is_boundary) {
//                            hessian.block(cs0.gen_idx * optDims, hessian.rows() - tessellation->boundary->nfree,
//                                          optDims, tessellation->boundary->nfree) +=
//                                    cellValue.gradient(cs0.cell_node_idx * 3 + i) *
//                                    nodePos0.hess[i].block(cs0.nodepos_start_idx, cs1.nodepos_start_idx, optDims, 3) *
//                                    tessellation->boundary->v[cs1.gen_idx].grad; // dFdx * d2xdcdv * dvdp (transpose of previous case)
//                        } else {
//                            hessian.block(cs0.gen_idx * optDims, cs1.gen_idx * optDims, optDims, optDims) +=
//                                    cellValue.gradient(cs0.cell_node_idx * 3 + i) *
//                                    nodePos0.hess[i].block(cs0.nodepos_start_idx, cs1.nodepos_start_idx, optDims,
//                                                           optDims); // dFdx * d2xdc2
//                        }
//                    }
//                }
//            }
//
//            for (auto n1: cell.nodeIndices) {
//                Node node1 = n1.first;
//                int nodeIdx1 = n1.second;
//                NodePosition nodePos1 = tessellation->nodes[node1];
//
//                std::vector<CoolStruct> coolStructs1;
//                switch (node1.type) {
//                    case STANDARD:
//                        for (int i = 0; i < 4; i++) {
//                            coolStructs1.emplace_back(false, node1.gen[i], nodeIdx1, i * 4);
//                        }
//                        break;
//                    case B_FACE:
//                        for (int i = 0; i < 3; i++) {
//                            coolStructs1.emplace_back(false, node1.gen[i + 1], nodeIdx1, i * 4);
//                        }
//                        for (int i = 0; i < 3; i++) {
//                            coolStructs1.emplace_back(true, tessellation->boundary->f[node1.gen[0]].vertices(i),
//                                                      nodeIdx1,
//                                                      3 * 4 + i * 3);
//                        }
//                        break;
//                    case B_EDGE:
//                        for (int i = 0; i < 2; i++) {
//                            coolStructs1.emplace_back(false, node1.gen[i + 2], nodeIdx1, i * 4);
//                        }
//                        for (int i = 0; i < 2; i++) {
//                            coolStructs1.emplace_back(true, node1.gen[i], nodeIdx1, 2 * 4 + i * 3);
//                        }
//                        break;
//                    case B_VERTEX:
//                        coolStructs1.emplace_back(true, node1.gen[0], nodeIdx1, 0);
//                    default:
//                        break;
//                }
//
//                for (CoolStruct cs0: coolStructs0) {
//                    for (CoolStruct cs1: coolStructs1) {
//                        if (cs0.gen_is_boundary && cs1.gen_is_boundary) {
//                            hessian.block(hessian.rows() - tessellation->boundary->nfree,
//                                          hessian.rows() - tessellation->boundary->nfree, tessellation->boundary->nfree,
//                                          tessellation->boundary->nfree) +=
//                                    tessellation->boundary->v[cs0.gen_idx].grad.transpose() *
//                                    (nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, 3).transpose() *
//                                     cellValue.hessian.block<3, 3>(cs0.cell_node_idx * 3,
//                                                                   cs1.cell_node_idx * 3) *
//                                     nodePos1.grad.block(0, cs1.nodepos_start_idx, 3, 3)).sparseView() *
//                                    tessellation->boundary->v[cs1.gen_idx].grad; // (dxdv * dvdp)^T * d2Fdx2 * (dxdv * dvdp)
//                        } else if (cs0.gen_is_boundary) {
//                            hessian.block(hessian.rows() - tessellation->boundary->nfree, cs1.gen_idx * optDims,
//                                          tessellation->boundary->nfree, optDims) +=
//                                    tessellation->boundary->v[cs0.gen_idx].grad.transpose() *
//                                    nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, 3).transpose() *
//                                    cellValue.hessian.block<3, 3>(cs0.cell_node_idx * 3, cs1.cell_node_idx * 3) *
//                                    nodePos1.grad.block(0, cs1.nodepos_start_idx, 3,
//                                                        optDims); // (dxdv * dvdp)^T * d2Fdx2 * dxdc
//                        } else if (cs1.gen_is_boundary) {
//                            hessian.block(cs0.gen_idx * optDims, hessian.rows() - tessellation->boundary->nfree,
//                                          optDims, tessellation->boundary->nfree) +=
//                                    nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, optDims).transpose() *
//                                    cellValue.hessian.block<3, 3>(cs0.cell_node_idx * 3, cs1.cell_node_idx * 3) *
//                                    nodePos1.grad.block(0, cs1.nodepos_start_idx, 3, 3) *
//                                    tessellation->boundary->v[cs1.gen_idx].grad; // dxdc^T * d2Fdx2 * (dxdv * dvdp) - transpose of previous case
//                        } else {
//                            hessian.block(cs0.gen_idx * optDims, cs1.gen_idx * optDims, optDims, optDims) +=
//                                    nodePos0.grad.block(0, cs0.nodepos_start_idx, 3, optDims).transpose() *
//                                    cellValue.hessian.block<3, 3>(cs0.cell_node_idx * 3, cs1.cell_node_idx * 3) *
//                                    nodePos1.grad.block(0, cs1.nodepos_start_idx, 3, optDims); // dxdc^T * d2Fdx2 * dxdc
//                        }
//                    }
//                }
//            }
//        }
//        hessian.block(cell.cellIndex * optDims, cell.cellIndex * optDims, optDims, optDims) +=
//                cellValue.hessian.block(cellValue.hessian.rows() - 4, cellValue.hessian.cols() - 4, optDims,
//                                        optDims); // d2Fdc2
//    }
//    hessian.bottomRightCorner(tessellation->boundary->nfree,
//                              tessellation->boundary->nfree) += tessellation->boundary->computeEnergyHessian(); // d2Fdp2
//
//    return hessian.sparseView();
//}
