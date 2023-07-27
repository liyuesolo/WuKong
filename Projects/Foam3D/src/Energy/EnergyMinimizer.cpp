
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/edges.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <Eigen/CholmodSupport>
#include "../../include/Energy/EnergyMinimizer.h"

#include "../../include/Globals.h"

bool EnergyMinimizer::advanceOneStep(int step, VectorXT &u, bool dynamic_, bool optimizeWeights_) {
    Timer step_timer(true);

    dynamic = dynamic_;

    optimizeWeights = optimizeWeights_;
    optDims = optimizeWeights ? 4 : 3;
    VectorXT u_ = u;
    if (!optimizeWeights) {
        VectorXT verts;
        tessellation->separateVerticesParams(u.head(u.rows() - tessellation->boundary->nfree), verts, paramsSave);
        u_.resize(verts.rows() + tessellation->boundary->nfree);
        u_ << verts, u.tail(tessellation->boundary->nfree);
    }

    if (dynamic) {
        if (!dynamic_initialized) {
            dynamic_v_prev = VectorXT::Zero(u_.rows());
            dynamic_y_prev = u_;
            dynamic_initialized = true;
        } else if (dynamic_new_step) {
            dynamic_v_prev = (u_ - dynamic_y_prev) / dt;
            dynamic_y_prev = u_;
        }
        dynamic_new_step = false;
    }

    VectorXT residual(u_.rows());
    residual.setZero();

    T residual_norm = computeResidual(u_, residual);
    std::cout << "[Newton] computeResidual takes " << step_timer.elapsed_sec() << "s" << std::endl;
    step_timer.restart();
    // if (save_mesh)
    //     saveCellMesh(step);
    // std::cout << "[Newton] saveCellMesh takes " << step_timer.elapsed_sec() << "s" << std::endl;
    // if (verbose)
    std::cout << "[Newton] iter " << step << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: "
              << newton_tol << std::endl;

    if (residual_norm < newton_tol)
        return true;

    T dq_norm = lineSearchNewton(u_, residual);

    if (optimizeWeights) {
        u = u_;
    } else {
        VectorXT c = tessellation->combineVerticesParams(u_.head(u_.rows() - tessellation->boundary->nfree),
                                                         paramsSave);
        u.resize(c.rows() + tessellation->boundary->nfree);
        u << c, u_.tail(tessellation->boundary->nfree);
    }

    step_timer.stop();
    if (verbose)
        std::cout << "[Newton] step takes " << step_timer.elapsed_sec() << "s" << std::endl;

    if (step == max_newton_iter || dq_norm > 1e10)
        return true;

    return false;
}

T EnergyMinimizer::computeLineSearchInitStepsize(const VectorXT &_u, const VectorXT &du) {
    if (verbose)
        std::cout << "** step size **" << std::endl;
    T step_size = 1.0;
//    if (use_ipc) {
//        T ipc_step_size = computeCollisionFreeStepsize(_u, du);
//        step_size = std::min(step_size, ipc_step_size);
//        if (verbose)
//            std::cout << "after ipc step size: " << step_size << std::endl;
//    }

    // if (use_sphere_radius_bound && !sphere_bound_penalty && !use_sdf_boundary)
    // {
    //     T inside_membrane_step_size = computeInsideMembraneStepSize(_u, du);
    //     step_size = std::min(step_size, inside_membrane_step_size);
    //     if (verbose)
    //         std::cout << "after inside membrane step size: " << step_size << std::endl;
    // }

    // if (add_tet_vol_barrier)
    // {
    //     T inversion_free_step_size = computeInversionFreeStepSize(_u, du);
    //     // std::cout << "cell tet inversion free step size: " << inversion_free_step_size << std::endl;
    //     step_size = std::min(step_size, inversion_free_step_size);
    //     if (verbose)
    //         std::cout << "after tet inverison step size: " << step_size << std::endl;
    // }

    // if (add_yolk_tet_barrier)
    // {
    //     T inversion_free_step_size = computeYolkInversionFreeStepSize(_u, du);
    //     // std::cout << "yolk inversion free step size: " << inversion_free_step_size << std::endl;
    //     step_size = std::min(step_size, inversion_free_step_size);
    // }
    if (verbose)
        std::cout << "**       **" << std::endl;
    return step_size;
}

bool EnergyMinimizer::solveWoodburyCholmod(StiffnessMatrix &K, MatrixXT &UV,
                                           VectorXT &residual, VectorXT &du) {
    MatrixXT UVT = UV.transpose();

    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;

    T alpha = 10e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++) {
        // std::cout << i << std::endl;
        solver.factorize(K);
        // T time_factorize = t.elapsed_sec() - time_analyze;
        // std::cout << "\t factorize takes " << time_factorize << "s" << std::endl;
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue) {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1) {
            VectorXT v = UV.col(0);
            MatrixXT rhs(K.rows(), 2);
            rhs.col(0) = residual;
            rhs.col(1) = v;
            // VectorXT A_inv_g = solver.solve(residual);
            // VectorXT A_inv_u = solver.solve(v);
            MatrixXT A_inv_gu = solver.solve(rhs);

            T dem = 1.0 + v.dot(A_inv_gu.col(1));

            du.noalias() = A_inv_gu.col(0) - (A_inv_gu.col(0).dot(v)) * A_inv_gu.col(1) / dem;
        }
            // UV is actually only U, since UV is the same in the case
            // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            // for (int col = 0; col < UV.cols(); col++)
            // A_inv_U.col(col) = solver.solve(UV.col(col));
            A_inv_U = solver.solve(UV);

            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UVT * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UVT * A_inv_g;
        }


        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;

        bool solve_success = true;//(K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success) {
            t.stop();
            if (verbose) {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i
                          << " indefinite " << indefinite_count_reg_cnt
                          << " invalid search dir " << invalid_search_dir_cnt
                          << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        } else {
            // K = H + alpha * I;       
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            // {
            //     K.coeffRef(row, row) += alpha;
            // });  
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

bool EnergyMinimizer::linearSolve(StiffnessMatrix &K, VectorXT &residual, VectorXT &du) {
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;

    T alpha = 10e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    int i = 0;
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (; i < 50; i++) {
        // std::cout << i << std::endl;
        solver.factorize(K);
        // T time_factorize = t.elapsed_sec() - time_analyze;
        // std::cout << "\t factorize takes " << time_factorize << "s" << std::endl;
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue) {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;

        bool solve_success = true;//(K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction "
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success) {
            t.stop();
            if (verbose) {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i
                          << " indefinite " << indefinite_count_reg_cnt
                          << " invalid search dir " << invalid_search_dir_cnt
                          << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        } else {
            // K = H + alpha * I;
            // tbb::parallel_for(0, (int)K.rows(), [&](int row)
            // {
            //     K.coeffRef(row, row) += alpha;
            // });
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

void EnergyMinimizer::preProcess(const VectorXT &y, bool needGradients) const {
    VectorXT c = y.head(y.rows() - tessellation->boundary->nfree);
    if (!optimizeWeights) {
        c = tessellation->combineVerticesParams(y.head(y.rows() - tessellation->boundary->nfree), paramsSave);
    }

    VectorXT vertices, params;
    tessellation->separateVerticesParams(c, vertices, params);
    tessellation->tessellate(vertices, params, y.tail(tessellation->boundary->nfree), needGradients);
}

void EnergyMinimizer::buildSystemMatrixWoodbury(const VectorXT &_u, StiffnessMatrix &K, MatrixXT &UV) {
    preProcess(_u);

    const auto tstart = std::chrono::high_resolution_clock::now();

    MatrixXT hessian_K = MatrixXT::Zero(_u.rows(), _u.rows());
    if (!tessellation->isValid) {
        K = hessian_K.sparseView();
    }

    int nc = tessellation->cells.size() * optDims;
    int nx = tessellation->nodes.size() * 3;
    int nv = tessellation->boundary->v.size() * 3;
    int np = tessellation->boundary->nfree;

    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsdFdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdc2;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdcdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdx2;

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

    Eigen::SparseMatrix<double> dFdx(nx, 1);
    Eigen::SparseMatrix<double> d2Fdc2(nc, nc);
    Eigen::SparseMatrix<double> d2Fdcdx(nc, nx);
    Eigen::SparseMatrix<double> d2Fdx2(nx, nx);

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

    Eigen::SparseMatrix<double> d2Fdp2_K;
    MatrixXT d2Fdp2_UV;
    tessellation->boundary->computeEnergyHessianWoodbury(d2Fdp2_K, d2Fdp2_UV);

    MatrixXT sum_dFdx_d2xdc2 = MatrixXT::Zero(nc, nc);
    MatrixXT sum_dFdx_d2xdcdv = MatrixXT::Zero(nc, nv);
    MatrixXT sum_dFdx_d2xdv2 = MatrixXT::Zero(nv, nv);
    MatrixXT sum_dFdx_dxdv_d2vdp2 = MatrixXT::Zero(np, np);
    {
        tbb::task_group g;
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdc2 += dFdx.coeff(i, 0) * tessellation->d2xdc2[i];
            }
        });
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdcdv += dFdx.coeff(i, 0) * tessellation->d2xdcdv[i];
            }
        });
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdv2 += dFdx.coeff(i, 0) * tessellation->d2xdv2[i];
            }
        });
        g.run([&] {
            VectorXT temp_dFdx_dxdv = (dFdx.transpose() * tessellation->dxdv).transpose();
            for (int i = 0; i < nv; i++) {
                sum_dFdx_dxdv_d2vdp2 += temp_dFdx_dxdv(i) * tessellation->boundary->d2vdp2[i];
            }
        });
        g.wait();
    }

    Eigen::SparseMatrix<double> dxdc = tessellation->dxdc;
    Eigen::SparseMatrix<double> dxdcT = dxdc.transpose();
    Eigen::SparseMatrix<double> d2Fdxdc = d2Fdcdx.transpose();
    Eigen::SparseMatrix<double> dvdp = tessellation->boundary->dvdp;
    Eigen::SparseMatrix<double> dvdpT = dvdp.transpose();
    Eigen::SparseMatrix<double> dxdv_dvdp = tessellation->dxdv * dvdp;
    Eigen::SparseMatrix<double> dxdv_dvdpT = dxdv_dvdp.transpose();

    MatrixXT D2FDC2, D2FDCDP, D2FDP2;
    {
        Eigen::SparseMatrix<double> dp20;
        MatrixXT dp21;
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
            dp20 = dxdv_dvdpT * d2Fdx2 * dxdv_dvdp;
        });
        g.run([&] {
            dp21 = dvdpT * sum_dFdx_d2xdv2 * dvdp;
        });
        g.wait();
        D2FDP2 = dp20 + dp21 + sum_dFdx_dxdv_d2vdp2 + d2Fdp2_K;
    }

    hessian_K.block(0, 0, nc, nc) = D2FDC2;
    hessian_K.block(nc, 0, np, nc) = D2FDCDP.transpose();
    hessian_K.block(0, nc, nc, np) = D2FDCDP;
    hessian_K.block(nc, nc, np, np) = D2FDP2;

    if (dynamic) {
        for (int i = 0; i < _u.size(); ++i) {
            hessian_K(i, i) += m / pow(dt, 2);
            hessian_K(i, i) += eta / dt;
        }
    }

    K = hessian_K.sparseView();

    UV = MatrixXT::Zero(_u.rows(), d2Fdp2_UV.cols());
    UV.bottomRightCorner(np, 2) = d2Fdp2_UV;
}

void EnergyMinimizer::buildSystemMatrix(const VectorXT &_u, StiffnessMatrix &K) {
    preProcess(_u);

    const auto tstart = std::chrono::high_resolution_clock::now();

    MatrixXT hessian = MatrixXT::Zero(_u.rows(), _u.rows());
    if (!tessellation->isValid) {
        K = hessian.sparseView();
    }

    int nc = tessellation->cells.size() * optDims;
    int nx = tessellation->nodes.size() * 3;
    int nv = tessellation->boundary->v.size() * 3;
    int np = tessellation->boundary->nfree;

    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsdFdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdc2;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdcdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsd2Fdx2;

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

    Eigen::SparseMatrix<double> dFdx(nx, 1);
    Eigen::SparseMatrix<double> d2Fdc2(nc, nc);
    Eigen::SparseMatrix<double> d2Fdcdx(nc, nx);
    Eigen::SparseMatrix<double> d2Fdx2(nx, nx);

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

    MatrixXT d2Fdp2;
    tessellation->boundary->computeEnergyHessian(d2Fdp2);

    MatrixXT sum_dFdx_d2xdc2 = MatrixXT::Zero(nc, nc);
    MatrixXT sum_dFdx_d2xdcdv = MatrixXT::Zero(nc, nv);
    MatrixXT sum_dFdx_d2xdv2 = MatrixXT::Zero(nv, nv);
    MatrixXT sum_dFdx_dxdv_d2vdp2 = MatrixXT::Zero(np, np);
    {
        tbb::task_group g;
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdc2 += dFdx.coeff(i, 0) * tessellation->d2xdc2[i];
            }
        });
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdcdv += dFdx.coeff(i, 0) * tessellation->d2xdcdv[i];
            }
        });
        g.run([&] {
            for (int i = 0; i < nx; i++) {
                sum_dFdx_d2xdv2 += dFdx.coeff(i, 0) * tessellation->d2xdv2[i];
            }
        });
        g.run([&] {
            VectorXT temp_dFdx_dxdv = (dFdx.transpose() * tessellation->dxdv).transpose();
            for (int i = 0; i < nv; i++) {
                sum_dFdx_dxdv_d2vdp2 += temp_dFdx_dxdv(i) * tessellation->boundary->d2vdp2[i];
            }
        });
        g.wait();
    }

    Eigen::SparseMatrix<double> dxdc = tessellation->dxdc;
    Eigen::SparseMatrix<double> dxdcT = dxdc.transpose();
    Eigen::SparseMatrix<double> d2Fdxdc = d2Fdcdx.transpose();
    Eigen::SparseMatrix<double> dvdp = tessellation->boundary->dvdp;
    Eigen::SparseMatrix<double> dvdpT = dvdp.transpose();
    Eigen::SparseMatrix<double> dxdv_dvdp = tessellation->dxdv * dvdp;
    Eigen::SparseMatrix<double> dxdv_dvdpT = dxdv_dvdp.transpose();

    MatrixXT D2FDC2, D2FDCDP, D2FDP2;
    {
        Eigen::SparseMatrix<double> dp20;
        MatrixXT dp21;
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
            dp20 = dxdv_dvdpT * d2Fdx2 * dxdv_dvdp;
        });
        g.run([&] {
            dp21 = dvdpT * sum_dFdx_d2xdv2 * dvdp;
        });
        g.wait();
        D2FDP2 = dp20 + dp21 + sum_dFdx_dxdv_d2vdp2 + d2Fdp2;
    }

    hessian.block(0, 0, nc, nc) = D2FDC2;
    hessian.block(nc, 0, np, nc) = D2FDCDP.transpose();
    hessian.block(0, nc, nc, np) = D2FDCDP;
    hessian.block(nc, nc, np, np) = D2FDP2;

    if (dynamic) {
        for (int i = 0; i < _u.size(); ++i) {
            hessian(i, i) += m / pow(dt, 2);
            hessian(i, i) += eta / dt;
        }
    }

    K = hessian.sparseView();
}

T EnergyMinimizer::computeTotalEnergy(const VectorXT &_u) {
    preProcess(_u, false);

    double value = 0;
    if (!tessellation->isValid) {
//        std::cout << "eval invalid" << std::endl;
        return 1e10;
    }

    std::vector<double> cellEnergies(tessellation->cells.size());
    tbb::parallel_for_each(tessellation->cells.begin(), tessellation->cells.end(), [&](Cell cell) {
        CellValue cellValue(cell);
        energyFunction.getValue(tessellation, cellValue);
        cellEnergies[cell.cellIndex] = cellValue.value;
    });
    for (double e: cellEnergies) {
        value += e;
    }

    value += tessellation->boundary->computeEnergy();

    if (dynamic) {
        VectorXT a = (_u - dynamic_y_prev) / (dt * dt) - dynamic_v_prev / dt;
        value += .5 * pow(dt, 2) * a.transpose() * m * a;
        value += (1.0 / dt) *
                 (0.5 * _u.transpose() * eta * _u -
                  _u.transpose() * eta * dynamic_y_prev).value();
    }

//    std::cout << "energy value: " << value << std::endl;
    return value;
}

T EnergyMinimizer::computeResidual(const VectorXT &_u, VectorXT &residual) {
    preProcess(_u);

    const auto tstart = std::chrono::high_resolution_clock::now();

    VectorXT gradient = VectorXT::Zero(_u.rows());
    if (!tessellation->isValid) {
        std::cout << "grad invalid" << std::endl;
        residual = -gradient;
        return residual.norm();
    }

    int nc = tessellation->cells.size() * optDims;
    int nx = tessellation->nodes.size() * 3;
    int nv = tessellation->boundary->v.size() * 3;
    int np = tessellation->boundary->nfree;

    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsdFdx;
    tbb::concurrent_vector<Eigen::Triplet<double>> tripletsdFdc;
    tbb::parallel_for_each(tessellation->cells.begin(), tessellation->cells.end(), [&](Cell cell) {
        CellValue cellValue(cell);
        energyFunction.getGradient(tessellation, cellValue);

        for (auto n: cell.nodeIndices) {
            Node node = n.first;
            int nodeIdxInCell = n.second;
            NodePosition nodePos = tessellation->nodes[node];

            for (int i = 0; i < 3; i++) {
                tripletsdFdx.emplace_back(nodePos.ix * 3 + i, 0, cellValue.gradient(nodeIdxInCell * 3 + i));
            }
        }
        for (int i = 0; i < optDims; i++) {
            tripletsdFdc.emplace_back(cell.cellIndex * optDims + i, 0,
                                      cellValue.gradient(cell.nodeIndices.size() * 3 + i));
        }
    });
    Eigen::SparseMatrix<double> dFdx_sp(nx, 1);
    dFdx_sp.setFromTriplets(tripletsdFdx.begin(), tripletsdFdx.end());
    Eigen::SparseMatrix<double> dFdc_sp(nc, 1);
    dFdc_sp.setFromTriplets(tripletsdFdc.begin(), tripletsdFdc.end());
    VectorXT dFdx = dFdx_sp;
    VectorXT dFdc = dFdc_sp;
    VectorXT dFdp;
    tessellation->boundary->computeEnergyGradient(dFdp);

    gradient.segment(0, nc) = dFdx.transpose() * tessellation->dxdc + dFdc.transpose();
    gradient.tail(np) = dFdx.transpose() * tessellation->dxdv * tessellation->boundary->dvdp + dFdp.transpose();

    if (dynamic) {
        VectorXT a = (_u - dynamic_y_prev) / (dt * dt) - dynamic_v_prev / dt;
        gradient += m * a;
        gradient += eta * (_u - dynamic_y_prev) / dt;
    }

    residual = -gradient;
    return residual.norm();
}

T EnergyMinimizer::lineSearchNewton(VectorXT &_u, VectorXT &residual, int ls_max) {
    VectorXT du = residual;
    du.setZero();
    StiffnessMatrix K(residual.rows(), residual.rows());

    bool success = false;
    Timer ti(true);

    MatrixXT UV;
    if (woodbury) {
        MatrixXT UV;
        buildSystemMatrixWoodbury(_u, K, UV);
        success = solveWoodburyCholmod(K, UV, residual, du);
//        StiffnessMatrix KK = K + (UV * UV.transpose()).sparseView();
//        success = linearSolve(KK, residual, du);
    } else {
        buildSystemMatrix(_u, K);
        success = linearSolve(K, residual, du);
    }
    if (!success) {
        std::cout << "linear solve failed" << std::endl;
        return 1e16;
    }


    T norm = du.norm();

    T E0 = computeTotalEnergy(_u);
    T alpha = computeLineSearchInitStepsize(_u, du);
    int cnt = 1;
    while (true) {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > ls_max) {
            _u = u_ls;
            if (cnt > ls_max)
                if (verbose)
                    std::cout << "---ls max---" << std::endl;
            if (verbose)
                std::cout << "# ls " << cnt << " |du| " << alpha * du.norm() << std::endl;
            break;
        }
        alpha *= 0.5;
        cnt++;
    }
    return norm;
}
