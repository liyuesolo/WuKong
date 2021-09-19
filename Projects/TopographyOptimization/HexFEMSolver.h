#ifndef HexFEMSolver_H
#define HexFEMSolver_H
#include "StaticGrid.h"
#include "getRSS.hpp"
#include <tbb/tbb.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Ziran/CS/Util/Debug.h>
#include <complex>
#include <unordered_set>
#include <Solver/LinearSolver.h>

namespace ZIRAN {

template <class Optimization, class BoundaryCondition, class T, int dim>
class HexFEMSolver {
public:
    using Objective = HexFEMSolver<Optimization, BoundaryCondition, T, dim>;
    typedef Matrix<T, Eigen::Dynamic, 1> VectorXT;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;

    Optimization& optimizer;
    BoundaryCondition &bc;
    Eigen::Ref<const TVStack> u;
    T dt;
    bool quasi_static;
    
    T kappa;
    StdVector<int> entryColK;
    StdVector<TM> entryValK;
    int final_dim_k;

    const static int grid_range = 3;
    int final_dim;

    bool verbose = false;
    T contact_eps;
    std::map<int, std::map<int, int>> contact_data;
    T contact_stiffness;
    

    HexFEMSolver(Optimization& in_optimizer, BoundaryCondition &in_bc, Eigen::Ref<const TVStack> u_in, T dt_in=1)
        : optimizer(in_optimizer), bc(in_bc), u(u_in), dt(dt_in), quasi_static(in_optimizer.quasi_static)
    {
        kappa = 1e5;
        dt = 1;
    }
    void implicitUpdate(Eigen::Ref<TVStack> du, T linear_tol, T newton_tol)
    {
        ZIRAN_TIMER();
        du.setZero();
        optimizer.updateMovedDeformationGradient(du);
        
        T alpha = 1;
        searchInitialStepSize(du, alpha);
        TVStack du_ = TVStack::Zero(dim, optimizer.num_nodes);
        searchNonContactStepSize(du_, du, alpha);
        du *= alpha;

        TVStack f = TVStack::Zero(dim, optimizer.num_nodes);
        bc.iterateNeumanData([&](const auto& dof, const auto& node, const auto& target) {
            f.col(dof) = target;
        });
        
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10) f(d, dof) = 0;
            }
        });

        bool project_dirichlet = false;
        TVStack lambda(dim, optimizer.num_nodes);
        lambda.setZero();

        int cnt = 0;
        while (true) {
            
            optimizer.updateMovedDeformationGradient(du);
            TVStack residual;
            computeResidual(du, f, residual, lambda, kappa);
            
            T norm = newtonLinesearch(1000, du, f, residual, true, project_dirichlet, lambda, kappa, linear_tol);
            if(project_dirichlet && !optimizer.nonlinear) break;
            // checkGradient(du, Ep, f, lambda, kappa);
            // checkHessian(du, Ep, f, lambda, kappa);
            ZIRAN_INFO("Iter: ", cnt, ", H-scaled norm: ", norm, ", Target tol: ", newton_tol); 
            if(norm < newton_tol) 
                break;
            else if(norm > 1e10)
            {
                ZIRAN_WARN("DTopo: Linesearch get stuck");
                writeOutHessian(du, true, kappa);
                break;
                checkGradient(du, f, lambda, kappa);
                checkHessian(du, f, lambda, kappa);
                ZIRAN_FATAL();
                // continue;
            }
            cnt++;
        }
        T cons_vio = constraintViolation(du);
        ZIRAN_INFO("DTopo: # of newton solve: " + std::to_string(cnt) + " constraint violation: " + std::to_string(cons_vio));
    }

    
    T newtonLinesearch(int search_time, Eigen::Ref<TVStack> du, const TVStack& f, TVStack& nagtive_gradient, bool project_spd, bool project_dirichlet, const TVStack& lambda, T kappa=1e6, T linear_tol=1e-7)
    {
        optimizer.updateMovedDeformationGradient(du);
        TVStack ddu(dim, optimizer.num_nodes);
        ddu.setZero();
        StdVector<int> entryCol;
        StdVector<TM> entryVal;
        buildMatrix(entryCol, entryVal, du, project_spd, kappa);
        linearSolver(entryCol, entryVal, nagtive_gradient, ddu, linear_tol, project_spd, project_dirichlet);
        // if (nagtive_gradient.cwiseProduct(delta_u).sum() < 0) return false;
        // T norm = ddu.cwiseAbs().maxCoeff();
        T norm = std::sqrt(ddu.cwiseProduct(ddu).sum());
        // T norm = ddu.cwiseAbs().sum();
        if (norm < 1e-5) return norm;
        T alpha = 1;
        searchInitialStepSize(ddu, alpha);
        if(alpha < 1e-5)
            ZIRAN_INFO("DTopo limited step size for non-invertible quadrature F", alpha);
        searchNonContactStepSize(du, ddu, alpha);
        if (verbose) { ZIRAN_INFO("initial alpha: ", alpha); }
        T E = totalEnergy(du, f, lambda, kappa);
        int cnt = 0;
        while (true) {
            TVStack du_hat = du + alpha * ddu;
            optimizer.updateMovedDeformationGradient(du_hat);
            T E_hat = totalEnergy(du_hat, f, lambda, kappa);
            if (E_hat - E < 0) {
                du = du_hat;
                break;
            }
            alpha *= T(0.5);
            cnt += 1;
            if (cnt == search_time) return 1e30;
        }
        if (verbose) { ZIRAN_INFO("backtracking times: ", cnt); }
        return norm;
    }

    T computeDirichletCriteria(const TVStack& du)
    {
        T num = 0;
        T den = 0;
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10 && target(d) != 0) {
                    num += std::pow(du(d,dof) - target(d), 2);
                    den += std::pow(target(d), 2);
                }
            }
        });
        if (num == 0 && den == 0) return 1;
        return 1 - std::sqrt(num / den);
    }

    void backSolve(Eigen::Ref<VectorXT> x, Eigen::Ref<VectorXT> b, T linear_tol)
    {
        ZIRAN_TIMER();
        verbose = true;
        // Solve A(du)x = b
        x.setZero();
        VectorXT x_tmp = VectorXT::Zero(x.size() + bc.numSyncPairs() * dim);
        VectorXT b_tmp = VectorXT::Zero(x.size() + bc.numSyncPairs() * dim);
        b_tmp.segment(0, b.size()) = b;
        StdVector<int> entryCol;
        StdVector<TM> entryVal;
        TVStack du = TVStack::Zero(dim, optimizer.num_nodes);
        buildMatrix<true>(entryCol, entryVal, du, false, kappa);
        projectMatrix(entryCol, entryVal, true);
        StiffnessMatrix A;
        compressCSR(entryCol, entryVal, A);
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for (int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10)
                    b_tmp(dim * dof + d) = 0;
            }
        });
        if (!optimizer.use_iterative_solver || !IterativeSolver<StiffnessMatrix, T>(A, b_tmp, x_tmp, dim, linear_tol, false)) {
            // outputSparse(optimizer.output_dir.path + "/matrix_failed_backsolve.mtx", A);
            // outputDense<T>(optimizer.output_dir.path + "/rhs_failed_backsolve.mtx", b_tmp);
            DirectSolver<StiffnessMatrix, T>(A, x_tmp, b_tmp, false);
        }
        x = x_tmp.segment(0, x.size());
    }


    void searchNonContactStepSize(Eigen::Ref<const TVStack> du, Eigen::Ref<const TVStack> ddu, T &alpha)
    {
        optimizer.iterateGridSerial([&](const auto& Xi, const auto& grid_state, const auto& grid) {
            if (!grid_state.contact_interface) return;
            int dofi = grid_state.grid_idx;
            bc.iterateContactSphere([&](const auto& dofj, const auto& Xj, T radius, const auto& component_id) {
                TV xi = Xi.template cast<T>() * optimizer.dx + u.col(dofi) + du.col(dofi);
                TV xj = Xj.template cast<T>() * optimizer.dx + u.col(dofj) + du.col(dofj);
                TV diff = xi - xj;
                TV delta_ddu = (ddu.col(dofi) - ddu.col(dofj));
                T b = delta_ddu.squaredNorm();
                T c = 2 * diff.dot(delta_ddu);
                T d = diff.squaredNorm() - 0.1 * std::pow(diff.norm() - radius, 2);
                T t = getSmallestPositiveRealQuadRoot(b, c, d, 1e-9);
                if (t < 0 || t > 1) t = 1;
                alpha = t < alpha ? t : alpha;
                while ((diff + alpha * delta_ddu).norm() - radius < 0)
                    alpha *= 0.5;
            });
        });
    }

    void searchInitialStepSize(Eigen::Ref<const TVStack> delta_u, T& alpha)
    {
        if (!optimizer.nonlinear) { return; }
        if (optimizer.nonlinear && optimizer.invertable) { return; }

        VectorXT alphas(optimizer.ppc * optimizer.num_cells);
        alphas.setOnes();
        optimizer.iterateCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            for (int qp_loop = 0; qp_loop < optimizer.ppc; qp_loop++) {
                TM Fn;
                Fn.setZero();
                grid.iterateKernel(
                    cell, qp_loop, optimizer.dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        Fn += delta_u.col(dof) * dw.transpose();
                    });
                if (Fn.norm() == 0) continue;
                TM A = optimizer.F_moved[optimizer.ppc * cell_state.cell_idx + qp_loop].partialPivLu().solve(Fn);

                T a, b, c, d;
                if constexpr (dim == 2) {
                    a = 0;
                    b = A.determinant();
                }
                else {
                    a = A.determinant();
                    b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
                }
                c = A.diagonal().sum();
                d = 0.8;

                T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
                if (t < 0 || t > 1) t = 1;
                alphas(optimizer.ppc * cell_state.cell_idx + qp_loop) = t;
            }
        });
        alpha = alphas.minCoeff();
    }

    double getSmallestPositiveRealCubicRoot(double a, double b, double c, double d, double tol = 1e-10)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::complex;
        using std::pow;
        using std::sqrt;
        double t = -1;
        if (abs(a) <= tol)
            t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
        else {
            complex<double> i(0, 1);
            complex<double> delta0(b * b - 3 * a * c, 0);
            complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<double> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            if (abs(C) < tol)
                C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
            complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;
            complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
            complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
            complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        return t;
    }

    double getSmallestPositiveRealQuadRoot(double a, double b, double c, double tol)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::sqrt;
        double t;
        if (abs(a) <= tol) {
            if (abs(b) <= tol) // f(x) = c > 0 for all x
                t = -1;
            else
                t = -c / b;
        }
        else {
            double desc = b * b - 4 * a * c;
            if (desc > 0) {
                t = (-b - sqrt(desc)) / (2 * a);
                if (t < 0)
                    t = (-b + sqrt(desc)) / (2 * a);
            }
            else // desv<0 ==> imag
                t = -1;
        }
        return t;
    }

    T constraintViolation(Eigen::Ref<const TVStack> du)
    {
        T violation = 0.0;
        for (auto dirichlet_cell : bc.cell_dirichlet_data)
        {
            
            
            TV cell_center = TV::Zero();
            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                cell_center += T(1) / optimizer.ppc * du.col(grid_state.grid_idx);
            });

            violation += (cell_center - dirichlet_cell.second).norm();
            
        }
        return violation;
    }

    // e = Psi - lambda(u - u*) + mu_k/2 (u - u*)^2
    T totalEnergy(Eigen::Ref<const TVStack> du,  const TVStack& f, const TVStack& lambda, T kappa)
    {
        int ppc = 1 << dim;

        T Psi = 0;
        {
            VectorXT psis(ppc * optimizer.num_cells);
            psis.setZero();
            optimizer.iterateCell([&](const IV& cell, const auto& cell_state, auto& grid) {
                if (!cell_state.active_cell) return;
                T vol = optimizer.vol;
                for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                    psis(cell_state.cell_idx * ppc + qp_loop) = vol * optimizer.psi(optimizer.F_moved[cell_state.cell_idx * ppc + qp_loop], cell_state.E, cell_state.nu);
                }
            });
            Psi = psis.sum();
            
        }
        T energy = Psi - du.cwiseProduct(f).sum();
        
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10 && target(d) != 0)
                    energy += -lambda(d, dof) * (du(d, dof) - target(d)) + 0.5 * kappa * std::pow(du(d, dof) - target(d), 2);
            }
        });

        if (std::abs(optimizer.gravity[1]) > 1e-6)
        {
            VectorXT gmTu(optimizer.num_nodes);
            gmTu.setZero();
            optimizer.coloredParEachCell([&](auto const& basenode, auto const& cell_state, auto& grid){
                if (!cell_state.active_cell) return;
                for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                    T density = 1.0 / T(optimizer.ppc);
                    T vol = optimizer.vol;
                    grid.iterateKernel(
                        basenode, qp_loop, optimizer.dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                            if (dof < 0) return;
                            gmTu[dof] += vol * density * optimizer.gravity.dot(du.col(dof)) * w;
                        });
                }
            });
            energy -= gmTu.sum();
            // std::cout << "gmTu: " << gmTu.sum() << std::endl;
        }

        

        for (auto dirichlet_cell : bc.cell_dirichlet_data)
        {
            
            
            TV cell_center = TV::Zero();
            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                cell_center += T(1) / optimizer.ppc * du.col(grid_state.grid_idx);
            });

            for (int d = 0; d < dim; d++)
                energy += 0.5 * kappa * std::pow(cell_center[d] - dirichlet_cell.second[d], 2);
            
        }
        return energy;
    }

    T residualNorm(TVStack residual)
    {
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10)
                    residual(d, dof) = 0;
            }
        });
 
        T vol = optimizer.vol;
        VectorXT nodal_dpdf(optimizer.num_nodes);
        nodal_dpdf.setZero();
        optimizer.coloredParEachCell([&](auto const& basenode, auto const& cell_state, auto& grid){
            if (!cell_state.active_cell) return;
            for (int qp_loop = 0; qp_loop < optimizer.ppc; qp_loop++) {
                T density = (optimizer.rho(optimizer.ppc * cell_state.cell_idx + qp_loop) + optimizer.alpha) * cell_state.density;
                TM F = TM::Identity();
                T dpdf_norm = Ep(optimizer.ppc * cell_state.cell_idx + qp_loop) * optimizer.firstPiolaDerivative(F, cell_state.E, cell_state.nu, false).norm();
                grid.iterateKernel(
                    basenode, qp_loop, optimizer.dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        nodal_dpdf(dof) += dpdf_norm * vol * density * w / optimizer.gridMass(dof);
                    });
            }
        });
        if constexpr (dim == 2)
            nodal_dpdf *= T(8) * optimizer.dx;
        else if constexpr (dim == 3)
            nodal_dpdf *= T(24) * optimizer.dx * optimizer.dx;

        tbb::parallel_for(0, optimizer.num_nodes, [&](int i) {
            residual.col(i) /= nodal_dpdf(i);
        });
        
        return residual.norm();
        
    }

    // Psi + lambda(u - u*) - mu_k/2 * (u - u*)^2
    void computeResidual(const TVStack& du, const TVStack& f, TVStack& residual, const TVStack& lambda, T kappa)
    {
        //dPsi/du
        TVStack dPsidu = TVStack::Zero(dim, optimizer.num_nodes);
        TVStack mg = TVStack::Zero(dim, optimizer.num_nodes);
        int ppc = 1 << dim;
        optimizer.coloredParEachCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            T vol = optimizer.vol;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                
                TM firstPiola = optimizer.firstPiola(optimizer.F_moved[ppc * cell_state.cell_idx + qp_loop], cell_state.E, cell_state.nu);
                grid.iterateKernel(
                    cell, qp_loop, optimizer.dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        TV vPFTw = vol * firstPiola * dw;
                        dPsidu.col(dof) += vPFTw;
                        if (std::abs(optimizer.gravity[1]) > 1e-6)
                            mg.col(dof) += vol * optimizer.gravity * w / T(optimizer.ppc);
                        
                    });
            }
        });

        residual = -(dPsidu - mg - f);

        // -lambda + mu_k * (u - u*)
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10 && target(d) != 0)
                    residual(d, dof) += lambda(d, dof) - kappa * (du(d, dof) - target(d));
            }
        });


        for (auto dirichlet_cell : bc.cell_dirichlet_data)
        {
            
            
            TV cell_center = TV::Zero();
            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                cell_center += T(1) / optimizer.ppc * du.col(grid_state.grid_idx);
            });

            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                for (int d = 0; d < dim; d++)
                    residual(d, grid_state.grid_idx) -= kappa * T(1) / optimizer.ppc * (cell_center[d] - dirichlet_cell.second[d]);
                    
            });
            
        }

    }

    template <bool augmented=false>
    void initializeSystemMatrix(Eigen::Ref<const TVStack> du, StdVector<int> &entryCol, StdVector<TM> &entryVal)
    {
        final_dim = std::pow(grid_range, dim) + bc.spring_data.size() + 3;
        int n_rows = optimizer.num_nodes;
        entryCol.resize(n_rows * final_dim);
        entryVal.resize(n_rows * final_dim);
        tbb::parallel_for(0, n_rows, [&](int g) {
            for (int i = 0; i < final_dim; ++i) {
                entryCol[g * final_dim + i] = -1;
                entryVal[g * final_dim + i] = TM::Zero();
            }
        });
    }


    template <bool augmented=false>
    void addLagrangianMultiplierMatrix(StdVector<int> &entryCol, StdVector<TM> &entryVal, T kappa)
    {
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for (int d = 0; d < dim; ++d) {
                if (std::abs(target(d)) <= 1e10 && target(d) != 0) {
                    entryCol[dof * final_dim + linearOffset(IV::Zero())] = dof;
                    entryVal[dof * final_dim + linearOffset(IV::Zero())](d, d) += kappa;
                }
            }
        });

        for (auto dirichlet_cell : bc.cell_dirichlet_data)
        {
            TM penalty_hessian = TM::Zero();
            T h_entry = kappa / T(optimizer.ppc) / T(optimizer.ppc);
            for(int d = 0; d < dim; d++)
                penalty_hessian(d,d) = h_entry;
            
            
            std::vector<int> cached_id;
            std::vector<IV> cached_node;
            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& node, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                cached_id.push_back(grid_state.grid_idx);
                cached_node.push_back(node);
            });

            optimizer.grid.iterateCellKernel(dirichlet_cell.first, [&](const IV& nodej, auto& grid_state) {
                if (grid_state.grid_idx == -1) 
                    return;
                int dofj = grid_state.grid_idx;
                int cnt = 0;
                for(int dofi : cached_id)
                {
                    IV& nodei = cached_node[cnt++];
                    if (dofi < dofj) continue;
                    entryCol[dofi * final_dim + linearOffset(nodei - nodej)] = dofj;
                    entryVal[dofi * final_dim + linearOffset(nodei - nodej)] += penalty_hessian;  
                    if (dofj != dofi)
                    {
                        entryCol[dofj * final_dim + linearOffset(nodej - nodei)] = dofi;
                        entryVal[dofj * final_dim + linearOffset(nodej - nodei)] += penalty_hessian.transpose();
                    }
                }
            });
            
        }

    }

    void addStiffnessMatrix(StdVector<int> &entryCol, StdVector<TM> &entryVal, Eigen::Ref<const TVStack> du, T weight, bool project_spd)
    {
        // ZIRAN_TIMER();
        int ppc = 1 << dim;
        optimizer.coloredParEachCell([&](const IV& cell, const auto& cell_state, auto& grid) {
            if (!cell_state.active_cell) return;
            for (int qp_loop = 0; qp_loop < ppc; qp_loop++) {
                std::vector<int> cached_id(ppc);
                std::vector<IV> cached_node(ppc);
                std::vector<TV> cached_dw(ppc);
                int cnt = 0;
                grid.iterateKernel(
                    cell, qp_loop, optimizer.dx, [&](const int& dof, const IV& node, const T& w, const TV& dw) {
                        if (dof < 0) return;
                        cached_id[cnt] = dof;
                        cached_node[cnt] = node;
                        cached_dw[cnt++] = dw;
                    });
                
                T vol = optimizer.vol;
                Hessian dPdF = optimizer.firstPiolaDerivative(optimizer.F_moved[ppc * cell_state.cell_idx + qp_loop], cell_state.E, cell_state.nu, project_spd);
                for (int i = 0; i < cnt; ++i) {
                    int& dofi = cached_id[i];
                    IV& nodei = cached_node[i];
                    TV& dwi = cached_dw[i];
                    for (int j = 0; j < cnt; ++j) {
                        int& dofj = cached_id[j];
                        IV& nodej = cached_node[j];
                        TV& dwj = cached_dw[j];
                        if (dofj < dofi) continue;
                        TM dFdX = TM::Zero();
                        for (int q = 0; q < dim; q++)
                            for (int v = 0; v < dim; v++)
                                dFdX += dPdF.template block<dim, dim>(dim * v, dim * q) * dwi(v) * dwj(q);
                        TM delta = weight * vol * dFdX; // normal
                        entryCol[dofi * final_dim + linearOffset(nodei - nodej)] = dofj;
                        entryVal[dofi * final_dim + linearOffset(nodei - nodej)] += delta;
                        if (dofj != dofi) {
                            entryCol[dofj * final_dim + linearOffset(nodej - nodei)] = dofi;
                            entryVal[dofj * final_dim + linearOffset(nodej - nodei)] += delta.transpose();
                        }
                    }
                }
            }
        });
    }

    template <bool augmented=false>
    void buildMatrix(StdVector<int>& entryCol, StdVector<TM>& entryVal, Eigen::Ref<const TVStack> du, bool project_spd, T kappa)
    {
        ZIRAN_TIMER();
        initializeSystemMatrix<augmented>(du, entryCol, entryVal);
        
        addStiffnessMatrix(entryCol, entryVal, du, 1.0, project_spd);
        
        addLagrangianMultiplierMatrix<augmented>(entryCol, entryVal, kappa);
    }

    static inline int linearOffset(const IV& node_offset)
    {
        IV rel_offset = node_offset.array() + 1;
        if constexpr (dim == 2)
            return rel_offset[0] * Objective::grid_range + rel_offset[1];
        else if constexpr (dim == 3)
            return ((rel_offset[0] * Objective::grid_range) + rel_offset[1]) * Objective::grid_range + rel_offset[2];
    }
    
    void checkHessian(TVStack du, const TVStack &f, const TVStack& lambda, T kappa)
    {
        T epsilon = 1e-3;
        optimizer.updateMovedDeformationGradient(du);
        StdVector<int> entryCol;
        StdVector<TM> entryVal;
        buildMatrix(entryCol, entryVal, du, false, kappa);
        char input = 0;
        bool jump_out = false;
        optimizer.iterateGridSerial([&](const IV& grid, auto& grid_state, auto&) {
        // bc.iterateSyncDOF([&](int dof, IV grid) {
        // bc.iterateContactSphere([&](const auto& dof, const auto& grid, const auto& radius, const auto& component_id) {
            // if (jump_out) return;
            // if (!grid_state.contact_interface && bc.contact_sphere.find(&grid_state) == bc.contact_sphere.end()) return;
            int dof = grid_state.grid_idx;
            for (int d = 0; d < dim; ++d) {
                if (bc.dirichlet_data.find(grid) != bc.dirichlet_data.end()) {
                    if (bc.dirichlet_data[grid](d) == 0) continue;
                }
                du(d, dof) += epsilon;
                TVStack g0;
                optimizer.updateMovedDeformationGradient(du);
                computeResidual(du, f, g0, lambda, kappa);
                du(d, dof) -= 2 * epsilon;
                TVStack g1;
                optimizer.updateMovedDeformationGradient(du);
                computeResidual(du, f, g1, lambda, kappa);
                du(d, dof) += epsilon;
                TVStack row_FD = (g1 - g0) / (2 * epsilon);
                int i = dof;
                int st = i * final_dim;
                int ed = st + final_dim;
                for (; st < ed; ++st) {
                    int j = entryCol[st];
                    if (j == -1) continue;
                    std::cout << i << " "<< j <<" "<< row_FD.col(j).transpose() << " " << entryVal[st].row(d) << std::endl;
                    input = getchar();
                    if (input == 'q') {
                        jump_out = true;
                        return;
                    }
                }
            }
        });
    }


    void checkGradient(TVStack du, const TVStack &f, const TVStack& lambda, T kappa)
    {
        T epsilon = 1e-6;
        optimizer.updateMovedDeformationGradient(du);
        TVStack gradient;
        computeResidual(du, f, gradient, lambda, kappa);
        TVStack gradient_FD(gradient);
        gradient_FD.setZero();
        char input = 0;
        bool jump_out = false;
        optimizer.iterateGridSerial([&](const IV& grid, const auto& grid_state, auto&){
        // bc.iterateSyncDOF([&](int dof, IV grid) {
            // if (jump_out) return;
            // if (!grid_state.contact_interface) return;
            int dof = grid_state.grid_idx;
            for (int d = 0; d < dim; ++d) {
                if (bc.dirichlet_data.find(grid) != bc.dirichlet_data.end()) {
                    if (bc.dirichlet_data[grid](d) == 0) continue;
                }
                du(d, dof) += epsilon;
                optimizer.updateMovedDeformationGradient(du);
                T E0 = totalEnergy(du, f, lambda, kappa);
                du(d, dof) -= 2 * epsilon;
                optimizer.updateMovedDeformationGradient(du);
                T E1 = totalEnergy(du, f, lambda, kappa);
                du(d, dof) += epsilon;
                gradient_FD(d, dof) = (E1 - E0) / (2*epsilon);
                std::cout << dof << " " << gradient_FD(d, dof) << " " << gradient(d, dof) << std::endl;
                std::getchar();
                if(std::abs(gradient(d, dof)) + std::abs(gradient_FD(d, dof)) > epsilon * epsilon) {
                    std::cout << dof << " " << gradient_FD(d, dof) << " " << gradient(d, dof) << std::endl;
                    input = getchar();
                    if (input == 'q') {
                        jump_out = true;
                        return;
                    }
                }
            }
        });
        T err = (gradient_FD - gradient).norm();
        T norm = gradient.norm();
        std::cout << "err: " << err << " gradient norm: " << norm << std::endl;
        exit(0);
    }

    void compressCSR(StdVector<int> &entryCol, StdVector<TM> &entryVal, StiffnessMatrix& A)
    {
        std::vector<StorageIndex> ptr;
        std::vector<StorageIndex> col;
        std::vector<T> val;
        int colsize = final_dim;
        int rowsize = entryVal.size() / colsize;
        ptr.resize(rowsize * dim + 1);
        ptr[0] = 0;

        tbb::parallel_for(0, rowsize, [&](int row_idx) {
            int idx = row_idx * colsize;
            int cnt = 0;
            for (int j = 0; j < colsize; ++idx, ++j) {
                int col_idx = entryCol[idx];
                if (col_idx != -1)
                    cnt++;
            }
            for (int d = 0; d < dim; ++d)
                ptr[dim * row_idx + d + 1] = cnt * dim;
        });

        for (int i = 0; i < rowsize * dim; ++i) {
            ptr[i + 1] += ptr[i];
        }

        col.resize(ptr.back()); // for parallelism
        val.resize(ptr.back()); // for parallelism
        tbb::parallel_for(0, rowsize, [&](int row_idx) {
            std::vector<std::pair<TM, int>> zipped;
            zipped.resize(ptr[row_idx * dim + 1] / dim - ptr[row_idx * dim] / dim); // avoid memery reallocation
            int idx = row_idx * colsize;
            int cnt = 0;
            for (int j = 0; j < colsize; ++idx, ++j) {
                if (entryCol[idx] != -1) {
                    zipped[cnt].first = entryVal[idx];
                    zipped[cnt++].second = entryCol[idx];
                }
            }
            // sort each row's entry by col index;
            std::sort(std::begin(zipped), std::end(zipped),
                [&](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

            for (int c = 0; c < (int)zipped.size(); ++c) {
                int col_idx = zipped[c].second;
                TM value = zipped[c].first;
                for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = 0; d2 < dim; ++d2) {
                        col[ptr[row_idx * dim + d1] + c * dim + d2] = col_idx * dim + d2;
                        val[ptr[row_idx * dim + d1] + c * dim + d2] = value(d1, d2);
                    }
            }
        });

        if (verbose) { ZIRAN_INFO("K nonzeros: ", ptr.back()); }
        rowsize = ptr.size() - 1;
        A.resize(rowsize, rowsize);
        A.setZero();
        A.reserve(val.size());

        memcpy(A.valuePtr(), val.data(),
            val.size() * sizeof(val[0]));
        std::vector<T>().swap(val);
        memcpy(A.innerIndexPtr(), col.data(),
            col.size() * sizeof(col[0]));
        std::vector<StorageIndex>().swap(col);
        memcpy(A.outerIndexPtr(), ptr.data(),
            ptr.size() * sizeof(ptr[0]));
        std::vector<StorageIndex>().swap(ptr);
        A.finalize();
    }

    void projectMatrix(StdVector<int> &entryCol, StdVector<TM> &entryVal, bool project_dirichlet) 
    {
        // project matrix
        int rowsize = entryVal.size() / final_dim;
        std::unordered_set<int> bCollide;
        bCollide.reserve(bc.dirichlet_data.size() * dim);
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for (int d = 0; d < dim; ++d) {
                if (project_dirichlet && std::abs(target(d)) <= 1e10)
                    bCollide.insert(dim * dof + d);
                else if (!project_dirichlet && target(d) == 0)
                    bCollide.insert(dim * dof + d);
            }
        });

        tbb::parallel_for(0, optimizer.num_nodes, [&](int i) {
            int st = i * final_dim;
            int ed = st + final_dim;
            for (; st < ed; ++st) {
                int j = entryCol[st];
                if (j == -1) continue;
                for (int d1 = 0; d1 < dim; ++d1) {
                    for (int d2 = 0; d2 < dim; ++d2) {
                        int row_idx = dim * i + d1;
                        int col_idx = dim * j + d2;
                        bool iCollide = bCollide.find(row_idx) != bCollide.end();
                        bool jCollide = bCollide.find(col_idx) != bCollide.end();
                        if (iCollide || jCollide) {
                            if (row_idx == col_idx) {
                                entryVal[st](d1, d2) = abs(entryVal[st](d1, d2));
                            }
                            else {
                                entryVal[st](d1, d2) = 0;
                            }
                        }
                    }
                }
            }
        });
    }

    void projectSystem(StdVector<int> &entryCol, StdVector<TM> &entryVal, TVStack& residual, bool project_dirichlet=true)
    {
        // project residual
        bc.iterateDirichletData([&](const auto& dof, const auto& node, const auto& target) {
            for(int d = 0; d < dim; ++d) {
                if (project_dirichlet && std::abs(target(d)) <= 1e10) 
                    residual(d, dof) = 0;
                else if (!project_dirichlet && target(d) == 0)
                    residual(d, dof) = 0;
            }
        });
        projectMatrix(entryCol, entryVal, project_dirichlet);
    }

    bool linearSolver(StdVector<int> &entryCol, StdVector<TM> &entryVal, TVStack& residual, Eigen::Ref<TVStack> x, T linear_tol=1e-5, bool spd=true, bool project_dirichlet=true, bool try_direct=true)
    {
        ZIRAN_TIMER();
        projectSystem(entryCol, entryVal, residual, project_dirichlet);
        StiffnessMatrix A;
        compressCSR(entryCol, entryVal, A);
        if (optimizer.debug)
        {
            outputSparse(optimizer.output_dir.path + "/hessian.mtx", A);
            outputDense<T>(optimizer.output_dir.path + "/rhs.mtx", residual);
        }
        if (!optimizer.use_iterative_solver || !IterativeSolver<StiffnessMatrix, T>(A, Eigen::Map<MatrixXT>(residual.data(), residual.size(), 1), Eigen::Map<MatrixXT>(x.data(), x.size(), 1), dim, linear_tol, spd)) {
            if (try_direct)
                DirectSolver<StiffnessMatrix, T>(A, Eigen::Map<VectorXT>(x.data(), x.size(), 1), Eigen::Map<VectorXT>(residual.data(), residual.size(), 1), spd);
            // outputSparse(optimizer.output_dir.path + "/matrix_failed.mtx", A);
            // outputDense<T>(optimizer.output_dir.path + "/rhs_failed.mtx", residual);
            return false;
        }
        return true;
    }

    void multiply(const StdVector<int> &entryCol, const StdVector<TM>& entryVal, Eigen::Ref<const TVStack> x, TVStack& b)
    {
        int row_cnt = x.cols();
        int colsize = entryCol.size() / row_cnt;
        tbb::parallel_for(0, row_cnt, [&](int i) {
            TV sum = TV::Zero();
            int idx = i * colsize;
            for (int j = 0; j < colsize; ++idx, ++j) {
                int col_idx = entryCol[idx];
                if (col_idx == -1) continue;
                sum += entryVal[idx] * x.col(col_idx);
            }
            b.col(i) = sum;
        });
    }

    void writeOutHessian(Eigen::Ref<const TVStack> du, bool project_spd, T kappa)
    {
        StdVector<int> entryCol;
        StdVector<TM> entryVal;
        buildMatrix(entryCol, entryVal, du, project_spd, kappa);
        StiffnessMatrix A;
        compressCSR(entryCol, entryVal, A);
        outputSparse(optimizer.output_dir.path + "/hessian.mtx", A);
    }

    template<int order>
    T barrier(T d, T eps)
    {
        ZIRAN_ASSERT(d > 0);
        if constexpr (order == 0)
            return - std::pow(d / eps - 1, 2) * std::log(d / eps);
        else if constexpr (order == 1)
            return (d - eps) * (-2 * d * std::log(d / eps) + eps - d) / (eps * eps * d);
        else
            return 1 / (d * d) + (2 / (d * eps) - 2 * std::log(d / eps) - 3) / (eps * eps);
    }

    void testBarrier()
    {
        T eps = 1;
        T delta = 1e-7;
        for (auto x: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
           T first_order = barrier<2>(x, eps);
            T fd = (barrier<1>(x + delta, eps) - barrier<1>(x - delta, eps)) / (2 * delta);
            std::cout << first_order << " " << fd << std::endl;
        }
            getchar();
    }
};

} // namespace ZIRAN

#endif