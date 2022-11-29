#pragma once

#include <IpTNLP.hpp>
#include <cassert>
#include <iostream>

#include "IpIpoptCalculatedQuantities.hpp"
#include "IpIpoptData.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpOrigIpoptNLP.hpp"
#include <IpIpoptApplication.hpp>

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_set>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "ImageMatchNLP.h"

class ImageMatchSolver : public Ipopt::TNLP {
public:
    ImageMatchNLP *imageMatchNLP;

    int nx = 0;
    int ng = 0;
    double *primal;

    /** default constructor */
    ImageMatchSolver(ImageMatchNLP *_imageMatchNlp)
            : imageMatchNLP(_imageMatchNlp) {
        nx = imageMatchNLP->x_guess.rows();
        primal = new double[nx];
        ng = nx;
        std::cout << "[ipopt]: #variable: " << nx << std::endl;
    }


    /** default destructor */
    virtual ~ImageMatchSolver() {
        delete[] primal;
    }

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Ipopt::Index &n,
                              Ipopt::Index &m,
                              Ipopt::Index &nnz_jac_g,
                              Ipopt::Index &nnz_h_lag,
                              Ipopt::TNLP::IndexStyleEnum &index_style) {
        std::cout << "[ipopt] get nlp info" << std::endl;

        n = nx;
        m = ng;
        nnz_jac_g = nx * nx;

        int cnt = 0;
        for (int row = 0; row < n; row++) {
            for (int col = 0; col <= row; col++) {
                cnt++;
            }
        }
        nnz_h_lag = cnt;
        index_style = TNLP::C_STYLE;

        return true;
    }

#define IDX_C(k, i) (((k) * n_free + (i)) * dims)

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Ipopt::Index n,
                                 Ipopt::Number *x_l,
                                 Ipopt::Number *x_u,
                                 Ipopt::Index m,
                                 Ipopt::Number *g_l,
                                 Ipopt::Number *g_u) {
        std::cout << "[ipopt] get bounds" << std::endl;

        int dims = imageMatchNLP->energy->tessellation->getNumVertexParams() + 2;
        int n_free = imageMatchNLP->energy->n_free;

        tbb::parallel_for(0, n, [&](int i) {
            x_l[i] = -1e19;
            x_u[i] = 1e19;
            if (i % dims == 0 && i < n_free * dims) {
                x_l[i] = -imageMatchNLP->objective->dx;
                x_u[i] = imageMatchNLP->objective->dx;
            }
            if (i % dims == 1 && i < n_free * dims) {
                x_l[i] = -imageMatchNLP->objective->dy;
                x_u[i] = imageMatchNLP->objective->dy;
            }
            if (i % dims == 2 && i < n_free * dims) {
                x_l[i] = -10;
                x_u[i] = 10;
            }
        });

        tbb::parallel_for(0, m, [&](int i) {
            g_l[i] = 0;
            g_u[i] = 0;
        });

        return true;
    }

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Ipopt::Index n,
                                    bool init_x,
                                    Ipopt::Number *x,
                                    bool init_z,
                                    Ipopt::Number *z_L,
                                    Ipopt::Number *z_U,
                                    Ipopt::Index m,
                                    bool init_lambda,
                                    Ipopt::Number *lambda) {
        std::cout << "[ipopt] get starting points" << std::endl;
        assert(init_x == true);
        assert(init_z == false);
        assert(init_lambda == false);

        for (int i = 0; i < n; ++i)
            x[i] = imageMatchNLP->x_guess[i];

        return true;
    }

    /** Method to return the objective value */
    virtual bool eval_f(Ipopt::Index n,
                        const Ipopt::Number *x,
                        bool new_x,
                        Ipopt::Number &obj_value) {
        Eigen::Map<const VectorXd> x_eigen(x, n);
        obj_value = imageMatchNLP->eval_f(x_eigen);

//        std::cout << "[ipopt] eval_f: " << obj_value << std::endl;
        return true;
    }

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Ipopt::Index n,
                             const Ipopt::Number *x,
                             bool new_x,
                             Ipopt::Number *grad_f) {
//        std::cout << "[ipopt] eval_grad" << std::endl;

        Eigen::Map<const VectorXd> x_eigen(x, n);
        Eigen::VectorXd grad_f_eigen = imageMatchNLP->eval_grad_f(x_eigen);

        tbb::parallel_for(0, n, [&](int i) {
            grad_f[i] = grad_f_eigen(i);
        });
        return true;
    }

    /** Method to return the constraint residuals */
    virtual bool eval_g(Ipopt::Index n,
                        const Ipopt::Number *x,
                        bool new_x,
                        Ipopt::Index m,
                        Ipopt::Number *g) {
//        std::cout << "[ipopt] eval_g" << std::endl;

        Eigen::Map<const VectorXd> x_eigen(x, n);
        Eigen::VectorXd g_eigen = imageMatchNLP->eval_g(x_eigen);

        tbb::parallel_for(0, m, [&](int i) {
            g[i] = g_eigen(i);
        });
        return true;
    }

    /** Method to return:s
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
    virtual bool eval_jac_g(Ipopt::Index n,
                            const Ipopt::Number *x,
                            bool new_x,
                            Ipopt::Index m,
                            Ipopt::Index nele_jac,
                            Ipopt::Index *iRow,
                            Ipopt::Index *jCol,
                            Ipopt::Number *values) {
//        std::cout << "[ipopt] eval_jac_g" << std::endl;
        // Reminder: nnz_jac_g = (N * nc * nc) + ((2 * N - 3) * nc) + (N * 2);

        if (iRow != NULL) {
            assert(jCol != NULL);
            assert(values == NULL);

            // First call. Only provide sparsity structure.
            int idx;
            for (int i = 0; i < ng; i++) {
                for (int j = 0; j < nx; j++) {
                    idx = i * nx + j;
                    iRow[idx] = i;
                    jCol[idx] = j;
                }
            }
        } else {
            assert(jCol == NULL);
            assert(values != NULL);

            // Subsequent calls. Provide constraint Jacobian values.
            Eigen::Map<const VectorXd> x_eigen(x, n);
            Eigen::SparseMatrix<double> jac_g_eigen = imageMatchNLP->eval_jac_g_sparsematrix(x_eigen);

            // First call. Only provide sparsity structure.
            int idx;
            for (int i = 0; i < ng; i++) {
                for (int j = 0; j < nx; j++) {
                    idx = i * nx + j;
                    values[idx] = jac_g_eigen.coeff(i, j);
                }
            }
        }

        return true;
    }

//    /** Method to return:
//   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
//   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
//   */
//    virtual bool eval_h(Ipopt::Index n,
//                        const Ipopt::Number *x,
//                        bool new_x,
//                        Ipopt::Number obj_factor,
//                        Ipopt::Index m,
//                        const Ipopt::Number *lambda,
//                        bool new_lambda,
//                        Ipopt::Index nele_hess,
//                        Ipopt::Index *iRow,
//                        Ipopt::Index *jCol,
//                        Ipopt::Number *values) {
//        std::cout << "[ipopt] eval_h" << std::endl;
//        if (values == NULL) {
//            int cnt = 0;
//            for (int row = 0; row < n; row++) {
//                for (int col = 0; col <= row; col++) {
//                    iRow[cnt] = row;
//                    jCol[cnt] = col;
//                    cnt++;
//                }
//            }
//        } else {
//            VectorXT p_curr(variable_num);
//            for (int i = 0; i < variable_num; i++)
//                p_curr[i] = x[i];
//            MatrixXT HGN;
//            objective.hessianGN(p_curr, HGN, true, new_x);
//            objective.equilibrium_prev = objective.vertex_model.u;
//            int cnt = 0;
//            for (int row = 0; row < n; row++) {
//                for (int col = 0; col <= row; col++) {
//                    values[cnt] = HGN(row, col);
//                    cnt++;
//                }
//            }
//        }
//        return true;
//    }

    //@}

    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status,
                                   Ipopt::Index n,
                                   const Ipopt::Number *x,
                                   const Ipopt::Number *z_L,
                                   const Ipopt::Number *z_U,
                                   Ipopt::Index m,
                                   const Ipopt::Number *g,
                                   const Ipopt::Number *lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData *ip_data,
                                   Ipopt::IpoptCalculatedQuantities *ip_cq) {
        for (int i = 0; i < n; i++) {
            imageMatchNLP->x_sol[i] = x[i];
        }
    }
    //@}

    virtual bool intermediate_callback(Ipopt::AlgorithmMode mode,
                                       Ipopt::Index iter, Ipopt::Number obj_value,
                                       Ipopt::Number inf_pr, Ipopt::Number inf_du,
                                       Ipopt::Number mu, Ipopt::Number d_norm,
                                       Ipopt::Number regularization_size,
                                       Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
                                       Ipopt::Index ls_trials,
                                       const Ipopt::IpoptData *ip_data,
                                       Ipopt::IpoptCalculatedQuantities *ip_cq) {
        Ipopt::TNLP::intermediate_callback(mode, iter, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size,
                                           alpha_du, alpha_pr, ls_trials, ip_data, ip_cq);

        using namespace Ipopt;
        Ipopt::TNLPAdapter *tnlp_adapter = NULL;
        if (ip_cq != NULL) {
            Ipopt::OrigIpoptNLP *orignlp;
            orignlp = dynamic_cast<OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
            if (orignlp != NULL)
                tnlp_adapter = dynamic_cast<TNLPAdapter *>(GetRawPtr(orignlp->nlp()));

            if (tnlp_adapter != NULL) {
                tnlp_adapter->ResortX(*ip_data->curr()->x(), primal);

                double *intermediate = new double[nx];
                tnlp_adapter->ResortX(*ip_data->curr()->x(), intermediate);

                for (int i = 0; i < nx; i++) {
                    imageMatchNLP->x_sol[i] = intermediate[i];
                }
            }
        }

        return true;
    }

//    virtual bool intermediate_callback(Ipopt::AlgorithmMode mode,
//                                       Ipopt::Index iter, Ipopt::Number obj_value,
//                                       Ipopt::Number inf_pr, Ipopt::Number inf_du,
//                                       Ipopt::Number mu, Ipopt::Number d_norm,
//                                       Ipopt::Number regularization_size,
//                                       Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
//                                       Ipopt::Index ls_trials,
//                                       const Ipopt::IpoptData *ip_data,
//                                       Ipopt::IpoptCalculatedQuantities *ip_cq) {
//        using namespace Ipopt;
//        Ipopt::TNLPAdapter *tnlp_adapter = NULL;
//        if (ip_cq != NULL) {
//            Ipopt::OrigIpoptNLP *orignlp;
//            orignlp = dynamic_cast<OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
//            if (orignlp != NULL)
//                tnlp_adapter = dynamic_cast<TNLPAdapter *>(GetRawPtr(orignlp->nlp()));
//            tnlp_adapter->ResortX(*ip_data->curr()->x(), primals);
//
//            if (tnlp_adapter != NULL) {
//                double *intermediate = new double[variable_num];
//                tnlp_adapter->ResortX(*ip_data->curr()->x(), intermediate);
//
//                delete[] primals;
//                primals = intermediate;
//
//                if (obj_value < min_objective) {
//                    VectorXT p_curr(variable_num);
//                    for (int i = 0; i < variable_num; i++)
//                        p_curr[i] = primals[i];
//
//                    T E = objective.value(p_curr, true, false);
//                    objective.saveState(data_folder + "/" + std::to_string(count) + ".obj");
//                    saveDesignParameters(data_folder + "/" + std::to_string(count) + ".txt", p_curr);
//                    min_objective = obj_value;
//                    std::cout << "[ipopt]\t real obj: " << E << std::endl;
//                    count++;
//                }
//            }
//            std::cout << "[ipopt]\tIter:\t" << iter << "\tIpopt obj:\t" << obj_value << "\tls #" << ls_trials
//                      << "\tconstraint voilation: " << inf_pr << std::endl;
//        }
//
//        return true;
//    }

private:
//    void saveDesignParameters(const std::string &filename, const VectorXT &params) {
//        std::ofstream out(filename);
//        out << std::setprecision(20) << params << std::endl;
//        out.close();
//    }
    /**@name Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *
   */
    //@{
    //  IpoptSolver();
    ImageMatchSolver(const ImageMatchSolver &);

    ImageMatchSolver &operator=(const ImageMatchSolver &);
    //@}
};
