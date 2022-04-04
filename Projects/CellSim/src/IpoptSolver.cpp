// #include "../include/IpoptSolver.h"

// virtual bool IpoptSolver::get_nlp_info(Ipopt::Index& n,
//     Ipopt::Index& m,
//     Ipopt::Index& nnz_jac_g,
//     Ipopt::Index& nnz_h_lag,
//     Ipopt::TNLP::IndexStyleEnum& index_style)
// {
//     std::cout << "[ipopt] get nlp info" << std::endl;

//     n = variable_num * 2;
//     m = 0;
//     nnz_jac_g = 0;
// }

// virtual bool IpoptSolver::get_bounds_info(Ipopt::Index n,
//     Ipopt::Number* x_l,
//     Ipopt::Number* x_u,
//     Ipopt::Index m,
//     Ipopt::Number* g_l,
//     Ipopt::Number* g_u)
// {
//     std::cout << "[ipopt] get bounds" << std::endl;

//     tbb::parallel_for(0, variable_num, [&](int i) {
//         x_l[i] = objective.bound[0];
//         x_u[i] = objective.bound[1];
//     });

//     return true;
// }

// virtual bool IpoptSolver::get_starting_point(Ipopt::Index n,
//     bool init_x,
//     Ipopt::Number* x,
//     bool init_z,
//     Ipopt::Number* z_L,
//     Ipopt::Number* z_U,
//     Ipopt::Index m,
//     bool init_lambda,
//     Ipopt::Number* lambda)
// {
//     std::cout << "[ipopt] get starting points" << std::endl;
//     assert(init_x == true);
//     assert(init_z == false);
//     assert(init_lambda == false);

//     VectorXT p_curr;
//     objective.getDesignParameters(p_curr);
//     for (int i = 0; i < variable_num; ++i) 
//         x[i] = p_curr[i];

//     return true;
// }

// virtual bool IpoptSolver::eval_f(Ipopt::Index n,
//     const Ipopt::Number* x,
//     bool new_x,
//     Ipopt::Number& obj_value)
// {

//     std::cout << "[ipopt] eval_f" << std::endl;
//     VectorXT p_curr(variable_num);
//     for (int i = 0; i < variable_num; i++)
//         p[i] = x[i];

//     T E = objective.value(p_curr, true, true);
    
//     obj_value = (Ipopt::Number)E;
//     return true;
// }

// virtual bool IpoptSolver::eval_grad_f(Ipopt::Index n,
//     const Ipopt::Number* x,
//     bool new_x,
//     Ipopt::Number* grad_f)
// {
//     // std::cout << "[ipopt] eval_grad" << std::endl;
//     // TVStack dL;
//     // if (!simulation->computedL(dL))
//     //     return false;

//     // std::cout << "[ipopt] grad norm: " << dL.cwiseAbs().maxCoeff() << std::endl;

//     // tbb::parallel_for(0, variable_num, [&](int i) {
//     //     for (int d = 0; d < dim; ++d) {
//     //         grad_f[i * dim + d] = dL(d, simulation->variable2particle[i]);
//     //     }
//     // });
//     return true;
// }

// virtual bool IpoptSolver::eval_g(Ipopt::Index n,
//     const Ipopt::Number* x,
//     bool new_x,
//     Ipopt::Index m,
//     Ipopt::Number* g)
// {
//     //        std::cout << "[ipopt] eval_g" << std::endl;
//     //        tbb::parallel_for(0, simulation->particles.count, [&](int i) {
//     //            g[i] = 0;
//     //            for (int j = 0; j < dim; ++j)
//     //                g[i] += std::pow(x[dim * i + j] - primals[dim * i + j], 2);
//     //        });
//     return true;
// }

// virtual bool IpoptSolver::eval_jac_g(Ipopt::Index n,
//     const Ipopt::Number* x,
//     bool new_x,
//     Ipopt::Index m,
//     Ipopt::Index nele_jac,
//     Ipopt::Index* iRow,
//     Ipopt::Index* jCol,
//     Ipopt::Number* values)
// {
//     //        std::cout << "[ipopt] eval_jac_g" << std::endl;
//     //        if (values == NULL) {
//     //            tbb::parallel_for(0, simulation->particles.count, [&](int i) {
//     //                auto& Xp = simulation->particles.X[i];
//     //                for (int j = 0; j < dim; ++j) {
//     //                    iRow[dim * i + j] = i;
//     //                    jCol[dim * i + j] = dim * i + j;
//     //                }
//     //            });
//     //        }
//     //        else {
//     //            tbb::parallel_for(0, simulation->particles.count, [&](int i) {
//     //                auto& Xp = simulation->particles.X[i];
//     //                for (int j = 0; j < dim; ++j) {
//     //                    values[dim * i + j] = 2 * (x[dim * i + j] - primals[dim * i + j]);
//     //                }
//     //            });
//     //        }
//     return true;
// }

// virtual bool IpoptSolver::eval_h(Ipopt::Index n,
//     const Ipopt::Number* x,
//     bool new_x,
//     Ipopt::Number obj_factor,
//     Ipopt::Index m,
//     const Ipopt::Number* lambda,
//     bool new_lambda,
//     Ipopt::Index nele_hess,
//     Ipopt::Index* iRow,
//     Ipopt::Index* jCol,
//     Ipopt::Number* values)
// {
//     std::cout << "[ipopt] eval_h" << std::endl;
//     return false;
// }

// virtual void IpoptSolver::finalize_solution(Ipopt::SolverReturn status,
//     Ipopt::Index n,
//     const Ipopt::Number* x,
//     const Ipopt::Number* z_L,
//     const Ipopt::Number* z_U,
//     Ipopt::Index m,
//     const Ipopt::Number* g,
//     const Ipopt::Number* lambda,
//     Ipopt::Number obj_value,
//     const Ipopt::IpoptData* ip_data,
//     Ipopt::IpoptCalculatedQuantities* ip_cq)
// {
//     // tbb::parallel_for(0, variable_num, [&](int i) {
//     //     auto& Xp = simulation->particles.X[simulation->variable2particle[i]];
//     //     for (int d = 0; d < dim; ++d) {
//     //         Xp[d] = x[i * dim + d];
//     //     }
//     // });
// }

// virtual bool IpoptSolver::intermediate_callback(Ipopt::AlgorithmMode mode,
//     Ipopt::Index iter, Ipopt::Number obj_value,
//     Ipopt::Number inf_pr, Ipopt::Number inf_du,
//     Ipopt::Number mu, Ipopt::Number d_norm,
//     Ipopt::Number regularization_size,
//     Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
//     Ipopt::Index ls_trials,
//     const Ipopt::IpoptData* ip_data,
//     Ipopt::IpoptCalculatedQuantities* ip_cq)
// {
//     using namespace Ipopt;
//     Ipopt::TNLPAdapter* tnlp_adapter = NULL;
//     if (ip_cq != NULL) {
//         Ipopt::OrigIpoptNLP* orignlp;
//         orignlp = dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
//         if (orignlp != NULL)
//             tnlp_adapter = dynamic_cast<TNLPAdapter*>(GetRawPtr(orignlp->nlp()));
//         //            tnlp_adapter->ResortX(*ip_data->curr()->x(), primals);

//         if (tnlp_adapter != NULL) {
//             double* intermediate = new double[variable_num];
//             tnlp_adapter->ResortX(*ip_data->curr()->x(), intermediate);

//             // tbb::parallel_for(0, variable_num, [&](int i) {
//             //     auto& Xp = simulation->particles.X[simulation->variable2particle[i]];
//             //     for (int d = 0; d < dim; ++d) {
//             //         Xp[d] = primals[i * dim + d];
//             //     }
//             // });
//             //simulation->visualize_particle_displacement(count, primals, intermediate);

//             delete[] primals;
//             primals = intermediate;
//             //if (obj_value < min_objective) {
//             // simulation->saveToBgeo(count);
//             // T L;
//             // simulation->computeL(L);
//             // std::cout << "[ipopt] " << count << ".bgeo saved L: " << L << std::endl;
//             // min_objective = obj_value;
//             // //simulation->visualize_particle_grad(count);
//             // //simulation->visualize_grid_u(count);
//             // count++;
//             //}
//         }
//         std::cout << "[ipopt]\tIter:\t" << iter << "\tObjective:\t" << obj_value << "\tls #" << ls_trials << "\tconstraint voilation: " << inf_pr << std::endl;
//     }

//     return true;
// }