#ifndef IPOPT_FORWARD_SOLVER_H
#define IPOPT_FORWARD_SOLVER_H

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
// #include "IpTNLP.hpp"
#include <cassert>
#include <iostream>

#include "Simulation.h"

#include "VecMatDef.h"
class Simulation;

class IpoptForwardSolver : public Ipopt::TNLP 
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using Edge = Vector<int, 2>;


    Simulation& simulation;
    int current_iteration = 0;
    int function_evaluation_cnt = 0;
    int variable_num = 0;
    int count = 0;
    double* primals;
    T min_objective = T(1e10);
    std::string data_folder;
    bool use_full_hessian = true;
    bool save_results = false;
    

    /** default constructor */
    IpoptForwardSolver(Simulation& _simulation, const std::string& _data_folder, 
        bool _use_full_hessian, bool _save_results)
        : simulation(_simulation), data_folder(_data_folder), 
        use_full_hessian(_use_full_hessian),
        save_results(_save_results)
    {
        variable_num = simulation.deformed.rows();
        primals = new double[variable_num];
        std::cout << "[ipopt]: #variable: " << variable_num << std::endl;
    }


    /** default destructor */
    virtual ~IpoptForwardSolver()
    {
        delete[] primals;
    }

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Ipopt::Index& n,
        Ipopt::Index& m,
        Ipopt::Index& nnz_jac_g,
        Ipopt::Index& nnz_h_lag,
        Ipopt::TNLP::IndexStyleEnum& index_style)
    {
        std::cout << "[ipopt] get nlp info" << std::endl;

        n = variable_num;
        m = 0;
        nnz_jac_g = 0;
        if (use_full_hessian)
        {
            int cnt = 0;
            for( int row = 0; row < n; row++ )
            {
                for( int col = 0; col <= row; col++ )
                {
                    cnt++;
                }
            }
            nnz_h_lag = cnt;
        }
        index_style = TNLP::C_STYLE;
        
        return true;
    }

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Ipopt::Index n,
        Ipopt::Number* x_l,
        Ipopt::Number* x_u,
        Ipopt::Index m,
        Ipopt::Number* g_l,
        Ipopt::Number* g_u)
    {
        std::cout << "[ipopt] get bounds" << std::endl;
        
        tbb::parallel_for(0, n, [&](int i) {
                x_l[i] = -1e19;
                x_u[i] = 1e19;
            });

        simulation.cells.iterateDirichletDoF([&](int offset, double target)
        {
            x_l[offset] = target;
            x_u[offset] = target;
        });

        return true;
    }

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Ipopt::Index n,
        bool init_x,
        Ipopt::Number* x,
        bool init_z,
        Ipopt::Number* z_L,
        Ipopt::Number* z_U,
        Ipopt::Index m,
        bool init_lambda,
        Ipopt::Number* lambda)
    {
        std::cout << "[ipopt] get starting points" << std::endl;
        assert(init_x == true);
        assert(init_z == false);
        assert(init_lambda == false);

        for (int i = 0; i < n; ++i) 
            x[i] = simulation.u[i];

        return true;
    }

    /** Method to return the objective value */
    virtual bool eval_f(Ipopt::Index n,
        const Ipopt::Number* x,
        bool new_x,
        Ipopt::Number& obj_value)
    {
        std::cout << "[ipopt] eval_f" << std::endl;
        VectorXT u(n);
        for (int i = 0; i < n; i++)
            u[i] = x[i];

        T E = 0.0;
        E = simulation.computeTotalEnergy(u, false);
        
        std::cout << "[ipopt] eval_f: " << E << std::endl;
        
        obj_value = (Ipopt::Number)E;
        return true;
    }

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Ipopt::Index n,
        const Ipopt::Number* x,
        bool new_x,
        Ipopt::Number* grad_f)
    {
        std::cout << "[ipopt] eval_grad" << std::endl;
        VectorXT u(n);
        for (int i = 0; i < n; i++)
            u[i] = x[i];
        VectorXT residual(n);
        residual.setZero();
        simulation.computeResidual(u, residual);
        tbb::parallel_for(0, variable_num, [&](int i) 
        {
            grad_f[i] = -residual[i];
        });

        std::cout << "[ipopt] |g|: " << residual.norm() << std::endl;
        return true;
    }

    /** Method to return the constraint residuals */
    virtual bool eval_g(Ipopt::Index n,
        const Ipopt::Number* x,
        bool new_x,
        Ipopt::Index m,
        Ipopt::Number* g)
    {
        std::cout << "[ipopt] eval_g" << std::endl;
        return false;
    }
    /** Method to return:s
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
    virtual bool eval_jac_g(Ipopt::Index n,
        const Ipopt::Number* x,
        bool new_x,
        Ipopt::Index m,
        Ipopt::Index nele_jac,
        Ipopt::Index* iRow,
        Ipopt::Index* jCol,
        Ipopt::Number* values)
    {
        std::cout << "[ipopt] eval_jac_g" << std::endl;
        return false;
    }

    /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
    virtual bool eval_h(Ipopt::Index n,
        const Ipopt::Number* x,
        bool new_x,
        Ipopt::Number obj_factor,
        Ipopt::Index m,
        const Ipopt::Number* lambda,
        bool new_lambda,
        Ipopt::Index nele_hess,
        Ipopt::Index* iRow,
        Ipopt::Index* jCol,
        Ipopt::Number* values)
    {
        std::cout << "[ipopt] eval_h" << std::endl;
        // if (!use_GN_hessian)
        //     return false;
        // if (values == NULL)
        // {
        //     int cnt = 0;
        //     for( int row = 0; row < n; row++ )
        //     {
        //         for( int col = 0; col <= row; col++ )
        //         {
        //             iRow[cnt] = row;
        //             jCol[cnt] = col;
        //             cnt++;
        //         }
        //     }
        // }
        // else
        // {
        //     VectorXT p_curr(variable_num);
        //     for (int i = 0; i < variable_num; i++)
        //         p_curr[i] = x[i];
        //     MatrixXT HGN;
        //     if (restart_all_the_time)
        //     {
        //         objective.hessianGN(p_curr, HGN, true, false);
        //     }
        //     else
        //     {
        //         if (new_x)
        //             objective.hessianGN(p_curr, HGN, new_x, false);
        //         else
        //             objective.hessianGN(p_curr, HGN, false);
        //     }
        //     // objective.hessianGN(p_curr, HGN, true, false);
        //     HGN.diagonal().array() += 1e-6;
        //     objective.equilibrium_prev = objective.simulation.u;
        //     int cnt = 0;
        //     for( int row = 0; row < n; row++ )
        //     {
        //         for( int col = 0; col <= row; col++ )
        //         {
        //             values[cnt] = HGN(row, col);
        //             cnt++;
        //         }
        //     }
        // }
        return true;
    }

    //@}

    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status,
        Ipopt::Index n,
        const Ipopt::Number* x,
        const Ipopt::Number* z_L,
        const Ipopt::Number* z_U,
        Ipopt::Index m,
        const Ipopt::Number* g,
        const Ipopt::Number* lambda,
        Ipopt::Number obj_value,
        const Ipopt::IpoptData* ip_data,
        Ipopt::IpoptCalculatedQuantities* ip_cq)
    {
        // VectorXT p_curr(n);
        // for (int i = 0; i < n; i++)
        //     p_curr[i] = x[i];
        // objective.updateDesignParameters(p_curr);
        
        // if (save_results)
        // {
        //     objective.saveDesignParameters(data_folder + "/p_ipopt.txt", p_curr);
        //     objective.saveState(data_folder + "/x_ipopt.obj");
        // }

        // here is where we would store the solution to variables, or write to a file, etc
        // so we could use the solution.
        
        // For this example, we write the solution to the console
        // std::cout << std::endl << std::endl << "Solution of the primal variables, x" << std::endl;
        // for( Index i = 0; i < n; i++ )
        // {
        //     std::cout << "x[" << i << "] = " << x[i] << std::endl;
        // }
        
        // std::cout << std::endl << std::endl << "Solution of the bound multipliers, z_L and z_U" << std::endl;
        // for( Index i = 0; i < n; i++ )
        // {
        //     std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
        // }
        // for( Index i = 0; i < n; i++ )
        // {
        //     std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
        // }
        
        // std::cout << std::endl << std::endl << "Objective value" << std::endl;
        // std::cout << "f(x*) = " << obj_value << std::endl;
        
        // std::cout << std::endl << "Final value of the constraints:" << std::endl;
        // for( Index i = 0; i < m; i++ )
        // {
        //     std::cout << "g(" << i << ") = " << g[i] << std::endl;
        // }
    }
    //@}


    virtual bool intermediate_callback(Ipopt::AlgorithmMode mode,
        Ipopt::Index iter, Ipopt::Number obj_value,
        Ipopt::Number inf_pr, Ipopt::Number inf_du,
        Ipopt::Number mu, Ipopt::Number d_norm,
        Ipopt::Number regularization_size,
        Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
        Ipopt::Index ls_trials,
        const Ipopt::IpoptData* ip_data,
        Ipopt::IpoptCalculatedQuantities* ip_cq)
    {
        using namespace Ipopt;
        Ipopt::TNLPAdapter* tnlp_adapter = NULL;
        if (ip_cq != NULL) {
            Ipopt::OrigIpoptNLP* orignlp;
            orignlp = dynamic_cast<OrigIpoptNLP*>(GetRawPtr(ip_cq->GetIpoptNLP()));
            if (orignlp != NULL)
                tnlp_adapter = dynamic_cast<TNLPAdapter*>(GetRawPtr(orignlp->nlp()));
                       tnlp_adapter->ResortX(*ip_data->curr()->x(), primals);

            if (tnlp_adapter != NULL) 
            {
                current_iteration = iter;
                function_evaluation_cnt = 0;
                double* intermediate = new double[variable_num];
                tnlp_adapter->ResortX(*ip_data->curr()->x(), intermediate);

                delete[] primals;
                primals = intermediate;

                // if (obj_value < min_objective) 
                // {
                //     VectorXT p_curr(variable_num);
                //     for (int i = 0; i < variable_num; i++)
                //         p_curr[i] = primals[i];

                //     T E = objective.value(p_curr, true, true);
                //     if (save_results)
                //     {
                //         objective.saveState(data_folder + "/" + std::to_string(count) + ".obj");
                //         objective.saveDesignParameters(data_folder + "/" + std::to_string(count) + ".txt", p_curr);
                //     }
                //     min_objective = obj_value;   
                //     std::cout << "[ipopt]\t real obj: " << E << std::endl; 
                //     count++;
                // }
            }
            std::cout << "[ipopt]\tIter:\t" << iter << "\tIpopt obj:\t" << obj_value << "\tls #" << ls_trials << "\tconstraint voilation: " << inf_pr << std::endl;
        }

        return true;
    }

private:
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
    //  IpoptForwardSolver();
    IpoptForwardSolver(const IpoptForwardSolver&);
    IpoptForwardSolver& operator=(const IpoptForwardSolver&);
    //@}
};


#endif