// #ifndef KINITRO_SOLVER_H
// #define KINITRO_SOLVER_H

// #include <utility>
// #include <iostream>
// #include <fstream>
// #include <Eigen/Geometry>
// #include <Eigen/Core>
// #include <Eigen/Sparse>
// #include <Eigen/Dense>
// #include <tbb/tbb.h>
// #include <unordered_set>
// // #include "IpTNLP.hpp"
// #include <cassert>
// #include <iostream> 


// #include "KNSolver.h"
// #include "KNProblem.h"
// #include "KNCallbacks.h"
// #include "Objectives.h"
// class ObjNucleiTracking;

// #include "knitro.h"
// using namespace knitro;

// class InverseProblem : public knitro::KNProblem 
// {
// public:
// 	using TV = Vector<double, 3>;
//     using TM = Matrix<double, 3, 3>;
//     using IV = Vector<int, 3>;

//     typedef int StorageIndex;
//     using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
//     // using StiffnessMatrix = Eigen::SparseMatrix<T>;
//     using Entry = Eigen::Triplet<T>;
//     using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
//     using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
//     using VectorXi = Vector<int, Eigen::Dynamic>;
//     using Edge = Vector<int, 2>;

// 	std::string data_folder;
//     bool use_GN_hessian = true;
//     bool save_results = false;
// public:

//     /*------------------------------------------------------------------*/
//     /*     FUNCTION callbackEvalF                                       */
//     /*------------------------------------------------------------------*/
//     /** The signature of this function matches KN_eval_callback in knitro.h.
//      *  Only "obj" is set in the KN_eval_result structure.
//      */
// 	static int callbackEvalF (KN_context_ptr             kc,
//                    CB_context_ptr             cb,
//                    KN_eval_request_ptr const  evalRequest,
//                    KN_eval_result_ptr  const  evalResult,
//                    void              * const  userParams)
// 	{
// 		std::cout << "[knitro] eval_f" << std::endl;
//     	const double *x;
//     	double *obj;
    	

//     	if (evalRequest->type != KN_RC_EVALFC)
//     	{
//         	printf ("*** callbackEvalF incorrectly called with eval type %d\n", evalRequest->type);
//         	return( -1 );
//     	}
//     	x = evalRequest->x;
//     	obj = evalResult->obj;

// 		Objectives* objective = (Objectives*) userParams;
// 		int n_dof_design = objective->n_dof_design;
//     	VectorXT p_curr(n_dof_design);
//         for (int i = 0; i < n_dof_design; i++)
//             p_curr[i] = x[i];

// 		T E = objective->value(p_curr, true, true);
// 		if (E < 100)
//             objective->equilibrium_prev = objective->simulation.u;

//     	*obj = E;
		
//     	return( 0 );
// 	}

// 	/*------------------------------------------------------------------*/
// 	/*     FUNCTION callbackEvalG                                       */
// 	/*------------------------------------------------------------------*/
// 	/** The signature of this function matches KN_eval_callback in knitro.h.
// 	 *  Only "objGrad" is set in the KN_eval_result structure.
// 	 */
// 	static int callbackEvalG (KN_context_ptr             kc,
// 	                   CB_context_ptr             cb,
// 	                   KN_eval_request_ptr const  evalRequest,
// 	                   KN_eval_result_ptr  const  evalResult,
// 	                   void              * const  userParams)
// 	{
// 		std::cout << "[knitro] eval_grad" << std::endl;
// 	    const double *x;
// 	    double *objGrad;
// 	    double dTmp;

// 	    if (evalRequest->type != KN_RC_EVALGA)
// 	    {
// 	        printf ("*** callbackEvalG incorrectly called with eval type %d\n",
// 	                evalRequest->type);
// 	        return( -1 );
// 	    }
// 	    x = evalRequest->x;
// 	    objGrad = evalResult->objGrad;
		
// 		Objectives* objective = (Objectives*) userParams;
// 		int n_dof_design = objective->n_dof_design;

// 		VectorXT p_curr(n_dof_design);
//         for (int i = 0; i < n_dof_design; i++)
//             p_curr[i] = x[i];

// 	    T O;
//         VectorXT dOdp;
// 		objective->gradient(p_curr, dOdp, O, true, true);
// 		objective->equilibrium_prev = objective->simulation.u;
// 		tbb::parallel_for(0, n_dof_design, [&](int i) 
//         {
//             objGrad[i] = dOdp[i];
//         });

// 	    return( 0 );
// 	}

// 	/*------------------------------------------------------------------*/
// 	/*     FUNCTION callbackEvalH                                       */
// 	/*------------------------------------------------------------------*/
// 	/** The signature of this function matches KN_eval_callback in knitro.h.
// 	 *  Only "hess" and "hessVec" are set in the KN_eval_result structure.
// 	 */
// 	static int callbackEvalH (KN_context_ptr             kc,
// 	                   CB_context_ptr             cb,
// 	                   KN_eval_request_ptr const  evalRequest,
// 	                   KN_eval_result_ptr  const  evalResult,
// 	                   void              * const  userParams)
// 	{
// 	    const double *x;
// 	    double sigma;
// 	    double *hess;

// 	    if (   evalRequest->type != KN_RC_EVALH
// 	        && evalRequest->type != KN_RC_EVALH_NO_F)
// 	    {
// 	        printf ("*** callbackEvalH incorrectly called with eval type %d\n",
// 	                evalRequest->type);
// 	        return( -1 );
// 	    }

// 	    x = evalRequest->x;
// 	    /** Scale objective component of hessian by sigma */
// 	    sigma = *(evalRequest->sigma);
// 	    hess = evalResult->hess;

// 	    /** Evaluate the hessian of the nonlinear objective.
// 	     *  Note: Since the Hessian is symmetric, we only provide the
// 	     *        nonzero elements in the upper triangle (plus diagonal).
// 	     *        These are provided in row major ordering as specified
// 	     *        by the setting KN_DENSE_ROWMAJOR in "KN_set_cb_hess()".
// 	     *  Note: The Hessian terms for the quadratic constraints
// 	     *        will be added internally by Knitro to form
// 	     *        the full Hessian of the Lagrangian. */
// 	    hess[0] = sigma * ( (-400.0 * x[1]) + (1200.0 * x[0]*x[0]) + 2.0); // (0,0)
// 	    hess[1] = sigma * (-400.0 * x[0]); // (0,1)
// 	    hess[2] = sigma * 200.0;           // (1,1)

// 	    return( 0 );
// 	}

// 	InverseProblem(const std::string& _data_folder, 
//         bool _use_GN_hessian, bool _save_results) : 
// 		data_folder(_data_folder), 
//         use_GN_hessian(_use_GN_hessian), 
//         save_results(_save_results)
// 	{
// 		// std::vector<T> lower_bounds(n_dof_design), upper_bounds(n_dof_design);

// 		// std::fill(lower_bounds.begin(), lower_bounds.end(), 0);
// 		// std::fill(upper_bounds.begin(), upper_bounds.end(), 40);

// 		// VectorXT p_curr;
//         // objective.getDesignParameters(p_curr);
// 		// std::vector<T> x_initial(p_curr.rows());
// 		// for (int i = 0; i < p_curr.rows(); i++)
// 		// 	x_initial[i] = p_curr[i];
// 		// this->_nVars = n_dof_design;
// 		// this->_nCons = 0;
// 		// this->setVarLoBnds({lower_bounds});
// 		// this->setVarUpBnds({upper_bounds});
// 		// this->setXInitial({x_initial});
// 		// this->setObjEvalCallback(&InverseProblem::callbackEvalF);
// 		// this->setGradEvalCallback(&InverseProblem::callbackEvalG);
// 	}

// 	// InverseProblem() : KNProblem(2,2) {
// 	// 	// Variables
// 	// 	this->setVarLoBnds({{-KN_INFINITY, -KN_INFINITY}});
// 	// 	this->setVarUpBnds({{0.5, KN_INFINITY}});
// 	// 	this->setXInitial({{-2.0, 1.0}});

// 	// 	// Constraints
// 	// 	this->setConLoBnds({{1.0, 0.0}});

// 	// 	// Structural patterns
// 	// 	/** First load quadratic structure x0*x1 for the first constraint */
// 	// 	this->getConstraintsQuadraticParts().add(0, { {0}, {1}, {1.0} } );

// 	// 	/** Load structure for the second constraint.  below we add the linear
//     //  	*  structure and the quadratic structure separately
//     //  	*/
// 	// 	/** Add linear term x0 in the second constraint */
// 	// 	this->getConstraintsLinearParts().add(1, { {0}, {1.0} } );

//     // 	/** Add quadratic term x1^2 in the second constraint */
//     // 	this->getConstraintsQuadraticParts().add(1, { {1}, {1}, {1.0} } );

// 	// 	// Evaluations callbacks
// 	// 	this->setObjEvalCallback(&InverseProblem::callbackEvalF);
// 	// 	this->setGradEvalCallback(&InverseProblem::callbackEvalG);
// 	// 	this->setHessEvalCallback(&InverseProblem::callbackEvalH);
// 	// }
// };

// #endif