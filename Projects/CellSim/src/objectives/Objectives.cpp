#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>

// void ObjUMatching::setTargetFromMesh(const std::string& filename)
// {
//     simulation.loadDeformedState(filename);
//     // target = 0.1 * simulation.u;
//     target = simulation.u;
// }

// T ObjUMatching::value(const VectorXT& p_curr, bool use_prev_equil)
// {
//     simulation.reset();
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();
//     return 0.5 * (simulation.u - target).dot(simulation.u - target);
// }

// T ObjUMatching::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
// {
//     simulation.reset();
    
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();

//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = simulation.u - target;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
//     return dOdp.norm();
// }

// T ObjUMatching::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
// {
//     simulation.reset();
    
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();
    
//     energy = 0.5 * (simulation.u - target).dot(simulation.u - target);
//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = simulation.u - target;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
//     equilibrium_prev = simulation.u;
//     return dOdp.norm();
// }

// T ObjUMatching::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
// {
//     updateDesignParameters(p_curr);
//     energy = 0.5 * (simulation.u - target).dot(simulation.u - target);
//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = simulation.u - target;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
//     return dOdp.norm();
// }

// void ObjUMatching::updateDesignParameters(const VectorXT& design_parameters)
// {
//     simulation.cells.edge_weights = design_parameters;
// }

// void ObjUMatching::getDesignParameters(VectorXT& design_parameters)
// {
//     design_parameters = simulation.cells.edge_weights;
// }

// void ObjUMatching::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
// {
//     _sim_dof = simulation.num_nodes * 3;
//     _design_dof = simulation.cells.edge_weights.rows();
//     n_dof_sim = _sim_dof;
//     n_dof_design = _design_dof;
// }



// T ObjUMatching::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool simulate, bool use_prev_equil)
// {
//     updateDesignParameters(p_curr);
//     if (simulate)
//     {
//         simulation.reset();
//         simulation.staticSolve();
//     }

//     MatrixXT dxdp;
//     simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

//     H = (dxdp.transpose() * dxdp).sparseView();
// }

// T ObjUTU::value(const VectorXT& p_curr, bool use_prev_equil)
// {
//     simulation.reset();
//     if (use_prev_equil)
//         simulation.u = equilibrium_prev;
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();
//     return -0.5 * simulation.u.dot(simulation.u);
// }

// T ObjUTU::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
// {
//     simulation.reset();
//     if (use_prev_equil)
//         simulation.u = equilibrium_prev;
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();

//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = -simulation.u;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
//     equilibrium_prev = simulation.u;
//     return dOdp.norm();
// }   

// T ObjUTU::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
// {
//     updateDesignParameters(p_curr);
//     energy = -0.5 * simulation.u.dot(simulation.u);
//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = -simulation.u;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
//     return dOdp.norm();
// }

// T ObjUTU::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
// {
//     simulation.reset();
//     if (use_prev_equil)
//         simulation.u = equilibrium_prev;
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();
//     energy = -0.5 * simulation.u.dot(simulation.u);
//     StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
//     if (simulation.woodbury)
//     {
//         MatrixXT UV;
//         simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
//     }
//     else
//     {   
//         simulation.buildSystemMatrix(simulation.u, d2edx2);
//     }
    
//     VectorXT dOdu = -simulation.u;
    
//     Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
//     solver.analyzePattern(d2edx2);
//     solver.factorize(d2edx2);
//     if (solver.info() == Eigen::NumericalIssue)
//         std::cout << "Forward simulation hessian indefinite" << std::endl;
    
//     // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
//     VectorXT lambda = solver.solve(dOdu);
    
//     simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
//     equilibrium_prev = simulation.u;
//     return dOdp.norm();
// }

// T ObjUTU::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil)
// {
//     simulation.reset();
//     if (use_prev_equil)
//         simulation.u = equilibrium_prev;
//     updateDesignParameters(p_curr);
//     simulation.staticSolve();

//     MatrixXT dxdp;
//     simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

//     H = (dxdp.transpose() * dxdp).sparseView();
    
// }

// void ObjUTU::getDesignParameters(VectorXT& design_parameters)
// {
//     design_parameters = simulation.cells.edge_weights;
// }

// void ObjUTU::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
// {
//     _sim_dof = simulation.num_nodes * 3;
//     _design_dof = simulation.cells.edge_weights.rows();
//     n_dof_sim = _sim_dof;
//     n_dof_design = _design_dof;
// }

// void ObjUTU::updateDesignParameters(const VectorXT& design_parameters)
// {
//     simulation.cells.edge_weights = design_parameters;
// }

void Objectives::setSimulationAndDesignDoF(int _sim_dof, int _design_dof)
{
    n_dof_design = _design_dof;
    n_dof_sim = _sim_dof;
}

void Objectives::saveDesignParameters(const std::string& filename, const VectorXT& design_parameters)
{
    std::ofstream out(filename);
    for (int i = 0; i < design_parameters.rows(); i++)
        out << design_parameters[i] << std::endl;
    out.close();
}

void Objectives::assembleSGNHessianBCZero(StiffnessMatrix &A, const StiffnessMatrix &dfdx,
	    const StiffnessMatrix &dfdp, StiffnessMatrix &KKT)
{
	StiffnessMatrix dfdxT = dfdx.transpose();
	StiffnessMatrix dfdpT = dfdp.transpose();

    int nx = A.rows();
	int np = dfdp.cols();
	int npnx = np + nx;

    int nl = dfdx.rows();

    KKT.resize(npnx + nl, npnx + nl);

    for (int k = 0; k < nx; k++)
	{
		KKT.startVec(k);
		for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
		{
			KKT.insertBack(it.row(), k) = it.value();
		}
		for (StiffnessMatrix::InnerIterator it(dfdx, k); it; ++it)
		{
			KKT.insertBack(it.row() + npnx, k) = it.value();
		}
	}

    for (int k = 0; k < np; k++)
	{
		KKT.startVec(k + nx);
		for (StiffnessMatrix::InnerIterator it(dfdp, k); it; ++it)
		{
			KKT.insertBack(it.row() + npnx, k + nx) = it.value();
		}
        
	}
    
	for (int k = 0; k < nl; k++)
	{
		KKT.startVec(k + npnx);
		for (StiffnessMatrix::InnerIterator it(dfdxT, k); it; ++it)
		{
			KKT.insertBack(it.row(), k + npnx) = it.value();
		}
        for (StiffnessMatrix::InnerIterator it(dfdpT, k); it; ++it)
		{
			KKT.insertBack(it.row()+nx, k + npnx) = it.value();
		}
	}
    KKT.finalize();
}

void Objectives::assembleSGNHessian(StiffnessMatrix &A,
	const StiffnessMatrix &B,
	const StiffnessMatrix &C,
	const StiffnessMatrix &dfdx,
	const StiffnessMatrix &dfdp,
	StiffnessMatrix &KKT)
{
	//START_SECTION_TIMER_METHOD_WITH_HELPER();
	//we are constructing [A, B', c[0]';
	//                     B, C,  c[1]';
	//                     c[0], c[1], 0]
	//A = Hessian of objective wrt. x (regularizer) 
	//C = Hessian of objective wrt. p (real objective)
	//B = Mixed partials of objective wrt. p and x
	StiffnessMatrix Btransposed = B.transpose();
	//StiffnessMatrix constraintJacobianTransposed = constraintJacobian.transpose();
	StiffnessMatrix dfdxT = dfdx.transpose();
	StiffnessMatrix dfdpT = dfdp.transpose();
	int nx = A.rows();
	int np = C.rows();
	int npnx = np + nx;

	int nl = dfdx.rows();
	//int nlx = dfdx.rows();
	//int nlp = dfdp.rows();
	//int nl = nlx + nlp

	//KKT.resize(npnx + nl, npnx + nl);
	KKT.resize(npnx + nl, npnx + nl);
	for (int k = 0; k < nx; k++)
	{
		KKT.startVec(k);
		for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
		{
			KKT.insertBack(it.row(), k) = it.value();
		}
		// for (StiffnessMatrix::InnerIterator it(B, k); it; ++it)
		// {
		// 	KKT.insertBack(it.row() + nx, k) = it.value();
		// }
		// for (StiffnessMatrix::InnerIterator it(dfdx, k); it; ++it)
		// {
		// 	KKT.insertBack(it.row() + npnx, k) = it.value();
		// }
	}
    
	// for (int k = 0; k < np; k++)
	// {
	// 	KKT.startVec(k + nx);
	// 	for (StiffnessMatrix::InnerIterator it(Btransposed, k); it; ++it)
	// 	{
	// 		KKT.insertBack(it.row(), k + nx) = it.value();
	// 	}
        
	// 	for (StiffnessMatrix::InnerIterator it(C, k); it; ++it)
	// 	{
	// 		KKT.insertBack(it.row() + nx, k + nx) = it.value();
	// 	}
	// 	//for (StiffnessMatrix::InnerIterator it(constraintJacobian, k + np); it; ++it)
	// 	//{
	// 	//	KKT.insertBack(it.row() + npnx, k + np) = it.value();
	// 	//}
        
	// 	for (StiffnessMatrix::InnerIterator it(dfdp, k); it; ++it)
	// 	{
	// 		KKT.insertBack(it.row() + npnx, k + nx) = it.value();
	// 	}
        
	// }
    
	// for (int k = 0; k < nl; k++)
	// {
	// 	KKT.startVec(k + npnx);
	// 	for (StiffnessMatrix::InnerIterator it(dfdxT, k); it; ++it)
	// 	{
	// 		KKT.insertBack(it.row(), k + npnx) = it.value();
	// 	}
		
	// 	//KKT.insertBack(npnx + k, npnx + k) = 1.0;
	// }
    // for (int k = 0; k < np; k++)
    // {
    //     KKT.startVec(k + npnx);
    //     for (StiffnessMatrix::InnerIterator it(dfdpT, k); it; ++it)
	// 	{
	// 		KKT.insertBack(it.row()+np, k + npnx) = it.value();
	// 	}
    // }
    
	KKT.finalize();

}


void Objectives::diffTestHessian()
{
    T epsilon = 1e-5;
    VectorXT p;
    getDesignParameters(p);
    StiffnessMatrix A;
    hessian(p, A);

    for(int dof_i = 0; dof_i < n_dof_design; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        p[dof_i] += epsilon;
        VectorXT g0(n_dof_design), g1(n_dof_design);
        g0.setZero(); g1.setZero();
        T E0;
        gradient(p, g0, E0);

        p[dof_i] -= 2.0 * epsilon;
        T E1;
        gradient(p, g1, E1);
        p[dof_i] += epsilon;
        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof_design; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Hessian Diff Test Passed" << std::endl;
}

void Objectives::diffTestHessianScale()
{
    VectorXT p;
    getDesignParameters(p);
    StiffnessMatrix A;
    hessian(p, A);
    
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    dp *= 0.001;

    VectorXT f0(n_dof_design);
    f0.setZero();
    T E0, E1;
    gradient(p, f0, E0);

    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof_design);
        f1.setZero();
        gradient(p + dp, f1, E1);

        T df_norm = (f0 + (A * dp) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dp *= 0.5;
    }
}

void Objectives::diffTestPartial2OPartialp2()
{
    VectorXT p;
    getDesignParameters(p);
    
    StiffnessMatrix A(n_dof_design, n_dof_design);
    std::vector<Entry> hessian_entries;
    computed2Odp2(p, hessian_entries);
    A.setFromTriplets(hessian_entries.begin(), hessian_entries.end());

    
    T epsilon = 1e-6;
    for(int dof_i = 0; dof_i < n_dof_design; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        p(dof_i) += epsilon;
        VectorXT g0(n_dof_design), g1(n_dof_design);
        g0.setZero(); g1.setZero();
        computedOdp(p, g0);

        p(dof_i) -= 2.0 * epsilon;
        computedOdp(p, g1);
        p(dof_i) += epsilon;
        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof_design; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Diff Test Passed" << std::endl;
}

void Objectives::diffTestPartial2OPartialp2Scale()
{
    VectorXT p;
    getDesignParameters(p);
    p.setConstant(-10);
    StiffnessMatrix A(n_dof_design, n_dof_design);
    std::vector<Entry> hessian_entries;
    computed2Odp2(p, hessian_entries);
    A.setFromTriplets(hessian_entries.begin(), hessian_entries.end());
    
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    
    VectorXT f0(n_dof_design);
    f0.setZero();
    T E0, E1;
    computedOdp(p, f0);

    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof_design);
        f1.setZero();
        computedOdp(p + dp, f1);

        T df_norm = (f0 + (A * dp) - f1).norm();
        // std::cout << "df_norm " << df_norm << std::endl;
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dp *= 0.5;
    }
}


void Objectives::diffTestPartialOPartialp()
{
    T epsilon = 1e-6;
    VectorXT dOdp(n_dof_design);
    dOdp.setZero();
    VectorXT p;
    getDesignParameters(p);
    
    p.setConstant(bound[1]);
    T _dummy;
    computedOdp(p, dOdp);

    for(int _dof_i = 0; _dof_i < n_dof_design; _dof_i++)
    {
        int dof_i = _dof_i;
        p[dof_i] += epsilon;
        T E1; computeOp(p, E1);
        p[dof_i] -= 2.0 * epsilon;
        T E0; computeOp(p, E0);
        p[dof_i] += epsilon;
        T fd = (E1 - E0) / (2.0 *epsilon);
        std::cout << "dof " << dof_i << " symbolic " << dOdp[dof_i] << " fd " << fd << std::endl;
        std::getchar();
    }
}

void Objectives::diffTestPartialOPartialpScale()
{
    std::cout << "###################### CHECK partial O partial p SCALE ######################" << std::endl; 
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    p.setConstant(-2);
    penalty_weight = 1;
    T E0;
    computeOp(p, E0);
    computedOdp(p, dOdp);
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    // std::cout << dp.minCoeff() << " " << dp.maxCoeff() << std::endl;
    // dp *= 0.001;
    T previous = 0.0;
    
    for (int i = 0; i < 10; i++)
    {
        // T E1 = value(p + dp, true, true);
        // VectorXT p1 = (p + dp).cwiseMax(bound[0]).cwiseMin(bound[1]);
        // dp = p1 - p;
        T E1;
        computeOp(p + dp, E1);
        T dE = E1 - E0;
        
        dE -= dOdp.dot(dp);
        // std::cout << "dE " << dE << std::endl;
        // std::cout << E1 << " " << E0 << std::endl;
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dp *= 0.5;
    }
}

void Objectives::diffTestGradient()
{
    T epsilon = 1e-4;
    VectorXT dOdp(n_dof_design);
    dOdp.setZero();
    VectorXT p;
    getDesignParameters(p);
    T _dummy;
    // simulation.newton_tol = 1e-9;
    simulation.staticSolve();
    VectorXT init = simulation.u;
    equilibrium_prev = init;
    // target_obj_weights.setZero();
    // w_reg_spacial = 0.0;
    gradient(p, dOdp, _dummy, true, true);

    for(int _dof_i = 0; _dof_i < n_dof_design; _dof_i++)
    {
        std::cout << "dof i " << _dof_i << std::endl;
        int dof_i = _dof_i;
        p[dof_i] += epsilon;
        // std::cout << "p_i+1: " << p[dof_i] << " ";
        equilibrium_prev = init;
        T E1 = value(p, true, true);
        // saveState("debug_p_i_plus_1.obj");
        p[dof_i] -= 2.0 * epsilon;
        // std::cout << "p_i-1: " << p[dof_i] << std::endl;
        equilibrium_prev = init;
        T E0 = value(p, true, true);
        // saveState("debug_p_i_minus_1.obj");
        p[dof_i] += epsilon;
        T fd = (E1 - E0) / (2.0 *epsilon);
        // if(dOdp[dof_i] == 0 && fd == 0)
        //     continue;
        // if (std::abs(dOdp[dof_i] - fd) < 1e-3 * std::abs(dOdp[dof_i]))
        //     continue;
        std::cout << "dof " << dof_i << " symbolic " << dOdp[dof_i] << " fd " << fd << std::endl;
        std::getchar();
    }
}

void Objectives::diffTestGradientScale()
{
    std::cout << "###################### CHECK GRADIENT SCALE ######################" << std::endl;   
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    T E0;
    // gradient(p, dOdp, E0, false);
    simulation.staticSolve();
    VectorXT init = simulation.u;
    equilibrium_prev = init;
    gradient(p, dOdp, E0, true, true);
    // std::cout << dOdp.minCoeff() << " " << dOdp.maxCoeff() << std::endl;
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    // dp *= 0.001;
    T previous = 0.0;
    
    for (int i = 0; i < 10; i++)
    {
        // T E1 = value(p + dp, true, true);
        // VectorXT p1 = (p + dp).cwiseMax(bound[0]).cwiseMin(bound[1]);
        // dp = p1 - p;
        equilibrium_prev = init;
        T E1 = value(p + dp, true, true);
        T dE = E1 - E0;
        
        dE -= dOdp.dot(dp);
        // std::cout << "dE " << dE << std::endl;
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dp *= 0.5;
    }
}

void Objectives::diffTestdOdxScale()
{
    std::cout << "###################### CHECK dOdx SCALE ######################" << std::endl;   
    VectorXT dOdx(n_dof_sim);
    VectorXT dx(n_dof_sim);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    VectorXT x = simulation.deformed;
    computedOdx(x, dOdx);
    T E0;
    computeOx(x, E0);
    for (int i = 0; i < 10; i++)
    {
        T E1;
        computeOx(x + dx, E1);
        T dE = E1 - E0;
        
        dE -= dOdx.dot(dx);
        
        if (i > 0)
        {
            // std::cout << "scale" << std::endl;
            std::cout << (previous/dE) << std::endl;
            // std::getchar();
        }
        previous = dE;
        dx *= 0.5;
    }
}

void Objectives::diffTestdOdx()
{
    std::cout << "###################### CHECK dOdx ENTRY ######################" << std::endl;   
    VectorXT dOdx(n_dof_sim);
    VectorXT dx(n_dof_sim);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;
    T previous = 0.0;
    VectorXT x = simulation.deformed;
    computedOdx(x, dOdx);
    
    VectorXT gradient_FD(n_dof_sim);
    gradient_FD.setZero();
    int cnt = 0;
    T epsilon = 1e-6;
    for(int dof_i = 0; dof_i < n_dof_sim; dof_i++)
    {
        x(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0; computeOx(x, E0);
        
        x(dof_i) -= 2.0 * epsilon;
        T E1; computeOx(x, E1);
        x(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E0 - E1) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && dOdx(dof_i) == 0)
            continue;
        if (std::abs( gradient_FD(dof_i) - dOdx(dof_i)) < 1e-3 * std::abs(dOdx(dof_i)))
            continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << dOdx(dof_i) << std::endl;
        // std::getchar();
        cnt++;   
    }
    std::cout << "FD test passed" << std::endl;
}



void Objectives::diffTestd2Odx2Scale()
{
    std::cout << "###################### CHECK d2Odx2 SCALE ######################" << std::endl; 
    running_diff_test = true;   
    VectorXT x = simulation.deformed;
    StiffnessMatrix A(x.rows(), x.rows());
    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(x, d2Odx2_entries);
    A.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    VectorXT dx(n_dof_sim);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;

    VectorXT f0(n_dof_sim);
    f0.setZero();
    computedOdx(x, f0);
    
    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof_sim);
        f1.setZero();
        computedOdx(x + dx, f1);
        
        T df_norm = (f0 + (A * dx) - f1).norm();
        if (i > 0)
        {
            std::cout << (previous/df_norm) << std::endl;
        }
        previous = df_norm;
        dx *= 0.5;
    }
    running_diff_test = false;
}

void Objectives::diffTestd2Odx2()
{
    std::cout << "###################### CHECK dO2dx2 ENTRY ######################" << std::endl; 
    running_diff_test = true;   
    VectorXT x = simulation.deformed;
    StiffnessMatrix A(x.rows(), x.rows());
    std::vector<Entry> d2Odx2_entries;
    computed2Odx2(x, d2Odx2_entries);
    A.setFromTriplets(d2Odx2_entries.begin(), d2Odx2_entries.end());

    VectorXT dx(n_dof_sim);
    dx.setRandom();
    dx *= 1.0 / dx.norm();
    dx *= 0.001;


    T epsilon = 1e-6;
    for(int dof_i = 0; dof_i < n_dof_sim; dof_i++)
    {
        // std::cout << dof_i << std::endl;
        x(dof_i) += epsilon;
        VectorXT g0(n_dof_sim), g1(n_dof_sim);
        g0.setZero(); g1.setZero();
        computedOdx(x, g0);

        x(dof_i) -= 2.0 * epsilon;
        computedOdx(x, g1);
        x(dof_i) += epsilon;
        VectorXT row_FD = (g0 - g1) / (2.0 * epsilon);

        for(int i = 0; i < n_dof_sim; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-3 * std::abs(row_FD(i)))
                continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    std::cout << "Diff Test Passed" << std::endl;
    running_diff_test = false;
}