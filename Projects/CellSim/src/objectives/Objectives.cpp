#include "../../include/Objectives.h"
#include <Eigen/PardisoSupport>

void ObjUMatching::setTargetFromMesh(const std::string& filename)
{
    simulation.loadDeformedState(filename);
    // target = 0.1 * simulation.u;
    target = simulation.u;
}

T ObjUMatching::value(const VectorXT& p_curr, bool use_prev_equil)
{
    simulation.reset();
    updateDesignParameters(p_curr);
    simulation.staticSolve();
    return 0.5 * (simulation.u - target).dot(simulation.u - target);
}

T ObjUMatching::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
{
    simulation.reset();
    
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = simulation.u - target;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
    return dOdp.norm();
}

T ObjUMatching::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
{
    simulation.reset();
    
    updateDesignParameters(p_curr);
    simulation.staticSolve();
    
    energy = 0.5 * (simulation.u - target).dot(simulation.u - target);
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = simulation.u - target;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    equilibrium_prev = simulation.u;
    return dOdp.norm();
}

T ObjUMatching::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
{
    updateDesignParameters(p_curr);
    energy = 0.5 * (simulation.u - target).dot(simulation.u - target);
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = simulation.u - target;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
    return dOdp.norm();
}

void ObjUMatching::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.cells.edge_weights = design_parameters;
}

void ObjUMatching::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.cells.edge_weights;
}

void ObjUMatching::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.cells.edge_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

void Objectives::setSimulationAndDesignDoF(int _sim_dof, int _design_dof)
{
    n_dof_design = _design_dof;
    n_dof_sim = _sim_dof;
}

T ObjUMatching::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil)
{
    simulation.reset();
    
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

    H = (dxdp.transpose() * dxdp).sparseView();
}

T ObjUTU::value(const VectorXT& p_curr, bool use_prev_equil)
{
    simulation.reset();
    if (use_prev_equil)
        simulation.u = equilibrium_prev;
    updateDesignParameters(p_curr);
    simulation.staticSolve();
    return -0.5 * simulation.u.dot(simulation.u);
}

T ObjUTU::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
{
    simulation.reset();
    if (use_prev_equil)
        simulation.u = equilibrium_prev;
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = -simulation.u;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    equilibrium_prev = simulation.u;
    return dOdp.norm();
}   

T ObjUTU::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
{
    updateDesignParameters(p_curr);
    energy = -0.5 * simulation.u.dot(simulation.u);
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = -simulation.u;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    
    return dOdp.norm();
}

T ObjUTU::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
{
    simulation.reset();
    if (use_prev_equil)
        simulation.u = equilibrium_prev;
    updateDesignParameters(p_curr);
    simulation.staticSolve();
    energy = -0.5 * simulation.u.dot(simulation.u);
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    if (simulation.woodbury)
    {
        MatrixXT UV;
        simulation.buildSystemMatrixWoodbury(simulation.u, d2edx2, UV);
    }
    else
    {   
        simulation.buildSystemMatrix(simulation.u, d2edx2);
    }
    
    VectorXT dOdu = -simulation.u;
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdu);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);
    equilibrium_prev = simulation.u;
    return dOdp.norm();
}

T ObjUTU::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil)
{
    simulation.reset();
    if (use_prev_equil)
        simulation.u = equilibrium_prev;
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

    H = (dxdp.transpose() * dxdp).sparseView();
    
}

void ObjUTU::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.cells.edge_weights;
}

void ObjUTU::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.cells.edge_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

void ObjUTU::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.cells.edge_weights = design_parameters;
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
        gradient(p, g0);

        p[dof_i] -= 2.0 * epsilon;
        gradient(p, g1);
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
    gradient(p, f0);

    T previous = 0.0;
    for (int i = 0; i < 10; i++)
    {
        
        VectorXT f1(n_dof_design);
        f1.setZero();
        gradient(p + dp, f1);

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

void Objectives::diffTestGradient()
{
    T epsilon = 1e-5;
    VectorXT dOdp(n_dof_design);
    dOdp.setZero();
    VectorXT p;
    getDesignParameters(p);
    updateDesignParameters(p);
    
    gradient(p, dOdp);
    for(int _dof_i = 0; _dof_i < n_dof_design; _dof_i++)
    {
        int dof_i = _dof_i;
        p[dof_i] += epsilon;
        T E1 = value(p);
        p[dof_i] -= 2.0 * epsilon;
        T E0 = value(p);
        p[dof_i] += epsilon;
        T fd = (E1 - E0) / (2.0 *epsilon);
        std::cout << "dof " << dof_i << " symbolic " << dOdp[dof_i] << " fd " << fd << std::endl;
        std::getchar();
    }
    
}

void Objectives::diffTestGradientScale()
{
    
    VectorXT dOdp(n_dof_design);
    VectorXT p;
    getDesignParameters(p);
    gradient(p, dOdp);
    VectorXT dp(n_dof_design);
    dp.setRandom();
    dp *= 1.0 / dp.norm();
    dp *= 0.001;
    T previous = 0.0;
    T E0 = value(p);
    for (int i = 0; i < 10; i++)
    {
        T E1 = value(p + dp);
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