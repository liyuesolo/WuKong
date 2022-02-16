#include "../include/Objectives.h"
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

void ObjUMatching::setSimulationAndDesignDoF(int _sim_dof, int _design_dof)
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

void ObjUTU::setSimulationAndDesignDoF(int _sim_dof, int _design_dof)
{
    n_dof_design = _design_dof;
    n_dof_sim = _sim_dof;
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