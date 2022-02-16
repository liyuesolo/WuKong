#include "../include/Objectives.h"
#include <Eigen/PardisoSupport>

void ObjNucleiTracking::setSimulationAndDesignDoF(int _sim_dof, int _design_dof)
{
    n_dof_design = _design_dof;
    n_dof_sim = _sim_dof;
}

void ObjNucleiTracking::initializeTarget()
{
    std::vector<int> VF_cell_idx;
    simulation.cells.getVFCellIds(VF_cell_idx);
    // std::cout << "VF cells " << VF_cell_idx.size() << std::endl;
    for (int idx : VF_cell_idx)
    {
        TV cell_centroid;
        VtxList face_vtx_list = simulation.cells.faces[idx];
        simulation.cells.computeCellCentroid(face_vtx_list, cell_centroid);
        target_positions[idx] = cell_centroid;
    }
}

void ObjNucleiTracking::loadTarget(const std::string& filename)
{
    std::ifstream in(filename);
    int idx; T x, y, z;
    while(in >> idx >> x >> y >> z)
        target_positions[idx] = TV(x, y, z);
    in.close();
}


T ObjNucleiTracking::value(const VectorXT& p_curr, bool use_prev_equil)
{
    simulation.reset();
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    T energy = 0.0;
    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
    });
    
    return energy;
}

T ObjNucleiTracking::gradient(const VectorXT& p_curr, VectorXT& dOdp, bool use_prev_equil)
{

}

T ObjNucleiTracking::gradient(const VectorXT& p_curr, VectorXT& dOdp, T& energy, bool use_prev_equil)
{
    simulation.reset();
    updateDesignParameters(p_curr);
    simulation.staticSolve();
    
    energy = 0.0;
    VectorXT dOdx(n_dof_sim);
    dOdx.setZero();

    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
        T coeff = cell_vtx_list.size();
        for (int idx : cell_vtx_list)
        {
            dOdx.segment<3>(idx * 3) += (centroid - target_pos) / coeff;        
        }
    });
    
    
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

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdx);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);

    return dOdp.norm();
}

T ObjNucleiTracking::evaluteGradientAndEnergy(const VectorXT& p_curr, VectorXT& dOdp, T& energy)
{
    updateDesignParameters(p_curr);

    energy = 0.0;
    VectorXT dOdx(n_dof_sim);
    dOdx.setZero();

    iterateTargets([&](int cell_idx, TV& target_pos)
    {
        VtxList face_vtx_list = simulation.cells.faces[cell_idx];
        VtxList cell_vtx_list = face_vtx_list;
        for (int idx : face_vtx_list)
            cell_vtx_list.push_back(idx + simulation.cells.basal_vtx_start);
        // VectorXT positions;
        // simulation.cells.positionsFromIndices(positions, cell_vtx_list);
        TV centroid;
        simulation.cells.computeCellCentroid(face_vtx_list, centroid);
        energy += 0.5 * (centroid - target_pos).dot(centroid - target_pos);
        T coeff = cell_vtx_list.size();
        for (int idx : cell_vtx_list)
            dOdx.segment<3>(idx * 3) += (centroid - target_pos) / coeff;        
    });

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

    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    
    // here d2e/dx2 is -df/dx, negative sign is cancelled with -df/dp
    VectorXT lambda = solver.solve(dOdx);
    
    simulation.cells.dOdpEdgeWeightsFromLambda(lambda, dOdp);

    return dOdp.norm();
}

void ObjNucleiTracking::updateDesignParameters(const VectorXT& design_parameters)
{
    simulation.cells.edge_weights = design_parameters;
}

void ObjNucleiTracking::getDesignParameters(VectorXT& design_parameters)
{
    design_parameters = simulation.cells.edge_weights;
}

void ObjNucleiTracking::getSimulationAndDesignDoF(int& _sim_dof, int& _design_dof)
{
    _sim_dof = simulation.num_nodes * 3;
    _design_dof = simulation.cells.edge_weights.rows();
    n_dof_sim = _sim_dof;
    n_dof_design = _design_dof;
}

T ObjNucleiTracking::hessianGN(const VectorXT& p_curr, StiffnessMatrix& H, bool use_prev_equil)
{
    simulation.reset();
    
    updateDesignParameters(p_curr);
    simulation.staticSolve();

    MatrixXT dxdp;
    simulation.cells.dxdpFromdxdpEdgeWeights(dxdp);

    H = (dxdp.transpose() * dxdp).sparseView();
}