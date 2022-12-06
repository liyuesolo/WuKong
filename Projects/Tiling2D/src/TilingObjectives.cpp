#include "../include/TilingObjectives.h"


T UniaxialStressObjective::generateSingleTarget(const VectorXT& ti)
{
    std::string result_folder = "./";
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = ti[j];
    }
    std::vector<std::vector<TV>> polygons;
    std::vector<TV> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    tiling.fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    tiling.generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");

    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    tiling.initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    solver.verbose = false;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    solver.pbc_strain_w = 1e6;
    solver.project_block_PD = false;
    solver.strain_theta = theta;
    solver.uniaxial_strain = strain;
    bool solve_succeed = solver.staticSolve();
    TM secondPK_stress, Green_strain;
    T psi;
    solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
    TV d(std::cos(theta), std::sin(theta));
    T stress_d = d.transpose() * (secondPK_stress * d);
    return stress_d;
}

void UniaxialStressObjective::computeStressForDifferentStrain(const VectorXT& ti, 
    VectorXT& stress)
{
    std::cout << "ti " << ti.transpose() << std::endl;
    stress.resize(strain_samples.rows());
    std::string result_folder = "./";
    csk::IsohedralTiling a_tiling( csk::tiling_types[ IH ] );
    int num_params = a_tiling.numParameters();
    T new_params[ num_params ];
    a_tiling.getParameters( new_params );
    std::vector<T> params(num_params);
    std::vector<TV> params_range(num_params);
    for (int j = 0; j < num_params;j ++)
    {
        params[j] = ti[j];
    }
    std::vector<std::vector<TV>> polygons;
    std::vector<TV> pbc_corners; 
    Vector<T, 4> cubic_weights;
    cubic_weights << 0.25, 0, 0.75, 0;
    tiling.fetchUnitCellFromOneFamily(IH, 2, polygons, pbc_corners, params, 
        cubic_weights, result_folder + "structure.txt");
    
    tiling.generatePeriodicMesh(polygons, pbc_corners, true, result_folder + "structure");

    solver.pbc_translation_file = result_folder + "structure_translation.txt";
    bool valid_structure = tiling.initializeSimulationDataFromFiles(result_folder + "structure.vtk", PBC_XY);
    if (!valid_structure)
    {
        stress.setConstant(1e10);
        return;
    }
    solver.verbose = false;
    solver.prescribe_strain_tensor = false;
    solver.biaxial = false;
    solver.pbc_strain_w = 1e6;
    solver.project_block_PD = false;
    solver.strain_theta = theta;
    TV d(std::cos(theta), std::sin(theta));
    
    solver.reset();
    for (int i = 0; i < strain_samples.rows(); i++)
    {
        solver.uniaxial_strain = strain_samples[i];
        bool solve_succeed = solver.staticSolve();
        solver.saveToOBJ("./tmp/" + std::to_string(ti[0]) +"_"+ std::to_string(ti[1]) + "_" + std::to_string(i) + "_rest.obj", true);
        TM secondPK_stress, Green_strain;
        T psi;
        solver.computeHomogenizationData(secondPK_stress, Green_strain, psi);
        stress[i] = d.transpose() * (secondPK_stress * d);
    }
}

T UniaxialStressObjective::value(const VectorXT& p_curr, 
    bool simulate, bool use_prev_equil)
{
    // T Si = generateSingleTarget(p_curr);
    VectorXT stress_current;
    computeStressForDifferentStrain(p_curr, stress_current);
    if (stress_current.norm() > 1e10)
        return -1.0;
    T obj = 0.5 * (stress_current - targets).dot(stress_current-targets);
    return obj;
}

T UniaxialStressObjective::gradient(const VectorXT& p_curr, 
    VectorXT& dOdp, T& energy, bool simulate, bool use_prev_equil)
{
    // T Si = generateSingleTarget(p_curr);
    
    energy = value(p_curr);
    
    dOdp.resize(p_curr.rows()); dOdp.setZero();
    T epsilon = 1e-5;
    for (int i = 0; i < p_curr.rows(); i++)
    {
        VectorXT p_forward = p_curr;
        p_forward[i] += epsilon;
        T E1 = value(p_forward);
        p_forward[i] -= 2.0 * epsilon;
        T E0 = value(p_forward);
        dOdp[i] = (E1 - E0) / 2.0 / epsilon;
    }
    
    return dOdp.norm();
}
