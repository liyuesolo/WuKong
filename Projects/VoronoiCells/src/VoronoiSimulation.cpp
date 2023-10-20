#include "../include/VoronoiCells.h"
bool VoronoiCells::advanceOneStep(int step)
{
    T tol = 1e-6;
    int ls_max = 12;
    max_newton_iter = 500;
    if (step == 0)
    {
        dirichlet_data[0] = 0.0;
        dirichlet_data[1] = 0.0;
    }
    int n_dof = samples.size() * 2;
    VectorXT residual(n_dof); residual.setZero();
    T residual_norm = computeResidual(residual);

    std::cout << "[NEWTON] iter " << step << "/" 
        << max_newton_iter << ": residual_norm " 
        << residual_norm << " tol: " << newton_tol << std::endl;
    
    if (residual_norm < newton_tol || step == max_newton_iter)
    {
        return true;
    }

    T du_norm = lineSearchNewton(residual);
    if(step == max_newton_iter || du_norm > 1e10 || du_norm < 1e-8)
        return true;
    return false;
}

T VoronoiCells::computeTotalEnergy()
{
    T energy = 0.0;
    if (add_peri)
    {
        T perimeter_term = computePerimeterMinimizationEnergy(w_peri);
        energy += perimeter_term;
    }
    if (add_centroid)
    {
        T centroid_term = computeCentroidalVDEnergy(w_centroid);
        energy += centroid_term;
    }
    if (add_reg)
    {
        T reg_term = addRegEnergy(w_reg);
        energy += reg_term;
    }
    return energy;
}

T VoronoiCells::computeResidual(VectorXT& residual)
{
    if (add_peri)
        addPerimeterMinimizationForceEntries(residual, w_peri);
    if (add_centroid)
        addCentroidalVDForceEntries(residual, w_centroid);
    if (add_reg)
        addRegForceEntries(residual, w_reg);

    iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
    return residual.norm();
}

void VoronoiCells::buildSystemMatrix(StiffnessMatrix& K)
{
    std::vector<Entry> entries;

    if (add_peri)
        addCentroidalVDHessianEntries(entries, w_centroid);
    if (add_centroid)
        addCentroidalVDHessianEntries(entries, w_centroid);
    if (add_reg)
        addRegHessianEntries(entries, w_reg);
    int n_dof = samples.size() * 2;
    K.resize(n_dof, n_dof);
    
    K.setFromTriplets(entries.begin(), entries.end());
    
    projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

T VoronoiCells::lineSearchNewton(const VectorXT& residual)
{
    VectorXT search_direction = residual;
    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(K);
    bool success = linearSolve(K, residual, search_direction);
    T E0 = computeTotalEnergy();
    T alpha = 1.0;
    std::vector<SurfacePoint> samples_current = samples;
    for (int ls = 0; ls < ls_max; ls++)
    {
        samples = samples_current;
        tbb::parallel_for(0, (int)samples.size(), [&](int i)
        {
            updateSurfacePoint(samples[i], alpha * search_direction.segment<2>(i * 2));
        });
        
        constructVoronoiDiagram(true);
        T E1 = computeTotalEnergy();
        // std::cout << "E0 " << E0 << " E1 " << E1 << std::endl;
        // break;
        
        if (E1 < E0)
            break;
        alpha *= 0.5;
    }
    return alpha * search_direction.norm();
}