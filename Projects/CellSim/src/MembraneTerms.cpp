#include "../include/VertexModel.h"
#include "../include/autodiff/MembraneEnergy.h"

#include "../include/SDF.h"

bool cubic = true;

void VertexModel::addMembraneSDFBoundEnergy(T& energy)
{
    T sdf_energy = 0.0;
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;
    VectorXT energies(n_vtx);
    energies.setZero();
    tbb::parallel_for(0, n_vtx, [&](int i){
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        if (cubic)
            energies[i] += bound_coeff * std::pow(sdf.value(xi), 3);
        else
            energies[i] += 0.5 * bound_coeff * std::pow(sdf.value(xi), 2);
    });
    // for (int i = 0; i < n_vtx; i++)
    // {
    //     TV xi = deformed.segment<3>(i * 3);
    //     if (sdf.inside(xi) && !run_diff_test)
    //         continue;
    //     if (cubic)
    //         sdf_energy += bound_coeff * std::pow(sdf.value(xi), 3);
    //     else
    //         sdf_energy += 0.5 * bound_coeff * std::pow(sdf.value(xi), 2);
        
    // }
    // energy += sdf_energy;
    energy += energies.sum();
}

void VertexModel::addMembraneSDFBoundForceEntries(VectorXT& residual)
{
    
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;
    int outside_cnt = 0;
    // for (int i = 0; i < n_vtx; i++)
    // {
    //     TV xi = deformed.segment<3>(i * 3);

    //     if (sdf.inside(xi) && !run_diff_test)
    //         continue;
    //     outside_cnt++;
    //     Vector<T, 3> dedx;
    //     sdf.gradient(xi, dedx);
    //     T value = sdf.value(xi);
    //     if (cubic)
    //         addForceEntry<3>(residual, {i}, -3.0 * bound_coeff * std::pow(value, 2) * dedx);
    //     else
    //         addForceEntry<3>(residual, {i}, -bound_coeff * value * dedx);
    //     // std::cout << "force norm " << dedx.norm() << std::endl;
    //     // std::getchar();
    // }
    tbb::parallel_for(0, n_vtx, [&](int i){
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        outside_cnt++;
        Vector<T, 3> dedx;
        sdf.gradient(xi, dedx);
        T value = sdf.value(xi);
        if (cubic)
            addForceEntry<3>(residual, {i}, -3.0 * bound_coeff * std::pow(value, 2) * dedx);
        else
            addForceEntry<3>(residual, {i}, -bound_coeff * value * dedx);
    });
    // std::cout << "[SDF force ]" << outside_cnt << "/" << n_vtx << " are outside the sdf" << std::endl;
}

void VertexModel::addMembraneSDFBoundHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;

    std::vector<Matrix<T, 3, 3>> sub_hessian(n_vtx, Matrix<T, 3, 3>::Zero());
    tbb::parallel_for(0, n_vtx, [&](int i){
        TV xi = deformed.segment<3>(i * 3);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        Matrix<T, 3, 3> d2phidx2;
        sdf.hessian(xi, d2phidx2);
        Vector<T, 3> dphidx;
        sdf.gradient(xi, dphidx);
        T value = sdf.value(xi);
        Matrix<T, 3, 3> hessian;
        if (cubic)       
            hessian = bound_coeff * (6.0 * value * dphidx * dphidx.transpose() + 3.0 * value * value * d2phidx2);
        else
            hessian = bound_coeff * (dphidx * dphidx.transpose() + value * d2phidx2);
        
        if (projectPD) 
            projectBlockPD<3>(hessian);
        sub_hessian[i] = hessian;
    });
    for (int i = 0; i < n_vtx; i++)
    {
        // if (lower_triangular)
        addHessianEntry<3>(entries, {i}, sub_hessian[i]);    
    }

    // for (int i = 0; i < n_vtx; i++)
    // {
    //     TV xi = deformed.segment<3>(i * 3);
    //     if (sdf.inside(xi) && !run_diff_test)
    //     // if (sdf.inside(xi))
    //         continue;
    //     Matrix<T, 3, 3> d2phidx2;
    //     sdf.hessian(xi, d2phidx2);
    //     Vector<T, 3> dphidx;
    //     sdf.gradient(xi, dphidx);
    //     T value = sdf.value(xi);
    //     Matrix<T, 3, 3> hessian;
    //     if (cubic)       
    //         hessian = bound_coeff * (6.0 * value * dphidx * dphidx.transpose() + 3.0 * value * value * d2phidx2);
    //     else
    //         hessian = bound_coeff * (dphidx * dphidx.transpose() + value * d2phidx2);
        
    //     if (projectPD) 
    //         projectBlockPD<3>(hessian);
    //     addHessianEntry<3>(entries, {i}, hessian);    
    // }
}

void VertexModel::addMembraneBoundEnergy(T& energy)
{
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;

    T sphere_bound_term = 0.0;
    for (int i = 0; i < n_vtx; i++)
    {    
        TV xi = deformed.segment<3>(i * 3);
        T e = 0.0;
        T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
        if (sphere_bound_penalty)
        {
            if (Rk >= Rc)
            {
                computeRadiusPenalty(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                sphere_bound_term += e;
            }
        }
        else if (sphere_bound_barrier)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            T d = Rc - Rk;
            if (d < membrane_dhat)
            {
                computeRadiusBarrier(bound_coeff, membrane_dhat, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
                sphere_bound_term += e;
            }
        }
        else
        {
            sphereBoundEnergy(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, e);
            // std::cout << e << " Rk " << Rk << " Rc " << Rc << std::endl;
            // std::getchar();
            sphere_bound_term += e;
        }
    }
    energy += sphere_bound_term;
}
void VertexModel::addMembraneBoundForceEntries(VectorXT& residual)
{
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;
    for (int i = 0; i < n_vtx; i++)
    {
        Vector<T, 3> dedx;
        if (sphere_bound_penalty)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            if (Rk >= Rc)
            {
                computeRadiusPenaltyGradient(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, dedx);
                addForceEntry<3>(residual, {i}, -dedx);
            }
        }
        else if (sphere_bound_barrier)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            T d = Rc - Rk;
            if (d < membrane_dhat)
            {
                computeRadiusBarrierGradient(bound_coeff, membrane_dhat, Rc, deformed.segment<3>(i * 3), mesh_centroid, dedx);
                addForceEntry<3>(residual, {i}, -dedx);
            }
        }
        else
        {
            sphereBoundEnergyGradient(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, dedx);
            // std::cout << dedx.transpose() << std::endl;
            addForceEntry<3>(residual, {i}, -dedx);
        }
    }
}

void VertexModel::addMembraneBoundHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;

    for (int i = 0; i < n_vtx; i++)
    {
        Matrix<T, 3, 3> hessian;
        if (sphere_bound_penalty)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            if (Rk >= Rc)
            {
                computeRadiusPenaltyHessian(bound_coeff, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);    
            }
        }
        else if (sphere_bound_barrier)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            T d = Rc - Rk;
            if (d < membrane_dhat)
            {
                computeRadiusBarrierHessian(bound_coeff, membrane_dhat, Rc, deformed.segment<3>(i * 3), mesh_centroid, hessian);
                addHessianEntry<3>(entries, {i}, hessian);
            }
        }
        else
        {
            sphereBoundEnergyHessian(bound_coeff, Rc, deformed.segment<3>(i*3), mesh_centroid, hessian);
            addHessianEntry<3>(entries, {i}, hessian);
        }
    }
}

T VertexModel::computeInsideMembraneStepSize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    int n_vtx = check_all_vtx_membrane ? num_nodes : basal_vtx_start;

    while (true)
    {
        deformed = undeformed + _u + step_size * du;
        bool constraint_violated = false;

        for (int i = 0; i < n_vtx; i++)
        {
            T Rk = (deformed.segment<3>(i * 3) - mesh_centroid).norm();
            T d = Rc - Rk;
            if (d < 1e-8)
            {
                constraint_violated = true;
                break;
            }
        }
        if (constraint_violated)
            step_size *= 0.8;
        else
            return step_size;
    }
    
}