#include "../include/VertexModel.h"
#include "../include/autodiff/MembraneEnergy.h"

void VertexModel::addMembraneBoundEnergy(T& energy)
{
    T sphere_bound_term = 0.0;
    for (int i = 0; i < basal_vtx_start; i++)
    {
        T e = 0.0;;
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
    for (int i = 0; i < basal_vtx_start; i++)
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
    for (int i = 0; i < basal_vtx_start; i++)
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
    while (true)
    {
        deformed = undeformed + _u + step_size * du;
        bool constraint_violated = false;
        for (int i = 0; i < basal_vtx_start; i++)
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
            step_size *= 0.5;
        else
            return step_size;
    }
    
}