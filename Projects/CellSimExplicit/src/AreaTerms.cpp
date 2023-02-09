#include "../include/CellSim.h"
#include "../include/autodiff/AreaEnergy.h"


void CellSim::addFaceAreaEnergy(Region face_region, T w, T& energy)
{
    VectorXT energies = VectorXT::Zero(faces.size());
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4)
            computeArea4PointsSquaredSum(w, positions, energies[face_idx]);
        else if (face_vtx_list.size() == 5)
            computeArea5PointsSquaredSum(w, positions, energies[face_idx]);
        else if (face_vtx_list.size() == 6)
            computeArea6PointsSquaredSum(w, positions, energies[face_idx]);
        else if (face_vtx_list.size() == 7)
            computeArea7PointsSquaredSum(w, positions, energies[face_idx]);
        else if (face_vtx_list.size() == 8)
            computeArea8PointsSquaredSum(w, positions, energies[face_idx]);
        else if (face_vtx_list.size() == 9)
            computeArea9PointsSquaredSum(w, positions, energies[face_idx]);
        else
            std::cout << "unknown polygon edge case" << std::endl;
    });
    energy += energies.sum();
}

void CellSim::addFaceAreaForceEntries(Region face_region, T w, VectorXT& residual)
{
    
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4)
        {
            Vector<T, 12> dedx;
            computeArea4PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<12>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 5)
        {
            Vector<T, 15> dedx;
            computeArea5PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<15>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 6)
        {
            Vector<T, 18> dedx;
            computeArea6PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<18>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 7)
        {
            Vector<T, 21> dedx;
            computeArea7PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<21>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 8)
        {
            
            Vector<T, 24> dedx;
            computeArea8PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<24>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 9)
        {
            Vector<T, 27> dedx;
            computeArea9PointsSquaredSumGradient(w, positions, dedx);
            addForceEntry<27>(residual, face_vtx_list, -dedx);
            
        }
        else
        {
            std::cout << "error " << __FILE__ << std::endl;
        }
    });
}
void CellSim::addFaceAreaHessianEntries(Region face_region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4)
        {
            Matrix<T, 12, 12> hessian;
            computeArea4PointsSquaredSumHessian(w, positions, hessian);
            
            if (projectPD) 
                projectBlockPD<12>(hessian);
            addHessianEntry<12>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 5)
        {
            Matrix<T, 15, 15> hessian;
            computeArea5PointsSquaredSumHessian(w, positions, hessian);
            
            if (projectPD) 
                projectBlockPD<15>(hessian);
            addHessianEntry<15>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 6)
        {
            Matrix<T, 18, 18> hessian;
            computeArea6PointsSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<18>(hessian);
            addHessianEntry<18>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 7)
        {
            Matrix<T, 21, 21> hessian;
            computeArea7PointsSquaredSumHessian(w, positions, hessian);
            
            if (projectPD) 
                projectBlockPD<21>(hessian);
            addHessianEntry<21>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 8)
        {
            Matrix<T, 24, 24> hessian;
            computeArea8PointsSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<24>(hessian);
            addHessianEntry<24>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 9)
        {
            Matrix<T, 27, 27> hessian;
            computeArea9PointsSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<27>(hessian);
            addHessianEntry<27>(entries, face_vtx_list, hessian);
        }
        else
        {
            // std::cout << "unknown " << std::endl;
        }
    });
}

