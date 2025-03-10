#include "../include/VertexModel.h"
#include "../include/autodiff/AreaEnergy.h"



void VertexModel::addFaceContractionEnergy(T w, T& energy)
{
    iterateContractingFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        if (face_vtx_list.size() == 4)
        {
            if (use_face_centroid)
                computeArea4PointsSquaredSum(w, positions, area_energy);
            else
                computeQuadFaceAreaSquaredSum(w, positions, area_energy);
        }
        else if (face_vtx_list.size() == 5)
        {
            if (use_face_centroid)
                computeArea5PointsSquaredSum(w, positions, area_energy);
            else
                computePentFaceAreaSquaredSum(w, positions, area_energy);
        }
        else if (face_vtx_list.size() == 6)
        {
            if (use_face_centroid)
                computeArea6PointsSquaredSum(w, positions, area_energy);
            else
                computeHexFaceAreaSquaredSum(w, positions, area_energy);
        }
        else if (face_vtx_list.size() == 7)
        {
            if (use_face_centroid)
                computeArea7PointsSquaredSum(w, positions, area_energy);
        }
        else if (face_vtx_list.size() == 8)
        {
            if (use_face_centroid)
                computeArea8PointsSquaredSum(w, positions, area_energy);
        }
        else
            std::cout << "unknown polygon edge case" << std::endl;
        energy += area_energy;
    });
}

void VertexModel::addFaceContractionForceEntries(T w, VectorXT& residual)
{
    iterateContractingFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4)
        {
            Vector<T, 12> dedx;
            if (use_face_centroid)
                computeArea4PointsSquaredSumGradient(w, positions, dedx);
            else
                computeQuadFaceAreaSquaredSumGradient(w, positions, dedx);
            addForceEntry<12>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 5)
        {
            Vector<T, 15> dedx;
            if (use_face_centroid)
                computeArea5PointsSquaredSumGradient(w, positions, dedx);
            else
                computePentFaceAreaSquaredSumGradient(w, positions, dedx);
            addForceEntry<15>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 6)
        {
            Vector<T, 18> dedx;
            if (use_face_centroid)
                computeArea6PointsSquaredSumGradient(w, positions, dedx);
            else
                computeHexFaceAreaSquaredSumGradient(w, positions, dedx);
            addForceEntry<18>(residual, face_vtx_list, -dedx);
        }
        else if (face_vtx_list.size() == 7)
        {
            Vector<T, 21> dedx;
            if (use_face_centroid)
            {
                computeArea7PointsSquaredSumGradient(w, positions, dedx);
                addForceEntry<21>(residual, face_vtx_list, -dedx);
            }
        }
        else if (face_vtx_list.size() == 8)
        {
            Vector<T, 24> dedx;
            if (use_face_centroid)
            {
                computeArea8PointsSquaredSumGradient(w, positions, dedx);
                addForceEntry<24>(residual, face_vtx_list, -dedx);
            }
        }
        else
        {
            std::cout << "error " << __FILE__ << std::endl;
        }
    });
}

void VertexModel::addFaceContractionHessianEntries(T w, std::vector<Entry>& entries, bool projectPD)
{
    iterateContractingFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        if (face_vtx_list.size() == 4)
        {
            Matrix<T, 12, 12> hessian;
            if (use_face_centroid)
                computeArea4PointsSquaredSumHessian(w, positions, hessian);
            else
                computeQuadFaceAreaSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<12>(hessian);
            addHessianEntry<12>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 5)
        {
            Matrix<T, 15, 15> hessian;
            if (use_face_centroid)
                computeArea5PointsSquaredSumHessian(w, positions, hessian);
            else
                computePentFaceAreaSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<15>(hessian);
            addHessianEntry<15>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 6)
        {
            Matrix<T, 18, 18> hessian;
            if (use_face_centroid)
                computeArea6PointsSquaredSumHessian(w, positions, hessian);
            else
                computeHexFaceAreaSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<18>(hessian);
            addHessianEntry<18>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 7)
        {
            Matrix<T, 21, 21> hessian;
            if (use_face_centroid)
                computeArea7PointsSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<21>(hessian);
            addHessianEntry<21>(entries, face_vtx_list, hessian);
        }
        else if (face_vtx_list.size() == 8)
        {
            Matrix<T, 24, 24> hessian;
            if (use_face_centroid)
                computeArea8PointsSquaredSumHessian(w, positions, hessian);
            if (projectPD) 
                projectBlockPD<24>(hessian);
            addHessianEntry<24>(entries, face_vtx_list, hessian);
        }
        else
        {
            std::cout << "unknown " << std::endl;
        }
    });
}

void VertexModel::addFaceAreaEnergyWithRestShape(Region region, T w, T& energy)
{
    VectorXT energies = VectorXT::Zero(faces.size());
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(region, face_idx))
        {
            VectorXT positions, positions_undeformed;
            positionsFromIndices(positions, face_vtx_list);
            positionsFromIndices(positions_undeformed, face_vtx_list, true);

            if (use_face_centroid)
            {
                if (face_vtx_list.size() == 4)
                    computeArea4PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else if (face_vtx_list.size() == 5)
                    computeArea5PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else if (face_vtx_list.size() == 6)
                    computeArea6PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else if (face_vtx_list.size() == 7)
                    computeArea7PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else if (face_vtx_list.size() == 8)
                    computeArea8PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else if (face_vtx_list.size() == 9)
                    computeArea9PointsPenaltyWithRestShape(w, positions, positions_undeformed, energies[face_idx]);
                else
                    std::cout << "unknown polygon edge case" << std::endl;
            }
            else
            {

            }
        }
    });
    energy += energies.sum();
}

void VertexModel::addFaceAreaForceEntriesWithRestShape(Region region, T w, VectorXT& residual)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(region, face_idx))
        {
            VectorXT positions, positions_undeformed;
            positionsFromIndices(positions, face_vtx_list);
            positionsFromIndices(positions_undeformed, face_vtx_list, true);

            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                computeArea4PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                addForceEntry<12>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                computeArea5PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                addForceEntry<15>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                computeArea6PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                addForceEntry<18>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 7)
            {
                Vector<T, 21> dedx;
                computeArea7PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                addForceEntry<21>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 8)
            {
                
                Vector<T, 24> dedx;
                if (use_face_centroid)
                {
                    computeArea8PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                    addForceEntry<24>(residual, face_vtx_list, -dedx);
                }
            }
            else if (face_vtx_list.size() == 9)
            {
                Vector<T, 27> dedx;
                if (use_face_centroid)
                {
                    computeArea9PointsPenaltyWithRestShapeGradient(w, positions, positions_undeformed, dedx);
                    addForceEntry<27>(residual, face_vtx_list, -dedx);
                }
            }
            else
            {
                std::cout << "error " << __FILE__ << std::endl;
            }
        }
    });
}

void VertexModel::addFaceAreaHessianEntriesWithRestShape(Region region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(region, face_idx))
        {
            VectorXT positions, positions_undeformed;
            positionsFromIndices(positions, face_vtx_list);
            positionsFromIndices(positions_undeformed, face_vtx_list, true);

            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                computeArea4PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                computeArea5PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                computeArea6PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 7)
            {
                Matrix<T, 21, 21> hessian;
                computeArea7PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<21>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 8)
            {
                Matrix<T, 24, 24> hessian;
                computeArea8PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<24>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 9)
            {
                Matrix<T, 27, 27> hessian;
                computeArea9PointsPenaltyWithRestShapeHessian(w, positions, positions_undeformed, hessian);
                addHessianEntry<27>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
            }
        }
    });
}

void VertexModel::addFaceAreaEnergy(Region face_region, T w, T& energy)
{
    VectorXT energies = VectorXT::Zero(faces.size());
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(face_region, face_idx))
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (use_face_centroid)
            {
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
            }
            else
            {
                if (face_vtx_list.size() == 4)
                    computeQuadFaceAreaSquaredSum(w, positions, energies[face_idx]);
                else if (face_vtx_list.size() == 5)
                    computePentFaceAreaSquaredSum(w, positions, energies[face_idx]);
                else if (face_vtx_list.size() == 6)
                    computeHexFaceAreaSquaredSum(w, positions, energies[face_idx]);
                else if (face_vtx_list.size() == 7)
                    computeSepFaceAreaSquaredSum(w, positions, energies[face_idx]);
                else if (face_vtx_list.size() == 8)
                    computeOctFaceAreaSquaredSum(w, positions, energies[face_idx]);
                else
                    std::cout << "unknown polygon edge case" << std::endl;
            }
        }
    });
    energy += energies.sum();
}

void VertexModel::addFaceAreaForceEntries(Region face_region, T w, VectorXT& residual)
{
    
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(face_region, face_idx))
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Vector<T, 12> dedx;
                if (use_face_centroid)
                    computeArea4PointsSquaredSumGradient(w, positions, dedx);
                else
                    computeQuadFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<12>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                if (use_face_centroid)
                    computeArea5PointsSquaredSumGradient(w, positions, dedx);
                else
                    computePentFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<15>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                if (use_face_centroid)
                    computeArea6PointsSquaredSumGradient(w, positions, dedx);
                else
                    computeHexFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<18>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 7)
            {
                Vector<T, 21> dedx;
                if (use_face_centroid)
                    computeArea7PointsSquaredSumGradient(w, positions, dedx);
                else
                    computeSepFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<21>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 8)
            {
                
                Vector<T, 24> dedx;
                if (use_face_centroid)
                    computeArea8PointsSquaredSumGradient(w, positions, dedx);
                else
                    computeOctFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<24>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 9)
            {
                Vector<T, 27> dedx;
                if (use_face_centroid)
                {
                    computeArea9PointsSquaredSumGradient(w, positions, dedx);
                    addForceEntry<27>(residual, face_vtx_list, -dedx);
                }
            }
            else
            {
                std::cout << "error " << __FILE__ << std::endl;
            }
        }
    });
}
void VertexModel::addFaceAreaHessianEntries(Region face_region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(face_region, face_idx))
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            if (face_vtx_list.size() == 4)
            {
                Matrix<T, 12, 12> hessian;
                if (use_face_centroid)
                    computeArea4PointsSquaredSumHessian(w, positions, hessian);
                else
                    computeQuadFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<12>(hessian);
                addHessianEntry<12>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 5)
            {
                Matrix<T, 15, 15> hessian;
                if (use_face_centroid)
                    computeArea5PointsSquaredSumHessian(w, positions, hessian);
                else
                    computePentFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<15>(hessian);
                addHessianEntry<15>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 6)
            {
                Matrix<T, 18, 18> hessian;
                if (use_face_centroid)
                    computeArea6PointsSquaredSumHessian(w, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<18>(hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 7)
            {
                Matrix<T, 21, 21> hessian;
                if (use_face_centroid)
                    computeArea7PointsSquaredSumHessian(w, positions, hessian);
                else
                    computeSepFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<21>(hessian);
                addHessianEntry<21>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 8)
            {
                Matrix<T, 24, 24> hessian;
                if (use_face_centroid)
                    computeArea8PointsSquaredSumHessian(w, positions, hessian);
                else
                    computeOctFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<24>(hessian);
                addHessianEntry<24>(entries, face_vtx_list, hessian);
            }
            else if (face_vtx_list.size() == 9)
            {
                Matrix<T, 27, 27> hessian;
                if (use_face_centroid)
                    computeArea9PointsSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<27>(hessian);
                addHessianEntry<27>(entries, face_vtx_list, hessian);
            }
            else
            {
                // std::cout << "unknown " << std::endl;
            }
        }
    });
}



T VertexModel::computeAreaEnergy(const VectorXT& _u)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    T energy = 0.0;
    use_face_centroid = true;
    // std::cout << lateral_face_start << std::endl;
    // std::cout << faces.size() << std::endl;
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        T area_energy = 0.0;
        // cell-wise volume preservation term
        
        // if (face_idx >= basal_face_start + 43)
        // if (face_idx >= lateral_face_start)
        if (face_idx == 196)
        {
            // std::ofstream out("bug_face.obj");
            // for (int i = 0; i < face_vtx_list.size(); i++)
            // {
            //     out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
            // }
            // TV face_centroid = TV::Zero();
            // computeFaceCentroid(face_vtx_list, face_centroid);
            // out << "v " << face_centroid.transpose() << std::endl;
            
            // out << "f 5 1 2 " << std::endl;
            // out << "f 5 2 3 " << std::endl;
            // out << "f 5 3 4 " << std::endl;
            // out << "f 5 4 1 " << std::endl;
            // out.close();
            // std::getchar();
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            if (face_vtx_list.size() == 4)
            {
                // computeArea4PointsSquaredSum(coeff, positions, area_energy);
                if (use_face_centroid)
                {
                    computeArea4PointsSquared(coeff, positions, area_energy);
                }
                else
                    computeQuadFaceAreaSquaredSum(coeff, positions, area_energy);
                // computeArea4PointsSquared(coeff, positions, area_energy);
                // computeQuadFaceArea(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(coeff, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(coeff, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(coeff, positions, area_energy);
            }
            // else
            //     std::cout << "unknown polygon edge case" << std::endl;
        }
        energy += area_energy;
    });
    return energy;
}