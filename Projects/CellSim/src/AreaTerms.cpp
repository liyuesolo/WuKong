#include "../include/VertexModel.h"
#include "../include/autodiff/AreaEnergy.h"


void VertexModel::addFaceAreaEnergy(Region face_region, T w, T& energy)
{
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        if (validFaceIdx(face_region, face_idx))
        {
            VectorXT positions;
            positionsFromIndices(positions, face_vtx_list);
            T area_energy = 0.0;
            if (face_vtx_list.size() == 4)
            {
                if (use_face_centroid)
                //this is sum squared
                    computeArea4PointsSquared(w, positions, area_energy);
                else
                    computeQuadFaceAreaSquaredSum(w, positions, area_energy);
            }
            else if (face_vtx_list.size() == 5)
            {
                if (use_face_centroid)
                    computeArea5PointsSquared(w, positions, area_energy);
                else
                    computePentFaceAreaSquaredSum(w, positions, area_energy);
            }
            else if (face_vtx_list.size() == 6)
            {
                if (use_face_centroid)
                    computeArea6PointsSquared(w, positions, area_energy);
                else
                    computeHexFaceAreaSquaredSum(w, positions, area_energy);
            }
            else
                std::cout << "unknown polygon edge case" << std::endl;
            energy += area_energy;
        }
    });
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
                    computeArea4PointsSquaredGradient(w, positions, dedx);
                else
                    computeQuadFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<12>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 5)
            {
                Vector<T, 15> dedx;
                if (use_face_centroid)
                    computeArea5PointsSquaredGradient(w, positions, dedx);
                else
                    computePentFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<15>(residual, face_vtx_list, -dedx);
            }
            else if (face_vtx_list.size() == 6)
            {
                Vector<T, 18> dedx;
                if (use_face_centroid)
                    computeArea6PointsSquaredGradient(w, positions, dedx);
                else
                    computeHexFaceAreaSquaredSumGradient(w, positions, dedx);
                addForceEntry<18>(residual, face_vtx_list, -dedx);
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
                    computeArea4PointsSquaredHessian(w, positions, hessian);
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
                    computeArea5PointsSquaredHessian(w, positions, hessian);
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
                    computeArea6PointsSquaredHessian(w, positions, hessian);
                else
                    computeHexFaceAreaSquaredSumHessian(w, positions, hessian);
                if (projectPD) 
                    projectBlockPD<18>(hessian);
                addHessianEntry<18>(entries, face_vtx_list, hessian);
            }
            else
            {
                std::cout << "unknown " << std::endl;
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
            else
                std::cout << "unknown polygon edge case" << std::endl;
        }
        energy += area_energy;
    });
    return energy;
}