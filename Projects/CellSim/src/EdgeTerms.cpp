#include "../include/VertexModel.h"

#include "../include/autodiff/EdgeEnergy.h"

void VertexModel::addEdgeContractionEnergy(T w, T& energy)
{
    iterateContractingEdgeSerial([&](Edge& e){    
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeSquaredNorm(vi, vj);
        energy += w * edge_length;
    });
}

void VertexModel::addEdgeContractionForceEntries(T w, VectorXT& residual)
{
    iterateContractingEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -w;
        addForceEntry<6>(residual, {e[0], e[1]}, dedx);
    }); 
}

void VertexModel::addEdgeContractionHessianEntries(T w, std::vector<Entry>& entries, bool projectPD)
{
    iterateContractingEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeSquaredNormHessian(vi, vj, hessian);
        hessian *= w;
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });
}

void VertexModel::addEdgeEnergy(Region region, T w, T& energy)
{
    T edge_length_term = 0.0;
    if (region == Apical)
        iterateApicalEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            T edge_length = computeEdgeSquaredNorm(vi, vj);
            edge_length_term += w * edge_length;

        });
    else if (region == ALL)
        iterateEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            T edge_length = computeEdgeSquaredNorm(vi, vj);
            edge_length_term += w * edge_length;

        });
    energy += edge_length_term;
}

void VertexModel::addEdgeForceEntries(Region region, T w, VectorXT& residual)
{
    if (region == Apical)
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -w;
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    else if (region == ALL)
        iterateEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -w;
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
}

void VertexModel::addEdgeHessianEntries(Region region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    if (region == Apical)
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= w;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    else if (region == ALL)
        iterateEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= w;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
}