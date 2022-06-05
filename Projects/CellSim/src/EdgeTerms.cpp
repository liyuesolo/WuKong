#include "../include/VertexModel.h"

#include "../include/autodiff/EdgeEnergy.h"

void VertexModel::computeRestLength()
{
    rest_length.resize(edges.size());
    tbb::parallel_for(0, (int)edges.size(), [&](int edge_idx)
    {
        Edge e = edges[edge_idx];
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        rest_length[edge_idx] = (vi - vj).norm();
    });
}   

void VertexModel::addPerEdgeEnergy(T& energy)
{
    int cnt = 0;
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            T edge_length = computeEdgeSquaredNorm(vi, vj);
            energy += edge_weights[cnt++] * edge_length;
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if (apical || basal || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                energy += edge_weights[cnt++] * edge_length;    
            }
        }
    }
    
}

void VertexModel::addPerEdgeForceEntries(VectorXT& residual)
{
    int cnt = 0; 
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -edge_weights[cnt++];
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if (apical || basal || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -edge_weights[cnt++];
                addForceEntry<6>(residual, {e[0], e[1]}, dedx);
            }
        }
    }
}

void VertexModel::addPerEdgeHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    int cnt = 0;
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            computeEdgeSquaredNormHessian(vi, vj, hessian);
            hessian *= edge_weights[cnt++];
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if (apical || basal || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Matrix<T, 6, 6> hessian;
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= edge_weights[cnt++];
                addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
            }
        }
    }
        
}

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
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                T ei;
                computeEdgeEnergyRestLength3D(w, l0, x, ei);
                edge_length_term += ei;
            }
            else
            {
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                edge_length_term += w * edge_length;
            }

        });
    if (region == Lateral)
        iterateLateralEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                T ei;
                computeEdgeEnergyRestLength3D(w, l0, x, ei);
                edge_length_term += ei;
            }
            else
            {
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                edge_length_term += w * edge_length;
            }

        });
    else if (region == Basal)
        iterateBasalEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                T ei;
                computeEdgeEnergyRestLength3D(w, l0, x, ei);
                edge_length_term += ei;
            }
            else
            {
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                edge_length_term += w * edge_length;
            }

        });
    else if (region == ALL)
    {
        int cnt = 0;
        iterateEdgeSerial([&](Edge& e){    
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                T ei;
                computeEdgeEnergyRestLength3D(w * edge_weight_mask[cnt], l0, x, ei);
                edge_length_term += ei;
            }
            else
            {
                T edge_length = computeEdgeSquaredNorm(vi, vj);
                edge_length_term += w * edge_length;
            }
            cnt++;
        });
    }
        
    energy += edge_length_term;
}

void VertexModel::addEdgeForceEntries(Region region, T w, VectorXT& residual)
{
    if (region == Apical)
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DGradient(w, l0, x, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    else if (region == Basal)
        iterateBasalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DGradient(w, l0, x, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    else if (region == Lateral)
        iterateLateralEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DGradient(w, l0, x, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    else if (region == ALL)
    {
        int cnt = 0;
        iterateEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DGradient(w, l0, x, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<6>(residual, {e[0], e[1]}, dedx);
        });
    }
        
}

void VertexModel::addEdgeHessianEntries(Region region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    if (region == Apical)
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DHessian(w, l0, x, hessian);
            }
            else
            {
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= w;
            }
            // std::cout << hessian << std::endl;
            // std::cout << computeHessianBlockEigenValues<6>(hessian) << std::endl;
            // std::getchar();
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    else if (region == Basal)
        iterateBasalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DHessian(w, l0, x, hessian);
            }
            else
            {
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= w;
            }
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    else if (region == Lateral)
        iterateLateralEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DHessian(w, l0, x, hessian);
            }
            else
            {
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= w;
            }
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    else if (region == ALL)
    {
        int cnt = 0;
        iterateEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Matrix<T, 6, 6> hessian;
            if (has_rest_shape)
            {
                TV Xi = undeformed.segment<3>(e[0] * 3);
                TV Xj = undeformed.segment<3>(e[1] * 3);
                T l0 = (Xj - Xi).norm();
                Vector<T, 6> x;
                x << vi, vj;
                computeEdgeEnergyRestLength3DHessian(w * edge_weight_mask[cnt], l0, x, hessian);
            }
            else
            {
                computeEdgeSquaredNormHessian(vi, vj, hessian);
                hessian *= w;
            }
            cnt++;
            addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
        });
    }
        
}