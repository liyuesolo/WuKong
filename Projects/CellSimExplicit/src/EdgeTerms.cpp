#include "../include/CellSim.h"

#include "../include/autodiff/EdgeEnergy.h"

void CellSim::computeRestLength()
{
    rest_length.resize(cell_edges.size());
    tbb::parallel_for(0, (int)cell_edges.size(), [&](int edge_idx)
    {
        Edge e = cell_edges[edge_idx];
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        rest_length[edge_idx] = (vi - vj).norm();
    });
}   

void CellSim::addEdgeEnergy(Region region, T w, T& energy)
{
    T edge_length_term = 0.0;
    
    iterateEdgeSerial([&](Edge& e, int idx){    
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        TV Xi = undeformed.segment<3>(e[0] * 3);
        TV Xj = undeformed.segment<3>(e[1] * 3);
        T l0 = (Xj - Xi).norm();
        Vector<T, 6> x;
        x << vi, vj;
        T ei;
        computeEdgeEnergyRestLength3D(w, l0, x, ei);
        edge_length_term += ei;
    });
        
    energy += edge_length_term;
}

void CellSim::addEdgeForceEntries(Region region, T w, VectorXT& residual)
{
    iterateEdgeSerial([&](Edge& e, int idx){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        TV Xi = undeformed.segment<3>(e[0] * 3);
        TV Xj = undeformed.segment<3>(e[1] * 3);
        T l0 = (Xj - Xi).norm();
        Vector<T, 6> x;
        x << vi, vj;
        computeEdgeEnergyRestLength3DGradient(w, l0, x, dedx);
        dedx *= -1.0;
        addForceEntry<6>(residual, {e[0], e[1]}, dedx);
    });
        
}

void CellSim::addEdgeHessianEntries(Region region, T w, 
    std::vector<Entry>& entries, bool projectPD)
{
    iterateEdgeSerial([&](Edge& e, int idx){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        TV Xi = undeformed.segment<3>(e[0] * 3);
        TV Xj = undeformed.segment<3>(e[1] * 3);
        T l0 = (Xj - Xi).norm();
        Vector<T, 6> x;
        x << vi, vj;
        computeEdgeEnergyRestLength3DHessian(w, l0, x, hessian);
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });
}