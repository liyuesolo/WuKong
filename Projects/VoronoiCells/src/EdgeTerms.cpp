#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::addEdgeLengthEnergy(T w, T& energy)
{
    if (retrace)
        traceGeodesics();
    VectorXT energies = VectorXT::Zero(spring_edges.size());
    if (Euclidean)
    {
        for(int i = 0; i < spring_edges.size(); i++)
        {
            Edge eij = spring_edges[i];
            T l0 = rest_length[i];
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            TV xa = toTV(vA.interpolate(geometry->vertexPositions));
            TV xb = toTV(vB.interpolate(geometry->vertexPositions));
            T geo_dis = (xb - xa).norm();
            energies[i] = w * (geo_dis - l0) * (geo_dis-l0);
        }
    }
    else
    {
        for(int i = 0; i < spring_edges.size(); i++)
        {
            Edge eij = spring_edges[i];
            T l0 = rest_length[i];
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            std::vector<SurfacePoint> path = paths[i];
            // std::cout << path.size() << std::endl;
            T geo_dis = current_length[i];
            energies[i] = w * (geo_dis - l0) * (geo_dis-l0);
        }
    }
	
    // energies /= T(spring_edges.size());
    energy += energies.sum();
}


void IntrinsicSimulation::addEdgeLengthForceEntries(T w, VectorXT& residual)
{
    int n_springs = spring_edges.size();

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        T l = current_length[cnt];
        if (Euclidean)
        {
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            TV xa = toTV(vA.interpolate(geometry->vertexPositions));
            TV xb = toTV(vB.interpolate(geometry->vertexPositions));
            l = (xb - xa).norm();
        }
        T l0 = rest_length[cnt];
        T coeff = 2.0 * w * (l - l0);

        if (two_way_coupling)
        {
            VectorXT dldq;
            std::vector<int> dof_indices;
            computeGeodesicLengthGradientCoupled(eij, dldq, dof_indices);
            // dldq /= T(spring_edges.size());
            addForceEntry(residual, {eij[0], eij[1]}, -dldq.segment<4>(0) * coeff);
            
            addForceEntry<3>(residual, dof_indices, 
                -dldq.segment(4, dof_indices.size() * 3) * coeff, /*shift = */shell_dof_start);
        }
        else
        {
            Vector<T, 4> dldw;
            computeGeodesicLengthGradient(eij, dldw);
            // dldw /= T(spring_edges.size());
            addForceEntry(residual, {eij[0], eij[1]}, -dldw * coeff);
        }
        
        cnt++;
    }
    
}



void IntrinsicSimulation::addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries)
{
    int n_springs = spring_edges.size();

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        T l = current_length[cnt];
        if (Euclidean)
        {
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            TV xa = toTV(vA.interpolate(geometry->vertexPositions));
            TV xb = toTV(vB.interpolate(geometry->vertexPositions));
            l = (xb - xa).norm();
        }
        T l0 = rest_length[cnt];

        if (two_way_coupling)
        {
            VectorXT dldq; MatrixXT d2ldq2;
            std::vector<int> dof_indices;
            computeGeodesicLengthGradientAndHessianCoupled(eij, dldq, d2ldq2, dof_indices);            
            // d2ldq2 /= T(spring_edges.size());
            MatrixXT hessian = 
                2.0 * w * (dldq * dldq.transpose() + (l - l0) * d2ldq2);
            
            addHessianEntry(entries, {eij[0], eij[1]}, hessian.block(0, 0, 4, 4));
            
            addHessianEntry<3, 3>(entries, dof_indices, 
                hessian.block(4, 4, dof_indices.size() * 3, dof_indices.size() * 3), 
                shell_dof_start, shell_dof_start);

            addJacobianEntry<2, 3>(entries, {eij[0], eij[1]}, 
                dof_indices, 
                hessian.block(0, 4, 4, dof_indices.size() * 3), 
                0, shell_dof_start);
            
            addJacobianEntry<3, 2>(entries, dof_indices,
                {eij[0], eij[1]},  
                hessian.block(4, 0, dof_indices.size() * 3, 4), 
                shell_dof_start, 0);
        }
        else
        {
            Vector<T, 4> dldw; Matrix<T, 4, 4> d2ldw2;
            computeGeodesicLengthGradientAndHessian(eij, dldw, d2ldw2);
            // d2ldw2 /= T(spring_edges.size());
            Matrix<T, 4, 4> hessian = 
                2.0 * w * (dldw * dldw.transpose() + (l - l0) * d2ldw2);
            addHessianEntry(entries, {eij[0], eij[1]}, hessian);
        }
        
        cnt++;
    }
}
