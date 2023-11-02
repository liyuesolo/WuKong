#include "../include/VoronoiCells.h"

T VoronoiCells::addRegEnergy(T w)
{
    T energy = 0.0;
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        energy += 0.5 * (si - Si).squaredNorm();
    }
    return energy * w;
}
void VoronoiCells::addRegForceEntries(VectorXT& grad, T w)
{
    
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        TV dOdx = w * (si - Si);
        Matrix<T, 3, 2> dxdw;
        computeSurfacePointdxds(samples[i], dxdw);
        TV2 dOds = dOdx.transpose() * dxdw;
        addForceEntry<2>(grad, {i}, -dOds);
    }
    
}

void VoronoiCells::addRegHessianEntries(std::vector<Entry>& entries, T w)
{
    for (int i = 0; i < samples.size(); i++)
    {
        TV si = toTV(samples[i].interpolate(geometry->vertexPositions));
        TV Si = toTV(samples_rest[i].interpolate(geometry->vertexPositions));
        TV dOdx = (si - Si);
        Matrix<T, 3, 2> dxdw;
        computeSurfacePointdxds(samples[i], dxdw);
        TV2 dOds = dOdx.transpose() * dxdw;
        TM2 d2Ods2 = w * dOds * dOds.transpose();
        addHessianEntry<2, 2>(entries, {i}, d2Ods2);
    }
}

void VoronoiCells::diffTestScale()
{
    if (objective == Centroidal)
    {
        VectorXT grad;
        T E0;
        T g_norm = computeCentroidalVDGradient(grad, E0);    
        std::vector<SurfacePoint> samples_current = samples;
        VectorXT dx(samples.size() * 2);
        dx.setRandom();
        dx *= 1.0 / dx.norm();
        dx *= 0.01;
        T previous = 0.0;
        for (int i = 0; i < 10; i++)
        {
            samples = samples_current;
            tbb::parallel_for(0, (int)samples.size(), [&](int i){
                updateSurfacePoint(samples[i], dx.segment<2>(i * 2));
            });
            constructVoronoiDiagram(true);
            T E1 = computeCentroidalVDEnergy();
            T dE = E1 - E0;
            dE -= grad.dot(dx);
            // std::cout << "dE " << dE << std::endl;
            if (i > 0)
            {
                std::cout << (previous/dE) << std::endl;
            }
            previous = dE;
            dx *= 0.5;
        }
        samples = samples_current;
    }
    else if (objective == Perimeter)
    {
        VectorXT grad;
        T E0;
        T g_norm = computePerimeterMinimizationGradient(grad, E0);    
        std::vector<SurfacePoint> samples_current = samples;
        VectorXT dx(samples.size() * 2);
        dx.setRandom();
        dx *= 1.0 / dx.norm();
        dx *= 0.01;
        T previous = 0.0;
        for (int i = 0; i < 10; i++)
        {
            samples = samples_current;
            tbb::parallel_for(0, (int)samples.size(), [&](int i){
                updateSurfacePoint(samples[i], dx.segment<2>(i * 2));
            });
            constructVoronoiDiagram(true);
            T E1 = computePerimeterMinimizationEnergy();
            T dE = E1 - E0;
            dE -= grad.dot(dx);
            // std::cout << "dE " << dE << std::endl;
            if (i > 0)
            {
                std::cout << (previous/dE) << std::endl;
            }
            previous = dE;
            dx *= 0.5;
        }
        samples = samples_current;
    }
}

T VoronoiCells::computeCentroidalVDEnergy(T w)
{
    VectorXT energies(unique_ixn_points.size());
    energies.setZero();
#ifdef PARALLEL
    tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int i)
#else
    for (int i = 0; i < unique_ixn_points.size(); i++)
#endif
    {
        auto ixn = unique_ixn_points[i];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        for (int idx : site_indices)
        {
            SurfacePoint si = samples[idx];
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path;
            T dis;
            computeGeodesicDistance(xi, si, dis, path, ixn_data_site, false);
            energies[i] += 0.5 * w * dis * dis * cell_weights[idx];
        }   
        
    }
#ifdef PARALLEL
    );
#endif
    return energies.sum();
}

T VoronoiCells::computeCentroidalVDGradient(VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();
    VectorXT energies(unique_ixn_points.size());
    energies.setZero();
    std::vector<VectorXT> grads((int)unique_ixn_points.size(), 
        VectorXT::Zero(n_sites * 2));
#ifdef PARALLEL
    tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int ixn_idx)
#else
    for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
#endif
    {
        auto ixn = unique_ixn_points[ixn_idx];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        MatrixXT dx_ds;
        computeDxDs(xi, site_indices, dx_ds, true);
        
        for (int i = 0; i < site_indices.size(); i++)
        {
            int idx = site_indices[i];
            SurfacePoint si = samples[idx];
            Vector<T, 4> dldw;
            T dis = computeGeodesicLengthAndGradient(xi, si, dldw);
            energies[ixn_idx] += 0.5 * dis * dis * cell_weights[idx];
            T dOdl = dis * cell_weights[idx];
            TV2 pOpx = dldw.segment<2>(0);
            TV2 pOps = dldw.segment<2>(2);

            
            TV2 dOds = dOdl * ((pOpx.transpose() * 
                            dx_ds.block(0, i * 2, 2, 2)).transpose()
                        + pOps);
            
            addForceEntry<2>(grads[ixn_idx], {idx}, dOds);
            // x is function of all s
            for (int j = 0; j < site_indices.size(); j++)
            {
                if (j==i) continue;
                dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                addForceEntry<2>(grads[ixn_idx], {site_indices[j]}, dOds);
            }
        }
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < unique_ixn_points.size(); i++)
    {
        grad += w * grads[i];
    }
    energy = w * energies.sum();
    return grad.norm();
}

void VoronoiCells::addCentroidalVDForceEntries(VectorXT& grad, T w)
{
    int n_dof = samples.size() * 2;
    std::vector<VectorXT> grads((int)unique_ixn_points.size(), 
        VectorXT::Zero(n_dof));
#ifdef PARALLEL
    tbb::parallel_for(0, (int)unique_ixn_points.size(),[&](int ixn_idx)
#else
    for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
#endif
    {
        auto ixn = unique_ixn_points[ixn_idx];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        MatrixXT dx_ds;
        computeDxDs(xi, site_indices, dx_ds, true);
        
        for (int i = 0; i < site_indices.size(); i++)
        {
            int idx = site_indices[i];
            SurfacePoint si = samples[idx];
            Vector<T, 4> dldw;
            T dis = computeGeodesicLengthAndGradient(xi, si, dldw);
            
            T dOdl = dis * cell_weights[idx];
            TV2 pOpx = dldw.segment<2>(0);
            TV2 pOps = dldw.segment<2>(2);

            
            TV2 dOds = dOdl * ((pOpx.transpose() * 
                            dx_ds.block(0, i * 2, 2, 2)).transpose()
                        + pOps);
            
            addForceEntry<2>(grads[ixn_idx], {idx}, dOds);
            // x is function of all s
            for (int j = 0; j < site_indices.size(); j++)
            {
                if (j==i) continue;
                dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                addForceEntry<2>(grads[ixn_idx], {site_indices[j]}, dOds);
            }
        }
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < unique_ixn_points.size(); i++)
    {
        grad += -w * grads[i];
    }
}


void VoronoiCells::addCentroidalVDHessianEntries(std::vector<Entry>& entries, T w)
{
    for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
    {
        auto ixn = unique_ixn_points[ixn_idx];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        MatrixXT dx_ds;
        computeDxDs(xi, site_indices, dx_ds, true);
        VectorXT dOds_total(site_indices.size() * 2);
        dOds_total.setZero();

        MatrixXT d2Ods2_total(site_indices.size() * 2,
            site_indices.size() * 2);
        d2Ods2_total.setZero();

        TM2 d2Odx2 = TM2::Zero();

        MatrixXT d2Ods2 = d2Ods2_total;

        MatrixXT d2Odxds(2, site_indices.size() * 2);
        d2Odxds.setZero();

        for (int i = 0; i < site_indices.size(); i++)
        {
            int idx = site_indices[i];
            SurfacePoint si = samples[idx];
            Vector<T, 4> dldw;
            Matrix<T, 4, 4> d2ldw2;
            T dis = computeGeodesicLengthAndGradientAndHessian(xi, si, dldw, d2ldw2);
            
            T dOdl = dis * cell_weights[idx];
            TV2 pOpx = dldw.segment<2>(0);
            TV2 pOps = dldw.segment<2>(2);

            TV2 dOds = dOdl * ((pOpx.transpose() * 
                            dx_ds.block(0, i * 2, 2, 2)).transpose()
                        + pOps);
            
            addForceEntry<2>(dOds_total, {i}, dOds);


            d2Odx2 += dis * d2ldw2.block(2, 2, 2, 2);
            d2Ods2.block(i * 2, i * 2, 2, 2) += dis * d2ldw2.block(0, 0, 2, 2) * cell_weights[idx];
            d2Odxds.block(0, i * 2, 2, 2) += dis * d2ldw2.block(0, 2, 2, 2) * cell_weights[idx];
            
            // x is function of all s
            for (int j = 0; j < site_indices.size(); j++)
            {
                if (j==i) continue;
                dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                addForceEntry<2>(dOds_total, {j}, dOds);
            }
        }
        

        d2Ods2_total += dOds_total * dOds_total.transpose();
        d2Ods2_total += dx_ds.transpose() * d2Odx2 * dx_ds;
        d2Ods2_total += d2Ods2;
        
        addHessianEntry<2, 2>(entries, site_indices, w * d2Ods2_total);
    }
}

T VoronoiCells::computeCentroidalVDHessian(StiffnessMatrix& hess, VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    energy = 0.0;
    std::vector<Entry> entries;
    for (int ixn_idx = 0; ixn_idx < unique_ixn_points.size(); ixn_idx++)
    {
        auto ixn = unique_ixn_points[ixn_idx];
        SurfacePoint xi = ixn.first;
        std::vector<int> site_indices = ixn.second;
        if (site_indices.size() == 2)
            continue;
        MatrixXT dx_ds;
        computeDxDs(xi, site_indices, dx_ds, true);
        VectorXT dOds_total(site_indices.size() * 2);
        dOds_total.setZero();

        MatrixXT d2Ods2_total(site_indices.size() * 2,
            site_indices.size() * 2);
        d2Ods2_total.setZero();

        TM2 d2Odx2 = TM2::Zero();

        MatrixXT d2Ods2 = d2Ods2_total;

        MatrixXT d2Odxds(2, site_indices.size() * 2);
        d2Odxds.setZero();

        for (int i = 0; i < site_indices.size(); i++)
        {
            int idx = site_indices[i];
            SurfacePoint si = samples[idx];
            Vector<T, 4> dldw;
            Matrix<T, 4, 4> d2ldw2;
            T dis = computeGeodesicLengthAndGradientAndHessian(xi, si, dldw, d2ldw2);
            energy += 0.5 * dis * dis;
            T dOdl = dis;
            TV2 pOpx = dldw.segment<2>(0);
            TV2 pOps = dldw.segment<2>(2);

            TV2 dOds = dOdl * ((pOpx.transpose() * 
                            dx_ds.block(0, i * 2, 2, 2)).transpose()
                        + pOps);
            
            addForceEntry<2>(dOds_total, {i}, dOds);


            d2Odx2 += dis * d2ldw2.block(2, 2, 2, 2);
            d2Ods2.block(i * 2, i * 2, 2, 2) += dis * d2ldw2.block(0, 0, 2, 2);
            d2Odxds.block(0, i * 2, 2, 2) += dis * d2ldw2.block(0, 2, 2, 2);
            
            // x is function of all s
            for (int j = 0; j < site_indices.size(); j++)
            {
                if (j==i) continue;
                dOds = dOdl * (pOpx.transpose() * dx_ds.block(0, j * 2, 2, 2)).transpose();
                addForceEntry<2>(dOds_total, {j}, dOds);
            }
        }
        addForceEntry<2>(grad, site_indices, dOds_total);

        d2Ods2_total += dOds_total * dOds_total.transpose();
        d2Ods2_total += dx_ds.transpose() * d2Odx2 * dx_ds;
        d2Ods2_total += d2Ods2;
        // d2Ods2_total += d2Odxds.transpose() * dx_ds;
        // d2Ods2_total += dx_ds.transpose() * d2Odxds;

        addHessianEntry<2, 2>(entries, site_indices, d2Ods2_total);
    }
    hess.resize(n_sites * 2, n_sites * 2);
    hess.setFromTriplets(entries.begin(), entries.end());
    projectDirichletDoFMatrix(hess, dirichlet_data);
    hess.makeCompressed();
    return grad.norm();
}

T VoronoiCells::computePerimeterMinimizationEnergy(T w)
{
    T energy = 0.0;
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        TV v0 = toTV(unique_ixn_points[idx0].first.interpolate(geometry->vertexPositions));
        TV v1 = toTV(unique_ixn_points[idx1].first.interpolate(geometry->vertexPositions));
        energy += 0.5 * w * (v1 - v0).dot(v1 - v0);
    }
    return energy;
}

T VoronoiCells::computePerimeterMinimizationGradient(VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    int n_edges = valid_VD_edges.size();
    VectorXT energies(n_edges);
    energies.setZero();
    std::vector<VectorXT> grads(n_edges, 
        VectorXT::Zero(n_sites * 2));


#ifdef PARALLEL
    tbb::parallel_for(0, n_edges,[&](int i)
#else
    for (int i = 0; i < n_edges; i++)
#endif
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        energies[i] += 0.5 * (v1 - v0).dot(v1 - v0);
        // energy += (v1 - v0).norm();
        
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);
        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        // if (!valid)
        //     return 1e12;

        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;

        addForceEntry<2>(grads[i], x0_sites, dOds_site0);
        addForceEntry<2>(grads[i], x1_sites, dOds_site1);
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < n_edges; i++)
    {
        grad += w * grads[i];
    }
    energy = w * energies.sum();
    return grad.norm();
}

void VoronoiCells::addPerimeterMinimizationForceEntries(VectorXT& grad, T w)
{
    int n_dof = samples.size() * 2;
    int n_edges = valid_VD_edges.size();
    
    std::vector<VectorXT> grads(n_edges, 
        VectorXT::Zero(n_dof));


#ifdef PARALLEL
    tbb::parallel_for(0, n_edges,[&](int i)
#else
    for (int i = 0; i < n_edges; i++)
#endif
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);
        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        // if (!valid)
        //     return 1e12;

        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;

        addForceEntry<2>(grads[i], x0_sites, dOds_site0);
        addForceEntry<2>(grads[i], x1_sites, dOds_site1);
    }
#ifdef PARALLEL
    );
#endif
    for (int i = 0; i < n_edges; i++)
    {
        grad += -w * grads[i];
    }
}
void VoronoiCells::addPerimeterMinimizationHessianEntries(std::vector<Entry>& entries, T w)
{
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        

        // energy += (v1 - v0).norm();
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);

        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);


        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;
        
        
        TM d2Odx2 = (TM::Identity() - (v1 - v0).normalized() * (v1 - v0).normalized().transpose()) / (v1 - v0).norm();


        MatrixXT d2x0ds2 = dx0_ds.transpose() * (d2Odx2 + TM::Identity()) * dx0_ds;
        MatrixXT d2x1ds2 = dx1_ds.transpose() * (d2Odx2 + TM::Identity()) * dx1_ds;

        

        addHessianEntry<2, 2>(entries, x0_sites, w * d2x0ds2);
        addHessianEntry<2, 2>(entries, x1_sites, w * d2x1ds2);

    }
}

T VoronoiCells::computePerimeterMinimizationHessian(StiffnessMatrix& hess, 
    VectorXT& grad, T& energy, T w)
{
    grad.resize(n_sites * 2); grad.setZero();

    energy = 0.0;
    std::vector<Entry> entries;
    int n_edges = valid_VD_edges.size();
    for (int i = 0; i < n_edges; i++)
    {   
        int idx0 = valid_VD_edges[i][0];
        int idx1 = valid_VD_edges[i][1];
        SurfacePoint x0 = unique_ixn_points[idx0].first.inSomeFace();
        SurfacePoint x1 = unique_ixn_points[idx1].first.inSomeFace();
        
        TV v0 = toTV(x0.interpolate(geometry->vertexPositions));
        TV v1 = toTV(x1.interpolate(geometry->vertexPositions));
        energy += 0.5 * (v1 - v0).dot(v1 - v0);

        // energy += (v1 - v0).norm();
        std::vector<int> x0_sites = unique_ixn_points[idx0].second;
        std::vector<int> x1_sites = unique_ixn_points[idx1].second;

        TV dOdx0, dOdx1;
        dOdx0 = -(v1 - v0);
        dOdx1 = (v1 - v0);

        // dOdx0 = -(v1 - v0).normalized();
        // dOdx1 = (v1 - v0).normalized();
        MatrixXT dx0_ds, dx1_ds;

        bool valid = computeDxDs(x0, x0_sites, dx0_ds);
        valid &= computeDxDs(x1, x1_sites, dx1_ds);

        if (!valid)
            return 1e12;


        VectorXT dOds_site0 = dOdx0.transpose() * dx0_ds;
        VectorXT dOds_site1 = dOdx1.transpose() * dx1_ds;
        addForceEntry<2>(grad, x0_sites, dOds_site0);
        addForceEntry<2>(grad, x1_sites, dOds_site1);

        
        // TM d2Odx2 = (TM::Identity() - (v1 - v0).normalized() * (v1 - v0).normalized().transpose()) / (v1 - v0).norm();
        MatrixXT d2x0ds2 = dx0_ds.transpose() * dx0_ds;
        MatrixXT d2x1ds2 = dx1_ds.transpose() * dx1_ds;

        MatrixXT d2Odxds = -dx1_ds.transpose() * dx0_ds;

        addHessianEntry<2, 2>(entries, x0_sites, d2x0ds2);
        addHessianEntry<2, 2>(entries, x1_sites, d2x1ds2);

    }

    hess.resize(n_sites * 2, n_sites * 2);
    hess.setFromTriplets(entries.begin(), entries.end());
    hess *= w;
    projectDirichletDoFMatrix(hess, dirichlet_data);
    hess.makeCompressed();
    return grad.norm();
}

T VoronoiCells::computeDistanceMatchingEnergy(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current)
{
    T energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    T dis_to_site0;
    std::vector<IxnData> ixn_data_site;
    std::vector<SurfacePoint> path; 
    if (metric == Euclidean)
        dis_to_site0 = (current - site0_location).norm();
    else
        computeGeodesicDistance(samples[site_indices[0]], xi_current, 
            dis_to_site0, path, ixn_data_site, false);
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        if (metric == Euclidean)
            dis_to_site = (current - site_location).norm();
        else
            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, false);
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2);
    }
    return energy;
}

T VoronoiCells::computeDistanceMatchingGradient(const std::vector<int>& site_indices, 
    SurfacePoint& xi_current, TV2& grad, T& energy)
{
    grad = TV2::Zero();
    energy = 0.0;
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    // std::cout << "face# " << fi.getIndex() << " xi " << current.transpose() << std::endl;
    TV bary = toTV(xi_current.faceCoords);
    TV pt = v0 * bary[0] + v1 * bary[1] + v2 * bary[2];
    // std::cout << pt.transpose() << " " << current.transpose() << std::endl;
    // std::getchar();

    // std::cout << v0.transpose() << " " << v1.transpose() << " " << v2.transpose() << std::endl;
    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;
        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 

        T dOdl = (dis_to_site - dis_to_site0); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
    }
    return grad.norm();
}

void VoronoiCells::computeDistanceMatchingHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess)
{
    hess = TM2::Zero();
    
    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        // grad += dOdl * dldw
        // hess += dOdl * d2ldw2 + dldw^T d2Odl2 * dldw 
        hess += dldw * dldw.transpose();
    }
    
}

T VoronoiCells::computeDistanceMatchingEnergyGradientHessian(const std::vector<int>& site_indices, 
        SurfacePoint& xi_current, TM2& hess, TV2& grad, T& energy)
{
    hess = TM2::Zero();
    grad = TV2::Zero();
    energy = 0.0;

    TV current = toTV(xi_current.interpolate(geometry->vertexPositions));
    TV site0_location = toTV(samples[site_indices[0]].interpolate(geometry->vertexPositions));
    TV v0 = toTV(geometry->vertexPositions[xi_current.face.halfedge().vertex()]);
    TV v1 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().vertex()]);
    TV v2 = toTV(geometry->vertexPositions[xi_current.face.halfedge().next().next().vertex()]);

    T dis_to_site0;
    Matrix<T, 3, 2> dxdw; 
    dxdw.col(0) = v0 - v2;
    dxdw.col(1) = v1 - v2;
    
    TV dldx0;
    TM d2ldx02;
    if (metric == Euclidean)
    {
        dis_to_site0 = (current - site0_location).norm();
        dldx0 = (current - site0_location).normalized();
        d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
    }
    else
    {
        std::vector<IxnData> ixn_data_site;
        std::vector<SurfacePoint> path;
        computeGeodesicDistance(samples[site_indices[0]], xi_current, dis_to_site0, path, 
            ixn_data_site, true);
        int length = path.size();
        
        TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
        if (length == 2)
        {
            dldx0 = (vtx1 - vtx0).normalized();
            d2ldx02 = (TM::Identity() - dldx0 * dldx0.transpose()) / dis_to_site0;
        }
        else
        {
            TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
            dldx0 = -(ixn1 - vtx1).normalized();
        }
    }
    
    for (int j = 1; j < site_indices.size(); j++)
    {
        int site_idx = site_indices[j];
        TV site_location = toTV(samples[site_indices[j]].interpolate(geometry->vertexPositions));
        T dis_to_site;
        TV dldxj;
        TM d2ldxj2;
        if (metric == Euclidean)
        {
            dis_to_site = (current - site_location).norm();
            dldxj = (current - site_location).normalized();
            d2ldxj2 = (TM::Identity() - dldxj * dldxj.transpose()) / dis_to_site;
        }
        else
        {
            std::vector<IxnData> ixn_data_site;
            std::vector<SurfacePoint> path; 

            computeGeodesicDistance(samples[site_idx], xi_current, dis_to_site,
                path, ixn_data_site, true);

            int length = path.size();
        
            TV vtx0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV vtx1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldxj = (vtx1 - vtx0).normalized();
            }
            else
            {
                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldxj = -(ixn1 - vtx1).normalized();

                int ixn_dof = (length - 2) * 3;
                MatrixXT dfdc(ixn_dof, 3); dfdc.setZero();
                MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                MatrixXT d2gdcdx(ixn_dof, 3); d2gdcdx.setZero();

                for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
                {
                    // std::cout << "inside" << std::endl;
                    Matrix<T, 6, 6> hess;
                    TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
                    TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

                    edgeLengthHessian(ixn_i, ixn_j, hess);
                    dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
                }
                for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
                {
                    TV x_start = ixn_data_site[1+ixn_id].start;
                    TV x_end = ixn_data_site[1+ixn_id].end;
                    dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
                }
                TM dlndxn = (TM::Identity() - dldxj * dldxj.transpose()) / (ixn1 - vtx1).norm();
                dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                dfdc.block(ixn_dof-3, 0, 3, 3) += -dlndxn;
                d2gdcdx.block(ixn_dof-3, 0, 3, 3) += -dlndxn;

                MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
                MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


                TM d2gdc2 = dlndxn;

                d2ldxj2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
            }
        }
        
        // std::cout << site0_location.transpose() << " " << site_location.transpose() << std::endl;
        // std::cout << "dis_to_site0 " << dis_to_site0 << " dis_to_site " << dis_to_site << std::endl;

        T dOdl = (dis_to_site - dis_to_site0); 
        TV2 dldw = dxdw.transpose() * (dldxj - dldx0);

        energy += 0.5 * std::pow(dis_to_site - dis_to_site0, 2); 
        
        grad += dOdl * -dxdw.transpose() * dldx0;
        grad += dOdl * dxdw.transpose() * dldxj;
        hess += dldw * dldw.transpose();
    }
    return grad.norm();
}