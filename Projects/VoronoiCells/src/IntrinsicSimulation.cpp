
#include <Eigen/CholmodSupport>
// #include <Spectra/SymEigsShiftSolver.h>
// #include <Spectra/MatOp/SparseSymShiftSolve.h>
// #include <Spectra/SymEigsSolver.h>
// #include <Spectra/MatOp/SparseSymMatProd.h>

#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    ixn_data.clear();
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    
    gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);

    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);
    
    mmp.propagate(va_sub);
    if (trace_path)
    {
        path = mmp.traceBack(vb_sub, dis);
        std::reverse(path.begin(), path.end());
        for (auto& pt : path)
        {
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = he.tailVertex();
                SurfacePoint end_extrinsic = he.tipVertex();
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                    std::swap(start, end);
                
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    std::exit(0);
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }
            ixn_data.push_back(IxnData(start, end, (1.0-edge_t)));
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt.face = mesh->face(pt.face.getIndex());
            
            pt = pt.inSomeFace();
        }
        TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        TV v1 = toTV(path[path.size() - 1].interpolate(geometry->vertexPositions));
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[path.size() - 2].interpolate(geometry->vertexPositions));
        
        if (path.size() > 2)
            if ((v0 - ixn0).norm() < 1e-5)
                path.erase(path.begin() + 1);
        if (path.size() > 2)
            if ((v1 - ixn1).norm() < 1e-5)
                path.erase(path.end() - 2);
        
    }
    else
        dis = mmp.getDistance(vb_sub);
}

bool IntrinsicSimulation::hasSmallSegment(const std::vector<SurfacePoint>& path)
{
    for (int i = 0; i < path.size() - 1; i++)
    {
        TV ixn0 = toTV(path[i].interpolate(geometry->vertexPositions));
        TV ixn1 = toTV(path[i+1].interpolate(geometry->vertexPositions));
        if ((ixn0 - ixn1).norm() < 1e-6)
        {
            return true;
        }
    }
    return false;
}

void IntrinsicSimulation::computeExactGeodesicEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    ixn_data.clear();
    int n_tri = extrinsic_indices.rows() / 3;
    std::vector<std::vector<size_t>> mesh_indices_gc(n_tri, std::vector<size_t>(3));
    for (int i = 0; i < n_tri; i++)
        for (int d = 0; d < 3; d++)
            mesh_indices_gc[i][d] = extrinsic_indices[i * 3 + d];
    int n_vtx_extrinsic = extrinsic_vertices.rows() / 3;
    std::vector<gc::Vector3> mesh_vertices_gc(n_vtx_extrinsic);
    for (int i = 0; i < n_vtx_extrinsic; i++)
        mesh_vertices_gc[i] = gc::Vector3{extrinsic_vertices(i * 3 + 0), 
            extrinsic_vertices(i * 3 + 1), extrinsic_vertices(i * 3 + 2)};
    // START_TIMING(contruct)
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    // FINISH_TIMING_PRINT(contruct)
    std::unique_ptr<gcs::ManifoldSurfaceMesh> sub_mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> sub_geometry;
    std::tie(sub_mesh, sub_geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    
    
    std::unique_ptr<gcs::FlipEdgeNetwork> sub_edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*sub_mesh, *sub_geometry, {}));
    sub_edgeNetwork->posGeom = sub_geometry.get();
    
    // gcs::GeodesicAlgorithmExact mmp(*sub_mesh, *sub_geometry);
    
    SurfacePoint va_sub(sub_mesh->face(va.face.getIndex()), va.faceCoords);
    SurfacePoint vb_sub(sub_mesh->face(vb.face.getIndex()), vb.faceCoords);

    SurfacePoint va_intrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(va_sub);
    gcVertex va_vtx = sub_edgeNetwork->tri->insertVertex(va_intrinsic);

    SurfacePoint vb_intrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(vb_sub);
    gcVertex vb_vtx = sub_edgeNetwork->tri->insertVertex(vb_intrinsic);
    
    std::vector<gcs::Halfedge> path_geo = shortestEdgePath(*sub_edgeNetwork->tri, va_vtx, vb_vtx);
    sub_edgeNetwork->addPath(path_geo);
    sub_edgeNetwork->nFlips = 0;
    sub_edgeNetwork->nShortenIters = 0;
    sub_edgeNetwork->EPS_ANGLE = 1e-5;
    sub_edgeNetwork->straightenAroundMarkedVertices = true;
    size_t iterLim = gc::INVALID_IND;
    double lengthLim = 0.;
    sub_edgeNetwork->addAllWedgesToAngleQueue();
    sub_edgeNetwork->iterativeShorten(iterLim, lengthLim);
    
    gcEdge ei = sub_edgeNetwork->tri->intrinsicMesh->connectingEdge(va_vtx, vb_vtx);
    if (ei == gcEdge())
    {
        std::cout << "Failed Connection" <<std::endl;
    }
    dis = sub_edgeNetwork->tri->edgeLengths[ei];
    if (trace_path)
    {
        gcs::Halfedge he = ei.halfedge();
        if (he.tailVertex() == vb_vtx)
            he = ei.halfedge().twin();
        path = sub_edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);
        for (auto& pt : path)
        {
            // std::cout << pt.inSomeFace().face.halfedge().getIndex() << std::endl;
            T edge_t = -1.0; TV start = TV::Zero(), end = TV::Zero();
            bool is_edge_point = (pt.type == gcs::SurfacePointType::Edge);
            if (is_edge_point)
            {
                auto he = pt.edge.halfedge();
                SurfacePoint start_extrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(he.tailVertex());
                SurfacePoint end_extrinsic = sub_edgeNetwork->tri->equivalentPointOnIntrinsic(he.tipVertex());
                start = toTV(start_extrinsic.interpolate(sub_geometry->vertexPositions));
                end = toTV(end_extrinsic.interpolate(sub_geometry->vertexPositions));
                TV ixn = toTV(pt.interpolate(sub_geometry->vertexPositions));
                TV test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((test_interp - ixn).norm() > 1e-6)
                    std::swap(start, end);
                
                TV dir0 = (end-start).normalized();
                TV dir1 = (ixn-start).normalized();
                if ((dir0.cross(dir1)).norm() > 1e-6)
                {
                    std::cout << "error in cross product" << std::endl;
                    std::exit(0);
                }
                test_interp = pt.tEdge * start + (1.0 - pt.tEdge) * end;
                if ((ixn - test_interp).norm() > 1e-6)
                {
                    std::cout << "error in interpolation" << std::endl;
                    std::exit(0);
                }
                edge_t = pt.tEdge;
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }
            ixn_data.push_back(IxnData(start, end, (1.0-edge_t)));
            pt.face = mesh->face(pt.face.getIndex());
            pt.edge = mesh->edge(pt.edge.getIndex());
            pt.vertex = mesh->vertex(pt.vertex.getIndex());
            pt = pt.inSomeFace();
        }
    }

    
    // mmp.propagate(va_sub);
    // if (trace_path)
    // {
    //     path = mmp.traceBack(vb_sub, dis);
    //     std::reverse(path.begin(), path.end());
    //     for (auto& pt : path)
    //     {
    //         pt.face = mesh->face(pt.face.getIndex());
    //     }
        
    // }
    // else
    //     dis = mmp.getDistance(vb_sub);
}

void IntrinsicSimulation::getMarkerPointsPosition(VectorXT& positions)
{
    std::vector<int> marker_indices = {1, 5};
    positions.resize(marker_indices.size() * 3);
    for (int i = 0; i < marker_indices.size(); i++)
    {
        if (marker_indices[i] * 3 < undeformed.rows())
            positions.segment<3>(i * 3) = toTV(mass_surface_points[marker_indices[i]].first.interpolate(geometry->vertexPositions));
    }
    // std::cout << mass_surface_points[marker_indices[0]].first.faceCoords << std::endl;
}

void IntrinsicSimulation::getAllPointsPosition(VectorXT& positions)
{
    positions.resize(mass_surface_points.size() * 3);
    for (int i = 0; i < mass_surface_points.size(); i++)
    {
        positions.segment<3>(i * 3) = toTV(mass_surface_points[i].first.interpolate(geometry->vertexPositions));
    }
}

bool IntrinsicSimulation::closeToIrregular(const SurfacePoint& point)
{
    SurfacePoint end_point = point.inSomeFace();
    Vector3 bc = end_point.faceCoords;
    int close_to_zero_cnt = 0;
    for (int d = 0; d < 3; d++)
    {
        if (std::abs(bc[d]) < IRREGULAR_EPSILON)
            close_to_zero_cnt++;
    }
    if (close_to_zero_cnt > 0)
    {
        return true;
    }
    return false;
}

void IntrinsicSimulation::getCurrentMassPointConfiguration(
    std::vector<std::pair<SurfacePoint, gcFace>>& configuration)
{
    configuration = mass_surface_points_undeformed;
    for (int i = 0; i < mass_surface_points.size(); i++)
    {
        configuration[i] = mass_surface_points[i];
    }
}

void IntrinsicSimulation::updateCurrentState(bool trace)
{
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            delta_u[offset] = target;
        });
    }

    for (int i = 0; i < mass_surface_points.size(); i++)
    {
        gcFace fi = mass_surface_points[i].second;
        Vector3 start_bc = mass_surface_points[i].first.faceCoords;
        Vector3 trace_vec{delta_u[i*2+0],delta_u[i*2+1],0.0-delta_u[i*2+0]-delta_u[i*2+1]};
        
        if (trace_vec.norm() < 1e-10)
            continue;

        gcs::TraceOptions options; 
        options.includePath = true;
        
        // trace geodesic on the extrinsic mesh
        gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
        // std::cout << std::setprecision(10) << "trace length " << result.length << std::endl;
        // while (closeToIrregular(result.endPoint) && !run_diff_test)
        // {
        //     std::cout << "close to irregular" << std::endl;
        //     Vector3 trace_vec_unit = trace_vec.normalize();
        //     T trace_norm = trace_vec.norm();
        //     trace_vec = trace_vec_unit * (trace_norm + 2.0 * IRREGULAR_EPSILON);
        //     result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
        // }
        // result.endPoint.faceCoords
        
        // std::cout << result.pathPoints.size() << std::endl;
        if (result.pathPoints.size() != 1)
        {
            // find equivalent intrinc vertex
            SurfacePoint endpoint = result.endPoint.inSomeFace();
            mass_surface_points[i].first = endpoint;
            // change reference face
            mass_surface_points[i].second = endpoint.face;
            // std::cout << "face " << mass_surface_points[i].second.getIndex() << std::endl;
        }
        else
        {
            // std::cout << "can't trace it" << std::endl;
        }
    }
    if (trace)
    {
        retrace = true;
        traceGeodesics();
    }
}


bool IntrinsicSimulation::linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du)
{
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::CholmodSupernodalLLT<StiffnessMatrix> solver;
    
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-10;
    K += H;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;

    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "factorize" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                // std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
}

void IntrinsicSimulation::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }
}

T IntrinsicSimulation::computeTotalEnergy()
{
    T total_energy = 0.0;

    addEdgeLengthEnergy(we, total_energy);
    

    return total_energy;
}

T IntrinsicSimulation::computeResidual(VectorXT& residual)
{

    addEdgeLengthForceEntries(we, residual);

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void IntrinsicSimulation::buildSystemMatrix(StiffnessMatrix& K)
{
   
    std::vector<Entry> entries;
    addEdgeLengthHessianEntries(we, entries);

    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);
    
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

void IntrinsicSimulation::moveMassPoint(int idx, int bc)
{
    
    u[idx * 2 + bc] += 0.001;
    
    deformed = undeformed + u;

    std::cout << "on face " << mass_surface_points[idx].second.getIndex() << std::endl;
}

void IntrinsicSimulation::massPointPosition(int idx, TV& pos)
{
    pos = toTV(mass_surface_points[idx].first.interpolate(geometry->vertexPositions));
}

void IntrinsicSimulation::updateVisualization(bool all_edges)
{
    
    all_intrinsic_edges.resize(0);

    int n_springs = spring_edges.size();
    std::vector<std::vector<std::pair<TV, TV>>> sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
    
    if (retrace)
    {
        traceGeodesics();
    }

	for(int i = 0; i < n_springs; i++)
    {
        SurfacePoint vA = mass_surface_points[spring_edges[i][0]].first;
        SurfacePoint vB = mass_surface_points[spring_edges[i][1]].first;

        
        std::vector<SurfacePoint> path = paths[i];
        for(int j = 0; j < path.size() - 1; j++)
        {
            sub_pairs[i].push_back(std::make_pair(
                toTV(path[j].interpolate(geometry->vertexPositions)),
                toTV(path[j+1].interpolate(geometry->vertexPositions))
            ));
        }
    }
    for (int i = 0; i < n_springs; i++)
    {
        all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
    }
}

bool IntrinsicSimulation::advanceOneStep(int step)
{
    START_TIMING(NewtonStep)
    VectorXT residual(deformed.rows());
    residual.setZero();
    // START_TIMING(traceGeodesics)
    traceGeodesics();
    // FINISH_TIMING_PRINT(traceGeodesics)
    // START_TIMING(computeResidual)
    T residual_norm = computeResidual(residual);
    // FINISH_TIMING_PRINT(computeResidual)
    std::cout << "[NEWTON] iter " << step << "/" 
        << max_newton_iter << ": residual_norm " 
        << residual_norm << " tol: " << newton_tol << std::endl;
    
    if (residual_norm < newton_tol || step == max_newton_iter)
    {
        return true;
    }
    T du_norm = 1e10;
    // START_TIMING(lineSearchNewton)
    // dq_norm = lineSearchNewton(u, residual);
    du_norm = lineSearchNewton(residual);
    // checkHessian();
    // FINISH_TIMING_PRINT(lineSearchNewton)
    FINISH_TIMING_PRINT(NewtonStep)
    if(step == max_newton_iter || du_norm > 1e10 || du_norm < 1e-6)
    {
        std::cout << "ABNORMAL STOP with |du| " << du_norm << std::endl;
        return true;
    }
    
    return false;
}


T IntrinsicSimulation::lineSearchNewton(const VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    du = residual;

    StiffnessMatrix K(residual.rows(), residual.rows());
    if (use_Newton)
    {
        // Timer ti(true);
        // START_TIMING(buildSystemMatrix)
        buildSystemMatrix(K);
        // FINISH_TIMING_PRINT(buildSystemMatrix)
        // // std::cout << "\tbuild system takes " <<  ti.elapsed_sec() << std::endl;
        bool success = linearSolve(K, residual, du);
        if (!success)
        {
            std::cout << "Linear Solve Failed" << std::endl;
            return 1e16;
        }
    }
    T norm = du.norm();
    if (verbose)
        std::cout << "\t|du | " << norm << std::endl;
    T E0 = computeTotalEnergy();
    // std::cout << std::setprecision(8) << "ls E0  " << E0 << std::endl;

    auto lineSearchInDirection = [&](const VectorXT& direction, bool using_gradient) -> bool
    {
        T alpha = 1.0;
        int cnt = 0;
        std::vector<std::pair<SurfacePoint, gcFace>> current_state = mass_surface_points;
        while (true)
        {
            delta_u = alpha * direction;
            updateCurrentState(/*trace = */true);
            T E1 = computeTotalEnergy();
            if (verbose)
                std::cout << "\t[LS INFO] total energy: " << E1 << " #ls " << cnt << " |du| " << delta_u.norm() << std::endl;
            if (E1 - E0 < 0 || cnt > 15)
            {
                // std::cout << "|du| " << alpha * du.norm() << std::endl;
                // std::cout << "ls final " << E1 << " #ls " << cnt << std::endl;
                if (cnt > 15)
                {
                    if (!using_gradient)
                        std::cout << "-----line search max----- cnt > 15 switch to gradient" << std::endl;
                    else
                        std::cout << "-----line search max----- cnt > 15 along gradient [BUG ALERT]" << std::endl;
                    // std::cout << "\t[LS INFO] total energy: " << E1 << " #ls " << cnt << " |du| " << delta_u.norm() << std::endl;
                    // std::ofstream out("search_direction.txt");
                    // out << direction.rows() << std::endl;
                    // for (int i = 0; i < direction.rows(); i++)
                    //     out << direction[i] << " ";
                    // out << mass_surface_points.size() << std::endl;
                    // for (int i = 0; i < mass_surface_points.size(); i++)
                    // {
                    //     out << toTV(current_state[i].first.faceCoords).transpose() << std::endl;
                    //     out << current_state[i].second.getIndex();
                    //     out << std::endl;
                    // }
                    // out.close();
                    // std::exit(0);
                    if (!using_gradient)
                        mass_surface_points = current_state;
                    return false;
                    // Eigen::EigenSolver<MatrixXT> es(K);
                    // std::cout << es.eigenvalues().real().maxCoeff() << std::endl;
                    // return 1e16;
                }
                return true;
            }
            alpha *= 0.5;
            cnt += 1;
            mass_surface_points = current_state;
        }
    };

    bool ls_succeed = lineSearchInDirection(du, false);
    if (!ls_succeed)
        ls_succeed = lineSearchInDirection(residual, true);

    return delta_u.norm();
}

void IntrinsicSimulation::reset()
{
    mass_surface_points = mass_surface_points_undeformed;
    delta_u.setZero();
    updateCurrentState();
}

void IntrinsicSimulation::checkInformation()
{
    
}