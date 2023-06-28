
#include <Eigen/CholmodSupport>
#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, bool trace_path)
{
    
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
    
    // std::unique_ptr<gcs::FlipEdgeNetwork> sub_edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*mesh, *geometry, {}));
    
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
                
                // TV dir0 = (end-start).normalized();
                // TV dir1 = (ixn-start).normalized();
                // std::cout << pt.tEdge << " " << " cross " << (dir0.cross(dir1)).norm() << std::endl;
                // std::cout << ixn.transpose() << " " << (pt.tEdge * start + (1.0-pt.tEdge) * end).transpose() << std::endl;
                // std::getchar();
            }
            ixn_data.push_back(IxnData(start, end, pt.tEdge));
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


bool IntrinsicSimulation::simDoFToPosition(const VectorXT& sim_dof)
{
    if (use_intrinsic)
    {
        edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*mesh, *geometry, {}));
        edgeNetwork->posGeom = geometry.get();

        for (int i = 0; i < mass_vertices.size(); i++)
        {
            gcFace fi = mass_vertices[i].second;
            Vector3 start_bc{undeformed[i * 2 + 0], undeformed[i*2+1], 1.0 - undeformed[i * 2 + 0] - undeformed[i * 2 + 1]};
            Vector3 target{sim_dof[i*2+0],sim_dof[i*2+1],1.0-sim_dof[i*2+0]-sim_dof[i*2+1]};
            Vector3 trace_vec = target - start_bc;
            
            gcs::TraceOptions options; 
            options.includePath = true;
            
            // trace geodesic on the extrinsic mesh
            gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
            if (result.pathPoints.size() != 1)
            {
                // find equivalent intrinc vertex
                SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(result.endPoint);
                gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
                // std::cout << "insert" << std::endl;        
                mass_vertices[i].first = new_vtx;
                SurfacePoint endpoint = result.endPoint.inSomeFace();
                // DO NOT CHANGE FACE REFERENCE
                // mass_vertices[i].second = endpoint.face;
                // sim_dof[i*2+0] = endpoint.faceCoords.x;
                // sim_dof[i*2+1] = endpoint.faceCoords.y;
            }
            else 
            {
                SurfacePoint new_pt = SurfacePoint(fi, start_bc);
                SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(new_pt);
                gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
                mass_vertices[i].first = new_vtx;
            }
            // mass_vertices[i].first
        }
        
        for (int i = 0; i < mass_vertices.size() - 1; i++)
        {
            gcVertex vA = mass_vertices[i].first;
            gcVertex vB = mass_vertices[i + 1].first;
            
            std::vector<gcs::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
            edgeNetwork->addPath(path);
            edgeNetwork->nFlips = 0;
            edgeNetwork->nShortenIters = 0;
            edgeNetwork->EPS_ANGLE = 1e-5;
            edgeNetwork->straightenAroundMarkedVertices = true;
            size_t iterLim = gc::INVALID_IND;
            double lengthLim = 0.;
            edgeNetwork->addAllWedgesToAngleQueue();
            // std::cout << "empty " << edgeNetwork->wedgeAngleQueue.empty() << std::endl;
            edgeNetwork->iterativeShorten(iterLim, lengthLim);
            gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
            if (e == gcEdge())
            {
                std::cout << "non-existing edge connecting vtx " << vA.getIndex() << " and " << vB.getIndex()<< std::endl;   
                return false;
            }
        }
        
        return true;
    }
    else
    {
        for (int i = 0; i < mass_surface_points.size(); i++)
        {
            gcFace fi = mass_surface_points[i].second;
            Vector3 start_bc{undeformed[i * 2 + 0], undeformed[i*2+1], 1.0 - undeformed[i * 2 + 0] - undeformed[i * 2 + 1]};
            Vector3 target{sim_dof[i*2+0],sim_dof[i*2+1],1.0-sim_dof[i*2+0]-sim_dof[i*2+1]};
            Vector3 trace_vec = target - start_bc;
            
            gcs::TraceOptions options; 
            options.includePath = true;
            
            // trace geodesic on the extrinsic mesh
            gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
            // std::cout << result.pathPoints.size() << std::endl;
            if (result.pathPoints.size() != 1)
            {
                // find equivalent intrinc vertex
                SurfacePoint endpoint = result.endPoint.inSomeFace();
                mass_surface_points[i].first = endpoint;
                // DO NOT CHANGE FACE REFERENCE
            }
            // mass_vertices[i].first
        }
        return true;
        // std::cout << "simDoFToPosition failed" << std::endl;
    }
}

void IntrinsicSimulation::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
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
    bool update_succeed = simDoFToPosition(deformed);
    std::vector<Entry> entries;
    addEdgeLengthHessianEntries(we, entries);

    int n_dof = deformed.rows();
    K.resize(n_dof, n_dof);
    
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();
}

bool IntrinsicSimulation::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    
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


T IntrinsicSimulation::computeTotalEnergy(const VectorXT& _u)
{
    T total_energy = 0.0;
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }

    deformed = undeformed + projected;
    bool update_succeed = simDoFToPosition(deformed);
    if (!update_succeed)
        return 1e10;

    addEdgeLengthEnergy(we, total_energy);
    

    return total_energy;
}

T IntrinsicSimulation::computeResidual(const VectorXT& _u, VectorXT& residual)
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
    // START_TIMING(simToPos)
    bool update_succeed = simDoFToPosition(deformed);
    // FINISH_TIMING_PRINT(simToPos)
    if (!update_succeed)
        return residual.norm();
    addEdgeLengthForceEntries(we, residual);

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void IntrinsicSimulation::moveMassPoint(int idx, int bc)
{
    
    u[idx * 2 + bc] = 0.1;
    
    deformed = undeformed + u;
    simDoFToPosition(deformed);
    undeformed = deformed;
    if (use_intrinsic)
        std::cout << "on face " << mass_vertices[idx].second.getIndex() << std::endl;
    else
        std::cout << "on face " << mass_surface_points[idx].second.getIndex() << std::endl;
}

void IntrinsicSimulation::massPointPosition(int idx, TV& pos)
{
    Vector3 bary{deformed[idx*2+0], deformed[idx*2+1], 1.0-deformed[idx*2+0]-deformed[idx*2+1]};
    if (use_intrinsic)
    {
        SurfacePoint pi(mass_vertices[idx].second, bary);
        pos = toTV(pi.interpolate(geometry->vertexPositions));
    }
    else
    {
        SurfacePoint pi(mass_surface_points[idx].second, bary);
        pos = toTV(pi.interpolate(geometry->vertexPositions));
    }
}

void IntrinsicSimulation::updateVisualization(bool all_edges)
{
    // std::cout << edgeNetwork->paths.size() << std::endl;
    // std::cout << edgeNetwork->isMarkedVertex.size() << std::endl;
    // std::cout << edgeNetwork->tri->markedEdges.size() << std::endl;
    // std::cout << "update visualization" << std::endl;
    if (use_intrinsic)
    {
        all_intrinsic_edges.resize(0);
        gcs::EdgeData<std::vector<SurfacePoint>> tracedEdges(*edgeNetwork->tri->intrinsicMesh);

        if (all_edges)
        {
            gcs::EdgeData<std::vector<SurfacePoint>> traces = edgeNetwork->tri->traceAllIntrinsicEdgesAlongInput();
            for (gcEdge e : edgeNetwork->tri->mesh.edges()) 
            {
                std::vector<TV> loop;
                for (geometrycentral::surface::SurfacePoint& p : traces[e]) 
                {
                    Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
                    loop.push_back(TV(vtx.x, vtx.y, vtx.z));
                }
                for (int i = 0; i < loop.size()-1; i++)
                {
                    int j = (i + 1) % loop.size();
                    all_intrinsic_edges.push_back(std::make_pair(loop[i], loop[j]));
                }
            }
        }
        else
        {
            for (Edge eij : spring_edges) 
            {
                gcVertex vA = mass_vertices[eij[0]].first;
                gcVertex vB = mass_vertices[eij[1]].first;
                
                gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
                gcs::Halfedge he = e.halfedge();

                tracedEdges[e] = edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he);
                std::vector<TV> loop;
                for (gcs::SurfacePoint& p : tracedEdges[e]) 
                {
                    Vector3 vtx = p.interpolate(geometry->inputVertexPositions);
                    loop.push_back(TV(vtx.x, vtx.y, vtx.z));
                }
                for (int i = 0; i < loop.size()-1; i++)
                {
                    int j = (i + 1) % loop.size();
                    all_intrinsic_edges.push_back(std::make_pair(loop[i], loop[j]));
                }
            }  
        }
    }
    else
    {
        all_intrinsic_edges.resize(0);
        // for (const Edge& edge : spring_edges)
        // {
        //     SurfacePoint vA = mass_surface_points[edge[0]].first;
        //     SurfacePoint vB = mass_surface_points[edge[1]].first;
            
        //     T geo_dis = 0.0; std::vector<SurfacePoint> path;
        //     computeExactGeodesic(vA, vB, geo_dis, path, true);
        //     rest_length.push_back(geo_dis);
        //     for(int i = 0; i < path.size() - 1; i++)
        //     {
        //         all_intrinsic_edges.push_back(std::make_pair(
        //             toTV(path[i].interpolate(geometry->vertexPositions)),
        //             toTV(path[i+1].interpolate(geometry->vertexPositions))
        //         ));
        //     }
        // }
        int n_springs = spring_edges.size();
        std::vector<std::vector<std::pair<TV, TV>>> sub_pairs(n_springs, std::vector<std::pair<TV, TV>>());
        // START_TIMING(geodesic)
        
        // for(int i = 0; i < n_springs; i++)
        tbb::parallel_for(0, n_springs, [&](int i)
        {
            SurfacePoint vA = mass_surface_points[spring_edges[i][0]].first;
            SurfacePoint vB = mass_surface_points[spring_edges[i][1]].first;
            
            T geo_dis; std::vector<SurfacePoint> path;
            std::vector<IxnData> ixn_data;
            computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
            
            for(int j = 0; j < path.size() - 1; j++)
            {
                sub_pairs[i].push_back(std::make_pair(
                    toTV(path[j].interpolate(geometry->vertexPositions)),
                    toTV(path[j+1].interpolate(geometry->vertexPositions))
                ));
            }
        }
        );
        // FINISH_TIMING_PRINT(geodesic)
        // std::exit(0);
        for (int i = 0; i < n_springs; i++)
        {
            all_intrinsic_edges.insert(all_intrinsic_edges.end(), sub_pairs[i].begin(), sub_pairs[i].end());
        }
    }
}

bool IntrinsicSimulation::advanceOneStep(int step)
{
    // u = VectorXT::Zero(undeformed.rows());
    // u << 0, -0.1, 0.1, 0.1, 0, -0.1, 0, 0,0;
    // deformed = undeformed + u;
    // simDoFToPosition(deformed);
    // std::cout << "simDoFToPosition done" << std::endl;
    // updateVisualization();
    // std::cout << "advanceOneStep done" << std::endl;
    // return true;

    VectorXT residual(deformed.rows());
    residual.setZero();
    START_TIMING(computeResidual)
    T residual_norm = computeResidual(u, residual);
    FINISH_TIMING_PRINT(computeResidual)
    std::cout << "[NEWTON] iter " << step << "/" 
        << max_newton_iter << ": residual_norm " 
        << residual_norm << " tol: " << newton_tol << std::endl;
    
    if (residual_norm < newton_tol || step == max_newton_iter)
    {
        return true;
    }
    T dq_norm = 1e10;
    START_TIMING(lineSearchNewton)
    dq_norm = lineSearchNewton(u, residual);
    FINISH_TIMING_PRINT(lineSearchNewton)
    // u += residual;
    
    if(step == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
    {
        
        return true;
    }
    
    return false;
}

T IntrinsicSimulation::lineSearchNewton(VectorXT& _u,  VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    du = residual;

    if (use_Newton)
    {
        StiffnessMatrix K(residual.rows(), residual.rows());
        // Timer ti(true);
        buildSystemMatrix(_u, K);
        // // std::cout << "\tbuild system takes " <<  ti.elapsed_sec() << std::endl;
        bool success = linearSolve(K, residual, du);
        if (!success)
            return 1e16;
    }
    T norm = du.norm();
    
    T alpha = 1.0;
    T E0 = computeTotalEnergy(_u);
    // std::cout << std::setprecision(8) << "ls E0  " << E0 << std::endl;
    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        // std::cout << "u_ls: " << alpha * du.transpose() << std::endl;
        T E1 = computeTotalEnergy(u_ls);
        // std::cout << "ls final " << E1 << " #ls " << cnt << std::endl;
        if (E1 - E0 < 0 || cnt > 15)
        {
            // std::cout << "|du| " << alpha * du.norm() << std::endl;
            // std::cout << "ls final " << E1 << " #ls " << cnt << std::endl;
            if (cnt > 15)
            {
                // std::cout << "cnt > 15" << std::endl;
                return 1e16;
            }
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    return norm;
}


void IntrinsicSimulation::reset()
{
    deformed = undeformed;
    u.setZero();
}