#include <igl/readOBJ.h>
#include "../include/IntrinsicSimulation.h"

void IntrinsicSimulation::initializeMassPointScene()
{
    // using namespace geometrycentral;
    // using namespace gc::surface;

    MatrixXT V; MatrixXi F;
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/VoronoiCells/data/sphere.obj", 
        V, F);

    iglMatrixFatten<T, 3>(V, extrinsic_vertices);
    iglMatrixFatten<int, 3>(F, extrinsic_indices);

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
    
    auto lvals = gcs::makeManifoldSurfaceMeshAndGeometry(mesh_indices_gc, mesh_vertices_gc);
    std::tie(mesh, geometry) = std::tuple<std::unique_ptr<gcs::ManifoldSurfaceMesh>,
                    std::unique_ptr<gcs::VertexPositionGeometry>>(std::move(std::get<0>(lvals)),  // mesh
                                                             std::move(std::get<1>(lvals))); // geometry
    edgeNetwork = std::unique_ptr<gcs::FlipEdgeNetwork>(new gcs::FlipEdgeNetwork(*mesh, *geometry, {}));
    edgeNetwork->posGeom = geometry.get();

    // intrinsic_vertices_undeformed.resize(3 * 3);
    intrinsic_vertices_barycentric_coords.resize(3 * 2);
    std::vector<FacePoint> face_points;
    // face 55 36 39
    int cnt = 0;
    for (int face_idx : {55, 36, 49})
    {
        T alpha = 0.2, beta = 0.5, gamma = 1.0 - alpha - beta;
        TV vi = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 0] * 3);
        TV vj = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 1] * 3);
        TV vk = extrinsic_vertices.segment<3>(extrinsic_indices[face_idx * 3 + 2] * 3);
        // intrinsic_vertices_undeformed.segment<3>(cnt * 3) = vi * alpha + vj * beta + vk * gamma;
        intrinsic_vertices_barycentric_coords.segment<2>(cnt * 2) = TV2(alpha, beta);
        face_points.push_back(std::make_pair(face_idx, TV(alpha, beta, gamma)));
        cnt++;
    }
    undeformed = intrinsic_vertices_barycentric_coords;
    deformed = undeformed;
    u.resize(undeformed.rows()); u.setZero();

    for (const FacePoint& pt : face_points)
    {
        Vector3 bary{pt.second[0], 
                    pt.second[1], 
                    pt.second[2]};
        // this point is on the extrinsic mesh
        gcs::Face f = edgeNetwork->tri->inputMesh.face(pt.first);
        SurfacePoint new_pt(f, bary);
        // std::cout << "mass point position"<<std::endl;
        // printVec3(new_pt.interpolate(geometry->vertexPositions));
        // std::cout << "==================="<<std::endl;
        SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(new_pt);
        gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
        mass_vertices.push_back(std::make_pair(new_vtx, f));
    }

    // fix barycentric coordinates of the first point
    for (int i = 4; i < 6; i++)
    {
        dirichlet_data[i] = 0.0;
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
        edgeNetwork->iterativeShorten(iterLim, lengthLim);
        gcEdge ei = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        spring_edges.push_back(Edge(i, i+1));
        rest_length.push_back(edgeNetwork->tri->edgeLengths[ei] * 0.9);
        edgeNetwork->isMarkedVertex.setDefault(false);
        edgeNetwork->paths.clear();
        edgeNetwork->tri->clearMarkedEdges();
    }

    


    all_intrinsic_edges.resize(0);
    gcs::EdgeData<std::vector<SurfacePoint>> tracedEdges(edgeNetwork->tri->mesh);

    for (Edge eij : spring_edges) {
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

bool IntrinsicSimulation::simDoFToPosition(const VectorXT& sim_dof)
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
            // update extrinsic face
            SurfacePoint endpoint = result.endPoint.inSomeFace();
            mass_vertices[i].second = endpoint.face;
        }
        else 
        {
            SurfacePoint new_pt = SurfacePoint(fi, start_bc);
            SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(new_pt);
            gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
            mass_vertices[i].first = new_vtx;
        }    
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

    // for (int i = 0; i < mass_vertices.size(); i++)
    // {
    //     gcVertex vi = mass_vertices[i].first;
    //     gcFace fi = mass_vertices[i].second;
    //     Vector3 start_bc{undeformed[i * 2 + 0], undeformed[i*2+1], 1.0 - undeformed[i * 2 + 0] - undeformed[i * 2 + 1]};
    //     Vector3 target{sim_dof[i*2+0],sim_dof[i*2+1],1.0-sim_dof[i*2+0]-sim_dof[i*2+1]};

    //     Vector3 trace_vec = target - start_bc;
        
    //     gcs::TraceOptions options; 
    //     options.includePath = true;
        
    //     // trace geodesic on the extrinsic mesh
    //     gcs::TraceGeodesicResult result = gcs::traceGeodesic(*geometry, fi, start_bc, trace_vec, options);
        
        
    //     if (result.pathPoints.size() != 1)
    //     {
    //         // remove intrinsic vertex
    //         gcFace f = edgeNetwork->tri->removeInsertedVertex(vi);
    //         // find equivalent intrinc vertex
    //         SurfacePoint new_pt_intrinsic = edgeNetwork->tri->equivalentPointOnIntrinsic(result.endPoint);
    //         gcVertex new_vtx = edgeNetwork->tri->insertVertex(new_pt_intrinsic);
            
    //         mass_vertices[i].first = new_vtx;
    //         // update extrinsic face
    //         SurfacePoint endpoint = result.endPoint.inSomeFace();
    //         mass_vertices[i].second = endpoint.face;
    //     }
    //     else 
    //     {
    //     }

        
    // }
    
    // edgeNetwork->tri->refreshQuantities();
    // for (int i = 0; i < mass_vertices.size() - 1; i++)
    // {
    //     gcVertex vA = mass_vertices[i].first;
    //     gcVertex vB = mass_vertices[i + 1].first;
        
    //     std::vector<gcs::Halfedge> path = shortestEdgePath(*edgeNetwork->tri, vA, vB);
    //     edgeNetwork->addPath(path);
    //     edgeNetwork->nFlips = 0;
    //     edgeNetwork->nShortenIters = 0;
    //     edgeNetwork->EPS_ANGLE = 1e-5;
    //     edgeNetwork->straightenAroundMarkedVertices = true;
    //     size_t iterLim = gc::INVALID_IND;
    //     double lengthLim = 0.;
    //     edgeNetwork->addAllWedgesToAngleQueue();
    //     std::cout << "empty " << edgeNetwork->wedgeAngleQueue.empty() << std::endl;
    //     edgeNetwork->iterativeShorten(iterLim, lengthLim);
    //     gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
    //     if (e == gcEdge())
    //     {
    //         std::cout << "non-existing edge connecting vtx " << vA.getIndex() << " and " << vB.getIndex()<< std::endl;
    //         edgeNetwork->isMarkedVertex.setDefault(false);
    //         edgeNetwork->paths.clear();
    //         edgeNetwork->tri->clearMarkedEdges();
    //         return false;
    //     }
    //     edgeNetwork->isMarkedVertex.setDefault(false);
    //     edgeNetwork->paths.clear();
    //     edgeNetwork->tri->clearMarkedEdges();
    // }
    
    // return true;
    
}

void IntrinsicSimulation::addEdgeLengthEnergy(T w, T& energy)
{

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        gcVertex vA = mass_vertices[eij[0]].first;
        gcVertex vB = mass_vertices[eij[1]].first;

        gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);

        if (e.isDead())
        {
            std::cout << "non existing edge" << std::endl;
            continue;
        }

        T l = edgeNetwork->tri->edgeLengths[e];
        T l0 = rest_length[cnt];
        energy += w * (l - l0) * (l-l0);
        cnt++;
    }
}

void IntrinsicSimulation::addEdgeLengthForceEntries(T w, VectorXT& residual)
{
    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        gcVertex vA = mass_vertices[eij[0]].first;
        gcVertex vB = mass_vertices[eij[1]].first;

        gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        if (e.isDead())
        {
            std::cout << "non existing edge" << std::endl;
            continue;
        }
        T l = edgeNetwork->tri->edgeLengths[e];
        T l0 = rest_length[cnt];
        T coeff = 2.0 * w * (l - l0);
        
        gcs::Halfedge he = e.halfedge();
        if (he.tailVertex() == vB)
            he = e.halfedge().twin();
        std::vector<SurfacePoint> pts = 
            edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);    
        int length = pts.size();
        // std::cout << "===========================" << std::endl;

        
        // std::cout << "-----" << std::endl;
        // printVec3(pts[0].interpolate(geometry->vertexPositions));
        // printVec3(pts[length-1].interpolate(geometry->vertexPositions));
        // std::cout << "-----" << std::endl;
        // std::getchar();

        TV dldx0 = -toTV(pts[1].interpolate(geometry->vertexPositions) - 
            pts[0].interpolate(geometry->vertexPositions)).normalized();
        TV dldx1 = -toTV(pts[length - 2].interpolate(geometry->vertexPositions) - 
            pts[length - 1].interpolate(geometry->vertexPositions)).normalized();
        
        gcVertex v10 = mass_vertices[eij[0]].second.halfedge().vertex();
        T dldalpha1 = dldx0.dot(toTV(geometry->vertexPositions[v10])) * coeff;
        gcVertex v11 = mass_vertices[eij[0]].second.halfedge().next().vertex();
        T dldbeta1 = dldx0.dot(toTV(geometry->vertexPositions[v11])) * coeff;
        gcVertex v12 = mass_vertices[eij[0]].second.halfedge().next().next().vertex();
        T dldgamma1 = dldx0.dot(toTV(geometry->vertexPositions[v12])) * coeff;
    
        // TV pt0 = toTV(geometry->inputVertexPositions[v10] * undeformed[eij[0] * 2 + 0] + 
        // geometry->inputVertexPositions[v11] * undeformed[eij[0] * 2 + 1] + 
        // geometry->inputVertexPositions[v12] * (1.0-undeformed[eij[0] * 2 + 0]-undeformed[eij[0] * 2 + 1]));
        // std::cout << pt0.transpose() << std::endl;

        gcVertex v20 = mass_vertices[eij[1]].second.halfedge().vertex();
        T dldalpha2 = dldx1.dot(toTV(geometry->vertexPositions[v20])) * coeff;
        gcVertex v21 = mass_vertices[eij[1]].second.halfedge().next().vertex();
        T dldbeta2 = dldx1.dot(toTV(geometry->vertexPositions[v21])) * coeff;
        gcVertex v22 = mass_vertices[eij[1]].second.halfedge().next().next().vertex();
        T dldgamma2 = dldx1.dot(toTV(geometry->vertexPositions[v22])) * coeff;

        // TV pt1 = toTV(geometry->inputVertexPositions[v20] * undeformed[eij[1] * 2 + 0] + 
        // geometry->inputVertexPositions[v21] * undeformed[eij[1] * 2 + 1] + 
        // geometry->inputVertexPositions[v22] * (1.0-undeformed[eij[1] * 2 + 0]-undeformed[eij[1] * 2 + 1]));
        // std::cout << pt1.transpose() << std::endl;
        
        residual[eij[0] * 2 + 0] += -(dldalpha1-dldgamma1);
        residual[eij[0] * 2 + 1] += -(dldbeta1-dldgamma1);

        residual[eij[1] * 2 + 0] += -(dldalpha2-dldgamma2);
        residual[eij[1] * 2 + 1] += -(dldbeta2-dldgamma2);
        
        cnt++;
    }
}

void IntrinsicSimulation::addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries)
{
    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        gcVertex vA = mass_vertices[eij[0]].first;
        gcVertex vB = mass_vertices[eij[1]].first;

        gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);
        if (e.isDead())
        {
            std::cout << "non existing edge" << std::endl;
            continue;
        }
        T l = edgeNetwork->tri->edgeLengths[e];
        T l0 = rest_length[cnt];
        T coeff = 2.0 * w;
        
        gcs::Halfedge he = e.halfedge();
        if (he.tailVertex() == vB)
            he = e.halfedge().twin();
        std::vector<SurfacePoint> pts = 
            edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);    
        int length = pts.size();

        TV dldx0 = -toTV(pts[1].interpolate(geometry->vertexPositions) - 
            pts[0].interpolate(geometry->vertexPositions)).normalized();
        TV dldx1 = -toTV(pts[length - 2].interpolate(geometry->vertexPositions) - 
            pts[length - 1].interpolate(geometry->vertexPositions)).normalized();
        
        gcVertex v10 = mass_vertices[eij[0]].second.halfedge().vertex();
        T dldalpha1 = dldx0.dot(toTV(geometry->vertexPositions[v10]));
        gcVertex v11 = mass_vertices[eij[0]].second.halfedge().next().vertex();
        T dldbeta1 = dldx0.dot(toTV(geometry->vertexPositions[v11]));
        gcVertex v12 = mass_vertices[eij[0]].second.halfedge().next().next().vertex();
        T dldgamma1 = dldx0.dot(toTV(geometry->vertexPositions[v12]));

        gcVertex v20 = mass_vertices[eij[1]].second.halfedge().vertex();
        T dldalpha2 = dldx1.dot(toTV(geometry->vertexPositions[v20]));
        gcVertex v21 = mass_vertices[eij[1]].second.halfedge().next().vertex();
        T dldbeta2 = dldx1.dot(toTV(geometry->vertexPositions[v21]));
        gcVertex v22 = mass_vertices[eij[1]].second.halfedge().next().next().vertex();
        T dldgamma2 = dldx1.dot(toTV(geometry->vertexPositions[v22]));

        TV2 dldbary0(dldalpha1-dldgamma1, dldbeta1-dldgamma1);
        TV2 dldbary2(dldalpha2-dldgamma2, dldbeta2-dldgamma2);

        
        cnt++;
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
    simDoFToPosition(deformed);
    bool update_succeed = simDoFToPosition(deformed);
    if (!update_succeed)
        return residual.norm();
    addEdgeLengthForceEntries(we, residual);

    return residual.norm();
}

void IntrinsicSimulation::updateVisualization(bool all_edges)
{
    std::cout << edgeNetwork->paths.size() << std::endl;
    std::cout << edgeNetwork->isMarkedVertex.size() << std::endl;
    std::cout << edgeNetwork->tri->markedEdges.size() << std::endl;
    // std::cout << "update visualization" << std::endl;
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

void IntrinsicSimulation::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    vectorToIGLMatrix<int, 3>(extrinsic_indices, F);
    vectorToIGLMatrix<T, 3>(extrinsic_vertices, V);
    C.resize(F.rows(), 3);
    C.col(0).setZero(); C.col(1).setConstant(0.3); C.col(2).setOnes();
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

    T residual_norm = computeResidual(u, residual);
    std::cout << "[NEWTON] iter " << step << "/" 
        << max_newton_iter << ": residual_norm " 
        << residual_norm << " tol: " << newton_tol << std::endl;
    
    if (residual_norm < newton_tol || step == max_newton_iter)
    {
        return true;
    }
    T dq_norm = 1e10;
    // dq_norm = lineSearchNewton(u, residual);
    u += residual;
    

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
    // StiffnessMatrix K(residual.rows(), residual.rows());
    // Timer ti(true);
    // buildSystemMatrix(_u, K);
    // // std::cout << "\tbuild system takes " <<  ti.elapsed_sec() << std::endl;
    // bool success = linearSolve(K, residual, du);
    // if (!success)
    //     return 1e16;
    // T norm = du.norm();
    while (du.norm() > 10)
        du *= 0.1;
    T alpha = 1.0;
    T E0 = computeTotalEnergy(_u);
    std::cout << "ls E0  " << E0 << std::endl;
    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        std::cout << "u: " << u_ls.transpose() << std::endl;
        T E1 = computeTotalEnergy(u_ls);
        std::cout << "ls E1 " << E1 << std::endl;
        if (E1 - E0 < 0 || cnt > 20)
        {
            // if (cnt > 15)
            //     std::cout << "cnt > 15" << std::endl;
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    return 1.0;
}