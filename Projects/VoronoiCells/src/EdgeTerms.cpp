#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::addEdgeLengthEnergy(T w, T& energy)
{

    // int cnt = 0;
    VectorXT energies = VectorXT::Zero(spring_edges.size());
    tbb::parallel_for(0, (int)spring_edges.size(), [&](int i)
    {
        Edge eij = spring_edges[i];
        T l0 = rest_length[i];
        if (use_intrinsic)
        {
            gcVertex vA = mass_vertices[eij[0]].first;
            gcVertex vB = mass_vertices[eij[1]].first;

            gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);

            if (e.isDead())
            {
                std::cout << "non existing edge" << std::endl;
                return;
            }

            T l = edgeNetwork->tri->edgeLengths[e];
            energies[i] = w * (l - l0) * (l-l0);
        }
        else
        {
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            
            T geo_dis; std::vector<SurfacePoint> path;
			std::vector<IxnData> ixn_data;
            computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, false);
            energies[i] = w * (geo_dis - l0) * (geo_dis-l0);
        }
    });
    energy += energies.sum();
    // for (const auto& eij : spring_edges)
    // {
    //     T l0 = rest_length[cnt];
    //     if (use_intrinsic)
    //     {
    //         gcVertex vA = mass_vertices[eij[0]].first;
    //         gcVertex vB = mass_vertices[eij[1]].first;

    //         gcEdge e = edgeNetwork->tri->intrinsicMesh->connectingEdge(vA, vB);

    //         if (e.isDead())
    //         {
    //             std::cout << "non existing edge" << std::endl;
    //             continue;
    //         }

    //         T l = edgeNetwork->tri->edgeLengths[e];
    //         energy += w * (l - l0) * (l-l0);
    //     }
    //     else
    //     {
    //         SurfacePoint vA = mass_surface_points[eij[0]].first;
    //         SurfacePoint vB = mass_surface_points[eij[1]].first;
            
    //         T geo_dis; std::vector<SurfacePoint> path;
    //         computeExactGeodesic(vA, vB, geo_dis, path, false);
    //         energy += w * (geo_dis - l0) * (geo_dis-l0);
    //     }
    //     cnt++;
    // }
}

void IntrinsicSimulation::addEdgeLengthForceEntries(T w, VectorXT& residual)
{
    int n_springs = spring_edges.size();
    std::vector<T> current_length(n_springs);
    std::vector<std::vector<SurfacePoint>> paths(n_springs);
	std::vector<std::vector<IxnData>> ixn_data_list(n_springs);
    // START_TIMING(geodesic)
    // std::cout << "here" << std::endl;
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
	for(int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = mass_surface_points[spring_edges[i][0]].first;
        SurfacePoint vB = mass_surface_points[spring_edges[i][1]].first;
        
        T geo_dis; std::vector<SurfacePoint> path;
		std::vector<IxnData> ixn_data;
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        ixn_data_list[i] = ixn_data;
        paths[i] = path;
        current_length[i] = geo_dis;
    }
#ifdef PARALLEL_GEODESIC
    );
#endif
    // FINISH_TIMING_PRINT(geodesic)
    // std::exit(0);
    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        if (use_intrinsic)
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
            
            
            TV v0 = toTV(pts[0].interpolate(geometry->vertexPositions));
            TV ixn0 = toTV(pts[1].interpolate(geometry->vertexPositions));
            TV dldx0 = -(ixn0 - v0).normalized();

            TV ixn1 = toTV(pts[length - 2].interpolate(geometry->vertexPositions));
            TV v1 = toTV(pts[length - 1].interpolate(geometry->vertexPositions));
            TV dldx1 = -(ixn1 - v1).normalized();

            

            TV v10 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().vertex()]);
            TV v11 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().next().vertex()]);
            TV v12 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx0dw0;
            dx0dw0.col(0) = v10 - v12;
            dx0dw0.col(1) = v11 - v12;

            TV2 dldw0 = dldx0.transpose() * dx0dw0;

            TV v20 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().vertex()]);
            TV v21 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().next().vertex()]);
            TV v22 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx1dw1;
            dx1dw1.col(0) = v20 - v22;
            dx1dw1.col(1) = v21 - v22;

            TV2 dldw1 = dldx1.transpose() * dx1dw1;

            residual.segment<2>(eij[0]*2) += -dldw0 * coeff;
            residual.segment<2>(eij[1]*2) += -dldw1 * coeff;

        }
        else
        {
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            
            // T l = 0.0; std::vector<SurfacePoint> path;
            // computeExactGeodesic(vA, vB, l, path, true);
            T l = current_length[cnt];
            std::vector<SurfacePoint> path = paths[cnt];
            
            T l0 = rest_length[cnt];
            T coeff = 2.0 * w * (l - l0);
            
            int length = path.size();
            // std::cout << length << std::endl;
            // std::getchar();
            // std::cout << "===========================" << std::endl;
            TV dldx0, dldx1;
            TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));
            if (length == 2)
            {
                dldx0 = -(v1 - v0).normalized();
                dldx1 = -dldx0;
            }
            else
            {
                TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
                dldx0 = -(ixn0 - v0).normalized();

                TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                dldx1 = -(ixn1 - v1).normalized();
            }

            // std::cout << toTV(vA.interpolate(geometry->vertexPositions)).transpose() << std::endl;
            // std::cout << v0.transpose() << std::endl;
            // std::cout << ixn0.transpose() << std::endl;
            // std::cout << ixn1.transpose() << std::endl;
            // std::cout << v1.transpose() << std::endl;
            // std::cout << toTV(vB.interpolate(geometry->vertexPositions)).transpose() << std::endl;

            // std::getchar();
            TV v10 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().vertex()]);
            TV v11 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().vertex()]);
            TV v12 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx0dw0;
            dx0dw0.col(0) = v10 - v12;
            dx0dw0.col(1) = v11 - v12;

            TV2 dldw0 = dldx0.transpose() * dx0dw0;

            TV v20 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().vertex()]);
            TV v21 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().vertex()]);
            TV v22 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx1dw1;
            dx1dw1.col(0) = v20 - v22;
            dx1dw1.col(1) = v21 - v22;

            TV2 dldw1 = dldx1.transpose() * dx1dw1;

            residual.segment<2>(eij[0]*2) += -dldw0 * coeff;
            residual.segment<2>(eij[1]*2) += -dldw1 * coeff;
        }
        cnt++;
    }
}

void IntrinsicSimulation::addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries)
{
int n_springs = spring_edges.size();
    std::vector<T> current_length(n_springs);
    std::vector<std::vector<SurfacePoint>> paths(n_springs);
	std::vector<std::vector<IxnData>> ixn_data_list(n_springs);
    // START_TIMING(geodesic)
    // std::cout << "here" << std::endl;
#ifdef PARALLEL_GEODESIC
    tbb::parallel_for(0, n_springs, [&](int i)
#else
	for(int i = 0; i < n_springs; i++)
#endif
    {
        SurfacePoint vA = mass_surface_points[spring_edges[i][0]].first;
        SurfacePoint vB = mass_surface_points[spring_edges[i][1]].first;
        
        T geo_dis; std::vector<SurfacePoint> path;
		std::vector<IxnData> ixn_data;
        computeExactGeodesic(vA, vB, geo_dis, path, ixn_data, true);
        ixn_data_list[i] = ixn_data;
        paths[i] = path;
        current_length[i] = geo_dis;
    }
#ifdef PARALLEL_GEODESIC
    );
#endif

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        if (use_intrinsic)
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
            
            gcs::Halfedge he = e.halfedge();
            if (he.tailVertex() == vB)
                he = e.halfedge().twin();
            std::vector<SurfacePoint> pts = 
                edgeNetwork->tri->traceIntrinsicHalfedgeAlongInput(he, false);    
            int length = pts.size();
            TV v0 = toTV(pts[0].interpolate(geometry->vertexPositions));
            TV ixn0 = toTV(pts[1].interpolate(geometry->vertexPositions));
            TV dldx0 = -(ixn0 - v0).normalized();

            TV ixn1 = toTV(pts[length - 2].interpolate(geometry->vertexPositions));
            TV v1 = toTV(pts[length - 1].interpolate(geometry->vertexPositions));
            TV dldx1 = -(ixn1 - v1).normalized();
            

            TV v10 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().vertex()]);
            TV v11 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().next().vertex()]);
            TV v12 = toTV(geometry->vertexPositions[mass_vertices[eij[0]].second.halfedge().next().next().vertex()]);

            TV v20 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().vertex()]);
            TV v21 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().next().vertex()]);
            TV v22 = toTV(geometry->vertexPositions[mass_vertices[eij[1]].second.halfedge().next().next().vertex()]);
            

            Matrix<T, 6, 4> dxdw; dxdw.setZero();
            dxdw.block(0, 0, 3, 1) = v10 - v12;
            dxdw.block(0, 1, 3, 1) = v11 - v12;

            dxdw.block(3, 2, 3, 1) = v20 - v22;
            dxdw.block(3, 3, 3, 1) = v21 - v22;

            Vector<T, 6> dldx; dldx.setZero();
            dldx.segment<3>(0) = dldx0;
            dldx.segment<3>(3) = dldx1;

            Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();
            d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
            d2ldx2.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
            d2ldx2.block(0, 3, 3, 3) = -0.5 * (d2ldx2.block(0, 0, 3, 3) + d2ldx2.block(3, 3, 3, 3));
            d2ldx2.block(3, 0, 3, 3) = -0.5 * (d2ldx2.block(0, 0, 3, 3) + d2ldx2.block(3, 3, 3, 3));

            
            Vector<T, 4> dldw = dldx.transpose() * dxdw;

            Matrix<T, 4, 4> hessian_full = 
                2.0 * w * (dldw * dldw.transpose() + (l - l0) * (dxdw.transpose() * d2ldx2 * dxdw));
        
            addHessianEntry<4>(entries, {eij[0], eij[1]}, hessian_full);
        }
        else
        {
            SurfacePoint vA = mass_surface_points[eij[0]].first;
            SurfacePoint vB = mass_surface_points[eij[1]].first;
            
            // T l = 0.0; std::vector<SurfacePoint> path;
            // computeExactGeodesic(vA, vB, l, path, true);
            T l = current_length[cnt];
            std::vector<SurfacePoint> path = paths[cnt];
            
            T l0 = rest_length[cnt];
            T coeff = 2.0 * w * (l - l0);
            
            int length = path.size();
            // std::cout << "===========================" << std::endl;
            TV dldx0, dldx1;
            TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
            TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

            TV v10 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().vertex()]);
            TV v11 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().vertex()]);
            TV v12 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx0dw0;
            dx0dw0.col(0) = v10 - v12;
            dx0dw0.col(1) = v11 - v12;

            TV2 dldw0 = dldx0.transpose() * dx0dw0;

            TV v20 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().vertex()]);
            TV v21 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().vertex()]);
            TV v22 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().next().vertex()]);

            Matrix<T, 3, 2> dx1dw1;
            dx1dw1.col(0) = v20 - v22;
            dx1dw1.col(1) = v21 - v22;

            TV2 dldw1 = dldx1.transpose() * dx1dw1;
            Matrix<T, 6, 4> dxdw; dxdw.setZero();
            dxdw.block(0, 0, 3, 1) = v10 - v12;
            dxdw.block(0, 1, 3, 1) = v11 - v12;

            dxdw.block(3, 2, 3, 1) = v20 - v22;
            dxdw.block(3, 3, 3, 1) = v21 - v22;
            
            Vector<T, 6> dldx; dldx.setZero();
            
            Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();

            if (length == 2)
            {
                dldx0 = -(v1 - v0).normalized();
                dldx1 = -dldx0;
                dldx.segment<3>(0) = dldx0;
                dldx.segment<3>(3) = dldx1;
                d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
                d2ldx2.block(3, 3, 3, 3) = d2ldx2.block(0, 0, 3, 3);
                d2ldx2.block(3, 0, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
                d2ldx2.block(0, 3, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
            }
            else
            {
                if (length == 4)
                {

                    TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
                    dldx0 = -(ixn0 - v0).normalized();

                    TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
                    dldx1 = -(ixn1 - v1).normalized();

                    dldx.segment<3>(0) = dldx0;
                    dldx.segment<3>(3) = dldx1;

                    // old backup ==============================
                    // d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
                    // d2ldx2.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
                    
                    // d2ldx2.block(0, 3, 3, 3) = -0.5 * (d2ldx2.block(0, 0, 3, 3) + d2ldx2.block(3, 3, 3, 3));
                    // d2ldx2.block(3, 0, 3, 3) = -0.5 * (d2ldx2.block(0, 0, 3, 3) + d2ldx2.block(3, 3, 3, 3));
                    // old backup stop ===========================

                    // compute dxdc
                    // we know the derivative of x w.r.t. the intersection points are zero
                    // meaning that one cannot further change them to mimizie the geodesic.

                    // define dl/dintersection as f
                    
                    int ixn_dof = (length - 2) * 3;
                    MatrixXT dfdc(ixn_dof, 6); dfdc.setZero();
                    MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
                    MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
                    MatrixXT d2gdcdx(ixn_dof, 6); d2gdcdx.setZero();

                    TM dl0dx0 = (TM::Identity() - dldx0 * dldx0.transpose()) / (ixn0 - v0).norm();

                    dfdx.block(0, 0, 3, 3) += dl0dx0;
                    dfdc.block(0, 0, 3, 3) += -dl0dx0;
                    d2gdcdx.block(0, 0, 3, 3) += -dl0dx0;
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
                        TV x_start = ixn_data_list[cnt][1+ixn_id].start;
						TV x_end = ixn_data_list[cnt][1+ixn_id].end;
                        dxdt.block(ixn_id * 3, ixn_id, 3, 1) = x_end - x_start;
                    }
                    std::cout << "dxdt" << std::endl;
					std::cout << dxdt << std::endl;
					std::cout << "---------" << std::endl;
                    TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
                    dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
                    dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
                    d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

                    MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
                    MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
					std::cout << "dxdtd2gdx2dxdt" << std::endl;
					std::cout << dxdtd2gdx2dxdt << std::endl;
					std::cout << "dxdtd2gdxdc" << std::endl;
					std::cout << dxdtd2gdxdc << std::endl;
                    // MatrixXT dxdc = dfdx.colPivHouseholderQr().solve(-dfdc);

                    MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);

                    std::cout << (dxdtd2gdx2dxdt * dtdc + dxdtd2gdxdc).norm() << std::endl;
					std::cout << "dtdc" << std::endl;
					std::cout << dtdc << std::endl;
                    std::getchar();

                    Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
                    d2gdc2.block(0, 0, 3, 3) += dl0dx0;
                    d2gdc2.block(3, 3, 3, 3) += dlndxn;


                    // d2ldx2 = d2gdc2 + 2.0 * dxdc.transpose() * d2gdcdx + dxdc.transpose() * dfdx * dxdc;
                    // d2ldx2 = d2gdc2 + dxdc.transpose() * d2gdcdx;
                    d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;
                    // std::cout << dxdc.transpose() * dfdx * dxdc + dxdc.transpose() * d2gdcdx << std::endl;

                    Matrix<T, 6, 6> d2ldx2_old; d2ldx2_old.setZero();
                    d2ldx2_old.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
                    d2ldx2_old.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
                    d2ldx2_old.block(0, 3, 3, 3) = -0.5 * (d2ldx2_old.block(0, 0, 3, 3) + d2ldx2_old.block(3, 3, 3, 3));
                    d2ldx2_old.block(3, 0, 3, 3) = -0.5 * (d2ldx2_old.block(0, 0, 3, 3) + d2ldx2_old.block(3, 3, 3, 3));
                    // Eigen::EigenSolver<MatrixXT> es(dfdx);
                    // std::cout << es.eigenvalues() << std::endl;

                    // std::cout << computeHessianBlockEigenValues<6>(dfdx) << std::endl;

                    std::cout << "d2gdc2" << std::endl;
                    std::cout << d2ldx2 << std::endl;
                    std::cout << "------------------" << std::endl;
                    std::cout << d2ldx2_old << std::endl;
                    std::getchar();

                    // d2ldx2 = d2ldx2_old;

                    // Matrix<T, 3, 6> dfdc_ad;
                    // Matrix<T, 3, 3> dfdx_ad;
                    // Matrix<T, 6, 6> d2gdc2_ad;
                    // Matrix<T, 6, 3> d2gdcdx_ad;
                    // geodesic(v0, ixn0, v1, dfdc_ad, dfdx_ad, d2gdc2_ad, d2gdcdx_ad);

                    // TV ixn2 = toTV(path[2].interpolate(geometry->vertexPositions));
                    // Matrix<T, 9, 6> dfdc_ad;
                    // Matrix<T, 9, 9> dfdx_ad;
                    // Matrix<T, 6, 6> d2gdc2_ad;
                    // Matrix<T, 6, 9> d2gdcdx_ad;
                    // geodesic5points(v0, ixn0, ixn2, ixn1, v1, dfdc_ad, dfdx_ad, d2gdc2_ad, d2gdcdx_ad);

                    // Matrix<T, 6, 6> dfdc_ad;
                    // Matrix<T, 6, 6> dfdx_ad;
                    // Matrix<T, 6, 6> d2gdc2_ad;
                    // Matrix<T, 6, 6> d2gdcdx_ad;
                    // Matrix<T, 6, 6> d2gdxdc_ad;
                    // geodesic4points(v0, ixn0, ixn1, v1, dfdc_ad, dfdx_ad, d2gdc2_ad, d2gdcdx_ad, d2gdxdc_ad);
                    // std::cout << "dfdc" << std::endl;
                    // std::cout << dfdc << std::endl;
                    // std::cout << "------------------" << std::endl;
                    // std::cout << dfdc_ad << std::endl;

                    // std::cout << "dfdx" << std::endl;
                    // std::cout << dfdx << std::endl;
                    // std::cout << "------------------" << std::endl;
                    // std::cout << dfdx_ad << std::endl;

                    // std::cout << "d2gdc2" << std::endl;
                    // std::cout << d2gdc2 << std::endl;
                    // std::cout << "------------------" << std::endl;
                    // std::cout << d2gdc2_ad << std::endl;

                    // std::cout << "d2gdcdx" << std::endl;
                    // std::cout << d2gdcdx << std::endl;
                    // std::cout << "------------------" << std::endl;
                    // std::cout << d2gdcdx_ad.transpose() << std::endl;
                    // std::cout << "------------------" << std::endl;
                    // std::cout << d2gdxdc_ad << std::endl;

                    // std::cout << length << std::endl;
                    
                }
            }

            // std::cout << computeHessianBlockEigenValues<6>(d2ldx2) << std::endl;
            // std::getchar();

            
            Vector<T, 4> dldw = dldx.transpose() * dxdw;

            Matrix<T, 4, 4> hessian_full = 
                2.0 * w * (dldw * dldw.transpose() + (l - l0) * (dxdw.transpose() * d2ldx2 * dxdw));
            addHessianEntry<4>(entries, {eij[0], eij[1]}, hessian_full);
        }
        cnt++;
    }
}
