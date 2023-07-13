#include "../include/IntrinsicSimulation.h"


void IntrinsicSimulation::addEdgeLengthEnergy(T w, T& energy)
{
    if (retrace)
        traceGeodesics();
    VectorXT energies = VectorXT::Zero(spring_edges.size());
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

        
        // int length = path.size();
        // TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        // TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

        // energies[i] += w_reg * (v1-v0).squaredNorm();
    }
    energy += energies.sum();
}

void IntrinsicSimulation::addEdgeLengthForceEntries(T w, VectorXT& residual)
{
    int n_springs = spring_edges.size();

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        T l = current_length[cnt];
        std::vector<SurfacePoint> path = paths[cnt];
        
        T l0 = rest_length[cnt];
        T coeff = 2.0 * w * (l - l0);

        Vector<T, 4> dldw;
        computeGeodesicLengthGradient(eij, dldw);

        addForceEntry<4>(residual, {eij[0], eij[1]}, -dldw * coeff);


        // Vector<T, 6> dldx_reg; dldx_reg.setZero();
        // dldx_reg.segment<3>(0) = 2.0 * w_reg * -(v1 - v0);   
        // dldx_reg.segment<3>(3) = -dldx_reg.segment<3>(0);
        // addForceEntry<4>(residual, {eij[0], eij[1]}, -dldx_reg.transpose() * dxdw);
        
        cnt++;
    }
    
}


void IntrinsicSimulation::computeGeodesicLengthGradient(const Edge& edge, Vector<T, 4>& dldw)
{
    dldw.setZero();
    int edge_idx = edge_map[edge];
    SurfacePoint vA = mass_surface_points[edge[0]].first;
    SurfacePoint vB = mass_surface_points[edge[1]].first;

    // std::cout << "va barycentric " << vA.faceCoords << std::endl;
    // std::cout << "vb barycentric " << vB.faceCoords << std::endl;
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().next().vertex()]);
    
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
        // std::cout << dldx0.transpose() << " " << dldx1.transpose() << std::endl;
        // std::getchar();

        if ((ixn0-v0).norm() < 1e-6)
            std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << edge[0] << " vj " << edge[1] << std::endl;
        if ((v1-ixn1).norm() < 1e-6)
            std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << edge[0] << " vj " << edge[1] << std::endl;
    }

    if (hasSmallSegment(path) && !run_diff_test)
    {
        // std::cout << "has small segment" << std::endl;
        // std::getchar();
    }

    Vector<T, 6> dldx; dldx.setZero();
    dldx.segment<3>(0) = dldx0;
    dldx.segment<3>(3) = dldx1;

    Matrix<T, 6, 4> dxdw; dxdw.setZero();
    dxdw.block(0, 0, 3, 1) = v10 - v12;
    dxdw.block(0, 1, 3, 1) = v11 - v12;

    dxdw.block(3, 2, 3, 1) = v20 - v22;
    dxdw.block(3, 3, 3, 1) = v21 - v22;

    dldw = dldx.transpose() * dxdw;
}

void IntrinsicSimulation::computeGeodesicLengthHessian(const Edge& edge, Matrix<T, 4, 4>& d2ldw2)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = mass_surface_points[edge[0]].first;
    SurfacePoint vB = mass_surface_points[edge[1]].first;

    // std::cout << "va barycentric " << vA.faceCoords << std::endl;
    // std::cout << "vb barycentric " << vB.faceCoords << std::endl;
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().next().vertex()]);

    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = v10 - v12;
    dx0dw0.col(1) = v11 - v12;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = v20 - v22;
    dx1dw1.col(1) = v21 - v22;

    TV2 dldw0 = dldx0.transpose() * dx0dw0;
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
        
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldx0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldx1 = -(ixn1 - v1).normalized();

        // if ((ixn0-v0).norm() < 1e-5)
        //     std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;
        // if ((v1-ixn1).norm() < 1e-5)
        //     std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;

        dldx.segment<3>(0) = dldx0;
        dldx.segment<3>(3) = dldx1;
        
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
            TV x_start = ixn_data_list[edge_idx][1+ixn_id].start;
            TV x_end = ixn_data_list[edge_idx][1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
        }
        
        TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        // if intersections are too close to the mass points,
        // then there could appear super large value in the element hessian
        // we use an approximated hessian in this case
        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6) && !run_diff_test)
            d2ldx2 = d2ldx2_approx;
        
        // if (hasSmallSegment(path) && !run_diff_test)
        // {
        //     std::cout << "has small segment" << std::endl;
        //     d2ldx2 = d2ldx2_approx;
        // }
        // Eigen::EigenSolver<MatrixXT> es(dfdx);
        // std::cout << es.eigenvalues().real().transpose() << std::endl;

    }
    d2ldw2 = dxdw.transpose() * d2ldx2 * dxdw;
}

void IntrinsicSimulation::computeGeodesicLengthGradientAndHessian(const Edge& edge, 
    Vector<T, 4>& dldw, Matrix<T, 4, 4>& d2ldw2)
{
    int edge_idx = edge_map[edge];
    SurfacePoint vA = mass_surface_points[edge[0]].first;
    SurfacePoint vB = mass_surface_points[edge[1]].first;

    // std::cout << "va barycentric " << vA.faceCoords << std::endl;
    // std::cout << "vb barycentric " << vB.faceCoords << std::endl;
    
    T l = current_length[edge_idx];
    std::vector<SurfacePoint> path = paths[edge_idx];
    
    int length = path.size();
    
    TV dldx0, dldx1;
    TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
    TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

    TV v10 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().vertex()]);
    TV v11 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().vertex()]);
    TV v12 = toTV(geometry->vertexPositions[mass_surface_points[edge[0]].second.halfedge().next().next().vertex()]);

    TV v20 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().vertex()]);
    TV v21 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().vertex()]);
    TV v22 = toTV(geometry->vertexPositions[mass_surface_points[edge[1]].second.halfedge().next().next().vertex()]);

    Matrix<T, 3, 2> dx0dw0;
    dx0dw0.col(0) = v10 - v12;
    dx0dw0.col(1) = v11 - v12;

    Matrix<T, 3, 2> dx1dw1;
    dx1dw1.col(0) = v20 - v22;
    dx1dw1.col(1) = v21 - v22;

    TV2 dldw0 = dldx0.transpose() * dx0dw0;
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
        
        TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        dldx0 = -(ixn0 - v0).normalized();

        TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        dldx1 = -(ixn1 - v1).normalized();

        // if ((ixn0-v0).norm() < 1e-5)
        //     std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;
        // if ((v1-ixn1).norm() < 1e-5)
        //     std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;

        dldx.segment<3>(0) = dldx0;
        dldx.segment<3>(3) = dldx1;
        
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
            TV x_start = ixn_data_list[edge_idx][1+ixn_id].start;
            TV x_end = ixn_data_list[edge_idx][1+ixn_id].end;
            dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
        }
        
        TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
        dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        d2gdc2.block(3, 3, 3, 3) += dlndxn;

        d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        // if intersections are too close to the mass points,
        // then there could appear super large value in the element hessian
        // we use an approximated hessian in this case
        if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6) && !run_diff_test)
            d2ldx2 = d2ldx2_approx;
        
        // if (hasSmallSegment(path) && !run_diff_test)
        // {
        //     std::cout << "has small segment" << std::endl;
        //     d2ldx2 = d2ldx2_approx;
        // }
        // Eigen::EigenSolver<MatrixXT> es(dfdx);
        // std::cout << es.eigenvalues().real().transpose() << std::endl;

    }
    dldw = dldx.transpose() * dxdw;
    d2ldw2 = dxdw.transpose() * d2ldx2 * dxdw;
}

void IntrinsicSimulation::addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries)
{
    int n_springs = spring_edges.size();

    int cnt = 0;
    for (const auto& eij : spring_edges)
    {
        T l = current_length[cnt];
        
        
        T l0 = rest_length[cnt];

        Vector<T, 4> dldw; Matrix<T, 4, 4> d2ldw2;
        computeGeodesicLengthGradientAndHessian(eij, dldw, d2ldw2);
        Matrix<T, 4, 4> hessian_reg = 
        2.0 * w * (dldw * dldw.transpose() + (l - l0) * d2ldw2);
        addHessianEntry<4>(entries, {eij[0], eij[1]}, hessian_reg);
        
        // SurfacePoint vA = mass_surface_points[eij[0]].first;
        // SurfacePoint vB = mass_surface_points[eij[1]].first;
        
        // // T l = 0.0; std::vector<SurfacePoint> path;
        // // computeExactGeodesic(vA, vB, l, path, true);
        
        // int length = path.size();
        // // std::cout << "===========================" << std::endl;
        // TV dldx0, dldx1;
        // TV v0 = toTV(path[0].interpolate(geometry->vertexPositions));
        // TV v1 = toTV(path[length - 1].interpolate(geometry->vertexPositions));

        // TV v10 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().vertex()]);
        // TV v11 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().vertex()]);
        // TV v12 = toTV(geometry->vertexPositions[mass_surface_points[eij[0]].second.halfedge().next().next().vertex()]);

        // Matrix<T, 3, 2> dx0dw0;
        // dx0dw0.col(0) = v10 - v12;
        // dx0dw0.col(1) = v11 - v12;

        // TV2 dldw0 = dldx0.transpose() * dx0dw0;

        // TV v20 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().vertex()]);
        // TV v21 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().vertex()]);
        // TV v22 = toTV(geometry->vertexPositions[mass_surface_points[eij[1]].second.halfedge().next().next().vertex()]);

        // Matrix<T, 3, 2> dx1dw1;
        // dx1dw1.col(0) = v20 - v22;
        // dx1dw1.col(1) = v21 - v22;

        // TV2 dldw1 = dldx1.transpose() * dx1dw1;
        // Matrix<T, 6, 4> dxdw; dxdw.setZero();
        // dxdw.block(0, 0, 3, 1) = v10 - v12;
        // dxdw.block(0, 1, 3, 1) = v11 - v12;

        // dxdw.block(3, 2, 3, 1) = v20 - v22;
        // dxdw.block(3, 3, 3, 1) = v21 - v22;
        
        // Vector<T, 6> dldx; dldx.setZero();
        
        // Matrix<T, 6, 6> d2ldx2; d2ldx2.setZero();

        // if (length == 2)
        // {
        //     dldx0 = -(v1 - v0).normalized();
        //     dldx1 = -dldx0;
        //     dldx.segment<3>(0) = dldx0;
        //     dldx.segment<3>(3) = dldx1;
        //     d2ldx2.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        //     d2ldx2.block(3, 3, 3, 3) = d2ldx2.block(0, 0, 3, 3);
        //     d2ldx2.block(3, 0, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
        //     d2ldx2.block(0, 3, 3, 3) = -d2ldx2.block(0, 0, 3, 3);
        // }
        // else
        // {
            
        //     TV ixn0 = toTV(path[1].interpolate(geometry->vertexPositions));
        //     dldx0 = -(ixn0 - v0).normalized();

        //     TV ixn1 = toTV(path[length - 2].interpolate(geometry->vertexPositions));
        //     dldx1 = -(ixn1 - v1).normalized();

        //     // if ((ixn0-v0).norm() < 1e-5)
        //     //     std::cout << "small edge : |x0 - c0| " << (ixn0-v0).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;
        //     // if ((v1-ixn1).norm() < 1e-5)
        //     //     std::cout << "small edge : |xn - c1| " << (v1-ixn1).norm() << " vi " << eij[0] << " vj " << eij[1] << std::endl;

        //     dldx.segment<3>(0) = dldx0;
        //     dldx.segment<3>(3) = dldx1;
            
        //     int ixn_dof = (length - 2) * 3;
        //     MatrixXT dfdc(ixn_dof, 6); dfdc.setZero();
        //     MatrixXT dfdx(ixn_dof, ixn_dof); dfdx.setZero();
        //     MatrixXT dxdt(ixn_dof, length-2); dxdt.setZero();
        //     MatrixXT d2gdcdx(ixn_dof, 6); d2gdcdx.setZero();

        //     TM dl0dx0 = (TM::Identity() - dldx0 * dldx0.transpose()) / (ixn0 - v0).norm();

        //     dfdx.block(0, 0, 3, 3) += dl0dx0;
        //     dfdc.block(0, 0, 3, 3) += -dl0dx0;
        //     d2gdcdx.block(0, 0, 3, 3) += -dl0dx0;
        //     for (int ixn_id = 0; ixn_id < length - 3; ixn_id++)
        //     {
        //         // std::cout << "inside" << std::endl;
        //         Matrix<T, 6, 6> hess;
        //         TV ixn_i = toTV(path[1 + ixn_id].interpolate(geometry->vertexPositions));
        //         TV ixn_j = toTV(path[1 + ixn_id+1].interpolate(geometry->vertexPositions));

        //         edgeLengthHessian(ixn_i, ixn_j, hess);
        //         dfdx.block(ixn_id*3, ixn_id * 3, 6, 6) += hess;
        //     }
        //     for (int ixn_id = 0; ixn_id < length - 2; ixn_id++)
        //     {
        //         TV x_start = ixn_data_list[cnt][1+ixn_id].start;
        //         TV x_end = ixn_data_list[cnt][1+ixn_id].end;
        //         dxdt.block(ixn_id * 3, ixn_id, 3, 1) = (x_end - x_start);
        //     }
            
        //     TM dlndxn = (TM::Identity() - dldx1 * dldx1.transpose()) / (ixn1 - v1).norm();
        //     dfdx.block(ixn_dof-3, ixn_dof-3, 3, 3) += dlndxn;
        //     dfdc.block(ixn_dof-3, 3, 3, 3) += -dlndxn;
        //     d2gdcdx.block(ixn_dof-3, 3, 3, 3) += -dlndxn;

        //     MatrixXT dxdtd2gdxdc = dxdt.transpose() * d2gdcdx;
        //     MatrixXT dxdtd2gdx2dxdt = dxdt.transpose() * dfdx * dxdt;
        //     MatrixXT dtdc = dxdtd2gdx2dxdt.colPivHouseholderQr().solve(-dxdtd2gdxdc);


        //     Matrix<T, 6, 6> d2gdc2; d2gdc2.setZero();
        //     d2gdc2.block(0, 0, 3, 3) += dl0dx0;
        //     d2gdc2.block(3, 3, 3, 3) += dlndxn;

        //     d2ldx2 = d2gdc2 + d2gdcdx.transpose() * dxdt * dtdc;

        //     Matrix<T, 6, 6> d2ldx2_approx; d2ldx2_approx.setZero();
        //     d2ldx2_approx.block(0, 0, 3, 3) = (TM::Identity() - dldx0 * dldx0.transpose()) / l;
        //     d2ldx2_approx.block(3, 3, 3, 3) = (TM::Identity() - dldx1 * dldx1.transpose()) / l;
        //     d2ldx2_approx.block(0, 3, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));
        //     d2ldx2_approx.block(3, 0, 3, 3) = -0.5 * (d2ldx2_approx.block(0, 0, 3, 3) + d2ldx2_approx.block(3, 3, 3, 3));

        //     // if intersections are too close to the mass points,
        //     // then there could appear super large value in the element hessian
        //     // we use an approximated hessian in this case
        //     // if ((dfdx.maxCoeff() > 1e6 || dfdx.minCoeff() < -1e6) && !run_diff_test)
        //     //     d2ldx2 = d2ldx2_approx;
            
        //     // if (hasSmallSegment(path) && !run_diff_test)
        //     // {
        //     //     std::cout << "has small segment" << std::endl;
        //     //     d2ldx2 = d2ldx2_approx;
        //     // }
        //     // Eigen::EigenSolver<MatrixXT> es(dfdx);
        //     // std::cout << es.eigenvalues().real().transpose() << std::endl;

        // }

        
        // Vector<T, 4> dldw = dldx.transpose() * dxdw;

        // Matrix<T, 4, 4> hessian_full = 
        //     2.0 * w * (dldw * dldw.transpose() + (l - l0) * (dxdw.transpose() * d2ldx2 * dxdw));
        // addHessianEntry<4>(entries, {eij[0], eij[1]}, hessian_full);

        // {
        //     Matrix<T, 6, 6> d2ldx2_reg; d2ldx2_reg.setZero();
        //     Vector<T, 6> dldx_reg; dldx_reg.setZero();
        //     dldx_reg.segment<3>(0) = -(v1 - v0).normalized();
        //     dldx_reg.segment<3>(3) = -dldx_reg.segment<3>(0);
        //     Vector<T, 4> dldw_reg = dldx_reg.transpose() * dxdw;
        //     d2ldx2_reg.block(0, 0, 3, 3) = (TM::Identity() - dldx_reg.segment<3>(0) * dldx_reg.segment<3>(0).transpose()) / (v1-v0).norm();
        //     d2ldx2_reg.block(3, 3, 3, 3) = d2ldx2_reg.block(0, 0, 3, 3);
        //     d2ldx2_reg.block(3, 0, 3, 3) = -d2ldx2_reg.block(0, 0, 3, 3);
        //     d2ldx2_reg.block(0, 3, 3, 3) = -d2ldx2_reg.block(0, 0, 3, 3);

        //     Matrix<T, 4, 4> hessian_reg = 
        //     2.0 * w_reg * (dldw_reg * dldw_reg.transpose() + (v1-v0).norm() * (dxdw.transpose() * d2ldx2_reg * dxdw));
        //         addHessianEntry<4>(entries, {eij[0], eij[1]}, hessian_reg);
        // }

        // VectorXT ev = computeHessianBlockEigenValues<4>(hessian_full);
        // std::cout << ev.transpose() << std::endl;
        
        cnt++;
    }
}
