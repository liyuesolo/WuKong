#include "../../include/Tessellation/SmoothVoronoi.h"
#include "../../include/Tessellation/CellFunction.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

TV3 SmoothVoronoi::getWeightDerivatives(double t) {
    double k = 10;

    TV3 w = TV3::Zero();

    {
        double t1 = t * t;
        w(0) = exp(0.1e1 / (t1 - 0.1e1) * k + k);
    }

    {
        double t1 = t * t;
        double t2 = t1 - 0.1e1;
        double t3 = t2 * t2;
        double t9 = exp(0.1e1 / t2 * k + k);
        w(1) = -0.2e1 * t9 * t / t3 * k;
    }

    {
        double t1 = t * t;
        double t2 = t1 - 0.1e1;
        double t3 = t2 * t2;
        double t10 = exp(0.1e1 / t2 * k + k);
        double t11 = t10 * t1;
        double t18 = k * k;
        double t19 = t3 * t3;
        w(2) = 0.8e1 * t11 / t2 / t3 * k - 0.2e1 * t10 / t3 * k + 0.4e1 * t11 / t19 * t18;
    }

    return w;
}

void SmoothVoronoi::getNodeWrapper(Node &node, NodePosition &nodePos) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;

    if (node.gen[1] < n_vtx &&
        node.gen[2] < n_vtx) { // Only handle standard voronoi node cases for now, no boundary interaction smoothing...
        VectorXT v0 = c.segment(node.gen[0] * dims, dims);
        VectorXT v1 = c.segment(node.gen[1] * dims, dims);
        VectorXT v2 = c.segment(node.gen[2] * dims, dims);
        getStandardNode(v0, v1, v2, nodePos.pos);

        double R = (v0 - nodePos.pos.head(dims)).norm();

        for (int i = 0; i < n_vtx; i++) {
            if (i == node.gen[0] || i == node.gen[1] || i == node.gen[2]) { continue; }
            else {
                VectorXT v = c.segment(i * dims, dims);
                double d = ((v - nodePos.pos.head(dims)).norm() - R) / d0;
                assert(d >= 0);

                if (d < 1) {
                    node.gen.push_back(i);
                }
            }
        }

        if (node.gen.size() == 3) {
            // No nearly-encroaching sites, no smoothing needed
            Tessellation::getNodeWrapper(node, nodePos);
        } else {
            node.type = NodeType::LSQ;
            node.gen_nc = node.gen.size();
            getNodeLSQ(node, nodePos);
        }

    } else {
        Tessellation::getNodeWrapper(node, nodePos);
    }
}

void SmoothVoronoi::getNodeLSQ(Node &node, NodePosition &nodePos) {
    int dims = 2 + getNumVertexParams();
    int n = node.gen.size();
    int nq = dims, nx = 3, nu = 3 * n, nc = dims * n, nt = n - 3;

    SmoothVoronoiChainRuleStruct A;
    getChainRuleMatrices(node, A);

//     TEST INPUTS TO COMPARE WITH MAPLE RESULT
//    n = 4;
//    node.gen[0] = 0;
//    node.gen[1] = 1;
//    node.gen[2] = 2;
//    node.gen[3] = 3;
//    c.segment<2>(2 * 0) = TV(0.04, 0.05);
//    c.segment<2>(2 * 1) = TV(-0.02, 1);
//    c.segment<2>(2 * 2) = TV(1.1, 1.2);
//    c.segment<2>(2 * 3) = TV(0.9, 0.1);
//    node.genw[0] = 1;
//    node.genw[1] = 1;
//    node.genw[2] = 1;
//    node.genw[3] = 0.5;

    MatrixXT A_LSQ(n, 3);
    VectorXT b_LSQ(n);

    for (int i = 0; i < n; i++) {
        VectorXT vi = A.u.segment(i * 3, 3);
        double xi = vi(0);
        double yi = vi(1);
        double sqrtwi = sqrt(vi(2));
        A_LSQ.row(i) = TV3(2 * xi * sqrtwi, 2 * yi * sqrtwi, -sqrtwi);
        b_LSQ(i) = sqrtwi * (xi * xi + yi * yi);
    }

    VectorXT x_LSQ = (A_LSQ.transpose() * A_LSQ).fullPivHouseholderQr().solve(A_LSQ.transpose() * b_LSQ);
    nodePos.pos = x_LSQ;
    double xc = x_LSQ(0);
    double yc = x_LSQ(1);
    double C = x_LSQ(2);

    if (false) {
        nodePos.grad = MatrixXT::Zero(3, nc);
        nodePos.hess.resize(CellFunction::nx);
        for (int i = 0; i < CellFunction::nx; i++) {
            nodePos.hess[i] = MatrixXT::Zero(nc, nc);
        }
        return;
    }

//    std::cout << "LSQ solution: " << x_LSQ << std::endl;

    MatrixXT dgdx = MatrixXT::Zero(nx, nx);
    MatrixXT dgdu = MatrixXT::Zero(nx, nu);
    Eigen::Tensor<double, 3> d2gdxdu(nx, nx, nu);
    d2gdxdu.setZero();
    Eigen::Tensor<double, 3> d2gdu2(nx, nu, nu);
    d2gdu2.setZero();
    for (int i = 0; i < n; i++) {
        VectorXT vi = A.u.segment(i * 3, 3);
        double xi = vi(0);
        double yi = vi(1);
        double wi = vi(2);

        dgdx(0, 0) += 8 * wi * xi * xi;
        dgdx(0, 1) += 8 * wi * xi * yi;
        dgdx(0, 2) += -4 * wi * xi;
        dgdx(1, 0) += 8 * wi * xi * yi;
        dgdx(1, 1) += 8 * wi * yi * yi;
        dgdx(1, 2) += -4 * wi * yi;
        dgdx(2, 0) += -4 * wi * xi;
        dgdx(2, 1) += -4 * wi * yi;
        dgdx(2, 2) += 2 * wi;

        dgdu(0, 3 * i + 0) = -4 * wi * (3 * xi * xi - 4 * xi * xc + yi * yi - 2 * yi * yc + C);
        dgdu(0, 3 * i + 1) = -8 * wi * (yi - yc) * xi;
        dgdu(0, 3 * i + 2) = -4 * (xi * xi - 2 * xi * xc + yi * yi - 2 * yi * yc + C) * xi;
        dgdu(1, 3 * i + 0) = -8 * wi * (xi - xc) * yi;
        dgdu(1, 3 * i + 1) = -4 * wi * (xi * xi - 2 * xi * xc + 3 * yi * yi - 4 * yi * yc + C);
        dgdu(1, 3 * i + 2) = -4 * (xi * xi - 2 * xi * xc + yi * yi - 2 * yi * yc + C) * yi;
        dgdu(2, 3 * i + 0) = 4 * wi * (xi - xc);
        dgdu(2, 3 * i + 1) = 4 * wi * (yi - yc);
        dgdu(2, 3 * i + 2) = 2 * (xi * xi - 2 * xi * xc + yi * yi - 2 * yi * yc + C);

        MatrixXT d2gdxdu_xi(3, 3);
        d2gdxdu_xi << 16 * wi * xi, 8 * wi * yi, -4 * wi,
                8 * wi * yi, 0, 0,
                -4 * wi, 0, 0;
        MatrixXT d2gdxdu_yi(3, 3);
        d2gdxdu_yi << 0, 8 * wi * xi, 0,
                8 * wi * xi, 16 * wi * yi, -4 * wi,
                0, -4 * wi, 0;
        MatrixXT d2gdxdu_wi(3, 3);
        d2gdxdu_wi << 8 * xi * xi, 8 * xi * yi, -4 * xi,
                8 * xi * yi, 8 * yi * yi, -4 * yi,
                -4 * xi, -4 * yi, 2;

        MatrixXT d2gdu2_xi(3, 3);
        d2gdu2_xi << -4 * (6 * xi - 4 * xc) * wi, -8 * wi * (yi - yc), -4 * (3 * xi * xi - 4 * xi * xc + yi * yi -
                                                                             2 * yi * yc + C),
                -8 * wi * yi, -8 * wi * (xi - xc), -8 * (xi - xc) * yi,
                4 * wi, 0, 4 * (xi - xc);
        MatrixXT d2gdu2_yi(3, 3);
        d2gdu2_yi << -8 * wi * (yi - yc), -8 * wi * xi, -8 * (yi - yc) * xi,
                -8 * wi * (xi - xc), -4 * wi * (6 * yi - 4 * yc), -4 *
                                                                  (xi * xi - 2 * xi * xc + 3 * yi * yi - 4 * yi * yc +
                                                                   C),
                0, 4 * wi, 4 * (yi - yc);
        MatrixXT d2gdu2_wi(3, 3);
        d2gdu2_wi << -4 * (3 * xi * xi - 4 * xi * xc + yi * yi - 2 * yi * yc + C), -8 * (yi - yc) * xi, 0,
                -8 * (xi - xc) * yi, -4 * (xi * xi - 2 * xi * xc + 3 * yi * yi - 4 * yi * yc + C), 0,
                4 * (xi - xc), 4 * (yi - yc), 0;

//        std::cout << "Second partials " << i << std::endl;
//        std::cout << "d2gdxdu_xi" << std::endl << d2gdxdu_xi << std::endl;
//        std::cout << "d2gdxdu_yi" << std::endl << d2gdxdu_yi << std::endl;
//        std::cout << "d2gdxdu_wi" << std::endl << d2gdxdu_wi << std::endl;
//        std::cout << "d2gdu2_xi" << std::endl << d2gdu2_xi << std::endl;
//        std::cout << "d2gdu2_yi" << std::endl << d2gdu2_yi << std::endl;
//        std::cout << "d2gdu2_wi" << std::endl << d2gdu2_wi << std::endl;

        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                d2gdxdu(ii, jj, 3 * i + 0) = d2gdxdu_xi(ii, jj);
                d2gdxdu(ii, jj, 3 * i + 1) = d2gdxdu_yi(ii, jj);
                d2gdxdu(ii, jj, 3 * i + 2) = d2gdxdu_wi(ii, jj);

                d2gdu2(ii, 3 * i + jj, 3 * i + 0) = d2gdu2_xi(ii, jj);
                d2gdu2(ii, 3 * i + jj, 3 * i + 1) = d2gdu2_yi(ii, jj);
                d2gdu2(ii, 3 * i + jj, 3 * i + 2) = d2gdu2_wi(ii, jj);
            }
        }
    }

    MatrixXT dgdxinv = dgdx.inverse();
    MatrixXT dxdu = -dgdxinv * dgdu;
//    std::cout << "dxdu: " << dxdu << std::endl;

    Eigen::Tensor<double, 3> pSpx(nx, nu, nx);
    pSpx.setZero();
    Eigen::Tensor<double, 3> pSpu(nx, nu, nu);
    pSpu.setZero();

    for (int i = 0; i < nx; i++) {
        MatrixXT d2gdudx_slice(nx, nu);
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < nu; jj++) {
                d2gdudx_slice(ii, jj) = d2gdxdu(i, ii, jj);
            }
        }
        MatrixXT pSpx_slice = -dgdxinv * d2gdudx_slice;
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < nu; jj++) {
                pSpx(ii, jj, i) = pSpx_slice(ii, jj);
            }
        }

//        if (i == 0) {
//            std::cout << "PSPX Slice " << std::endl << pSpx_slice << std::endl;
//        }
    }

    for (int i = 0; i < nu; i++) {
        MatrixXT d2gdxdu_slice(nx, nx), d2gdu2_slice(nx, nu);
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < nx; jj++) {
                d2gdxdu_slice(ii, jj) = d2gdxdu(ii, jj, i);
            }
        }
        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < nu; jj++) {
                d2gdu2_slice(ii, jj) = d2gdu2(ii, i, jj);
            }
        }

        MatrixXT pSpu_slice = -dgdxinv * (d2gdxdu_slice * dxdu + d2gdu2_slice);

        for (int ii = 0; ii < nx; ii++) {
            for (int jj = 0; jj < nu; jj++) {
                pSpu(ii, jj, i) = pSpu_slice(ii, jj);
            }
        }

//        if (i == 0) {
//            std::cout << "PSPU Slice " << std::endl << pSpu_slice << std::endl;
//        }
    }

    std::vector<MatrixXT> d2xdu2(nx);
    for (int i = 0; i < nx; i++) {
        MatrixXT pSpx_slice(nu, nx), pSpu_slice(nu, nu);
        for (int ii = 0; ii < nu; ii++) {
            for (int jj = 0; jj < nx; jj++) {
                pSpx_slice(ii, jj) = pSpx(i, ii, jj);
            }
        }
        for (int ii = 0; ii < nu; ii++) {
            for (int jj = 0; jj < nu; jj++) {
                pSpu_slice(ii, jj) = pSpu(i, ii, jj);
            }
        }

        d2xdu2[i] = pSpx_slice * dxdu + pSpu_slice;

//        std::cout << "SECOND ORDER SENSITIVITY " << i << std::endl;
//        std::cout << d2xdu2[i] << std::endl;
    }

    MatrixXT dtdc = A.ptpq * A.dqdc + A.ptpc;
    MatrixXT dudc = A.pupt * dtdc + A.pupc;

    nodePos.grad = dxdu * dudc;

    nodePos.hess.resize(CellFunction::nx);
    for (int i = 0; i < CellFunction::nx; i++) {
        nodePos.hess[i] = MatrixXT::Zero(nc, nc);
    }
    for (int ix = 0; ix < dims; ix++) {
        nodePos.hess[ix] += dudc.transpose() * d2xdu2[ix] * dudc;
        for (int iu = 0; iu < nu; iu++) {
            nodePos.hess[ix] += dxdu(ix, iu) * dtdc.transpose() * A.p2upt2[iu] * dtdc;
        }
        for (int it = 0; it < nt; it++) {
            MatrixXT sum_ptpq_d2qdc2 = MatrixXT::Zero(nc, nc);
            for (int iq = 0; iq < nq; iq++) {
                sum_ptpq_d2qdc2 += A.ptpq(it, iq) * A.d2qdc2[iq];
            }
            nodePos.hess[ix] += (dxdu.row(ix) * A.pupt.col(it)) * (A.dqdc.transpose() * A.p2tpq2[it] * A.dqdc
                                                                   + A.dqdc.transpose() * A.p2tpqpc[it]
                                                                   + A.p2tpqpc[it].transpose() * A.dqdc
                                                                   + sum_ptpq_d2qdc2
                                                                   + A.p2tpc2[it]);
        }
    }
}

void SmoothVoronoi::getChainRuleMatrices(Node &node, SmoothVoronoiChainRuleStruct &chainRuleMatrices) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n = node.gen.size();
    int nq = dims, nu = 3 * n, nc = dims * n, nt = n - 3;

    if (node.gen[1] < n_vtx &&
        node.gen[2] < n_vtx) { // Only handle standard voronoi node cases for now, no boundary interaction smoothing...
        VectorXT v0 = c.segment(node.gen[0] * dims, dims);
        VectorXT v1 = c.segment(node.gen[1] * dims, dims);
        VectorXT v2 = c.segment(node.gen[2] * dims, dims);

        NodePosition nodePos;
        Voronoi::getStandardNode(v0, v1, v2, nodePos.pos);
        Voronoi::getStandardNodeGradient(v0, v1, v2, nodePos.grad);
        Voronoi::getStandardNodeHessian(v0, v1, v2, nodePos.hess);

        chainRuleMatrices.q = nodePos.pos.head(nq);
        chainRuleMatrices.dqdc = MatrixXT::Zero(nq, nc);
        chainRuleMatrices.dqdc.block(0, 0, nq, dims * 3) = nodePos.grad.block(0, 0, nq, dims * 3);
        chainRuleMatrices.d2qdc2.resize(nq);
        for (int i = 0; i < nq; i++) {
            chainRuleMatrices.d2qdc2[i] = MatrixXT::Zero(nc, nc);
            chainRuleMatrices.d2qdc2[i].block(0, 0, dims * 3, dims * 3) = nodePos.hess[i];
        }

        chainRuleMatrices.t = VectorXT::Zero(nt);
        chainRuleMatrices.ptpq = MatrixXT::Zero(nt, nq);
        chainRuleMatrices.ptpc = MatrixXT::Zero(nt, nc);
        chainRuleMatrices.p2tpq2.resize(nt);
        chainRuleMatrices.p2tpqpc.resize(nt);
        chainRuleMatrices.p2tpc2.resize(nt);
        for (int i = 0; i < nt; i++) {
            chainRuleMatrices.p2tpq2[i] = MatrixXT::Zero(nq, nq);
            chainRuleMatrices.p2tpqpc[i] = MatrixXT::Zero(nq, nc);
            chainRuleMatrices.p2tpc2[i] = MatrixXT::Zero(nc, nc);
        }

        {
            double x0 = v0(0);
            double y0 = v0(1);
            double xc = chainRuleMatrices.q(0);
            double yc = chainRuleMatrices.q(1);

            for (int i = 0; i < nt; i++) {
                VectorXT vt = c.segment(node.gen[3 + i] * dims, dims);
                double xt = vt(0);
                double yt = vt(1);

                {
                    double t2 = pow(xt - xc, 0.2e1);
                    double t4 = pow(yt - yc, 0.2e1);
                    double t6 = sqrt(t2 + t4);
                    double t8 = pow(x0 - xc, 0.2e1);
                    double t10 = pow(y0 - yc, 0.2e1);
                    double t12 = sqrt(t8 + t10);
                    chainRuleMatrices.t(i) = 0.1e1 / d0 * (t6 - t12);
                }

                {
                    double unknown[6];

                    double t1 = x0 - xc;
                    double t2 = t1 * t1;
                    double t3 = y0 - yc;
                    double t4 = t3 * t3;
                    double t6 = sqrt(t2 + t4);
                    double t7 = 0.1e1 / t6;
                    double t9 = 0.1e1 / d0;
                    double t15 = xt - xc;
                    double t16 = t15 * t15;
                    double t17 = yt - yc;
                    double t18 = t17 * t17;
                    double t20 = sqrt(t16 + t18);
                    double t21 = 0.1e1 / t20;
                    unknown[0] = -t9 * t1 * t7;
                    unknown[1] = -t9 * t3 * t7;
                    unknown[2] = t9 * t15 * t21;
                    unknown[3] = t9 * t17 * t21;
                    unknown[4] = t9 * (0.2e1 * t1 * t7 - 0.2e1 * t15 * t21) / 0.2e1;
                    unknown[5] = t9 * (-0.2e1 * t17 * t21 + 0.2e1 * t3 * t7) / 0.2e1;

                    VectorXT dt = Eigen::Map<VectorXT>(&unknown[0], 6);

                    chainRuleMatrices.ptpc.row(i).segment(0, dims) = dt.segment(0, dims);
                    chainRuleMatrices.ptpc.row(i).segment((3 + i) * dims, dims) = dt.segment(dims, dims);
                    chainRuleMatrices.ptpq.row(i).segment(0, nq) = dt.segment(2 * dims, nq);
                }

                {
                    double unknown[6][6];

                    double t1 = x0 - xc;
                    double t2 = t1 * t1;
                    double t3 = y0 - yc;
                    double t4 = t3 * t3;
                    double t5 = t2 + t4;
                    double t6 = sqrt(t5);
                    double t8 = 0.1e1 / t6 / t5;
                    double t10 = 0.4e1 * t1 * t1 * t8;
                    double t11 = 0.1e1 / d0;
                    double t14 = 0.1e1 / t6;
                    double t15 = t11 * t14;
                    double t17 = 0.2e1 * t1 * t8;
                    double t20 = t3 * t11 * t17 / 0.2e1;
                    double t21 = -0.2e1 * t1 * t11;
                    double t24 = t21 * t17 / 0.4e1 + t15;
                    double t25 = -0.2e1 * t3 * t11;
                    double t27 = t25 * t17 / 0.4e1;
                    double t29 = 0.4e1 * t3 * t3 * t8;
                    double t33 = 0.2e1 * t3 * t8;
                    double t35 = t21 * t33 / 0.4e1;
                    double t38 = t25 * t33 / 0.4e1 + t15;
                    double t39 = xt - xc;
                    double t40 = t39 * t39;
                    double t41 = yt - yc;
                    double t42 = t41 * t41;
                    double t43 = t40 + t42;
                    double t44 = sqrt(t43);
                    double t46 = 0.1e1 / t44 / t43;
                    double t48 = 0.4e1 * t39 * t39 * t46;
                    double t51 = 0.1e1 / t44;
                    double t52 = t11 * t51;
                    double t54 = 0.2e1 * t39 * t46;
                    double t57 = t41 * t11 * t54 / 0.2e1;
                    double t58 = -0.2e1 * t39 * t11;
                    double t61 = -t58 * t54 / 0.4e1 - t52;
                    double t62 = -0.2e1 * t41 * t11;
                    double t64 = t62 * t54 / 0.4e1;
                    double t66 = 0.4e1 * t41 * t41 * t46;
                    double t70 = 0.2e1 * t41 * t46;
                    double t72 = t58 * t70 / 0.4e1;
                    double t75 = -t62 * t70 / 0.4e1 - t52;
                    double t85 = t11 * (0.4e1 * t3 * t1 * t8 - 0.4e1 * t41 * t39 * t46) / 0.4e1;
                    unknown[0][0] = t11 * t10 / 0.4e1 - t15;
                    unknown[0][1] = t20;
                    unknown[0][2] = 0.0e0;
                    unknown[0][3] = 0.0e0;
                    unknown[0][4] = t24;
                    unknown[0][5] = t27;
                    unknown[1][0] = t20;
                    unknown[1][1] = t11 * t29 / 0.4e1 - t15;
                    unknown[1][2] = 0.0e0;
                    unknown[1][3] = 0.0e0;
                    unknown[1][4] = t35;
                    unknown[1][5] = t38;
                    unknown[2][0] = 0.0e0;
                    unknown[2][1] = 0.0e0;
                    unknown[2][2] = -t11 * t48 / 0.4e1 + t52;
                    unknown[2][3] = -t57;
                    unknown[2][4] = t61;
                    unknown[2][5] = -t64;
                    unknown[3][0] = 0.0e0;
                    unknown[3][1] = 0.0e0;
                    unknown[3][2] = -t57;
                    unknown[3][3] = -t11 * t66 / 0.4e1 + t52;
                    unknown[3][4] = -t72;
                    unknown[3][5] = t75;
                    unknown[4][0] = t24;
                    unknown[4][1] = t35;
                    unknown[4][2] = t61;
                    unknown[4][3] = -t72;
                    unknown[4][4] = t11 * (-t48 / 0.4e1 + t51 + t10 / 0.4e1 - t14);
                    unknown[4][5] = t85;
                    unknown[5][0] = t27;
                    unknown[5][1] = t38;
                    unknown[5][2] = -t64;
                    unknown[5][3] = t75;
                    unknown[5][4] = t85;
                    unknown[5][5] = t11 * (-t66 / 0.4e1 + t51 + t29 / 0.4e1 - t14);

                    MatrixXT d2t = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], 6, 6);

                    chainRuleMatrices.p2tpq2[i] = d2t.bottomRightCorner(nq, nq);

                    chainRuleMatrices.p2tpqpc[i].block(0, 0, nq, dims) = d2t.bottomLeftCorner(nq, dims);
                    chainRuleMatrices.p2tpqpc[i].block(0, (3 + i) * dims, nq, dims) = d2t.block(2 * dims, dims, nq,
                                                                                                dims);

                    chainRuleMatrices.p2tpc2[i].block(0, 0, dims, dims) = d2t.topLeftCorner(dims, dims);
                    chainRuleMatrices.p2tpc2[i].block(0, (3 + i) * dims, dims, dims) = d2t.block(0, dims, dims, dims);
                    chainRuleMatrices.p2tpc2[i].block((3 + i) * dims, 0, dims, dims) = d2t.block(dims, 0, dims, dims);
                    chainRuleMatrices.p2tpc2[i].block((3 + i) * dims, (3 + i) * dims, dims, dims) = d2t.block(dims,
                                                                                                              dims,
                                                                                                              dims,
                                                                                                              dims);
                }
            }
        }

        chainRuleMatrices.u = VectorXT::Zero(nu);
        chainRuleMatrices.pupt = MatrixXT::Zero(nu, nt);
        chainRuleMatrices.pupc = MatrixXT::Zero(nu, nc);
        chainRuleMatrices.p2upt2.resize(nu);
        for (int i = 0; i < nu; i++) {
            chainRuleMatrices.p2upt2[i] = MatrixXT::Zero(nt, nt);
        }

        for (int i = 0; i < n; i++) {
            chainRuleMatrices.u(i * 3 + 0) = c(node.gen[i] * dims + 0);
            chainRuleMatrices.u(i * 3 + 1) = c(node.gen[i] * dims + 1);
            chainRuleMatrices.u(i * 3 + 2) = 1; // Overwrite in next loop if necessary

            chainRuleMatrices.pupc(i * 3 + 0, i * dims + 0) = 1;
            chainRuleMatrices.pupc(i * 3 + 1, i * dims + 1) = 1;
        }
        for (int i = 0; i < nt; i++) {
            double t = chainRuleMatrices.t(i);
            TV3 w = getWeightDerivatives(t);
            chainRuleMatrices.u((3 + i) * 3 + 2) = w(0);
            chainRuleMatrices.pupt((3 + i) * 3 + 2, i) = w(1);
            chainRuleMatrices.p2upt2[(3 + i) * 3 + 2](i, i) = w(2);
        }
    }
}
