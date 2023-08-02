#include "../../include/Tessellation/SmoothVoronoi.h"
#include "../../include/Tessellation/CellFunction.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

double SmoothVoronoi::getWeight(double d) {
    double k = 10;
    return exp(k / (pow(d, 2) - 1) + k);
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

        node.genw.resize(3);
        std::fill(node.genw.begin(), node.genw.end(), 1.0);

        for (int i = 0; i < n_vtx; i++) {
            if (i == node.gen[0] || i == node.gen[1] || i == node.gen[2]) { continue; }
            else {
                VectorXT v = c.segment(i * dims, dims);
                double d = ((v - nodePos.pos.head(dims)).norm() - R) / d0;
                assert(d >= 0);

                if (d < 1) {
                    node.gen.push_back(i);
                    node.genw.push_back(getWeight(d));
                }
            }
        }

        if (node.gen.size() == 3) {
            // No nearly-encroaching sites, no smoothing needed
            Tessellation::getNodeWrapper(node, nodePos);
        } else {
            node.type = NodeType::LSQ;
            getNodeLSQ(node, nodePos);
        }

    } else {
        Tessellation::getNodeWrapper(node, nodePos);
    }
}

void SmoothVoronoi::getNodeLSQ(Node &node, NodePosition &nodePos) {
    int dims = 2 + getNumVertexParams();
    int n = node.gen.size();

    n = 4;
    node.gen[0] = 0;
    node.gen[1] = 1;
    node.gen[2] = 2;
    node.gen[3] = 3;
    c.segment<2>(2 * 0) = TV(0.04, 0.05);
    c.segment<2>(2 * 1) = TV(-0.02, 1);
    c.segment<2>(2 * 2) = TV(1.1, 1.2);
    c.segment<2>(2 * 3) = TV(0.9, 0.1);
    node.genw[0] = 1;
    node.genw[1] = 1;
    node.genw[2] = 1;
    node.genw[3] = 0.5;

    MatrixXT A_LSQ(n, 3);
    VectorXT b_LSQ(n);

    for (int i = 0; i < n; i++) {
        VectorXT vi = c.segment(node.gen[i] * dims, dims);
        double xi = vi(0);
        double yi = vi(1);
        double sqrtwi = sqrt(node.genw[i]);
        A_LSQ.row(i) = TV3(2 * xi * sqrtwi, 2 * yi * sqrtwi, -sqrtwi);
        b_LSQ(i) = sqrtwi * (xi * xi + yi * yi);
    }

    VectorXT x_LSQ = (A_LSQ.transpose() * A_LSQ).fullPivHouseholderQr().solve(A_LSQ.transpose() * b_LSQ);
    nodePos.pos = x_LSQ;
    double xc = x_LSQ(0);
    double yc = x_LSQ(1);
    double C = x_LSQ(2);

    std::cout << "LSQ solution: " << x_LSQ << std::endl;

    int nx = 3, nu = 3 * n;
    MatrixXT dgdx = MatrixXT::Zero(nx, nx);
    MatrixXT dgdu = MatrixXT::Zero(nx, nu);
    Eigen::Tensor<double, 3> d2gdxdu(nx, nx, nu);
    d2gdxdu.setZero();
    Eigen::Tensor<double, 3> d2gdu2(nx, nu, nu);
    d2gdu2.setZero();
    for (int i = 0; i < n; i++) {
        VectorXT vi = c.segment(node.gen[i] * dims, dims);
        double xi = vi(0);
        double yi = vi(1);
        double wi = node.genw[i];

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
    std::cout << "dxdu: " << dxdu << std::endl;

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

        if (i == 0) {
            std::cout << "PSPX Slice " << std::endl << pSpx_slice << std::endl;
        }
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

        if (i == 0) {
            std::cout << "PSPU Slice " << std::endl << pSpu_slice << std::endl;
        }
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

        std::cout << "SECOND ORDER SENSITIVITY " << i << std::endl;
        std::cout << d2xdu2[i] << std::endl;
    }

    nodePos.grad = VectorXT::Zero(0);

    nodePos.hess.resize(CellFunction::nx);
    for (int i = 0; i < CellFunction::nx; i++) {
        nodePos.hess[i] = MatrixXT::Zero(0, 0);
    }
}
