#include "../../include/Tessellation/Tessellation.h"
#include <set>
#include <chrono>
#include <iostream>

#include "Projects/Foam3D/include/Energy/CellFunctionPerTriangle.h"
#include "../../include/Energy/PerTriangleVolume.h"

void Tessellation::tessellate(const VectorXT &vertices, const VectorXT &params, const VectorXT &p_free) {
    VectorXT c_new = combineVerticesParams(vertices, params);

    // Check if inputs are the same as previous tessellation, do nothing if so.
    bool same = true;
    same = same && (c_new.rows() == c.rows() && (c_new - c).norm() < 1e-14) &&
           ((p_free - boundary->get_p_free()).norm() < 1e-14);
    if (same) {
//        std::cout << "same " << (c_new - c).norm() << " " << (p_free - boundary->get_p_free()).norm() << std::endl;
        return;
    }

    isValid = true;
    c = c_new;
    nodes.clear();
    faces.clear();
    cells.clear();

    for (int i = 0; i < c.rows() / 4 - 8; i++) {
        if (fabs(c(i * 4 + 0)) > 5
            || fabs(c(i * 4 + 1)) > 5
            || fabs(c(i * 4 + 2)) > 5) {
            isValid = false;
            break;
        }
    }
    if (!isValid) return;

    boundary->compute(p_free);

    if (!boundary->checkValid()) {
        isValid = false;
    }
    if (!isValid) return;

    getDualGraph();

    for (int i = 0; i < faces.size(); i++) {
        for (Node node: faces[i].nodes) {
            if (nodes.find(node) == nodes.end()) {
                NodePosition nodePosition;

                VectorXT v0 = c.segment<4>(node.gen[0] * 4);
                VectorXT v1 = c.segment<4>(node.gen[1] * 4);
                VectorXT v2 = c.segment<4>(node.gen[2] * 4);
                VectorXT v3 = c.segment<4>(node.gen[3] * 4);

                getNode(v0, v1, v2, v3, nodePosition);
                getNodeGradient(v0, v1, v2, v3, nodePosition);
                getNodeHessian(v0, v1, v2, v3, nodePosition);
                nodes[node] = nodePosition;
            }
        }
    }

    clipFaces2();
    clipFaces();
    nodes.clear(); // TODO: hack here to remove the out-of-bounds nodes involving the convex hull cells pre-clip.

    for (int i = 0; i < faces.size(); i++) {
        for (Node node: faces[i].nodes) {
            if (nodes.find(node) == nodes.end()) {
                NodePosition nodePosition;

                VectorXT v0, v1, v2, v3;
                TV3 b0, b1, b2;

                switch (node.type) {
                    case STANDARD:
                        v0 = c.segment<4>(node.gen[0] * 4);
                        v1 = c.segment<4>(node.gen[1] * 4);
                        v2 = c.segment<4>(node.gen[2] * 4);
                        v3 = c.segment<4>(node.gen[3] * 4);

                        getNode(v0, v1, v2, v3, nodePosition);
                        getNodeGradient(v0, v1, v2, v3, nodePosition);
                        getNodeHessian(v0, v1, v2, v3, nodePosition);
                        break;
                    case B_FACE:
                        b0 = boundary->v[boundary->f[node.gen[0]].vertices(0)].pos;
                        b1 = boundary->v[boundary->f[node.gen[0]].vertices(1)].pos;
                        b2 = boundary->v[boundary->f[node.gen[0]].vertices(2)].pos;

                        v0 = c.segment<4>(node.gen[1] * 4);
                        v1 = c.segment<4>(node.gen[2] * 4);
                        v2 = c.segment<4>(node.gen[3] * 4);

                        getNodeBFace(b0, b1, b2, v0, v1, v2, nodePosition);
                        getNodeBFaceGradient(b0, b1, b2, v0, v1, v2, nodePosition);
                        getNodeBFaceHessian(b0, b1, b2, v0, v1, v2, nodePosition);
                        break;
                    case B_EDGE:
                        b0 = boundary->v[node.gen[0]].pos;
                        b1 = boundary->v[node.gen[1]].pos;

                        v0 = c.segment<4>(node.gen[2] * 4);
                        v1 = c.segment<4>(node.gen[3] * 4);

                        getNodeBEdge(b0, b1, v0, v1, nodePosition);
                        getNodeBEdgeGradient(b0, b1, v0, v1, nodePosition);
                        getNodeBEdgeHessian(b0, b1, v0, v1, nodePosition);
                        break;
                    case B_VERTEX:
                        nodePosition.pos = boundary->v[node.gen[0]].pos;
                        nodePosition.grad = MatrixXT::Identity(3, 3);
                        nodePosition.hess[0] = MatrixXT::Zero(3, 3);
                        nodePosition.hess[1] = MatrixXT::Zero(3, 3);
                        nodePosition.hess[2] = MatrixXT::Zero(3, 3);
                        break;
                    default:
                        assert(0);
                }
                nodes[node] = nodePosition;
            }
        }
    }
    int ix = 0;
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
        it->second.ix = ix;
        ix++;
    }

    computeCellData();

    PerTriangleVolume perTriVol;
    CellFunctionPerTriangle volFunc(&perTriVol);
    for (int i = 0; i < cells.size(); i++) {
        CellValue cellVal(cells[i]);
        volFunc.getValue(this, cellVal);
        std::cout << "Volume " << cellVal.value << std::endl;
        if (cellVal.value < 1e-10) {
            isValid = false;
            break;
        }
    }

    std::cout << "Num faccs " << faces.size() << std::endl;

    computeMatrices();
}

void Tessellation::computeCellData() {
    int n_cells = c.rows() / 4 - 8;
    cells.resize(n_cells);

    for (int f = 0; f < faces.size(); f++) {
        if (faces[f].site0 >= 0) {
            cells[faces[f].site0].facesPos.push_back(f);
        }
        if (faces[f].site1 >= 0) {
            cells[faces[f].site1].facesNeg.push_back(f);
        }
    }

    int ic = 0;
    for (Cell &cell: cells) {
        cell.cellIndex = ic;
        ic++;

        int i = 0;
        auto func = [&](const int &f) {
            for (Node n: faces[f].nodes) {
                if (cell.nodeIndices.find(n) == cell.nodeIndices.end()) {
                    cell.nodeIndices[n] = i;
                    i++;
                }
            }
        };
        std::for_each(cell.facesPos.begin(), cell.facesPos.end(), func);
        std::for_each(cell.facesNeg.begin(), cell.facesNeg.end(), func);

        std::cout << "Cellsssss " << i << " has nodes: " << cell.nodeIndices.size() << std::endl;
    }
}

struct CoolStructy {
    bool gen_is_boundary;
    int gen_idx;
    int nodepos_start_idx;

    CoolStructy(bool b, int i0, int i1) {
        gen_is_boundary = b;
        gen_idx = i0;
        nodepos_start_idx = i1;
    }
};

#define cdims 3

void Tessellation::computeMatrices() {
    int nc = cdims * cells.size();
    int nx = 3 * nodes.size();
    int nv = 3 * boundary->v.size();
    int np = boundary->nfree;

    std::vector<Eigen::Triplet<double>> tripletsDXDC;
    std::vector<Eigen::Triplet<double>> tripletsDXDV;
    std::vector<std::vector<Eigen::Triplet<double>>> tripletsD2XDC2(nx);
    std::vector<std::vector<Eigen::Triplet<double>>> tripletsD2XDCDV(nx);
    std::vector<std::vector<Eigen::Triplet<double>>> tripletsD2XDV2(nx);
    std::vector<Eigen::Triplet<double>> tripletsDVDP;
    std::vector<std::vector<Eigen::Triplet<double>>> tripletsD2VDP2(nv);

    int nodeIdx = 0;
    for (auto pair: nodes) {
        Node node = pair.first;
        NodePosition nodePos = pair.second;

        std::vector<CoolStructy> coolStructs;
        switch (node.type) {
            case STANDARD:
                for (int i = 0; i < 4; i++) {
                    coolStructs.emplace_back(false, node.gen[i], i * 4);
                }
                break;
            case B_FACE:
                for (int i = 0; i < 3; i++) {
                    coolStructs.emplace_back(false, node.gen[i + 1], i * 4);
                }
                for (int i = 0; i < 3; i++) {
                    coolStructs.emplace_back(true, boundary->f[node.gen[0]].vertices(i),
                                             3 * 4 + i * 3);
                }
                break;
            case B_EDGE:
                for (int i = 0; i < 2; i++) {
                    coolStructs.emplace_back(false, node.gen[i + 2], i * 4);
                }
                for (int i = 0; i < 2; i++) {
                    coolStructs.emplace_back(true, node.gen[i], 2 * 4 + i * 3);
                }
                break;
            case B_VERTEX:
                coolStructs.emplace_back(true, node.gen[0], 0);
            default:
                break;
        }

        for (CoolStructy cs0: coolStructs) {
            if (cs0.gen_is_boundary) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        tripletsDXDV.emplace_back(nodeIdx * 3 + i, cs0.gen_idx * 3 + j,
                                                  nodePos.grad(i, cs0.nodepos_start_idx + j));
                    }
                }
            } else {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < cdims; j++) {
                        tripletsDXDC.emplace_back(nodeIdx * 3 + i, cs0.gen_idx * cdims + j,
                                                  nodePos.grad(i, cs0.nodepos_start_idx + j));
                    }
                }
            }

            for (CoolStructy cs1: coolStructs) {
                if (cs0.gen_is_boundary && cs1.gen_is_boundary) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            for (int k = 0; k < 3; k++) {
                                tripletsD2XDV2[nodeIdx * 3 + i].emplace_back(cs0.gen_idx * 3 + j,
                                                                             cs1.gen_idx * 3 + k,
                                                                             nodePos.hess[i](cs0.nodepos_start_idx + j,
                                                                                             cs1.nodepos_start_idx +
                                                                                             k));
                            }
                        }
                    }
                } else if (cs0.gen_is_boundary) {
                    // Do nothing
                } else if (cs1.gen_is_boundary) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < cdims; j++) {
                            for (int k = 0; k < 3; k++) {
                                tripletsD2XDCDV[nodeIdx * 3 + i].emplace_back(cs0.gen_idx * cdims + j,
                                                                              cs1.gen_idx * 3 + k,
                                                                              nodePos.hess[i](cs0.nodepos_start_idx + j,
                                                                                              cs1.nodepos_start_idx +
                                                                                              k));
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < cdims; j++) {
                            for (int k = 0; k < cdims; k++) {
                                tripletsD2XDC2[nodeIdx * 3 + i].emplace_back(cs0.gen_idx * cdims + j,
                                                                             cs1.gen_idx * cdims + k,
                                                                             nodePos.hess[i](cs0.nodepos_start_idx + j,
                                                                                             cs1.nodepos_start_idx +
                                                                                             k));
                            }
                        }
                    }
                }
            }
        }

        nodeIdx++;
    }

    int vertIdx = 0;
    for (BoundaryVertex v: boundary->v) {
        for (int i = 0; i < v.grad.outerSize(); i++) {
            for (typename Eigen::SparseMatrix<double>::InnerIterator it(v.grad, i); it; ++it) {
                tripletsDVDP.emplace_back(vertIdx * 3 + it.row(), it.col(), it.value());
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < v.hess[i].outerSize(); j++) {
                for (typename Eigen::SparseMatrix<double>::InnerIterator it(v.hess[i], j); it; ++it) {
                    // TODO: I think this is redundant, this basically reconstructs the same matrices again...
                    tripletsD2VDP2[vertIdx * 3 + i].emplace_back(it.row(), it.col(), it.value());
                }
            }
        }

        vertIdx++;
    }

    dxdc.resize(nx, nc);
    dxdv.resize(nx, nv);
    d2xdc2.resize(nx);
    d2xdcdv.resize(nx);
    d2xdv2.resize(nx);
    for (int i = 0; i < nx; i++) {
        d2xdc2[i].resize(nc, nc);
        d2xdcdv[i].resize(nc, nv);
        d2xdv2[i].resize(nv, nv);
    }
    dvdp.resize(nv, np);
    d2vdp2.resize(nv);
    for (int i = 0; i < nv; i++) {
        d2vdp2[i].resize(np, np);
    }

    dxdc.setFromTriplets(tripletsDXDC.begin(), tripletsDXDC.end());
    dxdv.setFromTriplets(tripletsDXDV.begin(), tripletsDXDV.end());
    for (int i = 0; i < nx; i++) {
        d2xdc2[i].setFromTriplets(tripletsD2XDC2[i].begin(), tripletsD2XDC2[i].end());
        d2xdcdv[i].setFromTriplets(tripletsD2XDCDV[i].begin(), tripletsD2XDCDV[i].end());
        d2xdv2[i].setFromTriplets(tripletsD2XDV2[i].begin(), tripletsD2XDV2[i].end());
    }
    dvdp.setFromTriplets(tripletsDVDP.begin(), tripletsDVDP.end());
    for (int i = 0; i < nv; i++) {
        d2vdp2[i].setFromTriplets(tripletsD2VDP2[i].begin(), tripletsD2VDP2[i].end());
    }
}
