#include "../../include/Tessellation/Tessellation.h"
#include <set>
#include <chrono>
#include <iostream>

void Tessellation::tessellate(const VectorXT &vertices, const VectorXT &params) {
    VectorXT c_new = combineVerticesParams(vertices, params);

    // Check if inputs are the same as previous tessellation, do nothing if so.
    bool same = true;
    same = same && (c_new.rows() == c.rows() && c_new.isApprox(c));
    if (same) return;

    isValid = true;
    c = c_new;
    nodes.clear();
    faces.clear();
    cells.clear();

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
                nodes[node] = nodePosition;
            }
        }
    }

    clipFaces();

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
                        b0 = bv[bf[node.gen[0]].vertices(0)].pos;
                        b1 = bv[bf[node.gen[0]].vertices(1)].pos;
                        b2 = bv[bf[node.gen[0]].vertices(2)].pos;

                        v0 = c.segment<4>(node.gen[1] * 4);
                        v1 = c.segment<4>(node.gen[2] * 4);
                        v2 = c.segment<4>(node.gen[3] * 4);

                        getNodeBFace(b0, b1, b2, v0, v1, v2, nodePosition);
                        getNodeBFaceGradient(b0, b1, b2, v0, v1, v2, nodePosition);
                        getNodeBFaceHessian(b0, b1, b2, v0, v1, v2, nodePosition);
                        break;
                    case B_EDGE:
                        b0 = bv[node.gen[0]].pos;
                        b1 = bv[node.gen[1]].pos;

                        v0 = c.segment<4>(node.gen[2] * 4);
                        v1 = c.segment<4>(node.gen[3] * 4);

                        getNodeBEdge(b0, b1, v0, v1, nodePosition);
                        getNodeBEdgeGradient(b0, b1, v0, v1, nodePosition);
                        getNodeBEdgeHessian(b0, b1, v0, v1, nodePosition);
                        break;
                    case B_VERTEX:
                        nodePosition.pos = bv[node.gen[0]].pos;
                        break;
                    default:
                        assert(0);
                }
                nodes[node] = nodePosition;
            }
        }
    }

    computeCellData();

//
//    auto cmp = [](IV4 a, IV4 b) {
//        for (int i = 0; i < 4; i++) {
//            if (a(i) < b(i)) return true;
//            if (a(i) > b(i)) return false;
//        }
//        return false;
//    };
//    std::set<IV4, decltype(cmp)> triplets(cmp);
//
//    for (int i = 0; i < n_free; i++) {
//        Cell cell = cells[i];
//        for (int j = 0; j < cell.edges.size(); j++) {
//            int n1 = cell.edges[j].neighbor;
//            int n2 = cell.edges[cell.edges[j].nextEdge].neighbor;
//            int flag = cell.edges[cell.edges[j].nextEdge].flag;
//
//            IV4 triplet(i, n1, n2, flag);
//            std::sort(triplet.data(), triplet.data() + 3);
//
//            triplets.insert(triplet);
//        }
//    }
//    std::vector<IV4> faces(triplets.begin(), triplets.end());
//
//    int dims = 2 + getNumVertexParams();
//    int n_vtx = c.rows() / dims;
//    int n_bdy = bdry->v.rows() / 2;
//    for (int i = 0; i < n_free; i++) {
//        Cell cell = cells[i];
//        for (int j = 0; j < cell.edges.size(); j++) {
//            int n1 = cell.edges[j].neighbor;
//            int n2 = cell.edges[cell.edges[j].nextEdge].neighbor;
//            int flag = cell.edges[cell.edges[j].nextEdge].flag;
//
//            IV4 triplet(i, n1, n2, flag);
//            std::sort(triplet.data(), triplet.data() + 3);
//
//            auto lower = std::lower_bound(faces.begin(), faces.end(), triplet, cmp);
//            assert(lower != faces.end() && *lower == triplet);
//        }
//    }
//
//    dual.resize(faces.size() * 4);
//    for (int i = 0; i < faces.size(); i++) {
//        dual.segment<4>(i * 4) = faces[i];
//    }
//
//    x.resize(faces.size() * CellFunction::nx);
//    x.setZero();
//    MatrixXT dxdc_dense = MatrixXT::Zero(x.rows(), n_free * dims);
//    MatrixXT dxdv_dense = MatrixXT::Zero(x.rows(), bdry->v.rows());
//    MatrixXT dxdq_dense = MatrixXT::Zero(x.rows(), bdry->q.rows());
//    d2xdy2.resize(x.rows());
//    for (int i = 0; i < faces.size(); i++) {
//        IV4 face = faces[i];
//
//        VectorXT node;
//        MatrixXT nodeGrad;
//        std::vector<MatrixXT> nodeHess;
//        int type;
//        getNodeWrapper(face[0], face[1], face[2], face[3], node, nodeGrad, nodeHess,
//                       type);
//
//        x.segment<CellFunction::nx>(i * CellFunction::nx) = node;
//
//        for (int j = 0; j < CellFunction::nx; j++) {
//            d2xdy2[i * CellFunction::nx + j] = nodeHess[j];
//        }
//
//        // Assemble global Jacobian matrix dxdc.
//        int n_sites = c.rows() / dims;
//        switch (type) {
//            case 0:
//                for (int j = 0; j < 3; j++) {
//                    if (face[j] >= n_free) {
//                        continue;
//                    }
//                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
//                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
//                }
//                break;
//            case 1:
//                for (int j = 0; j < 2; j++) {
//                    if (face[j] >= n_free) {
//                        continue;
//                    }
//                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
//                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
//                }
//                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
//                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
//                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
//                                 2) =
//                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 2);
//
//                break;
//            case 2:
//                for (int j = 0; j < 2; j++) {
//                    if (face[j] >= n_free) {
//                        continue;
//                    }
//                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
//                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
//                }
//                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
//                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
//                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
//                                 2) =
//                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 2);
//
//                dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].q_idx, CellFunction::nx, 1) =
//                        nodeGrad.block(0, 2 * dims + 4, CellFunction::nx, 1);
//                break;
//            case 3:
//                for (int j = 0; j < 2; j++) {
//                    if (face[j] >= n_free) {
//                        continue;
//                    }
//                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
//                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
//                }
//                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
//                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
//                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
//                                 2) =
//                        nodeGrad.block(0, 2 * dims + 3, CellFunction::nx, 2);
//
//                dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].q_idx, CellFunction::nx, 1) =
//                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 1);
//                dxdq_dense.block(i * CellFunction::nx, bdry->edges[bdry->edges[face[2] - n_sites].nextEdge].q_idx,
//                                 CellFunction::nx, 1) =
//                        nodeGrad.block(0, 2 * dims + 5, CellFunction::nx, 1);
//                break;
//            case 4:
//                dxdv_dense.block(i * CellFunction::nx, (face[1] - n_sites) * 2, CellFunction::nx, 2) =
//                        nodeGrad.block(0, 0, CellFunction::nx, 2);
//                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
//                        nodeGrad.block(0, 3, CellFunction::nx, 2);
//
//                if (bdry->edges[face[1] - n_sites].q_idx >= 0) {
//                    dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[1] - n_sites].q_idx, CellFunction::nx, 1) =
//                            nodeGrad.block(0, 2, CellFunction::nx, 1);
//                }
//                if (bdry->edges[face[2] - n_sites].q_idx >= 0) {
//                    dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].q_idx, CellFunction::nx, 1) =
//                            nodeGrad.block(0, 5, CellFunction::nx, 1);
//                }
//                break;
//            default:
//                assert(0);
//                break;
//        }
//    }
//
//    dxdc = dxdc_dense.sparseView();
//    dxdv = dxdv_dense.sparseView();
//    dxdq = dxdq_dense.sparseView();
//
//    for (int i = 0; i < n_free; i++) {
//        if (cells[i].edges.size() < 2) {
//            isValid = false;
//            return;
//        }
//    }
}

#include "../../include/Tessellation/CellFunctionPerTriangle.h"
#include "../../include/Energy/PerTriangleVolume.h"

void Tessellation::computeCellData() {
    int n_cells = c.rows() / 4;
    cells.resize(n_cells);

    for (int f = 0; f < faces.size(); f++) {
        if (faces[f].site0 >= 0) {
            cells[faces[f].site0].facesPos.push_back(f);
        }
        if (faces[f].site1 >= 0) {
            cells[faces[f].site1].facesNeg.push_back(f);
        }
    }

    double totalv = 0;
    PerTriangleVolume perTriVol;

    CellFunctionPerTriangle volFunc;
    volFunc.perTriangleFunction = &perTriVol;
    for (Cell cell: cells) {
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

        CellValue cellVal(cell);
        volFunc.addValue(this, cellVal);
        volFunc.addGradient(this, cellVal);
        volFunc.addHessian(this, cellVal);
        totalv += cellVal.value;
        std::cout << cellVal.value << std::endl;
    }
    std::cout << "total " << totalv << std::endl;
}
