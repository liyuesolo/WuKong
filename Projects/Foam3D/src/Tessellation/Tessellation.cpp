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
    if (same)
        std::cout << "same " << (c_new - c).norm() << " " << (p_free - boundary->get_p_free()).norm() << std::endl;
    if (same) return;

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

    computeCellData();

    PerTriangleVolume perTriVol;
    CellFunctionPerTriangle volFunc(&perTriVol);
    for (int i = 0; i < cells.size(); i++) {
        CellValue cellVal(cells[i]);
        volFunc.getValue(this, cellVal);
//        std::cout << "Volume " << cellVal.value << std::endl;
        if (cellVal.value < 1e-10) {
            isValid = false;
            break;
        }
    }

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
    }

//    double totalv = 0;
//    PerTriangleSurfaceArea perTriVol;
//
//    CellFunctionPerTriangle volFunc(&perTriVol);
//    for (Cell cell: cells) {
//        CellValue cellVal(cell);
//        volFunc.getValue(this, cellVal);
//        volFunc.getGradient(this, cellVal);
//        volFunc.getHessian(this, cellVal);
//        totalv += cellVal.value;
//        std::cout << cellVal.value << std::endl;
//    }
//    std::cout << "total " << totalv << std::endl;
}
