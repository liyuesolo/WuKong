#include "../../include/Tessellation/Tessellation.h"
#include "../../include/Tessellation/CellFunction.h"
#include "../../include/Boundary/Boundary.h"
#include <set>
#include <iostream>

bool
Tessellation::getNeighborsClipped(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual, int n_cells) {
    int n_vtx = vertices.rows() / 2, n_bdy = bdry->v.rows() / 2;

    cells.clear();
    cells.resize(n_cells);

    std::vector<std::vector<int>> neighborsRaw = getNeighbors(vertices, dual, n_cells);
    if (n_bdy == 0) {
        for (int i = 0; i < n_cells; i++) {
            cells[i].edges.resize(neighborsRaw[i].size());
            for (int j = 0; j < cells[i].edges.size(); j++) {
                cells[i].edges[j].neighbor = neighborsRaw[i][j];
                cells[i].edges[j].nextEdge = (j + 1) % cells[i].edges.size();
            }
        }
        return true;
    };

    VectorXT c = combineVerticesParams(vertices, params);
    int dims = 2 + getNumVertexParams();

    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &neighbors = neighborsRaw[i];
        size_t degree = neighbors.size();
        if (degree == 0) {
            return false;
        }

        VectorXT c0 = c.segment(i * dims, dims);
        std::vector<TV> nodes(degree);

        for (size_t j = 0; j < degree; j++) {
            int n1 = neighbors[j];
            int n2 = neighbors[(j + 1) % degree];

            VectorXT v;
            getNode(c0, c.segment(n1 * dims, dims), c.segment(n2 * dims, dims), v);
            nodes[(j + 1) % degree] = v.segment<2>(0);
        }

        std::vector<BoundaryIntersection> intersections;
        if (!bdry->getCellIntersections(nodes, intersections)) {
            return false;
        }
        if (intersections.empty()) {
            cells[i].edges.resize(neighborsRaw[i].size());
            for (int j = 0; j < cells[i].edges.size(); j++) {
                cells[i].edges[j].neighbor = neighborsRaw[i][j];
                cells[i].edges[j].nextEdge = (j + 1) % neighborsRaw[i].size();
            }
        } else {
            int loopStartIndex = 0;
            BoundaryIntersection loopStartIntersection = intersections[1];
            int startIndex = 1;
            bool isLoopStart = true;
            while (!intersections.empty()) {
                assert(intersections.size() % 2 == 0);
                BoundaryIntersection intersect1 = intersections[startIndex];

                VectorXi segmentDists = -1 * VectorXi::Ones(n_bdy);
                int curr = intersect1.i_bdry;
                int segmentDist = 0;
                do {
                    segmentDists(curr) = segmentDist;
                    curr = bdry->edges[curr].nextEdge;
                    segmentDist++;
                } while (curr != intersect1.i_bdry);

                // Find next intersection along boundary.
                double minDist = 1e10;
                int minIdx = -1;
                for (int j = 0; j < intersections.size(); j++) {
                    if (j == startIndex) continue;

                    BoundaryIntersection intersectCurr = intersections[j];
                    if (segmentDists(intersectCurr.i_bdry) == -1) continue;

                    double dist = segmentDists(intersectCurr.i_bdry) + intersectCurr.t_bdry - intersect1.t_bdry;
                    if (dist < 0) dist += n_bdy;

                    if (dist < minDist) {
                        minDist = dist;
                        minIdx = j;
                    }
                }

                BoundaryIntersection intersectNext = intersections[minIdx];
                int i_bdry = intersect1.i_bdry;
                int flag = intersect1.flag;
                bool bad1 = (intersectNext.i_bdry == intersect1.i_bdry && intersectNext.t_bdry < intersect1.t_bdry);
                while (true) {
                    cells[i].edges.emplace_back();
                    cells[i].edges.back().neighbor = n_vtx + i_bdry;
                    cells[i].edges.back().nextEdge = cells[i].edges.size();
                    cells[i].edges.back().flag = flag;
                    flag = 0;
                    if (i_bdry == intersectNext.i_bdry && !bad1) break;
                    bad1 = false;
                    i_bdry = bdry->edges[i_bdry].nextEdge;
                }

                assert(minIdx + 1 < intersections.size());
                BoundaryIntersection intersectNext2 = (minIdx == 0 ? loopStartIntersection : intersections[minIdx + 1]);
                int i_cell = intersectNext.i_cell;
                flag = intersectNext.flag;
                bool bad2 = (minIdx == 0 && isLoopStart && intersectNext.t_cell > intersect1.t_cell);
                while (true) {
                    cells[i].edges.emplace_back();
                    cells[i].edges.back().neighbor = neighbors[i_cell];
                    cells[i].edges.back().nextEdge = cells[i].edges.size();
                    cells[i].edges.back().flag = flag;
                    flag = 0;
                    if (i_cell == intersectNext2.i_cell && !bad2) break;
                    bad2 = false;
                    i_cell = (i_cell + 1) % degree;
                }

                if (minIdx == 0) {
                    intersections.erase(intersections.begin() + startIndex);
                    intersections.erase(intersections.begin());

                    cells[i].edges.back().nextEdge = loopStartIndex;
                    loopStartIndex = cells[i].edges.size();
                    loopStartIntersection = intersections[1];
                    startIndex = 1;
                    isLoopStart = true;
                } else {
                    intersections.erase(intersections.begin() + std::max(minIdx, startIndex));
                    intersections.erase(intersections.begin() + std::min(minIdx, startIndex));
                    startIndex = minIdx - 1;
                    isLoopStart = false;
                }
            }
        }

        // Check for boundary sections completely enclosed by the cell
        for (int b = 0; b < n_bdy; b++) {
            if (bdry->edges[b].nextEdge != b + 1) {
                TV point = bdry->v.segment<2>(b * 2);

                double w = 0; // Winding number
                for (int j = 0; j < degree; j++) {
                    double x1 = nodes[j](0);
                    double y1 = nodes[j](1);
                    double x2 = nodes[(j + 1) % degree](0);
                    double y2 = nodes[(j + 1) % degree](1);

                    double a = atan2(y2 - point.y(), x2 - point.x()) - atan2(y1 - point.y(), x1 - point.x());
                    if (a > M_PI) a -= 2 * M_PI;
                    if (a < -M_PI) a += 2 * M_PI;
                    w += a;
                }

                bool isEnclosed = w > M_PI;

                for (int j = 0; j < cells[i].edges.size(); j++) {
                    if (cells[i].edges[j].neighbor == n_vtx + b) {
                        isEnclosed = false;
                        break;
                    }
                }

                if (isEnclosed) {
                    int b2 = b;
                    int startEdge = cells[i].edges.size();
                    do {
                        cells[i].edges.emplace_back();
                        cells[i].edges.back().neighbor = n_vtx + b2;
                        cells[i].edges.back().nextEdge = cells[i].edges.size();
                        b2 = bdry->edges[b2].nextEdge;
                    } while (b2 != b);
                    cells[i].edges.back().nextEdge = startEdge;
                }
            }
        }
    }

    return true;
}

std::vector<std::vector<int>>
Tessellation::getNeighbors(const VectorXT &vertices, const VectorXi &dual, int n_cells) {
    int n_faces = dual.rows() / 3;

    std::vector<std::set<int>> cells(n_cells);

    for (int i = 0; i < n_faces; i++) {
        int v0 = dual(i * 3 + 0);
        int v1 = dual(i * 3 + 1);
        int v2 = dual(i * 3 + 2);

        if (v0 < n_cells) {
            cells[v0].insert(v1);
            cells[v0].insert(v2);
        }

        if (v1 < n_cells) {
            cells[v1].insert(v2);
            cells[v1].insert(v0);
        }

        if (v2 < n_cells) {
            cells[v2].insert(v0);
            cells[v2].insert(v1);
        }
    }

    std::vector<std::vector<int>> neighborLists;

    for (int i = 0; i < n_cells; i++) {
        std::vector<int> neighbors(cells[i].begin(), cells[i].end());

        double xc = vertices(i * 2 + 0);
        double yc = vertices(i * 2 + 1);

        std::sort(neighbors.begin(), neighbors.end(), [vertices, xc, yc](int a, int b) {
            double xa = vertices(a * 2 + 0);
            double ya = vertices(a * 2 + 1);
            double angle_a = atan2(ya - yc, xa - xc);

            double xb = vertices(b * 2 + 0);
            double yb = vertices(b * 2 + 1);
            double angle_b = atan2(yb - yc, xb - xc);

            return angle_a < angle_b;
        });

        neighborLists.push_back(neighbors);
    }

    return neighborLists;
}

//void
//Tessellation::getNodeWrapper(int i0, int i1, int i2, TV &node) {
//    int dims = 2 + getNumVertexParams();
//    int n_vtx = c.rows() / dims;
//
//    if (i2 < i1 && i2 < n_vtx) {
//        std::swap(i1, i2);
//    }
//
//    VectorXT v0 = c.segment(i0 * dims, dims); // i0 is always a site, never a boundary edge.
//    VectorXT v1, v2;
//    TV b0, b1;
//    if (i1 < n_vtx && i2 < n_vtx) {
//        // Normal node.
//        v1 = c.segment(i1 * dims, dims);
//        v2 = c.segment(i2 * dims, dims);
//        getNode(v0, v1, v2, node);
//    } else if (i1 < n_vtx && i2 >= n_vtx) {
//        // Boundary node with n2 a boundary edge.
//        v1 = c.segment(i1 * dims, dims);
//        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
//        b1 = bdry->v.segment<2>(bdry->next((i2 - n_vtx)) * 2);
//        getBoundaryNode(v0, v1, b0, b1, node);
//    } else {
//        // Boundary vertex.
//        assert(i1 >= n_vtx && i2 >= n_vtx);
//        node = bdry->v.segment<2>((i2 - n_vtx) * 2);
//    }
//}

void
Tessellation::getNodeWrapper(int i0, int i1, int i2, int flag, VectorXT &node, MatrixXT &nodeGrad,
                             std::vector<MatrixXT> &nodeHess, int &mode) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;

    VectorXT v0 = c.segment(i0 * dims, dims); // i0 is always a site, never a boundary edge.
    VectorXT v1, v2;
    TV b0, b1;
    double r, q0, q1;
    if (i1 < n_vtx && i2 < n_vtx) {
        // Normal node.
        mode = 0;

        v1 = c.segment(i1 * dims, dims);
        v2 = c.segment(i2 * dims, dims);
        getNode(v0, v1, v2, node);

        getNodeGradient(v0, v1, v2, nodeGrad);

        getNodeHessian(v0, v1, v2, nodeHess);
    } else if (i1 < n_vtx && i2 >= n_vtx && bdry->edges[i2 - n_vtx].btype == 0) {
        // Boundary intersection node with n2 a straight boundary segment.
        mode = 1;

        v1 = c.segment(i1 * dims, dims);
        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
        b1 = bdry->v.segment<2>(bdry->edges[i2 - n_vtx].nextEdge * 2);
        getBoundaryNode(v0, v1, b0, b1, node);

        getBoundaryNodeGradient(v0, v1, b0, b1, nodeGrad);

        getBoundaryNodeHessian(v0, v1, b0, b1, nodeHess);
    } else if (i1 < n_vtx && i2 >= n_vtx && bdry->edges[i2 - n_vtx].btype == 1) {
        // Boundary intersection node with n2 an arc boundary segment.
        mode = 2;

        v1 = c.segment(i1 * dims, dims);
        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
        b1 = bdry->v.segment<2>(bdry->edges[i2 - n_vtx].nextEdge * 2);
        r = bdry->q(bdry->edges[i2 - n_vtx].q_idx);
        getArcBoundaryNode(v0, v1, b0, b1, r, flag, node);

        getArcBoundaryNodeGradient(v0, v1, b0, b1, r, flag, nodeGrad);

//        double eps = 1e-7;
//        for (int i = 0; i < 11; i++) {
//            TV3 dv0 = TV3::Zero();
//            TV3 dv1 = TV3::Zero();
//            TV db0 = TV::Zero();
//            TV db1 = TV::Zero();
//            double dr = 0;
//
//            switch (i) {
//                case 0:
//                    dv0(0) = eps;
//                    break;
//                case 1:
//                    dv0(1) = eps;
//                    break;
//                case 2:
//                    dv0(2) = eps;
//                    break;
//                case 3:
//                    dv1(0) = eps;
//                    break;
//                case 4:
//                    dv1(1) = eps;
//                    break;
//                case 5:
//                    dv1(2) = eps;
//                    break;
//                case 6:
//                    db0(0) = eps;
//                    break;
//                case 7:
//                    db0(1) = eps;
//                    break;
//                case 8:
//                    db1(0) = eps;
//                    break;
//                case 9:
//                    db1(1) = eps;
//                    break;
//                case 10:
//                    dr = eps;
//                    break;
//            }
//
//            TV nodep;
//            getArcBoundaryNode(v0 + dv0, v1 + dv1, b0 + db0, b1 + db1, r + dr, flag, nodep);
//            std::cout << "grad " << i << " x " << gradX(i) << " " << (nodep(0) - node(0)) / eps << " "
//                      << nodep(0) - node(0) - eps * gradX[i] << std::endl;
//            std::cout << "grad " << i << " y " << gradY(i) << " " << (nodep(1) - node(1)) / eps << " "
//                      << nodep(1) - node(1) - eps * gradY[i] << std::endl;
//        }

        getArcBoundaryNodeHessian(v0, v1, b0, b1, r, flag, nodeHess);

//        double eps = 1e-7;
//        for (int i = 0; i < 11; i++) {
//            TV3 dv0 = TV3::Zero();
//            TV3 dv1 = TV3::Zero();
//            TV db0 = TV::Zero();
//            TV db1 = TV::Zero();
//            double dr = 0;
//
//            switch (i) {
//                case 0:
//                    dv0(0) = eps;
//                    break;
//                case 1:
//                    dv0(1) = eps;
//                    break;
//                case 2:
//                    dv0(2) = eps;
//                    break;
//                case 3:
//                    dv1(0) = eps;
//                    break;
//                case 4:
//                    dv1(1) = eps;
//                    break;
//                case 5:
//                    dv1(2) = eps;
//                    break;
//                case 6:
//                    db0(0) = eps;
//                    break;
//                case 7:
//                    db0(1) = eps;
//                    break;
//                case 8:
//                    db1(0) = eps;
//                    break;
//                case 9:
//                    db1(1) = eps;
//                    break;
//                case 10:
//                    dr = eps;
//                    break;
//            }
//
//            VectorXT gradXp(11), gradYp(11);
//            getArcBoundaryNodeGradient(v0 + dv0, v1 + dv1, b0 + db0, b1 + db1, r + dr, flag, gradXp, gradYp);
//
////            hessX(i, i) = (gradXp(i) - gradX(i)) / eps;
////            hessY(i, i) = (gradYp(i) - gradY(i)) / eps;
//
//            for (int j = 0; j < 11; j++) {
//                std::cout << "hess " << i << " " << j << " x " << hessX(j, i) << " " << (gradXp(j) - gradX(j)) / eps
//                          << std::endl;
//                std::cout << "hess " << i << " " << j << " y " << hessY(j, i) << " " << (gradYp(j) - gradY(j)) / eps
//                          << std::endl;
//            }
//        }
    } else if (i1 < n_vtx && i2 >= n_vtx && bdry->edges[i2 - n_vtx].btype == 2) {
        // Boundary intersection node with n2 an arc boundary segment.
        mode = 3;

        v1 = c.segment(i1 * dims, dims);
        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
        b1 = bdry->v.segment<2>(bdry->edges[i2 - n_vtx].nextEdge * 2);
        q0 = bdry->q(bdry->edges[i2 - n_vtx].q_idx);
        q1 = bdry->q(bdry->edges[bdry->edges[i2 - n_vtx].nextEdge].q_idx);
        getBezierBoundaryNode(v0, v1, b0, b1, q0, q1, flag, node);

        getBezierBoundaryNodeGradient(v0, v1, b0, b1, q0, q1, flag, nodeGrad);

//        double eps = 1e-7;
//        for (int i = 0; i < 11; i++) {
//            TV3 dv0 = TV3::Zero();
//            TV3 dv1 = TV3::Zero();
//            TV db0 = TV::Zero();
//            TV db1 = TV::Zero();
//            double dr = 0;
//
//            switch (i) {
//                case 0:
//                    dv0(0) = eps;
//                    break;
//                case 1:
//                    dv0(1) = eps;
//                    break;
//                case 2:
//                    dv0(2) = eps;
//                    break;
//                case 3:
//                    dv1(0) = eps;
//                    break;
//                case 4:
//                    dv1(1) = eps;
//                    break;
//                case 5:
//                    dv1(2) = eps;
//                    break;
//                case 6:
//                    db0(0) = eps;
//                    break;
//                case 7:
//                    db0(1) = eps;
//                    break;
//                case 8:
//                    db1(0) = eps;
//                    break;
//                case 9:
//                    db1(1) = eps;
//                    break;
//                case 10:
//                    dr = eps;
//                    break;
//            }
//
//            TV nodep;
//            getArcBoundaryNode(v0 + dv0, v1 + dv1, b0 + db0, b1 + db1, r + dr, flag, nodep);
//            std::cout << "grad " << i << " x " << gradX(i) << " " << (nodep(0) - node(0)) / eps << " "
//                      << nodep(0) - node(0) - eps * gradX[i] << std::endl;
//            std::cout << "grad " << i << " y " << gradY(i) << " " << (nodep(1) - node(1)) / eps << " "
//                      << nodep(1) - node(1) - eps * gradY[i] << std::endl;
//        }

        getBezierBoundaryNodeHessian(v0, v1, b0, b1, q0, q1, flag, nodeHess);

//        double eps = 1e-7;
//        for (int i = 0; i < 11; i++) {
//            TV3 dv0 = TV3::Zero();
//            TV3 dv1 = TV3::Zero();
//            TV db0 = TV::Zero();
//            TV db1 = TV::Zero();
//            double dr = 0;
//
//            switch (i) {
//                case 0:
//                    dv0(0) = eps;
//                    break;
//                case 1:
//                    dv0(1) = eps;
//                    break;
//                case 2:
//                    dv0(2) = eps;
//                    break;
//                case 3:
//                    dv1(0) = eps;
//                    break;
//                case 4:
//                    dv1(1) = eps;
//                    break;
//                case 5:
//                    dv1(2) = eps;
//                    break;
//                case 6:
//                    db0(0) = eps;
//                    break;
//                case 7:
//                    db0(1) = eps;
//                    break;
//                case 8:
//                    db1(0) = eps;
//                    break;
//                case 9:
//                    db1(1) = eps;
//                    break;
//                case 10:
//                    dr = eps;
//                    break;
//            }
//
//            VectorXT gradXp(11), gradYp(11);
//            getArcBoundaryNodeGradient(v0 + dv0, v1 + dv1, b0 + db0, b1 + db1, r + dr, flag, gradXp, gradYp);
//
////            hessX(i, i) = (gradXp(i) - gradX(i)) / eps;
////            hessY(i, i) = (gradYp(i) - gradY(i)) / eps;
//
//            for (int j = 0; j < 11; j++) {
//                std::cout << "hess " << i << " " << j << " x " << hessX(j, i) << " " << (gradXp(j) - gradX(j)) / eps
//                          << std::endl;
//                std::cout << "hess " << i << " " << j << " y " << hessY(j, i) << " " << (gradYp(j) - gradY(j)) / eps
//                          << std::endl;
//            }
//        }
    } else {
        // Boundary vertex.
        assert(i1 >= n_vtx && i2 >= n_vtx);
        mode = 4;

        TV b1s = bdry->v.segment<2>((i1 - n_vtx) * 2);
        TV b1e = bdry->v.segment<2>(bdry->edges[i1 - n_vtx].nextEdge * 2);

        assert(bdry->edges[i1 - n_vtx].nextEdge == i2 - n_vtx || bdry->edges[i2 - n_vtx].nextEdge == i1 - n_vtx);
        nodeGrad = MatrixXT::Zero(CellFunction::nx, 4);

        double q = 0;
        if (bdry->edges[i1 - n_vtx].nextEdge == i2 - n_vtx) {
            if (bdry->edges[i2 - n_vtx].btype != 0) {
                q = bdry->q(bdry->edges[i2 - n_vtx].q_idx);
            }
            node = TV3(b1e(0), b1e(1), q);
            nodeGrad(0, 2) = 1;
            nodeGrad(1, 3) = 1;
        } else {
            if (bdry->edges[i1 - n_vtx].btype != 0) {
                q = bdry->q(bdry->edges[i1 - n_vtx].q_idx);
            }
            node = TV3(b1s(0), b1s(1), q);
            nodeGrad(0, 0) = 1;
            nodeGrad(1, 1) = 1;
        }

        nodeHess.resize(CellFunction::nx);
        for (int i = 0; i < CellFunction::nx; i++) {
            nodeHess[i] = MatrixXT::Zero(0, 0);
        }
    }
}

void
Tessellation::addSingleCellFunctionValue(int cellIndex, const CellFunction &function, double &value,
                                         const CellInfo *cellInfo) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;

    Cell cell = cells[cellIndex];

    VectorXT site = VectorXT::Zero(dims);
    VectorXT nodes = VectorXT::Zero(cell.edges.size() * CellFunction::nx);
    VectorXi next = VectorXi::Zero(cell.edges.size());
    VectorXi btype = VectorXi::Zero(cell.edges.size());

    site = c.segment(cellIndex * dims, dims);
    for (int i = 0; i < cell.edges.size(); i++) {
        nodes.segment<CellFunction::nx>(i * CellFunction::nx) = x.segment<CellFunction::nx>(
                cell.edges[i].startNode * CellFunction::nx);
        next(i) = cell.edges[i].nextEdge;
        if (cell.edges[i].neighbor > n_vtx) {
            btype(i) = bdry->edges[cell.edges[i].neighbor - n_vtx].btype;
        }
    }

    function.addValue(site, nodes, next, btype, value, cellInfo);
}

void
Tessellation::addSingleCellFunctionGradient(int cellIndex, const CellFunction &function, VectorXT &gradient,
                                            const CellInfo *cellInfo) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_cells = cells.size();

    VectorXT dOdc = VectorXT::Zero(dims * n_cells);
    VectorXT dOdx = VectorXT::Zero(x.rows());

    Cell cell = cells[cellIndex];

    VectorXT site = VectorXT::Zero(dims);
    VectorXT nodes = VectorXT::Zero(cell.edges.size() * CellFunction::nx);
    VectorXi next = VectorXi::Zero(cell.edges.size());
    VectorXi btype = VectorXi::Zero(cell.edges.size());

    site = c.segment(cellIndex * dims, dims);
    for (int i = 0; i < cell.edges.size(); i++) {
        nodes.segment<CellFunction::nx>(i * CellFunction::nx) = x.segment<CellFunction::nx>(
                cell.edges[i].startNode * CellFunction::nx);
        next(i) = cell.edges[i].nextEdge;
        if (cell.edges[i].neighbor > n_vtx) {
            btype(i) = bdry->edges[cell.edges[i].neighbor - n_vtx].btype;
        }
    }

    VectorXT gradient_c = VectorXT::Zero(site.rows());
    VectorXT gradient_x = VectorXT::Zero(nodes.rows());
    function.addGradient(site, nodes, next, btype, gradient_c, gradient_x, cellInfo);

    dOdc.segment(cellIndex * dims, dims) += gradient_c;
    for (int i = 0; i < cell.edges.size(); i++) {
        dOdx.segment<CellFunction::nx>(
                cell.edges[i].startNode * CellFunction::nx) += gradient_x.segment<CellFunction::nx>(
                i * CellFunction::nx);
    }

    gradient.segment(0, dims * n_cells) += dxdc.transpose() * dOdx + dOdc;
    if (bdry->nfree > 0) {
        gradient.segment(dims * n_cells, bdry->nfree) += (dxdv * bdry->dvdp).transpose() * dOdx;
        if (bdry->q.rows() > 0) {
            gradient.segment(dims * n_cells, bdry->nfree) += (dxdq * bdry->dqdp).transpose() * dOdx;
        }
    }
}

void
Tessellation::addFunctionValue(const CellFunction &function, double &value,
                               const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;

    for (int cellIndex = 0; cellIndex < cells.size(); cellIndex++) {
        Cell cell = cells[cellIndex];

        VectorXT site = VectorXT::Zero(dims);
        VectorXT nodes = VectorXT::Zero(cell.edges.size() * CellFunction::nx);
        VectorXi next = VectorXi::Zero(cell.edges.size());
        VectorXi btype = VectorXi::Zero(cell.edges.size());

        site = c.segment(cellIndex * dims, dims);
        for (int i = 0; i < cell.edges.size(); i++) {
            nodes.segment<CellFunction::nx>(i * CellFunction::nx) = x.segment<CellFunction::nx>(
                    cell.edges[i].startNode * CellFunction::nx);
            next(i) = cell.edges[i].nextEdge;
            if (cell.edges[i].neighbor > n_vtx) {
                btype(i) = bdry->edges[cell.edges[i].neighbor - n_vtx].btype;
            }
        }

        function.addValue(site, nodes, next, btype, value, &cellInfos[cellIndex]);
    }
}

void Tessellation::addFunctionGradient(const CellFunction &function, VectorXT &gradient,
                                       const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_cells = cells.size();

    VectorXT partial_c = VectorXT::Zero(dims * n_cells);
    VectorXT partial_x = VectorXT::Zero(x.rows());

    for (int cellIndex = 0; cellIndex < n_cells; cellIndex++) {
        Cell cell = cells[cellIndex];

        VectorXT site = VectorXT::Zero(dims);
        VectorXT nodes = VectorXT::Zero(cell.edges.size() * CellFunction::nx);
        VectorXi next = VectorXi::Zero(cell.edges.size());
        VectorXi btype = VectorXi::Zero(cell.edges.size());

        site = c.segment(cellIndex * dims, dims);
        for (int i = 0; i < cell.edges.size(); i++) {
            nodes.segment<CellFunction::nx>(i * CellFunction::nx) = x.segment<CellFunction::nx>(
                    cell.edges[i].startNode * CellFunction::nx);
            next(i) = cell.edges[i].nextEdge;
            if (cell.edges[i].neighbor > n_vtx) {
                btype(i) = bdry->edges[cell.edges[i].neighbor - n_vtx].btype;
            }
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, next, btype, gradient_c, gradient_x, &cellInfos[cellIndex]);

        partial_c.segment(cellIndex * dims, dims) += gradient_c;
        for (int i = 0; i < cell.edges.size(); i++) {
            partial_x.segment<CellFunction::nx>(
                    cell.edges[i].startNode * CellFunction::nx) += gradient_x.segment<CellFunction::nx>(
                    i * CellFunction::nx);
        }
    }

    Eigen::Ref<VectorXT> gradient_c = gradient.segment(0, dims * n_cells);
    Eigen::Ref<VectorXT> gradient_p = gradient.segment(dims * n_cells, bdry->nfree);

    gradient_c += dxdc.transpose() * partial_x + partial_c;
    if (bdry->nfree > 0) {
        gradient_p += (dxdv * bdry->dvdp).transpose() * partial_x;
        if (bdry->q.rows() > 0) {
            gradient_p += (dxdq * bdry->dqdp).transpose() * partial_x;
        }
    }
}

void Tessellation::addFunctionHessian(const CellFunction &function, MatrixXT &hessian,
                                      const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_cells = cells.size();

    MatrixXT partial_cc = MatrixXT::Zero(dims * n_cells, dims * n_cells);
    MatrixXT partial_cx = MatrixXT::Zero(dims * n_cells, x.rows());
    MatrixXT partial_xx = MatrixXT::Zero(x.rows(), x.rows());
    VectorXT partial_x = VectorXT::Zero(x.rows());

    for (int cellIndex = 0; cellIndex < n_cells; cellIndex++) {
        Cell cell = cells[cellIndex];

        VectorXT site = VectorXT::Zero(dims);
        VectorXT nodes = VectorXT::Zero(cell.edges.size() * CellFunction::nx);
        VectorXi next = VectorXi::Zero(cell.edges.size());
        VectorXi btype = VectorXi::Zero(cell.edges.size());

        site = c.segment(cellIndex * dims, dims);
        for (int i = 0; i < cell.edges.size(); i++) {
            nodes.segment<CellFunction::nx>(i * CellFunction::nx) = x.segment<CellFunction::nx>(
                    cell.edges[i].startNode * CellFunction::nx);
            next(i) = cell.edges[i].nextEdge;
            if (cell.edges[i].neighbor > n_vtx) {
                btype(i) = bdry->edges[cell.edges[i].neighbor - n_vtx].btype;
            }
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, next, btype, gradient_c, gradient_x, &cellInfos[cellIndex]);

        for (int i = 0; i < cell.edges.size(); i++) {
            partial_x.segment<CellFunction::nx>(
                    cell.edges[i].startNode * CellFunction::nx) += gradient_x.segment<CellFunction::nx>(
                    i * CellFunction::nx);
        }

        MatrixXT hessian_local = MatrixXT::Zero(site.rows() + nodes.rows(), site.rows() + nodes.rows());
        function.addHessian(site, nodes, next, btype, hessian_local, &cellInfos[cellIndex]);

        Eigen::Ref<MatrixXT> hessian_local_cc = hessian_local.block(0, 0, site.rows(), site.rows());
        Eigen::Ref<MatrixXT> hessian_local_cx = hessian_local.block(0, site.rows(), site.rows(), nodes.rows());
        Eigen::Ref<MatrixXT> hessian_local_xx = hessian_local.block(site.rows(), site.rows(), nodes.rows(),
                                                                    nodes.rows());

        partial_cc.block(cellIndex * dims, cellIndex * dims, dims, dims) += hessian_local_cc;
        for (int i = 0; i < cell.edges.size(); i++) {
            partial_cx.block(cellIndex * dims, cell.edges[i].startNode * CellFunction::nx, dims,
                             CellFunction::nx) += hessian_local_cx.block(0, i *
                                                                            CellFunction::nx,
                                                                         dims, CellFunction::nx);
            for (int j = 0; j < cell.edges.size(); j++) {
                partial_xx.block<CellFunction::nx, CellFunction::nx>(cell.edges[i].startNode * CellFunction::nx,
                                                                     cell.edges[j].startNode *
                                                                     CellFunction::nx) += hessian_local_xx.block<CellFunction::nx, CellFunction::nx>(
                        i * CellFunction::nx,
                        j * CellFunction::nx);
            }
        }
    }

    Eigen::Ref<MatrixXT> hessian_cc = hessian.block(0, 0, dims * n_cells, dims * n_cells);
    Eigen::Ref<MatrixXT> hessian_cp = hessian.block(0, dims * n_cells, dims * n_cells, bdry->nfree);
    Eigen::Ref<MatrixXT> hessian_pp = hessian.block(dims * n_cells, dims * n_cells, bdry->nfree, bdry->nfree);

    MatrixXT sum_f_cc = MatrixXT::Zero(dims * n_cells, dims * n_cells);
    MatrixXT sum_f_cv = MatrixXT::Zero(dims * n_cells, bdry->v.rows());
    MatrixXT sum_f_vv = MatrixXT::Zero(bdry->v.rows(), bdry->v.rows());
    MatrixXT sum_f_cq = MatrixXT::Zero(dims * n_cells, bdry->q.rows());
    MatrixXT sum_f_qq = MatrixXT::Zero(bdry->q.rows(), bdry->q.rows());
    MatrixXT sum_f_vq = MatrixXT::Zero(bdry->v.rows(), bdry->q.rows());
    for (int ii = 0; ii < x.rows(); ii++) {
        IV3 face = dual.segment<3>(4 * ((int) (ii / CellFunction::nx)));
        MatrixXT hess = d2xdy2[ii];

        // Boundary corner vertices and nonexistent curve parameters
        if (hess.rows() == 0) continue;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (face(i) >= n_cells || face(j) >= n_cells) {
                    continue;
                }
                sum_f_cc.block(face(i) * dims, face(j) * dims, dims, dims) +=
                        partial_x(ii) * hess.block(i * dims, j * dims, dims, dims);
            }
        }

        int n_sites = c.rows() / dims;
        // Boundary edge vertices
        if (face(2) >= n_sites) {
            int ib0 = (face(2) - n_sites) * 2;
            int ib1 = bdry->edges[face(2) - n_sites].nextEdge * 2;

            for (int i = 0; i < 2; i++) {
                if (face(i) >= n_cells) continue;

                sum_f_cv.block(face(i) * dims, ib0, dims, 2) +=
                        partial_x(ii) * hess.block(i * dims, 2 * dims + 0, dims, 2);
                sum_f_cv.block(face(i) * dims, ib1, dims, 2) +=
                        partial_x(ii) * hess.block(i * dims, 2 * dims + 2, dims, 2);
            }

            sum_f_vv.block(ib0, ib0, 2, 2) += partial_x(ii) * hess.block(2 * dims + 0, 2 * dims + 0, 2, 2);
            sum_f_vv.block(ib0, ib1, 2, 2) += partial_x(ii) * hess.block(2 * dims + 0, 2 * dims + 2, 2, 2);
            sum_f_vv.block(ib1, ib0, 2, 2) += partial_x(ii) * hess.block(2 * dims + 2, 2 * dims + 0, 2, 2);
            sum_f_vv.block(ib1, ib1, 2, 2) += partial_x(ii) * hess.block(2 * dims + 2, 2 * dims + 2, 2, 2);

            if (bdry->q.rows() > 0) {
                if (bdry->edges[face(2) - n_sites].btype == 1) {
                    int iq = bdry->edges[face(2) - n_sites].q_idx;

                    for (int i = 0; i < 2; i++) {
                        if (face(i) >= n_cells) continue;

                        sum_f_cq.block(face(i) * dims, iq, dims, 1) +=
                                partial_x(ii) * hess.block(i * dims, 2 * dims + 4, dims, 1);
                    }

                    sum_f_qq.block(iq, iq, 1, 1) += partial_x(ii) * hess.block(2 * dims + 4, 2 * dims + 4, 1, 1);

                    sum_f_vq.block(ib0, iq, 2, 1) += partial_x(ii) * hess.block(2 * dims + 0, 2 * dims + 4, 2, 1);
                    sum_f_vq.block(ib1, iq, 2, 1) += partial_x(ii) * hess.block(2 * dims + 2, 2 * dims + 4, 2, 1);
                } else if (bdry->edges[face(2) - n_sites].btype == 2) {
                    int iq0 = bdry->edges[face(2) - n_sites].q_idx;
                    int iq1 = bdry->edges[bdry->edges[face(2) - n_sites].nextEdge].q_idx;

                    for (int i = 0; i < 2; i++) {
                        if (face(i) >= n_cells) continue;

                        sum_f_cq.block(face(i) * dims, iq0, dims, 1) +=
                                partial_x(ii) * hess.block(i * dims, 2 * dims + 2, dims, 1);
                        sum_f_cq.block(face(i) * dims, iq1, dims, 1) +=
                                partial_x(ii) * hess.block(i * dims, 2 * dims + 5, dims, 1);
                    }

                    sum_f_qq(iq0, iq0) += partial_x(ii) * hess(2 * dims + 2, 2 * dims + 2);
                    sum_f_qq(iq0, iq1) += partial_x(ii) * hess(2 * dims + 2, 2 * dims + 5);
                    sum_f_qq(iq1, iq0) += partial_x(ii) * hess(2 * dims + 5, 2 * dims + 2);
                    sum_f_qq(iq1, iq1) += partial_x(ii) * hess(2 * dims + 5, 2 * dims + 5);

                    sum_f_vq.block(ib0, iq0, 2, 1) += partial_x(ii) * hess.block(2 * dims + 0, 2 * dims + 2, 2, 1);
                    sum_f_vq.block(ib1, iq0, 2, 1) += partial_x(ii) * hess.block(2 * dims + 2, 2 * dims + 2, 2, 1);
                    sum_f_vq.block(ib0, iq1, 2, 1) += partial_x(ii) * hess.block(2 * dims + 0, 2 * dims + 5, 2, 1);
                    sum_f_vq.block(ib1, iq1, 2, 1) += partial_x(ii) * hess.block(2 * dims + 2, 2 * dims + 5, 2, 1);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> dxdc_T = dxdc.transpose();

    hessian_cc +=
            partial_cc + partial_cx * dxdc + dxdc_T * (partial_cx.transpose() + partial_xx * dxdc) + sum_f_cc;

    if (bdry->nfree > 0) {
        if (bdry->q.rows() > 0) {
            Eigen::SparseMatrix<double> dxdp = (dxdv * bdry->dvdp + dxdq * bdry->dqdp).sparseView();
            hessian_cp +=
                    dxdc_T * partial_xx * dxdp + partial_cx * dxdp + sum_f_cv * bdry->dvdp + sum_f_cq * bdry->dqdp;

            MatrixXT sum_x_pp = MatrixXT::Zero(bdry->nfree, bdry->nfree);
            MatrixXT partial_x_T = partial_x.transpose();
            for (int ii = 0; ii < bdry->v.rows(); ii++) {
                sum_x_pp += (partial_x_T * dxdv.col(ii)).coeff(0, 0) * bdry->d2vdp2[ii];
            }
            for (int ii = 0; ii < bdry->q.rows(); ii++) {
                sum_x_pp += (partial_x_T * dxdq.col(ii)).coeff(0, 0) * bdry->d2qdp2[ii];
            }
            hessian_pp +=
                    dxdp.transpose() * partial_xx * dxdp +
                    bdry->dvdp.transpose() * sum_f_vv * bdry->dvdp +
                    bdry->dqdp.transpose() * sum_f_qq * bdry->dqdp +
                    bdry->dvdp.transpose() * sum_f_vq * bdry->dqdp +
                    (bdry->dvdp.transpose() * sum_f_vq * bdry->dqdp).transpose() +
                    sum_x_pp;
        } else {
            Eigen::SparseMatrix<double> dxdp = (dxdv * bdry->dvdp).sparseView();
            hessian_cp += dxdc_T * partial_xx * dxdp + partial_cx * dxdp + sum_f_cv * bdry->dvdp;

            MatrixXT sum_x_pp = MatrixXT::Zero(bdry->nfree, bdry->nfree);
            MatrixXT partial_x_T = partial_x.transpose();
            for (int ii = 0; ii < bdry->v.rows(); ii++) {
                sum_x_pp += (partial_x_T * dxdv.col(ii)).coeff(0, 0) * bdry->d2vdp2[ii];
            }
            hessian_pp +=
                    dxdp.transpose() * partial_xx * dxdp + bdry->dvdp.transpose() * sum_f_vv * bdry->dvdp + sum_x_pp;
        }

        hessian.block(dims * n_cells, 0, bdry->nfree, dims * n_cells) = hessian_cp.transpose();
    }
}

void Tessellation::tessellate(const VectorXT &vertices, const VectorXT &params, Boundary *bdry_new, const int n_free) {
    VectorXT c_new = combineVerticesParams(vertices, params);

    // Check if inputs are the same as previous tessellation, do nothing if so.
    bool same = true;
    same = same && (c_new.rows() == c.rows() && c_new.isApprox(c));
    same = same && (bdry_new->v.rows() == last_boundary.rows() && bdry_new->v == last_boundary);
    same = same && (n_free == cells.size());
    if (same) return;

    isValid = true;
    bdry = bdry_new;
    last_boundary = bdry_new->v;
    c = c_new;

    if (!bdry->checkValid()) {
        isValid = false;
        return;
    }

    VectorXi dualRaw = getDualGraph(vertices, params);
    if (!getNeighborsClipped(vertices, params, dualRaw, n_free)) {
        isValid = false;
        return;
    }

    auto cmp = [](IV4 a, IV4 b) {
        for (int i = 0; i < 4; i++) {
            if (a(i) < b(i)) return true;
            if (a(i) > b(i)) return false;
        }
        return false;
    };
    std::set<IV4, decltype(cmp)> triplets(cmp);

    for (int i = 0; i < n_free; i++) {
        Cell cell = cells[i];
        for (int j = 0; j < cell.edges.size(); j++) {
            int n1 = cell.edges[j].neighbor;
            int n2 = cell.edges[cell.edges[j].nextEdge].neighbor;
            int flag = cell.edges[cell.edges[j].nextEdge].flag;

            IV4 triplet(i, n1, n2, flag);
            std::sort(triplet.data(), triplet.data() + 3);

            triplets.insert(triplet);
        }
    }
    std::vector<IV4> faces(triplets.begin(), triplets.end());

    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_bdy = bdry->v.rows() / 2;
    for (int i = 0; i < n_free; i++) {
        Cell cell = cells[i];
        for (int j = 0; j < cell.edges.size(); j++) {
            int n1 = cell.edges[j].neighbor;
            int n2 = cell.edges[cell.edges[j].nextEdge].neighbor;
            int flag = cell.edges[cell.edges[j].nextEdge].flag;

            IV4 triplet(i, n1, n2, flag);
            std::sort(triplet.data(), triplet.data() + 3);

            auto lower = std::lower_bound(faces.begin(), faces.end(), triplet, cmp);
            assert(lower != faces.end() && *lower == triplet);

            // Node between current and next edge is startNode of NEXT edge, not current.
            cells[i].edges[cell.edges[j].nextEdge].startNode = std::distance(faces.begin(), lower);
        }
    }

    dual.resize(faces.size() * 4);
    for (int i = 0; i < faces.size(); i++) {
        dual.segment<4>(i * 4) = faces[i];
    }

    x.resize(faces.size() * CellFunction::nx);
    x.setZero();
    MatrixXT dxdc_dense = MatrixXT::Zero(x.rows(), n_free * dims);
    MatrixXT dxdv_dense = MatrixXT::Zero(x.rows(), bdry->v.rows());
    MatrixXT dxdq_dense = MatrixXT::Zero(x.rows(), bdry->q.rows());
    d2xdy2.resize(x.rows());
    for (int i = 0; i < faces.size(); i++) {
        IV4 face = faces[i];

        VectorXT node;
        MatrixXT nodeGrad;
        std::vector<MatrixXT> nodeHess;
        int type;
        getNodeWrapper(face[0], face[1], face[2], face[3], node, nodeGrad, nodeHess,
                       type);

        x.segment<CellFunction::nx>(i * CellFunction::nx) = node;

        for (int j = 0; j < CellFunction::nx; j++) {
            d2xdy2[i * CellFunction::nx + j] = nodeHess[j];
        }

        // Assemble global Jacobian matrix dxdc.
        int n_sites = c.rows() / dims;
        switch (type) {
            case 0:
                for (int j = 0; j < 3; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
                }
                break;
            case 1:
                for (int j = 0; j < 2; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
                }
                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
                                 2) =
                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 2);

                break;
            case 2:
                for (int j = 0; j < 2; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
                }
                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
                                 2) =
                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 2);

                dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].q_idx, CellFunction::nx, 1) =
                        nodeGrad.block(0, 2 * dims + 4, CellFunction::nx, 1);
                break;
            case 3:
                for (int j = 0; j < 2; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    dxdc_dense.block(i * CellFunction::nx, face[j] * dims, CellFunction::nx, dims) =
                            nodeGrad.block(0, j * dims, CellFunction::nx, dims);
                }
                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
                        nodeGrad.block(0, 2 * dims + 0, CellFunction::nx, 2);
                dxdv_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].nextEdge * 2, CellFunction::nx,
                                 2) =
                        nodeGrad.block(0, 2 * dims + 3, CellFunction::nx, 2);

                dxdq_dense.block(i * CellFunction::nx, bdry->edges[face[2] - n_sites].q_idx, CellFunction::nx, 1) =
                        nodeGrad.block(0, 2 * dims + 2, CellFunction::nx, 1);
                dxdq_dense.block(i * CellFunction::nx, bdry->edges[bdry->edges[face[2] - n_sites].nextEdge].q_idx,
                                 CellFunction::nx, 1) =
                        nodeGrad.block(0, 2 * dims + 5, CellFunction::nx, 1);
                break;
            case 4:
                dxdv_dense.block(i * CellFunction::nx, (face[1] - n_sites) * 2, CellFunction::nx, 2) =
                        nodeGrad.block(0, 0, CellFunction::nx, 2);
                dxdv_dense.block(i * CellFunction::nx, (face[2] - n_sites) * 2, CellFunction::nx, 2) =
                        nodeGrad.block(0, 2, CellFunction::nx, 2);
                break;
            default:
                assert(0);
                break;
        }
    }

    dxdc = dxdc_dense.sparseView();
    dxdv = dxdv_dense.sparseView();
    dxdq = dxdq_dense.sparseView();

    for (int i = 0; i < n_free; i++) {
        if (cells[i].edges.size() < 2) {
            isValid = false;
            return;
        }
    }
}
