#include "../../include/Tessellation/Tessellation.h"
#include "../../include/Tessellation/CellFunction.h"
#include "../../include/Boundary/Boundary.h"
#include <set>
#include <iostream>

static bool pointInPolygon(const TV &point, const VectorXT &polygon) {
    int np = polygon.rows() / 2;

    double w = 0; // Winding number
    for (int i = 0; i < np; i++) {
        double x1 = polygon(2 * i + 0);
        double y1 = polygon(2 * i + 1);
        double x2 = polygon(2 * ((i + 1) % np) + 0);
        double y2 = polygon(2 * ((i + 1) % np) + 1);

        double a = atan2(y2 - point.y(), x2 - point.x()) - atan2(y1 - point.y(), x1 - point.x());
        if (a > M_PI) a -= 2 * M_PI;
        if (a < -M_PI) a += 2 * M_PI;
        w += a;
    }

    return fabs(w) > M_PI;
}

static bool lineSegmentIntersection(const TV &p0, const TV &p1, const TV &p2, const TV &p3, TV &intersect) {
    double s1_x, s1_y, s2_x, s2_y;
    s1_x = p1.x() - p0.x();
    s1_y = p1.y() - p0.y();
    s2_x = p3.x() - p2.x();
    s2_y = p3.y() - p2.y();

    double s, t;
    s = (-s1_y * (p0.x() - p2.x()) + s1_x * (p0.y() - p2.y())) / (-s2_x * s1_y + s1_x * s2_y);
    t = (s2_x * (p0.y() - p2.y()) - s2_y * (p0.x() - p2.x())) / (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
        // Collision detected
        intersect.x() = p0.x() + (t * s1_x);
        intersect.y() = p0.y() + (t * s1_y);
        return true;
    }

    return false; // No collision
}

std::vector<std::vector<int>>
Tessellation::getNeighborsClipped(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual, int n_cells) {
    int n_vtx = vertices.rows() / 2, n_bdy = bdry->v.rows() / 2;

    std::vector<std::vector<int>> neighborsRaw = getNeighbors(vertices, dual, n_cells);
    if (n_bdy == 0) return neighborsRaw;

    std::vector<std::vector<int>> neighborsClipped(n_cells);

    VectorXT c = combineVerticesParams(vertices, params);
    int dims = 2 + getNumVertexParams();

    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &neighbors = neighborsRaw[i];
        size_t degree = neighbors.size();

        VectorXT c0 = c.segment(i * dims, dims);
        std::vector<TV> nodes(degree);
        std::vector<bool> inPoly(degree);

        for (size_t j = 0; j < degree; j++) {
            int n1 = neighbors[j];
            int n2 = neighbors[(j + 1) % degree];

            TV v;
            getNode(c0, c.segment(n1 * dims, dims), c.segment(n2 * dims, dims), v);
            nodes[j] = v;
            inPoly[j] = pointInPolygon(v, bdry->v);
        }

        for (size_t j = 0; j < degree; j++) {
            bool inPoly0 = inPoly[j];
            bool inPoly1 = inPoly[(j + 1) % degree];

            TV v0 = nodes[j];
            TV v1 = nodes[(j + 1) % degree];

            if (inPoly0 && inPoly1) {
                // Just add neighbor.
                neighborsClipped[i].push_back(neighbors[(j + 1) % degree]);
            } else if (inPoly0 && !inPoly1) {
                // Add neighbor and then boundary edge.
                neighborsClipped[i].push_back(neighbors[(j + 1) % degree]);
                for (size_t k = 0; k < n_bdy; k++) {
                    TV v2 = bdry->v.segment<2>(k * 2);
                    TV v3 = bdry->v.segment<2>(((k + 1) % n_bdy) * 2);
                    TV intersect;
                    if (lineSegmentIntersection(v0, v1, v2, v3, intersect)) {
                        neighborsClipped[i].push_back(n_vtx + k);
                        break; // Can only be one intersection.
                    }
                }
            } else if (!inPoly0 && inPoly1) {
                // Add boundary edge and then neighbor.
                for (size_t k = 0; k < n_bdy; k++) {
                    TV v2 = bdry->v.segment<2>(k * 2);
                    TV v3 = bdry->v.segment<2>(((k + 1) % n_bdy) * 2);
                    TV intersect;
                    if (lineSegmentIntersection(v0, v1, v2, v3, intersect)) {
                        neighborsClipped[i].push_back(n_vtx + k);
                        break; // Can only be one intersection.
                    }
                }
                neighborsClipped[i].push_back(neighbors[(j + 1) % degree]);
            } else {
                // Check if zero or two intersections.
                assert(!inPoly0 && !inPoly1);
                std::vector<int> intersectIndices;
                std::vector<double> intersectDistances;
                for (size_t k = 0; k < n_bdy; k++) {
                    TV v2 = bdry->v.segment<2>(k * 2);
                    TV v3 = bdry->v.segment<2>(((k + 1) % n_bdy) * 2);
                    TV intersect;
                    if (lineSegmentIntersection(v0, v1, v2, v3, intersect)) {
                        intersectIndices.push_back(k);
                        intersectDistances.push_back((intersect - v0).norm());
                    }
                }

                assert(intersectIndices.size() == 0 || intersectIndices.size() == 2);
                // If zero, do nothing.
                // If two, add closer edge, then neighbor, then farther edge.
                if (intersectIndices.size() == 2) {
                    if (intersectDistances[0] < intersectDistances[1]) {
                        neighborsClipped[i].push_back(n_vtx + intersectIndices[0]);
                        neighborsClipped[i].push_back(neighbors[(j + 1) % degree]);
                        neighborsClipped[i].push_back(n_vtx + intersectIndices[1]);
                    } else {
                        neighborsClipped[i].push_back(n_vtx + intersectIndices[1]);
                        neighborsClipped[i].push_back(neighbors[(j + 1) % degree]);
                        neighborsClipped[i].push_back(n_vtx + intersectIndices[0]);
                    }
                }
            }
        }

        size_t clippedDegree = neighborsClipped[i].size();
        for (int j = 0; j < clippedDegree; j++) {
            int n1 = neighborsClipped[i][j];
            int n2 = neighborsClipped[i][(j + 1) % clippedDegree];

            if (n1 >= n_vtx && n2 >= n_vtx) {
                if (n1 == n2) {
                    neighborsClipped[i].erase(neighborsClipped[i].begin() + j);
                    clippedDegree--;
                    j--;
                } else if ((n1 - n_vtx + 1) % n_bdy + n_vtx == n2) {
                    // Do nothing
                } else {
                    neighborsClipped[i].insert(neighborsClipped[i].begin() + j + 1, (n1 - n_vtx + 1) % n_bdy + n_vtx);
                    clippedDegree++;
                }
            }
        }
    }

    return neighborsClipped;
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

void
Tessellation::getNodeWrapper(int i0, int i1, int i2, TV &node) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_bdy = bdry->v.rows() / 2;

    if (i2 < i1 && i2 < n_vtx) {
        std::swap(i1, i2);
    }

    VectorXT v0 = c.segment(i0 * dims, dims); // i0 is always a site, never a boundary edge.
    VectorXT v1, v2;
    TV b0, b1;
    if (i1 < n_vtx && i2 < n_vtx) {
        // Normal node.
        v1 = c.segment(i1 * dims, dims);
        v2 = c.segment(i2 * dims, dims);
        getNode(v0, v1, v2, node);
    } else if (i1 < n_vtx && i2 >= n_vtx) {
        // Boundary node with n2 a boundary edge.
        v1 = c.segment(i1 * dims, dims);
        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
        b1 = bdry->v.segment<2>(((i2 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);
    } else {
        // Boundary vertex.
        assert(i1 >= n_vtx && i2 >= n_vtx);
        node = bdry->v.segment<2>((i2 - n_vtx) * 2);
    }
}

void
Tessellation::getNodeWrapper(int i0, int i1, int i2, TV &node, VectorXT &gradX, VectorXT &gradY, MatrixXT &hessX,
                             MatrixXT &hessY, int &mode) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_bdy = bdry->v.rows() / 2;

    VectorXT v0 = c.segment(i0 * dims, dims); // i0 is always a site, never a boundary edge.
    VectorXT v1, v2;
    TV b0, b1;
    if (i1 < n_vtx && i2 < n_vtx) {
        // Normal node.
        mode = 0;

        v1 = c.segment(i1 * dims, dims);
        v2 = c.segment(i2 * dims, dims);
        getNode(v0, v1, v2, node);

        gradX.resize(dims * 3);
        gradY.resize(dims * 3);
        getNodeGradient(v0, v1, v2, gradX, gradY);

        hessX.resize(dims * 3, dims * 3);
        hessY.resize(dims * 3, dims * 3);
        getNodeHessian(v0, v1, v2, hessX, hessY);
    } else if (i1 < n_vtx && i2 >= n_vtx) {
        // Boundary node with n2 a boundary edge.
        mode = 1;

        v1 = c.segment(i1 * dims, dims);
        b0 = bdry->v.segment<2>((i2 - n_vtx) * 2);
        b1 = bdry->v.segment<2>(((i2 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);

        gradX.resize(dims * 2 + 4);
        gradY.resize(dims * 2 + 4);
        getBoundaryNodeGradient(v0, v1, b0, b1, gradX, gradY);

        hessX.resize(dims * 2 + 4, dims * 2 + 4);
        hessY.resize(dims * 2 + 4, dims * 2 + 4);
        getBoundaryNodeHessian(v0, v1, b0, b1, hessX, hessY);
    } else {
        // Boundary vertex.
        assert(i1 >= n_vtx && i2 >= n_vtx);
        mode = 2;

        TV b1s = bdry->v.segment<2>((i1 - n_vtx) * 2);
        TV b1e = bdry->v.segment<2>(((i1 + 1 - n_vtx) % n_bdy) * 2);
        TV b2s = bdry->v.segment<2>((i2 - n_vtx) * 2);
        TV b2e = bdry->v.segment<2>(((i2 + 1 - n_vtx) % n_bdy) * 2);

        assert(b1e == b2s || b2e == b1s);
        gradX = VectorXT::Zero(4);
        gradY = VectorXT::Zero(4);
        if (b1e == b2s) {
            node = b1e;
            gradX(2) = 1;
            gradY(3) = 1;
        } else {
            node = b1s;
            gradX(0) = 1;
            gradY(1) = 1;
        }

        hessX.resize(0, 0);
        hessY.resize(0, 0);
    }
}

void
Tessellation::addSingleCellFunctionValue(int cell, const CellFunction &function, double &value,
                                         const CellInfo *cellInfo) {
    VectorXi nodeIndices = cells[cell];

    VectorXT site(2);
    VectorXT nodes(nodeIndices.rows() * 2);

    int dims = 2 + getNumVertexParams();
    site = c.segment<2>(cell * dims);
    for (int i = 0; i < nodeIndices.rows(); i++) {
        nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
    }

    function.addValue(site, nodes, value, cellInfo);
}

void
Tessellation::addSingleCellFunctionGradient(int cell, const CellFunction &function, VectorXT &gradient,
                                            const CellInfo *cellInfo) {
    int dims = 2 + getNumVertexParams();
    int n_cells = cells.size();

    VectorXT dOdc = VectorXT::Zero(dims * n_cells);
    VectorXT dOdx = VectorXT::Zero(x.rows());

    VectorXi nodeIndices = cells[cell];

    VectorXT site(2);
    VectorXT nodes(nodeIndices.rows() * 2);

    site = c.segment<2>(cell * dims);
    for (int i = 0; i < nodeIndices.rows(); i++) {
        nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
    }

    VectorXT gradient_c = VectorXT::Zero(site.rows());
    VectorXT gradient_x = VectorXT::Zero(nodes.rows());
    function.addGradient(site, nodes, gradient_c, gradient_x, cellInfo);

    dOdc.segment<2>(cell * dims) += gradient_c;
    for (int i = 0; i < nodeIndices.rows(); i++) {
        dOdx.segment<2>(nodeIndices(i) * 2) += gradient_x.segment<2>(i * 2);
    }

    gradient += dxdc.transpose() * dOdx + dOdc;
}

void
Tessellation::addFunctionValue(const CellFunction &function, double &value,
                               const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    for (int cell = 0; cell < cells.size(); cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(dims);
        VectorXT nodes(nodeIndices.rows() * 2);

        site = c.segment(cell * dims, dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        function.addValue(site, nodes, value, &cellInfos[cell]);
    }
}

void Tessellation::addFunctionGradient(const CellFunction &function, VectorXT &gradient,
                                       const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    int n_cells = cells.size();

    VectorXT partial_c = VectorXT::Zero(dims * n_cells);
    VectorXT partial_x = VectorXT::Zero(x.rows());

    for (int cell = 0; cell < n_cells; cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(dims);
        VectorXT nodes(nodeIndices.rows() * 2);

        site = c.segment(cell * dims, dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, gradient_c, gradient_x, &cellInfos[cell]);

        partial_c.segment(cell * dims, dims) += gradient_c;
        for (int i = 0; i < nodeIndices.rows(); i++) {
            partial_x.segment<2>(nodeIndices(i) * 2) += gradient_x.segment<2>(i * 2);
        }
    }

    Eigen::Ref<VectorXT> gradient_c = gradient.segment(0, dims * n_cells);
    Eigen::Ref<VectorXT> gradient_p = gradient.segment(dims * n_cells, bdry->nfree);

    gradient_c += dxdc.transpose() * partial_x + partial_c;
    if (bdry->nfree > 0) {
        // TODO: Add partial_p
        gradient_p += (dxdv * bdry->dvdp).transpose() * partial_x;
    }
}

void Tessellation::addFunctionHessian(const CellFunction &function, MatrixXT &hessian,
                                      const std::vector<CellInfo> cellInfos) {
    int dims = 2 + getNumVertexParams();
    int n_cells = cells.size();

    MatrixXT partial_cc = MatrixXT::Zero(dims * n_cells, dims * n_cells);
    MatrixXT partial_cx = MatrixXT::Zero(dims * n_cells, x.rows());
    MatrixXT partial_xx = MatrixXT::Zero(x.rows(), x.rows());
    VectorXT partial_x = VectorXT::Zero(x.rows());

    for (int cell = 0; cell < n_cells; cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(dims);
        VectorXT nodes(nodeIndices.rows() * 2);

        site = c.segment(cell * dims, dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, gradient_c, gradient_x, &cellInfos[cell]);

        for (int i = 0; i < nodeIndices.rows(); i++) {
            partial_x.segment<2>(nodeIndices(i) * 2) += gradient_x.segment<2>(i * 2);
        }

        MatrixXT hessian_local = MatrixXT::Zero(site.rows() + nodes.rows(), site.rows() + nodes.rows());
        function.addHessian(site, nodes, hessian_local, &cellInfos[cell]);

        Eigen::Ref<MatrixXT> hessian_local_cc = hessian_local.block(0, 0, site.rows(), site.rows());
        Eigen::Ref<MatrixXT> hessian_local_cx = hessian_local.block(0, site.rows(), site.rows(), nodes.rows());
        Eigen::Ref<MatrixXT> hessian_local_xx = hessian_local.block(site.rows(), site.rows(), nodes.rows(),
                                                                    nodes.rows());

        partial_cc.block(cell * dims, cell * dims, dims, dims) += hessian_local_cc;
        for (int i = 0; i < nodeIndices.rows(); i++) {
            partial_cx.block<2, 2>(cell * dims, nodeIndices(i) * 2) += hessian_local_cx.block<2, 2>(0, i * 2);
            for (int j = 0; j < nodeIndices.rows(); j++) {
                partial_xx.block<2, 2>(nodeIndices(i) * 2, nodeIndices(j) * 2) += hessian_local_xx.block<2, 2>(i * 2,
                                                                                                               j * 2);
            }
        }
    }

    Eigen::Ref<MatrixXT> hessian_cc = hessian.block(0, 0, dims * n_cells, dims * n_cells);
    Eigen::Ref<MatrixXT> hessian_cp = hessian.block(0, dims * n_cells, dims * n_cells, bdry->nfree);
    Eigen::Ref<MatrixXT> hessian_pp = hessian.block(dims * n_cells, dims * n_cells, bdry->nfree, bdry->nfree);

    MatrixXT sum_f_cc = MatrixXT::Zero(dims * n_cells, dims * n_cells);
    MatrixXT sum_f_cv = MatrixXT::Zero(dims * n_cells, bdry->v.rows());
    MatrixXT sum_f_vv = MatrixXT::Zero(bdry->v.rows(), bdry->v.rows());
    for (int ii = 0; ii < x.rows(); ii++) {
        IV3 face = dual.segment<3>(3 * ((int) (ii / 2)));
        MatrixXT hess = d2xdy2[ii];

        // Boundary corner vertices
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
        int n_bdy = bdry->v.rows() / 2;
        // Boundary edge vertices
        if (face(2) >= n_sites) {
            int ib0 = (face(2) - n_sites) * 2;
            int ib1 = ((face(2) - n_sites + 1) % n_bdy) * 2;

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
        }
    }

    hessian_cc +=
            partial_cc + partial_cx * dxdc + dxdc.transpose() * (partial_cx.transpose() + partial_xx * dxdc) + sum_f_cc;

    if (bdry->nfree > 0) {
        MatrixXT dxdp = dxdv * bdry->dvdp;
        hessian_cp += dxdc.transpose() * partial_xx * dxdp + partial_cx * dxdp + sum_f_cv * bdry->dvdp;

        // TODO: Add partial_pp

        MatrixXT sum_x_pp = MatrixXT::Zero(bdry->nfree, bdry->nfree);
        for (int ii = 0; ii < bdry->v.rows(); ii++) {
            sum_x_pp += (partial_x.transpose() * dxdv.col(ii)) * bdry->d2vdp2[ii];
        }
        hessian_pp += dxdp.transpose() * partial_xx * dxdp + bdry->dvdp.transpose() * sum_f_vv * bdry->dvdp + sum_x_pp;

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
    bdry = bdry_new;
    last_boundary = bdry_new->v;

    c = c_new;

    VectorXi dualRaw = getDualGraph(vertices, params);
    std::vector<std::vector<int>> neighborhoods_stdvec = getNeighborsClipped(vertices, params, dualRaw, n_free);
    neighborhoods.resize(neighborhoods_stdvec.size());
    for (int i = 0; i < neighborhoods.size(); i++) {
        neighborhoods[i] = Eigen::Map<VectorXi>(neighborhoods_stdvec[i].data(), neighborhoods_stdvec[i].size());
    }

    auto cmp = [](IV3 a, IV3 b) {
        for (int i = 0; i < 3; i++) {
            if (a(i) < b(i)) return true;
            if (a(i) > b(i)) return false;
        }
        return false;
    };
    std::set<IV3, decltype(cmp)> triplets(cmp);

    for (int i = 0; i < n_free; i++) {
        VectorXi cell = neighborhoods[i];
        for (int j = 0; j < cell.size(); j++) {
            int n1 = cell[j];
            int n2 = cell[(j + 1) % cell.size()];

            IV3 triplet(i, n1, n2);
            std::sort(triplet.data(), triplet.data() + 3);

            triplets.insert(triplet);
        }
    }
    std::vector<IV3> faces(triplets.begin(), triplets.end());

    cells.resize(n_free);
    for (int i = 0; i < n_free; i++) {
        VectorXi neighborhood = neighborhoods[i];
        VectorXi cell(neighborhood.size());

        for (int j = 0; j < neighborhood.size(); j++) {
            int n1 = neighborhood[j];
            int n2 = neighborhood[(j + 1) % neighborhood.size()];

            IV3 triplet(i, n1, n2);
            std::sort(triplet.data(), triplet.data() + 3);

            auto lower = std::lower_bound(faces.begin(), faces.end(), triplet, cmp);
            assert (lower != faces.end() && *lower == triplet);

            cell(j) = std::distance(faces.begin(), lower);
        }

        cells[i] = cell;
    }

    dual.resize(faces.size() * 3);
    for (int i = 0; i < faces.size(); i++) {
        dual.segment<3>(i * 3) = faces[i];
    }

    int dims = 2 + getNumVertexParams();

    x.resize(faces.size() * 2);
    x.setZero();
    MatrixXT dxdc_dense = MatrixXT::Zero(faces.size() * 2, n_free * dims);
    MatrixXT dxdv_dense = MatrixXT::Zero(faces.size() * 2, bdry->v.rows());
    d2xdy2.resize(faces.size() * 2);
    for (int i = 0; i < faces.size(); i++) {
        IV3 face = faces[i];

        TV node;
        VectorXT gradX, gradY;
        int type;
        getNodeWrapper(face[0], face[1], face[2], node, gradX, gradY, d2xdy2[i * 2 + 0], d2xdy2[i * 2 + 1], type);

        x.segment<2>(i * 2) = node;

        // Assemble global Jacobian matrix dxdc.
        int n_sites = c.rows() / dims;
        int n_bdy = bdry->v.rows() / 2;
        switch (type) {
            case 0:
                for (int j = 0; j < 3; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    for (int k = 0; k < dims; k++) {
                        dxdc_dense(i * 2 + 0, face[j] * dims + k) = gradX(j * dims + k);
                        dxdc_dense(i * 2 + 1, face[j] * dims + k) = gradY(j * dims + k);
                    }
                }
                break;
            case 1:
                for (int j = 0; j < 2; j++) {
                    if (face[j] >= n_free) {
                        continue;
                    }
                    for (int k = 0; k < dims; k++) {
                        dxdc_dense(i * 2 + 0, face[j] * dims + k) = gradX(j * dims + k);
                        dxdc_dense(i * 2 + 1, face[j] * dims + k) = gradY(j * dims + k);
                    }
                }
                for (int k = 0; k < 2; k++) {
                    dxdv_dense(i * 2 + 0, (face[2] - n_sites) * 2 + k) = gradX(2 * dims + k);
                    dxdv_dense(i * 2 + 0, ((face[2] - n_sites + 1) % n_bdy) * 2 + k) = gradX(2 * dims + 2 + k);
                    dxdv_dense(i * 2 + 1, (face[2] - n_sites) * 2 + k) = gradY(2 * dims + k);
                    dxdv_dense(i * 2 + 1, ((face[2] - n_sites + 1) % n_bdy) * 2 + k) = gradY(2 * dims + 2 + k);
                }
                break;
            case 2:
                for (int k = 0; k < 2; k++) {
                    dxdv_dense(i * 2 + 0, (face[1] - n_sites) * 2 + k) = gradX(k);
                    dxdv_dense(i * 2 + 0, ((face[1] - n_sites + 1) % n_bdy) * 2 + k) = gradX(2 + k);
                    dxdv_dense(i * 2 + 1, (face[1] - n_sites) * 2 + k) = gradY(k);
                    dxdv_dense(i * 2 + 1, ((face[1] - n_sites + 1) % n_bdy) * 2 + k) = gradY(2 + k);
                }
                break;
            default:
                assert(0);
                break;
        }
    }
    dxdc = dxdc_dense.sparseView();
    dxdv = dxdv_dense.sparseView();

    isValid = true;
    for (int i = 0; i < n_free; i++) {
        if (cells[i].rows() < 3) {
            isValid = false;
        }
    }
}
