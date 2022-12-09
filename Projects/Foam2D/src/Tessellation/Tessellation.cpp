#include "../../include/Tessellation/Tessellation.h"
#include "../../include/Tessellation/CellFunction.h"
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
Tessellation::getNeighborsClipped(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual,
                                  const VectorXT &boundary,
                                  int n_cells) {
    int n_vtx = vertices.rows() / 2, n_bdy = boundary.rows() / 2;

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
            inPoly[j] = pointInPolygon(v, boundary);
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
                    TV v2 = boundary.segment<2>(k * 2);
                    TV v3 = boundary.segment<2>(((k + 1) % n_bdy) * 2);
                    TV intersect;
                    if (lineSegmentIntersection(v0, v1, v2, v3, intersect)) {
                        neighborsClipped[i].push_back(n_vtx + k);
                        break; // Can only be one intersection.
                    }
                }
            } else if (!inPoly0 && inPoly1) {
                // Add boundary edge and then neighbor.
                for (size_t k = 0; k < n_bdy; k++) {
                    TV v2 = boundary.segment<2>(k * 2);
                    TV v3 = boundary.segment<2>(((k + 1) % n_bdy) * 2);
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
                    TV v2 = boundary.segment<2>(k * 2);
                    TV v3 = boundary.segment<2>(((k + 1) % n_bdy) * 2);
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
//                    j++;
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
    int n_bdy = boundary.rows() / 2;

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
        b0 = boundary.segment<2>((i2 - n_vtx) * 2);
        b1 = boundary.segment<2>(((i2 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);
    } else if (i1 >= n_vtx && i2 < n_vtx) {
        // Boundary node with n1 a boundary edge.
        v1 = c.segment(i2 * dims, dims);
        b0 = boundary.segment<2>((i1 - n_vtx) * 2);
        b1 = boundary.segment<2>(((i1 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);
    } else {
        // Boundary vertex.
        assert(i1 >= n_vtx && i2 >= n_vtx);
        node = boundary.segment<2>((i2 - n_vtx) * 2);
    }
}

void
Tessellation::getNodeWrapper(int i0, int i1, int i2, TV &node, VectorXT &gradX, VectorXT &gradY, MatrixXT &hessX,
                             MatrixXT &hessY) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;
    int n_bdy = boundary.rows() / 2;

    VectorXT v0 = c.segment(i0 * dims, dims); // i0 is always a site, never a boundary edge.
    VectorXT v1, v2;
    TV b0, b1;
    if (i1 < n_vtx && i2 < n_vtx) {
        // Normal node.
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
        v1 = c.segment(i1 * dims, dims);
        b0 = boundary.segment<2>((i2 - n_vtx) * 2);
        b1 = boundary.segment<2>(((i2 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);

        gradX.resize(dims * 2);
        gradY.resize(dims * 2);
        getBoundaryNodeGradient(v0, v1, b0, b1, gradX, gradY);

        hessX.resize(dims * 2, dims * 2);
        hessY.resize(dims * 2, dims * 2);
        getBoundaryNodeHessian(v0, v1, b0, b1, hessX, hessY);
    } else if (i1 >= n_vtx && i2 < n_vtx) {
        // Boundary node with n1 a boundary edge.
        v1 = c.segment(i2 * dims, dims);
        b0 = boundary.segment<2>((i1 - n_vtx) * 2);
        b1 = boundary.segment<2>(((i1 - n_vtx + 1) % n_bdy) * 2);
        getBoundaryNode(v0, v1, b0, b1, node);

        gradX.resize(dims * 2);
        gradY.resize(dims * 2);
        getBoundaryNodeGradient(v0, v1, b0, b1, gradX, gradY);

        hessX.resize(dims * 2, dims * 2);
        hessY.resize(dims * 2, dims * 2);
        getBoundaryNodeHessian(v0, v1, b0, b1, hessX, hessY);
    } else {
        // Boundary vertex.
        assert(i1 >= n_vtx && i2 >= n_vtx);

        TV b1s = boundary.segment<2>((i1 - n_vtx) * 2);
        TV b1e = boundary.segment<2>(((i1 + 1 - n_vtx) % n_bdy) * 2);
        TV b2s = boundary.segment<2>((i2 - n_vtx) * 2);
        TV b2e = boundary.segment<2>(((i2 + 1 - n_vtx) % n_bdy) * 2);

        assert(b1e == b2s || b2e == b1s);
        if (b1e == b2s) {
            node = b1e;
        } else {
            node = b1s;
        }

        gradX.resize(0);
        gradY.resize(0);
        hessX.resize(0, 0);
        hessY.resize(0, 0);
    }
}

void
Tessellation::addSingleCellFunctionValue(int cell, const CellFunction &function, double &value) {
    VectorXi nodeIndices = cells[cell];

    VectorXT site(2);
    VectorXT nodes(nodeIndices.rows() * 2);

    int dims = 2 + getNumVertexParams();
    site = c.segment<2>(cell * dims);
    for (int i = 0; i < nodeIndices.rows(); i++) {
        nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
    }

    function.addValue(site, nodes, value);
}

void
Tessellation::addFunctionValue(const CellFunction &function, double &value) {
    for (int cell = 0; cell < cells.size(); cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(2);
        VectorXT nodes(nodeIndices.rows() * 2);

        int dims = 2 + getNumVertexParams();
        site = c.segment<2>(cell * dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        function.addValue(site, nodes, value);
    }
}

void Tessellation::addFunctionGradient(const CellFunction &function, VectorXT &gradient) {
    int dims = 2 + getNumVertexParams();
    int n_cells = cells.size();

    VectorXT dOdc = VectorXT::Zero(dims * n_cells);
    VectorXT dOdx = VectorXT::Zero(x.rows());

    for (int cell = 0; cell < n_cells; cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(2);
        VectorXT nodes(nodeIndices.rows() * 2);

        int dims = 2 + getNumVertexParams();
        site = c.segment<2>(cell * dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, gradient_c, gradient_x);

        dOdc.segment<2>(cell * dims) += gradient_c;
        for (int i = 0; i < nodeIndices.rows(); i++) {
            dOdx.segment<2>(nodeIndices(i) * 2) += gradient_x.segment<2>(i * 2);
        }
    }

    gradient += (dOdx.transpose() * dxdc).transpose() + dOdc;
}


void Tessellation::addFunctionHessian(const CellFunction &function, MatrixXT &hessian) {
    int dims = 2 + getNumVertexParams();
    int n_cells = cells.size();

    MatrixXT hessian_cc = MatrixXT::Zero(dims * n_cells, dims * n_cells);
    MatrixXT hessian_cx = MatrixXT::Zero(dims * n_cells, x.rows());
    MatrixXT hessian_xc = MatrixXT::Zero(x.rows(), dims * n_cells); // TODO: Probably redundant?
    MatrixXT hessian_xx = MatrixXT::Zero(x.rows(), x.rows());
    VectorXT dOdx = VectorXT::Zero(x.rows());

    for (int cell = 0; cell < n_cells; cell++) {
        VectorXi nodeIndices = cells[cell];

        VectorXT site(2);
        VectorXT nodes(nodeIndices.rows() * 2);

        site = c.segment<2>(cell * dims);
        for (int i = 0; i < nodeIndices.rows(); i++) {
            nodes.segment<2>(i * 2) = x.segment<2>(nodeIndices(i) * 2);
        }

        VectorXT gradient_c = VectorXT::Zero(site.rows());
        VectorXT gradient_x = VectorXT::Zero(nodes.rows());
        function.addGradient(site, nodes, gradient_c, gradient_x);

        for (int i = 0; i < nodeIndices.rows(); i++) {
            dOdx.segment<2>(nodeIndices(i) * 2) += gradient_x.segment<2>(i * 2);
        }

        MatrixXT hessian_local = MatrixXT::Zero(site.rows() + nodes.rows(), site.rows() + nodes.rows());
        function.addHessian(site, nodes, hessian_local);

        MatrixXT hessian_local_cc = hessian_local.block(0, 0, site.rows(), site.rows());
        MatrixXT hessian_local_cx = hessian_local.block(0, site.rows(), site.rows(), nodes.rows());
        MatrixXT hessian_local_xc = hessian_local.block(site.rows(), 0, nodes.rows(), site.rows());
        MatrixXT hessian_local_xx = hessian_local.block(site.rows(), site.rows(), nodes.rows(), nodes.rows());

        hessian_cc.block(cell * dims, cell * dims, 2, 2) += hessian_local_cc;
        for (int i = 0; i < nodeIndices.rows(); i++) {
            hessian_cx.block<2, 2>(cell * dims, nodeIndices(i) * 2) += hessian_local_cx.block<2, 2>(0, i * 2);
            hessian_xc.block<2, 2>(nodeIndices(i) * 2, cell * dims) += hessian_local_xc.block<2, 2>(i * 2, 0);
            for (int j = 0; j < nodeIndices.rows(); j++) {
                hessian_xx.block<2, 2>(nodeIndices(i) * 2, nodeIndices(j) * 2) += hessian_local_xx.block<2, 2>(i * 2,
                                                                                                               j * 2);
            }
        }
    }

    hessian += hessian_cc;
    hessian += hessian_cx * dxdc;
    hessian += dxdc.transpose() * hessian_xc;
    hessian += dxdc.transpose() * hessian_xx * dxdc;

    for (int ii = 0; ii < x.rows(); ii++) {
        IV3 face = dual.segment<3>(3 * ((int) (ii / 2)));
        MatrixXT hess = d2xdc2[ii];

        int degree = hess.rows() / dims;
        for (int i = 0; i < degree; i++) {
            for (int j = 0; j < degree; j++) {
                if (face(i) >= n_cells || face(j) >= n_cells) {
                    continue;
                }
                hessian.block(face(i) * dims, face(j) * dims, dims, dims) +=
                        dOdx(ii) * hess.block(i * dims, j * dims, dims, dims);
            }
        }
    }
}

void Tessellation::tessellate(const VectorXT &vertices, const VectorXT &params, const VectorXT &boundary_,
                              const int n_free) {
    VectorXT c_new = combineVerticesParams(vertices, params);

    // Check if inputs are the same as previous tessellation, do nothing if so.
    bool same = true;
    same = same && (c_new.rows() == c.rows() && c_new.isApprox(c));
    same = same && (boundary_.rows() == boundary.rows() && boundary_ == boundary);
    same = same && (n_free == cells.size());
    if (same) return;

    c = c_new;
    boundary = boundary_;

    VectorXi dualRaw = getDualGraph(vertices, params);
    std::vector<std::vector<int>> neighborhoods = getNeighborsClipped(vertices, params, dualRaw, boundary, n_free);
    todo_neighborhoods = neighborhoods;

    auto cmp = [](IV3 a, IV3 b) {
        for (int i = 0; i < 3; i++) {
            if (a(i) < b(i)) return true;
            if (a(i) > b(i)) return false;
        }
        return false;
    };
    std::set<IV3, decltype(cmp)> triplets(cmp);

    for (int i = 0; i < n_free; i++) {
        std::vector<int> cell = neighborhoods[i];
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
        std::vector<int> neighborhood = neighborhoods[i];
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
    d2xdc2.resize(faces.size() * 2);
    for (int i = 0; i < faces.size(); i++) {
        IV3 face = faces[i];

        TV node;
        VectorXT gradX, gradY;
        MatrixXT hessX, hessY;
        getNodeWrapper(face[0], face[1], face[2], node, gradX, gradY, hessX, hessY);

        x.segment<2>(i * 2) = node;

        // Assemble global Jacobian matrix dxdc.
        for (int j = 0; j < gradX.size() / dims; j++) {
            if (face[j] >= n_free) {
                continue;
            }
            for (int k = 0; k < dims; k++) {
                dxdc_dense(i * 2 + 0, face[j] * dims + k) = gradX(j * dims + k);
                dxdc_dense(i * 2 + 1, face[j] * dims + k) = gradY(j * dims + k);
            }
        }

        // Keep local Hessians.
        d2xdc2[i * 2 + 0] = hessX;
        d2xdc2[i * 2 + 1] = hessY;
    }
    dxdc = dxdc_dense.sparseView();

    isValid = true;
    for (int i = 0; i < n_free; i++) {
        if (cells[i].rows() < 3) {
            isValid = false;
        }
    }
}
