#include "../../include/Tessellation/Tessellation.h"
#include "../../include/Tessellation/CellFunction.h"
#include "../../include/Boundary/Boundary.h"
#include <set>
#include <iostream>

bool
Tessellation::getNeighborsClipped(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual, int n_cells) {
    int n_vtx = vertices.rows() / 2, n_bdy = bdry->v.rows() / 2;

    std::vector<std::vector<int>> neighborsRaw = getNeighbors(vertices, dual, n_cells);
    if (n_bdy == 0) {
        neighborhoods = neighborsRaw;
        return true;
    };

    neighborhoods.clear();
    neighborhoods.resize(n_cells);
    neighborhoodFlags.clear();
    neighborhoodFlags.resize(n_cells);

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

            TV v;
            getNode(c0, c.segment(n1 * dims, dims), c.segment(n2 * dims, dims), v);
            nodes[j] = v;
        }

        std::vector<BoundaryIntersection> intersections;
        if (!bdry->getCellIntersections(nodes, intersections)) {
            return false;
        }

        bool inPoly = bdry->pointInBounds(nodes[0]);
        int intersectIdx = 0;
        for (size_t j = 0; j < degree; j++) {
            if (inPoly) {
                neighborhoods[i].push_back(neighbors[(j + 1) % degree]);
            }
            while (intersectIdx < intersections.size() && intersections[intersectIdx].i_cell == j) {
                inPoly = !inPoly;

                neighborhoods[i].push_back(n_vtx + intersections[intersectIdx].i_bdry);
                if (inPoly) {
                    neighborhoods[i].push_back(neighbors[(j + 1) % degree]);
                }

                intersectIdx++;
            }
        }

        size_t clippedDegree = neighborhoods[i].size();
        for (int j = 0; j < clippedDegree; j++) {
            int n1 = neighborhoods[i][j];
            int n2 = neighborhoods[i][(j + 1) % clippedDegree];

            if (n1 >= n_vtx && n2 >= n_vtx) {
                if (n1 == n2) {
                    neighborhoods[i].erase(neighborhoods[i].begin() + j);
                    clippedDegree--;
                    j--;
                } else if (bdry->next(n1 - n_vtx) == n2 - n_vtx) {
                    // Do nothing
                } else {
                    neighborhoods[i].insert(neighborhoods[i].begin() + j + 1, bdry->next(n1 - n_vtx) + n_vtx);
                    clippedDegree++;
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
Tessellation::getNodeWrapper(int i0, int i1, int i2, TV &node, VectorXT &gradX, VectorXT &gradY, MatrixXT &hessX,
                             MatrixXT &hessY, int &mode) {
    int dims = 2 + getNumVertexParams();
    int n_vtx = c.rows() / dims;

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
        b1 = bdry->v.segment<2>(bdry->next(i2 - n_vtx) * 2);
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
        TV b1e = bdry->v.segment<2>(bdry->next(i1 - n_vtx) * 2);

        assert(bdry->next(i1 - n_vtx) == i2 - n_vtx || bdry->next(i2 - n_vtx) == i1 - n_vtx);
        gradX = VectorXT::Zero(4);
        gradY = VectorXT::Zero(4);
        if (bdry->next(i1 - n_vtx) == i2 - n_vtx) {
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
        IV3 face = dual.segment<3>(4 * ((int) (ii / 2)));
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
        // Boundary edge vertices
        if (face(2) >= n_sites) {
            int ib0 = (face(2) - n_sites) * 2;
            int ib1 = bdry->next(face(2) - n_sites) * 2;

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

    Eigen::SparseMatrix<double> dxdc_T = dxdc.transpose();

    hessian_cc +=
            partial_cc + partial_cx * dxdc + dxdc_T * (partial_cx.transpose() + partial_xx * dxdc) + sum_f_cc;

    if (bdry->nfree > 0) {
        Eigen::SparseMatrix<double> dxdp = (dxdv * bdry->dvdp).sparseView();
        hessian_cp += dxdc_T * partial_xx * dxdp + partial_cx * dxdp + sum_f_cv * bdry->dvdp;

        MatrixXT sum_x_pp = MatrixXT::Zero(bdry->nfree, bdry->nfree);
        MatrixXT partial_x_T = partial_x.transpose();
        for (int ii = 0; ii < bdry->v.rows(); ii++) {
            sum_x_pp += (partial_x_T * dxdv.col(ii)).coeff(0, 0) * bdry->d2vdp2[ii];
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

    // Fourth field is a flag set to 1 only if
    // - There are two nodes between the same neighbors because one is a curved boundary surface
    // - This node is the one with higher t (in the bezier parameterization)
    auto cmp = [](IV4 a, IV4 b) {
        for (int i = 0; i < 4; i++) {
            if (a(i) < b(i)) return true;
            if (a(i) > b(i)) return false;
        }
        return false;
    };
    std::set<IV4, decltype(cmp)> triplets(cmp);

    for (int i = 0; i < n_free; i++) {
        std::vector<int> cell = neighborhoods[i];
        for (int j = 0; j < cell.size(); j++) {
            int n1 = cell[j];
            int n2 = cell[(j + 1) % cell.size()];

            IV4 triplet(i, n1, n2, 0);
            std::sort(triplet.data(), triplet.data() + 3);

            triplets.insert(triplet);
        }
    }
    std::vector<IV4> faces(triplets.begin(), triplets.end());

    cells.resize(n_free);
    for (int i = 0; i < n_free; i++) {
        std::vector<int> neighborhood = neighborhoods[i];
        VectorXi cell(neighborhood.size());

        for (int j = 0; j < neighborhood.size(); j++) {
            int n1 = neighborhood[j];
            int n2 = neighborhood[(j + 1) % neighborhood.size()];

            IV4 triplet(i, n1, n2, 0);
            std::sort(triplet.data(), triplet.data() + 3);

            auto lower = std::lower_bound(faces.begin(), faces.end(), triplet, cmp);
            assert (lower != faces.end() && *lower == triplet);

            cell(j) = std::distance(faces.begin(), lower);
        }

        cells[i] = cell;
    }

    dual.resize(faces.size() * 4);
    for (int i = 0; i < faces.size(); i++) {
        dual.segment<4>(i * 4) = faces[i];
    }

    int dims = 2 + getNumVertexParams();

    x.resize(faces.size() * 2);
    x.setZero();
    MatrixXT dxdc_dense = MatrixXT::Zero(faces.size() * 2, n_free * dims);
    MatrixXT dxdv_dense = MatrixXT::Zero(faces.size() * 2, bdry->v.rows());
    d2xdy2.resize(faces.size() * 2);
    for (int i = 0; i < faces.size(); i++) {
        IV4 face = faces[i];

        TV node;
        VectorXT gradX, gradY;
        int type;
        getNodeWrapper(face[0], face[1], face[2], node, gradX, gradY, d2xdy2[i * 2 + 0], d2xdy2[i * 2 + 1], type);

        x.segment<2>(i * 2) = node;

        // Assemble global Jacobian matrix dxdc.
        int n_sites = c.rows() / dims;
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
                    dxdv_dense(i * 2 + 0, bdry->next(face[2] - n_sites) * 2 + k) = gradX(2 * dims + 2 + k);
                    dxdv_dense(i * 2 + 1, (face[2] - n_sites) * 2 + k) = gradY(2 * dims + k);
                    dxdv_dense(i * 2 + 1, bdry->next(face[2] - n_sites) * 2 + k) = gradY(2 * dims + 2 + k);
                }
                break;
            case 2:
                for (int k = 0; k < 2; k++) {
                    dxdv_dense(i * 2 + 0, (face[1] - n_sites) * 2 + k) = gradX(k);
                    dxdv_dense(i * 2 + 0, bdry->next(face[1] - n_sites) * 2 + k) = gradX(2 + k);
                    dxdv_dense(i * 2 + 1, (face[1] - n_sites) * 2 + k) = gradY(k);
                    dxdv_dense(i * 2 + 1, bdry->next(face[1] - n_sites) * 2 + k) = gradY(2 + k);
                }
                break;
            default:
                assert(0);
                break;
        }
    }
    dxdc = dxdc_dense.sparseView();
    dxdv = dxdv_dense.sparseView();

    for (int i = 0; i < n_free; i++) {
        if (cells[i].rows() < 3) {
            isValid = false;
            return;
        }
    }
}
