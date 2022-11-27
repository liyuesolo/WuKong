#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Energy/CodeGen.h"
#include <iostream>

TV Voronoi::getNode(const VectorXT &v1, const VectorXT &v2, const VectorXT &v3) {
    assert(v1.rows() == 2 && v2.rows() == 2 && v3.rows() == 2);

    double x1 = v1(0);
    double y1 = v1(1);
    double x2 = v2(0);
    double y2 = v2(1);
    double x3 = v3(0);
    double y3 = v3(1);

    double m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1));
    double xn = 0.5 * (x1 + x3) - m * (y3 - y1);
    double yn = 0.5 * (y1 + y3) + m * (x3 - x1);
    return {xn, yn};
}

TV Voronoi::getBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1) {
    assert(v1.rows() == 2 && v2.rows() == 2);

    double x1 = (v1(0) + v2(0)) / 2;
    double y1 = (v1(1) + v2(1)) / 2;
    double x2 = x1 + (v2(1) - v1(1));
    double y2 = y1 - (v2(0) - v1(0));
    double x3 = b0(0);
    double y3 = b0(1);
    double x4 = b1(0);
    double y4 = b1(1);

    double t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (-(x4 - x3) * (y2 - y1) + (x2 - x1) * (y4 - y3));
    double xn = x1 + t * (x2 - x1);
    double yn = y1 + t * (y2 - y1);

    return {xn, yn};
}

VectorXi Voronoi::delaunayNaive(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;
    std::vector<int> tri1;
    std::vector<int> tri2;
    std::vector<int> tri3;

    for (int i = 0; i < n_vtx; i++) {
        TV vi = vertices.segment<2>(i * 2);
        std::vector<int> neighbors;

        for (int j = 0; j < n_vtx; j++) {
            if (j == i) continue;

            TV vj = vertices.segment<2>(j * 2);
            TV line = {-(vj(1) - vi(1)), vj(0) - vi(0)};

            double dmin = INFINITY;
            double dmax = -INFINITY;

            for (int k = 0; k < n_vtx; k++) {
                if (k == i || k == j) continue;

                TV vk = vertices.segment<2>(k * 2);
                TV vc = getNode(vi, vj, vk);
                double d = vc.dot(line);

                if ((vk - vi).dot(line) > 0) {
                    dmin = fmin(dmin, d);
                } else {
                    dmax = fmax(dmax, d);
                }
                if (dmax > dmin) break;
            }

            if (dmax < dmin || (dmax == dmin)) {
                neighbors.push_back(j);
            }
        }

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

        if (neighbors.size() > 0) {
            assert(neighbors.size() > 1);
            for (int j = 0; j < neighbors.size(); j++) {
                int v1 = i;
                int v2 = neighbors[j];
                int v3 = neighbors[(j + 1) % neighbors.size()];

                if (v1 < v2 && v1 < v3) {
                    double x1 = vertices(v1 * 2 + 0);
                    double y1 = vertices(v1 * 2 + 1);
                    double x2 = vertices(v2 * 2 + 0);
                    double y2 = vertices(v2 * 2 + 1);
                    double x3 = vertices(v3 * 2 + 0);
                    double y3 = vertices(v3 * 2 + 1);

                    if (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2 > 0) {
                        tri1.push_back(v1);
                        tri2.push_back(v2);
                        tri3.push_back(v3);
                    }
                }
            }
        }
    }

    VectorXi tri(tri1.size() * 3);
    for (int i = 0; i < tri1.size(); i++) {
        tri(i * 3 + 0) = tri1[i];
        tri(i * 3 + 1) = tri2[i];
        tri(i * 3 + 2) = tri3[i];
    }

    return tri;
}

VectorXi Voronoi::delaunayJRS(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    MatrixXT P;
    P.resize(n_vtx, 2);
    for (int i = 0; i < n_vtx; i++) {
        P.row(i) = vertices.segment<2>(i * 2);
    }

    MatrixXT V;
    MatrixXi F;
    igl::triangle::triangulate(P,
                               MatrixXi(),
                               MatrixXT(),
                               "cQ", // Enclose convex hull with segments
                               V, F);

    VectorXi tri;
    tri.resize(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++)
        tri.segment<3>(i * 3) = F.row(i);
    return tri;
}

VectorXi Voronoi::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    return delaunayJRS(vertices);
}

VectorXT Voronoi::getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual) {
    int n_faces = dual.rows() / 3;
    VectorXT nodes(2 * n_faces);

    for (int i = 0; i < n_faces; i++) {
        int v1 = dual(i * 3 + 0);
        int v2 = dual(i * 3 + 1);
        int v3 = dual(i * 3 + 2);

        nodes.segment<2>(i * 2) = getNode(vertices.segment<2>(v1 * 2), vertices.segment<2>(v2 * 2),
                                          vertices.segment<2>(v3 * 2));
    }

    return nodes;
}

VectorXT Voronoi::getDefaultVertexParams(const VectorXT &vertices) {
    return {};
}

