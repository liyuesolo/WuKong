#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "Projects/Foam2D/include/CodeGen.h"
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_2<K> Regular_triangulation;

static TV3 getCircumcentre(const TV3 &v1, const TV3 &v2, const TV3 &v3) {
    double x1 = v1(0);
    double y1 = v1(1);
    double z1 = v1(2);
    double x2 = v2(0);
    double y2 = v2(1);
    double z2 = v2(2);
    double x3 = v3(0);
    double y3 = v3(1);
    double z3 = v3(2);

    double rsq2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    double d2 = 0.5 + 0.5 * (z2 - z1) / rsq2;
    double xp2 = x1 + d2 * (x2 - x1);
    double yp2 = y1 + d2 * (y2 - y1);
    double xl2 = -(y2 - y1);
    double yl2 = (x2 - x1);
    double rsq3 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1);
    double d3 = 0.5 + 0.5 * (z3 - z1) / rsq3;
    double xp3 = x1 + d3 * (x3 - x1);
    double yp3 = y1 + d3 * (y3 - y1);
    double xl3 = -(y3 - y1);
    double yl3 = (x3 - x1);

    double a2 = (yl3 * (xp3 - xp2) - xl3 * (yp3 - yp2)) / (xl2 * yl3 - xl3 * yl2);
    double xn = xp2 + a2 * xl2;
    double yn = yp2 + a2 * yl2;

    TV3 ret = {xn, yn, 0};

    return ret;
}

int idxClosest(const TV &p, const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices3d.segment<2>(i * 3);
        double d = (p2 - p).norm();
        if (d < dmin) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

VectorXi Power::powerDualCGAL(const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;

    std::vector<Regular_triangulation::Weighted_point> wpoints;
    for (int i = 0; i < n_vtx; i++) {
        TV3 v = vertices3d.segment<3>(i * 3);
        Regular_triangulation::Weighted_point wp({v(0), v(1)}, -v(2));
        wpoints.push_back(wp);
    }
    Regular_triangulation rt(wpoints.begin(), wpoints.end());

    VectorXi tri(rt.number_of_faces() * 3);
    int f = 0;
    for (auto it = rt.faces_begin(); it != rt.faces_end(); it++) {

        auto v0 = it->vertex(0)->point();
        TV V0 = {v0.x(), v0.y()};
        auto v1 = it->vertex(1)->point();
        TV V1 = {v1.x(), v1.y()};
        auto v2 = it->vertex(2)->point();
        TV V2 = {v2.x(), v2.y()};

        int i0 = idxClosest(V0, vertices3d);
        int i1 = idxClosest(V1, vertices3d);
        int i2 = idxClosest(V2, vertices3d);

        tri.segment<3>(f * 3) = IV3(i0, i1, i2);
        f++;
    }

    return tri;
}

VectorXi Power::powerDualNaive(const VectorXT &vertices3d) {
    int n_vtx = vertices3d.rows() / 3;
    std::vector<int> tri1;
    std::vector<int> tri2;
    std::vector<int> tri3;

    for (int i = 0; i < n_vtx; i++) {
        TV3 vi = vertices3d.segment<3>(i * 3);
        std::vector<int> neighbors;

        for (int j = 0; j < n_vtx; j++) {
            if (j == i) continue;

            TV3 vj = vertices3d.segment<3>(j * 3);
            TV3 line = (vj - vi).cross(TV3(0, 0, 1));

            double dmin = INFINITY;
            double dmax = -INFINITY;

            for (int k = 0; k < n_vtx; k++) {
                if (k == i || k == j) continue;

                TV3 vk = vertices3d.segment<3>(k * 3);
                TV3 vc = getCircumcentre(vi, vj, vk);
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

        double xc = vertices3d(i * 3 + 0);
        double yc = vertices3d(i * 3 + 1);

        std::sort(neighbors.begin(), neighbors.end(), [vertices3d, xc, yc](int a, int b) {
            double xa = vertices3d(a * 3 + 0);
            double ya = vertices3d(a * 3 + 1);
            double angle_a = atan2(ya - yc, xa - xc);

            double xb = vertices3d(b * 3 + 0);
            double yb = vertices3d(b * 3 + 1);
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
                    double x1 = vertices3d(v1 * 3 + 0);
                    double y1 = vertices3d(v1 * 3 + 1);
                    double x2 = vertices3d(v2 * 3 + 0);
                    double y2 = vertices3d(v2 * 3 + 1);
                    double x3 = vertices3d(v3 * 3 + 0);
                    double y3 = vertices3d(v3 * 3 + 1);

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

VectorXi Power::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    VectorXT vertices3d = combineVerticesParams(vertices, params);
    return powerDualCGAL(vertices3d);
}

VectorXT Power::getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual) {
    int n_faces = dual.rows() / 3;
    VectorXT nodes(2 * n_faces);

    for (int i = 0; i < n_faces; i++) {
        int v1 = dual(i * 3 + 0);
        int v2 = dual(i * 3 + 1);
        int v3 = dual(i * 3 + 2);

        double x1 = vertices(v1 * 2 + 0);
        double y1 = vertices(v1 * 2 + 1);
        double z1 = params(v1);
        double x2 = vertices(v2 * 2 + 0);
        double y2 = vertices(v2 * 2 + 1);
        double z2 = params(v2);
        double x3 = vertices(v3 * 2 + 0);
        double y3 = vertices(v3 * 2 + 1);
        double z3 = params(v3);

        double rsq2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        double d2 = 0.5 + 0.5 * (z2 - z1) / rsq2;
        double xp2 = x1 + d2 * (x2 - x1);
        double yp2 = y1 + d2 * (y2 - y1);
        double xl2 = -(y2 - y1);
        double yl2 = (x2 - x1);
        double rsq3 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1);
        double d3 = 0.5 + 0.5 * (z3 - z1) / rsq3;
        double xp3 = x1 + d3 * (x3 - x1);
        double yp3 = y1 + d3 * (y3 - y1);
        double xl3 = -(y3 - y1);
        double yl3 = (x3 - x1);

        double a2 = (yl3 * (xp3 - xp2) - xl3 * (yp3 - yp2)) / (xl2 * yl3 - xl3 * yl2);
        double xn = xp2 + a2 * xl2;
        double yn = yp2 + a2 * yl2;

        nodes.segment<2>(i * 2) = TV(xn, yn);
    }

    return nodes;
}

VectorXT Power::getDefaultVertexParams(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    VectorXT z = VectorXT::Zero(n_vtx);

    return z;
}

