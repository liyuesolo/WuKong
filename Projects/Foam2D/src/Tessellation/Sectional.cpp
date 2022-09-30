#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Sectional.h"
#include "Projects/Foam2D/include/CodeGen.h"
#include <iostream>

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

    double m2 = -(y2 - y1) / (x2 - x1);
    double c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 * z2 - z1 * z1) / (2 * (x2 - x1));
    double m3 = -(y3 - y1) / (x3 - x1);
    double c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 * z3 - z1 * z1) / (2 * (x3 - x1));

    double yn = (c3 - c2) / (m2 - m3);
    double xn = m2 * yn + c2;

    TV3 ret = {xn, yn, 0};

    return ret;
}

VectorXi Sectional::sectionalDualNaive(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 3;
    std::vector<int> tri1;
    std::vector<int> tri2;
    std::vector<int> tri3;

    for (int i = 0; i < n_vtx; i++) {
        TV3 vi = vertices.segment<3>(i * 3);
        std::vector<int> neighbors;

        for (int j = 0; j < n_vtx; j++) {
            if (j == i) continue;

            TV3 vj = vertices.segment<3>(j * 3);
            TV3 line = (vj - vi).cross(TV3(0, 0, 1));

            double dmin = INFINITY;
            double dmax = -INFINITY;

            for (int k = 0; k < n_vtx; k++) {
                if (k == i || k == j) continue;

                TV3 vk = vertices.segment<3>(k * 3);
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

        double xc = vertices(i * 3 + 0);
        double yc = vertices(i * 3 + 1);

        std::sort(neighbors.begin(), neighbors.end(), [vertices, xc, yc](int a, int b) {
            double xa = vertices(a * 3 + 0);
            double ya = vertices(a * 3 + 1);
            double angle_a = atan2(ya - yc, xa - xc);

            double xb = vertices(b * 3 + 0);
            double yb = vertices(b * 3 + 1);
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
                    double x1 = vertices(v1 * 3 + 0);
                    double y1 = vertices(v1 * 3 + 1);
                    double x2 = vertices(v2 * 3 + 0);
                    double y2 = vertices(v2 * 3 + 1);
                    double x3 = vertices(v3 * 3 + 0);
                    double y3 = vertices(v3 * 3 + 1);

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

VectorXi Sectional::getDualGraph(const VectorXT &vertices, const VectorXT &params) {
    VectorXT vertices3d = combineVerticesParams(vertices, params);
    return sectionalDualNaive(vertices3d);
}

VectorXT Sectional::getNodes(const VectorXT &vertices, const VectorXT &params, const VectorXi &dual) {
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

        double m2 = -(y2 - y1) / (x2 - x1);
        double c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 * z2 - z1 * z1) / (2 * (x2 - x1));
        double m3 = -(y3 - y1) / (x3 - x1);
        double c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 * z3 - z1 * z1) / (2 * (x3 - x1));

        double yn = (c3 - c2) / (m2 - m3);
        double xn = m2 * yn + c2;

        nodes.segment<2>(i * 2) = TV(xn, yn);
    }

    return nodes;
}

VectorXT Sectional::getDefaultVertexParams(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    VectorXT z = VectorXT::Zero(n_vtx);
    // Small initial z so that gradients are nonzero.
    z.segment<40>(0) = 1e-2 * VectorXT::Ones(40);

    return z;
}

