#include "../../include/Boundary/Boundary.h"

Boundary::Boundary(const VectorXT &p_, const VectorXi &free_) {
    p = p_;
    free_idx = free_;

    np = p.rows();
    nfree = free_idx.rows();

    free_map = -1 * VectorXi::Ones(np);
    for (int i = 0; i < nfree; i++) {
        free_map(free_idx(i)) = i;
    }
}

void Boundary::compute(const VectorXT &p_free) {
    for (int i = 0; i < nfree; i++) {
        p(free_idx(i)) = p_free(i);
    }

    computeVertices();
    computeGradient();
    computeHessian();
}

VectorXT Boundary::get_p_free() {
    VectorXT ret(nfree);
    for (int i = 0; i < nfree; i++) {
        ret(i) = p(free_idx(i));
    }
    return ret;
}

bool Boundary::pointInBounds(const TV &point) {
    int np = v.rows() / 2;

    double w = 0; // Winding number
    for (int i = 0; i < edges.size(); i++) {
        int j = edges[i].nextEdge;
        double x1 = v(2 * i + 0);
        double y1 = v(2 * i + 1);
        double x2 = v(2 * j + 0);
        double y2 = v(2 * j + 1);

        double a = atan2(y2 - point.y(), x2 - point.x()) - atan2(y1 - point.y(), x1 - point.x());
        if (a > M_PI) a -= 2 * M_PI;
        if (a < -M_PI) a += 2 * M_PI;

        if (edges[i].btype == 1) {
            double r = q(edges[i].q_idx);

            double q = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            double d = r / 2 * sqrt(4 - pow(q / r, 2));
            double theta = atan2(-(x2 - x1), (y2 - y1));
            double xc = (x1 + x2) / 2 - d * cos(theta);
            double yc = (y1 + y2) / 2 - d * sin(theta);

            if (a > 0 && r < 0 && (point - TV(xc, yc)).norm() < fabs(r)) {
                a -= 2 * M_PI;
            }
            if (a < 0 && r > 0 && (point - TV(xc, yc)).norm() < r) {
                a += 2 * M_PI;
            }
        }

        w += a;
    }

    return w > M_PI; // w == (2 * M_PI)
}

bool Boundary::straightBoundaryIntersection(const TV &p0, const TV &p1, int v_idx, BoundaryIntersection &intersect) {
    TV p2 = v.segment<2>(v_idx * 2);
    TV p3 = v.segment<2>(edges[v_idx].nextEdge * 2);

    if (std::min(p0.x(), p1.x()) > std::max(p2.x(), p3.x()) ||
        std::max(p0.x(), p1.x()) < std::min(p2.x(), p3.x()) ||
        std::min(p0.y(), p1.y()) > std::max(p2.y(), p3.y()) ||
        std::max(p0.y(), p1.y()) < std::min(p2.y(), p3.y())) {
        return false;
    }

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
        intersect.t_cell = t;
        intersect.t_bdry = s;
        intersect.flag = 0;
        return true;
    }

    return false; // No collision
}

void Boundary::curvedBoundaryIntersection(const TV &p0, const TV &p1, int v_idx, bool &isInt0,
                                          BoundaryIntersection &intersect0, bool &isInt1,
                                          BoundaryIntersection &intersect1) {
    TV p2 = v.segment<2>(v_idx * 2);
    TV p3 = v.segment<2>(edges[v_idx].nextEdge * 2);

    double x0 = p0.x();
    double y0 = p0.y();
    double x1 = p1.x();
    double y1 = p1.y();
    double x2 = p2.x();
    double y2 = p2.y();
    double x3 = p3.x();
    double y3 = p3.y();
    double r = q(edges[v_idx].q_idx);

    TV int0, int1;

    // @formatter:off
    {
        double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
                t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
                t41, t42, t43, t44, t45, t46, t47, t48, t49, t50;

        t1 = x1 - x0;
        t2 = y3 - y2;
        t3 = x2 * x2;
        t4 = pow(t3, 0.2e1);
        t5 = x2 * t3;
        t6 = pow(t2, 0.2e1);
        t7 = x3 * x3;
        t8 = pow(t7, 0.2e1);
        t9 = 0.2e1;
        t10 = x2 * x3;
        t11 = t10 * t9;
        t12 = -t11 + t3 + t6 + t7;
        t13 = -y1 + y0;
        t14 = -x3 + x2;
        t15 = sqrt(t12);
        t16 = x2 + x3;
        t17 = 0.1e1 / 0.2e1;
        t18 = t17 * t16;
        t19 = t18 - x1;
        t20 = y2 + y3;
        t21 = t17 * t20;
        t22 = y2 * y2;
        t23 = y3 * y3;
        t24 = y2 * y3;
        t25 = r * r;
        t26 = sqrt((t9 * (t10 + t24) + 0.4e1 * t25 - t22 - t23 - t3 - t7) * pow(r, -0.2e1));
        t27 = t17 * t7;
        t28 = t27 + t25;
        t29 = t17 * (t22 + t23) - t24 + t25;
        t30 = -pow(t6, 0.2e1) + t4 + t8;
        t31 = t12 * t16;
        t32 = t28 * t3;
        t33 = t25 * t7;
        t34 = t30 / 0.4e1;
        t11 = t11 * t29;
        t35 = 0.4e1 * t10;
        t25 = -(-t2 * (t17 * t2 * y2 + t25) + t7 * (-t17 * y2 + y3)) * x2 + (-t3 * (-t17 * y3 + y2) - t2 * (-t17 * t2 * y3 + t25) + t27 * y3) * x3;
        t27 = t5 * y2;
        t36 = t12 * t20;
        t37 = 0.3e1 / 0.2e1;
        t38 = t24 * t9;
        t39 = pow(t1, 0.2e1);
        t40 = y1 * y1;
        t41 = t9 * t1;
        t42 = pow(t13, 0.2e1);
        t4 = t12 * t42 * (-t41 * (-t17 * (-t36 * x0 + t27) - t25) * y1 + ((-t34 - t11 + (-t12 * x1 + t31) * x1 + t32 + t33) * y0 + (t17 * t30 - t9 * (t12 * t19 * x0 + t33) - t28 * t3 * t9 - t31 * x1 + t35 * t29) * y1 + t9 * (-t17 * (-t36 * x1 + t27) - t25) * t1) * y0 + (-t34 - t11 + (-t12 * x0 + t31) * x0 + t32 + t33) * t40 + t39 * (t4 / 0.4e1 + t8 / 0.4e1 + ((t37 * t7 - t10 - t24) * x2 + x3 * (t38 - t7)) * x2 - t24 * t7 + (-t21 + r) * t6 * (t21 + r)) + (t1 * t14 + t13 * t2) * r * t15 * (t21 * t1 + t19 * y0 + (-t18 + x0) * y1) * t26);
        t2 = -t1 * t2 + t13 * t14;
        t6 = y0 * y0;
        t8 = (t12 * (t9 * (t40 * x0 + t6 * x1) + t1 * (t1 * t16 + t20 * y1) + (-t1 * t20 - t9 * (x0 + x1) * y1) * y0) + r * t2 * t15 * t1 * t26) * t13;
        t10 = t41 * sqrt(t4);
        t6 = -t9 * y0 * y1 + t39 + t40 + t6;
        t11 = x0 * y1 - y0 * x1;
        t16 = t13 * t16 * t17 + t11;
        t14 = pow(t14, 0.2e1);
        t4 = sqrt(t4);
        t2 = -t22 * t42 * y2 + t13 * (r * t2 * t15 * t26 - t13 * y3 * (t14 + t23)) - t22 * (-t41 * t16 - t42 * y3) - (0.4e1 * t1 * t16 * y3 - t42 * (-t14 + t23)) * y2 + t13 * (x3 * t7 + t5) * t1;
        t3 = (t11 * t3 + t16 * t23 + t7 * (-t17 * t13 * x2 + t11)) * t1;
        t1 = t35 * t1 * (t13 * x2 / 0.4e1 + t11);
        t5 = -t38 + t22 + t14 + t23;
        t7 = -t9 * x0 * x1 + x0 * x0 + x1 * x1 + t42;
        t5 = 0.1e1 / t5;
        t11 = 0.1e1 / t13;
        t6 = 0.1e1 / t6;
        t7 = 0.1e1 / t7;
        t12 = 0.1e1 / t12;
        int0[0] = t17 * (t8 - t10) * t12 * t11 * t6;
        int0[1] = t17 * (t9 * (-t3 + t4) - t2 + t1) * t5 * t7;
        int1[0] = t17 * (t8 + t10) * t12 * t11 * t6;
        int1[1] = t17 * (-t9 * (t3 + t4) + t1 - t2) * t5 * t7;
    }
    // @formatter:on

    {
        double t0 = (int0 - p0).dot(p1 - p0) / (p1 - p0).squaredNorm();
        double s0 = (int0 - p2).dot(p3 - p2) / (p3 - p2).squaredNorm();
        double e0 = (int0 - p2).x() * (p3 - p2).y() - (p3 - p2).x() * (int0 - p2).y();
        if (e0 * r > 0 && t0 > 0 && t0 < 1 && s0 > 0 && s0 < 1) {
//            std::cout << "int0 " << int0.x() << " " << int0.y() << std::endl;
            isInt0 = true;
            intersect0.t_cell = t0;
            intersect0.t_bdry = s0;
            intersect0.flag = 0;
        } else {
            isInt0 = false;
        }
    }

    {
        double t1 = (int1 - p0).dot(p1 - p0) / (p1 - p0).squaredNorm();
        double s1 = (int1 - p2).dot(p3 - p2) / (p3 - p2).squaredNorm();
        double e1 = (int1 - p2).x() * (p3 - p2).y() - (p3 - p2).x() * (int1 - p2).y();
        if (e1 * r > 0 && t1 > 0 && t1 < 1 && s1 > 0 && s1 < 1) {
//            std::cout << "int1 " << int1.x() << " " << int1.y() << std::endl;
            isInt1 = true;
            intersect1.t_cell = t1;
            intersect1.t_bdry = s1;
            intersect1.flag = 1;
        } else {
            isInt1 = false;
        }
    }
}

bool Boundary::getCellIntersections(const std::vector<TV> &nodes, std::vector<BoundaryIntersection> &intersections) {
    size_t degree = nodes.size(), n_bdy = v.rows() / 2;

    intersections.clear();
    for (size_t i = 0; i < degree; i++) {
        TV v0 = nodes[i];
        TV v1 = nodes[(i + 1) % degree];

        for (size_t k = 0; k < n_bdy; k++) {
            if (edges[k].btype == 0) {
                BoundaryIntersection intersect;
                intersect.i_cell = i;
                intersect.i_bdry = k;
                if (straightBoundaryIntersection(v0, v1, k, intersect)) {
                    intersections.push_back(intersect);
                }
            } else if (edges[k].btype == 1) {
                BoundaryIntersection intersect0;
                intersect0.i_cell = i;
                intersect0.i_bdry = k;
                BoundaryIntersection intersect1;
                intersect1.i_cell = i;
                intersect1.i_bdry = k;

                bool isInt0, isInt1;
                curvedBoundaryIntersection(v0, v1, k, isInt0, intersect0, isInt1, intersect1);
                if (isInt0) {
                    intersections.push_back(intersect0);
                }
                if (isInt1) {
                    intersections.push_back(intersect1);
                }
            }
        }
    }

    std::sort(intersections.begin(), intersections.end(),
              [](const BoundaryIntersection &a, const BoundaryIntersection &b) {
                  return std::pair<int, double>(a.i_cell, a.t_cell) < std::pair<int, double>(b.i_cell, b.t_cell);
              });

    if (!intersections.empty()) {
        bool inPoly = pointInBounds(nodes[0]);
        if (inPoly) {
            intersections.push_back(intersections[0]);
            intersections.erase(intersections.begin());
        }
    }

    return true;
}
