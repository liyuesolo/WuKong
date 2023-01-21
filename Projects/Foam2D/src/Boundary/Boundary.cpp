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
    for (int i = 0; i < np; i++) {
        int j = next(i);
        double x1 = v(2 * i + 0);
        double y1 = v(2 * i + 1);
        double x2 = v(2 * j + 0);
        double y2 = v(2 * j + 1);

        double a = atan2(y2 - point.y(), x2 - point.x()) - atan2(y1 - point.y(), x1 - point.x());
        if (a > M_PI) a -= 2 * M_PI;
        if (a < -M_PI) a += 2 * M_PI;
        w += a;
    }

    return w > M_PI; // w == (2 * M_PI)
}

bool Boundary::boundarySegmentIntersection(const TV &p0, const TV &p1, int v_idx, BoundaryIntersection &intersect) {
    TV p2 = v.segment<2>(v_idx * 2);
    TV p3 = v.segment<2>(next(v_idx) * 2);

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

bool Boundary::getCellIntersections(const std::vector<TV> &nodes, std::vector<BoundaryIntersection> &intersections) {
    size_t degree = nodes.size(), n_bdy = v.rows() / 2;

    intersections.clear();
    for (size_t i = 0; i < degree; i++) {
        TV v0 = nodes[i];
        TV v1 = nodes[(i + 1) % degree];

        for (size_t k = 0; k < n_bdy; k++) {
            BoundaryIntersection intersect;
            intersect.i_cell = i;
            intersect.i_bdry = k;
            if (boundarySegmentIntersection(v0, v1, k, intersect)) {
                intersections.push_back(intersect);
            }
        }
        std::sort(intersections.begin(), intersections.end(),
                  [](const BoundaryIntersection &a, const BoundaryIntersection &b) {
                      return std::pair<int, double>(a.i_cell, a.t_cell) < std::pair<int, double>(b.i_cell, b.t_cell);
                  });
    }

    std::sort(intersections.begin(), intersections.end(),
              [](const BoundaryIntersection &a, const BoundaryIntersection &b) {
                  return std::pair<int, double>(a.i_cell, a.t_cell) < std::pair<int, double>(b.i_cell, b.t_cell);
              });

    bool inPoly = pointInBounds(nodes[0]);
    for (int i = 0; i < intersections.size(); i++) {
        if (!inPoly) {
            inPoly = true;
            continue;
        }

        BoundaryIntersection intersect0 = intersections[i];
        BoundaryIntersection intersect1 = intersections[(i + 1) % intersections.size()];

        VectorXi segmentDists = -1 * VectorXi::Ones(n_bdy);
        int curr = intersect0.i_bdry;
        int segmentDist = 0;
        do {
            segmentDists(curr) = segmentDist;
            curr = next(curr);
            segmentDist++;
        } while (curr != intersect0.i_bdry);

        // Find next intersection along boundary from intersect0.
        double minDist = 1e10;
        int minIdx = -1;
        for (int j = 0; j < intersections.size(); j++) {
            if (j == i) continue;

            BoundaryIntersection intersectCurr = intersections[j];
            if (segmentDists(intersectCurr.i_bdry) == -1) continue;

            double dist = segmentDists(intersectCurr.i_bdry) + intersectCurr.t_bdry - intersect0.t_bdry;
            if (dist < 0) dist += n_bdy;

            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        if (minIdx != (i + 1) % intersections.size()) return false;

        inPoly = false;
    }

    return true;
}
