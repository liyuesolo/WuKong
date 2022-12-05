#include "../../include/Tessellation/Tessellation.h"
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
                    j++;
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

