#include "../include/Tessellation.h"

std::vector<std::vector<int>>
Tessellation::getCells(const VectorXT &vertices, const VectorXi &dual, const VectorXT &nodes) {
    int n_vtx = vertices.rows() / 2, n_faces = dual.rows() / 3;

    std::vector<std::vector<int>> cells(n_vtx);

    for (int i = 0; i < n_faces; i++) {
        cells[dual(i * 3 + 0)].push_back(i);
        cells[dual(i * 3 + 1)].push_back(i);
        cells[dual(i * 3 + 2)].push_back(i);
    }

    for (int i = 0; i < n_vtx; i++) {
        double xc = vertices(i * 2 + 0);
        double yc = vertices(i * 2 + 1);

        std::sort(cells[i].begin(), cells[i].end(), [nodes, xc, yc](int a, int b) {
            double xa = nodes(a * 2 + 0);
            double ya = nodes(a * 2 + 1);
            double angle_a = atan2(ya - yc, xa - xc);

            double xb = nodes(b * 2 + 0);
            double yb = nodes(b * 2 + 1);
            double angle_b = atan2(yb - yc, xb - xc);

            return angle_a < angle_b;
        });
    }

    return cells;
}

