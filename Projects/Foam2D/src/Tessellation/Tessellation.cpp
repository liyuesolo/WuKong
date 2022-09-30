#include "../../include/Tessellation/Tessellation.h"
#include "../../include/Constants.h"
#include <set>

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
        std::vector<int> cell = cells[i];

        double xc = 0, yc = 0;
        for (int j = 0; j < cell.size(); j++) {
            xc += nodes(cell[j] * 2 + 0);
            yc += nodes(cell[j] * 2 + 1);
        }
        xc /= cell.size();
        yc /= cell.size();

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

    std::vector<std::vector<int>> neighborlists;

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

        neighborlists.push_back(neighbors);
    }

    return neighborlists;
}

