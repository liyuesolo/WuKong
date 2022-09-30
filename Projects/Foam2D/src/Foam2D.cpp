#include "../include/Foam2D.h"
#include "../include/CodeGen.h"
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/Sectional.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../include/Constants.h"
#include <random>

Foam2D::Foam2D() {
    tessellations.push_back(new Voronoi());
    tessellations.push_back(new Sectional());
    minimizers.push_back(new GradientDescentLineSearch(1, 1e-6, 15));
    minimizers.push_back(new NewtonFunctionMinimizer(1, 1e-6, 15));
}

void Foam2D::resetVertexParams() {
    params = tessellations[tesselation]->getDefaultVertexParams(vertices);
}

void Foam2D::initRandomSitesInCircle(int n_free_in, int n_fixed_in) {
    n_free = n_free_in;
    n_fixed = n_fixed_in;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(n_fixed * 2);
    for (int i = 0; i < n_fixed; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / n_fixed), sin(i * 2 * M_PI / n_fixed));
    }

    vertices = VectorXT::Zero((n_free + n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(n_free * 2, n_fixed * 2) = boundary_points;

    resetVertexParams();
}

void Foam2D::initBasicTestCase() {
    n_free = 2;
    n_fixed = 4;

    vertices.resize(6 * 2);

    vertices << 0, 0.5, 0, -0.4, 0, 1, -0.4, 0, 0, -1, 0.4, 0;

    resetVertexParams();
}

void Foam2D::optimize() {
    objective.tessellation = tessellations[tesselation];
    objective.n_free = n_free;
    objective.n_fixed = n_fixed;

    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    objective.c_fixed = c.segment(n_free * (2 + tessellations[tesselation]->getNumVertexParams()),
                                  n_fixed * (2 + tessellations[tesselation]->getNumVertexParams()));
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tesselation]->getNumVertexParams()));
    minimizers[opttype]->minimize(&objective, c_free);

    c.segment(0, n_free * (2 + tessellations[tesselation]->getNumVertexParams())) = c_free;

    tessellations[tesselation]->separateVerticesParams(c, vertices, params);
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < n_free) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Foam2D::moveVertex(int idx, const TV &pos) {
    vertices.segment<2>(idx * 2) = pos;
}

static TV3 getColor(double area, double target) {
    double r, g, b;
    double q = (area - target);
    if (q >= 0) q = sqrt(q); else q = -sqrt(-q);
    r = 1 - q / 0.2;
    g = 1;
    b = 1 + q / 0.2;

    r = fmin(fmax(r, 0), 1);
    g = fmin(fmax(g, 0), 1);
    b = fmin(fmax(b, 0), 1);
    return {r, g, b};
}

static VectorXT getCellAreas(VectorXT x, std::vector<std::vector<int>> cells, int n_cells) {
    VectorXT A(n_cells);
    for (int i = 0; i < n_cells; i++) {
        if (cells[i].size() < 3) {
            A(i) = 0;
        } else {
            vector<int> cell = cells[i];
            double x1 = x(cell[0] * 2 + 0);
            double y1 = x(cell[0] * 2 + 1);

            double area = 0;
            for (int j = 1; j < cell.size() - 1; j++) {
                double x2 = x(cell[j] * 2 + 0);
                double y2 = x(cell[j] * 2 + 1);
                double x3 = x(cell[j + 1] * 2 + 0);
                double y3 = x(cell[j + 1] * 2 + 1);
                area += 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
            }
            A(i) = area;
        }
    }
    return A;
}

void Foam2D::getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    X = S;

    int num_edges = n_faces * 3;
    E.resize(num_edges, 2);
    E.setZero();

    for (int i = 0; i < n_faces; i++) {
        int v0 = tri(i * 3 + 0);
        int v1 = tri(i * 3 + 1);
        int v2 = tri(i * 3 + 2);

        E.row(i * 3 + 0) = IV(v0, v1);
        E.row(i * 3 + 1) = IV(v1, v2);
        E.row(i * 3 + 2) = IV(v2, v0);
    }

    // Mesh points and faces
    V.resize(n_vtx, 3);
    V << S;

    F.resize(n_faces, 3);
    F.setZero();
    C.resize(n_faces, 3);
    C.setZero();
    for (int i = 0; i < n_faces; i++) {
        F.row(i) = tri.segment<3>(i * 3);
        C.row(i) = TV3(0.5, 0.5, 0.5);
    }
}

void Foam2D::getTessellationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    // Overlay points and edges
    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    VectorXT x = tessellations[tesselation]->getNodes(vertices, params, tri);
    X.resize(n_faces, 3);
    X.setZero();
    for (int i = 0; i < n_faces; i++) {
        X.row(i).segment<2>(0) = x.segment<2>(i * 2);
    }

    int num_voronoi_edges = 2 * n_faces - n_vtx + 1;
    E.resize(num_voronoi_edges * 2, 2);
    E.setZero();

    std::vector<std::vector<int>> cells = tessellations[tesselation]->getCells(vertices, tri, x);
    int edge = 0;
    int n_cells = n_free;
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &cell = cells[i];
        size_t degree = cell.size();

        for (size_t j = 0; j < degree; j++) {
            int v1 = cell[j];
            int v2 = cell[(j + 1) % degree];

            // TODO: This adds most edges twice (once per adjacent cell), fine for now.
            E.row(edge) = IV(v1, v2);
            edge++;
        }
    }

    // Mesh points and faces
    V.resize(n_vtx + n_faces, 3);
    V << S, X;

    F.resize(num_voronoi_edges * 2, 3);
    F.setZero();
    C.resize(num_voronoi_edges * 2, 3);
    C.setZero();
    VectorXT areas = getCellAreas(x, cells, n_free);
    edge = 0;
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &cell = cells[i];
        for (size_t j = 1; j < cells[i].size() - 1; j++) {
            int v1 = cell[0];
            int v2 = cell[j];
            int v3 = cell[j + 1];

            F.row(edge) = IV3(v1 + n_vtx, v2 + n_vtx, v3 + n_vtx);
            C.row(edge) = getColor(areas(i), objective.area_target);
            edge++;
        }
    }
}
