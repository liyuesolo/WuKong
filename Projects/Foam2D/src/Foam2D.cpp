#include "../include/Foam2D.h"
#include "../include/CodeGen.h"
#include "../include/Voronoi.h"
#include "../include/Sectional.h"
#include "../src/optLib/GradientDescentMinimizer.h"
#include <random>

#define NCELLS 40
#define NAREA 300
#define OBJWEIGHT 2

Foam2D::Foam2D() {
    tessellations.push_back(new Voronoi());
    tessellations.push_back(new Sectional());
}

static VectorXi getAreaTriangles(std::vector<std::vector<int>> cells) {
    VectorXi area_triangles(NAREA * 3);

    int edge = 0;
    for (size_t i = 0; i < NCELLS; i++) {
        std::vector<int> &cell = cells[i];
        size_t degree = cell.size();

        for (size_t j = 0; j < degree; j++) {
            area_triangles[edge * 3 + 0] = i;
            area_triangles[edge * 3 + 1] = cell[j];
            area_triangles[edge * 3 + 2] = cell[(j + 1) % degree];
            edge++;
        }
    }

    for (int i = edge; i < NAREA; i++) {
        area_triangles[i * 3 + 0] = 0; // TODO: this is a hack, degenerate triangle with area 0...
        area_triangles[i * 3 + 1] = 0;
        area_triangles[i * 3 + 2] = 0;
    }

    return area_triangles;
}

void Foam2D::checkGradients() {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices);
    Eigen::SparseMatrix<double> dxdc = tessellations[tesselation]->getNodesGradient(vertices, tri);

    VectorXT x0 = tessellations[tesselation]->getNodes(vertices, tri);

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT xh;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            xh = tessellations[tesselation]->getNodes(vertices + vert_offsets * eps, tri);

            VectorXT xfd = x0 + dxdc * vert_offsets * eps;

            std::cout << "dxdc error " << (xfd - xh).norm() << " " << error / (xfd - xh).norm() << std::endl;

            error = (xfd - xh).norm();
            eps *= 0.5;
        }
    }

    VectorXi area_triangles = getAreaTriangles(tessellations[tesselation]->getCells(vertices, tri, x0));

    VectorXT A = evaluate_A(vertices.segment<NCELLS * 2>(0), x0, area_triangles);
    Eigen::SparseMatrix<double> dAdx = evaluate_dAdx(vertices.segment<NCELLS * 2>(0), x0, area_triangles);

    for (int i = 0; i < A.rows(); i++) {
        std::cout << "area " << A[i] << std::endl;
    }

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT node_offsets = VectorXT::Zero(x0.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT Ah;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            Ah = evaluate_A(vertices.segment<NCELLS * 2>(0), x0 + eps * node_offsets, area_triangles);

            VectorXT Afd = A + dAdx * node_offsets * eps;

            std::cout << "dAdx error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;

            error = (Afd - Ah).norm();
            eps *= 0.5;
        }
    }

    Eigen::SparseMatrix<double> dAdc = dAdx * dxdc;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT xh;
        VectorXT Ah;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            xh = tessellations[tesselation]->getNodes(vertices + vert_offsets * eps, tri);
            Ah = evaluate_A((vertices + vert_offsets * eps).segment<NCELLS * 2>(0), xh, area_triangles);

            VectorXT Afd = A + dAdc * vert_offsets * eps;

            std::cout << "dAdc error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;

            error = (Afd - Ah).norm();
            eps *= 0.5;
        }
    }
}

void Foam2D::generateRandomVoronoi() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(40 * 2);
    for (int i = 0; i < 40; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / 40), sin(i * 2 * M_PI / 40));
    }

    vertices = VectorXT::Zero((40 + NCELLS) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment<40 * 2>(NCELLS * 2) = boundary_points;
}


class AreaObjective : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    double area_target = 0.1;

public:
    virtual double evaluate(const VectorXd &c) const {
        VectorXi tri;
        VectorXT x;
        VectorXi e;
        VectorXT A;

        tri = tessellation->getDualGraph(c);
        x = tessellation->getNodes(c, tri);

        e = getAreaTriangles(tessellation->getCells(c, tri, x));

        A = evaluate_A(c.segment<NCELLS * 2>(0), x, e);

        double O = 0;
        for (int i = 0; i < A.rows(); i++) {
            O += (A(i) - area_target) * (A(i) - area_target);
        }

        return OBJWEIGHT * O;
    }

    virtual void addGradientTo(const VectorXd &c, VectorXd &grad) const {
        grad += get_DODc(c);
    }

    VectorXd get_DODc(const VectorXd &c) const {
        VectorXi tri;
        VectorXT x;
        Eigen::SparseMatrix<double> dxdc;
        VectorXi e;
        VectorXT A;
        Eigen::SparseMatrix<double> dAdx;
        MatrixXT dOdA;

        tri = tessellation->getDualGraph(c);
        x = tessellation->getNodes(c, tri);
        dxdc = tessellation->getNodesGradient(c, tri);

        e = getAreaTriangles(tessellation->getCells(c, tri, x));

        A = evaluate_A(c.segment<NCELLS * 2>(0), x, e);
        dAdx = evaluate_dAdx(c.segment<NCELLS * 2>(0), x, e);

        VectorXT targets;
        targets.resize(A.rows());
        targets.setOnes();
        targets = targets * area_target;
        dOdA = 2 * (A - targets).transpose();

        VectorXT dOdc = (dOdA * dAdx * dxdc).transpose();
        dOdc.segment<40 * 2>(NCELLS * 2) *= 0; // Fix boundary sites...

        return OBJWEIGHT * dOdc.transpose();
    }
};

void Foam2D::optimize(double area_target) {
    AreaObjective objective;
    Voronoi voronoi;
    objective.tessellation = &voronoi;
    objective.area_target = area_target;

    GradientDescentLineSearch minimizer(1, 1e-6, 15);
    minimizer.minimize(&objective, vertices);
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < NCELLS) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Foam2D::moveVertex(int idx, const TV &pos) {
    vertices.segment<2>(idx * 2) = pos;
}

void Foam2D::getTriangulationViewerData(MatrixXT &C, MatrixXT &X, MatrixXi &E) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    C.resize(n_vtx, 3);
    C.setZero();
    for (int i = 0; i < n_vtx; i++) {
        C.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    X = C;

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
}

void Foam2D::getTessellationViewerData(MatrixXT &C, MatrixXT &X, MatrixXi &E) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    C.resize(n_vtx, 3);
    C.setZero();
    for (int i = 0; i < n_vtx; i++) {
        C.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    VectorXT x = tessellations[tesselation]->getNodes(vertices, tri);
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
    for (int i = 0; i < NCELLS; i++) {
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
}
