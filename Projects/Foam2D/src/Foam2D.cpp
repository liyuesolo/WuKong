#include "../include/Foam2D.h"
#include "../include/CodeGen.h"
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/Sectional.h"
#include "../src/optLib/GradientDescentMinimizer.h"
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

//void Foam2D::checkGradients() {
//    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
//    Eigen::SparseMatrix<double> dxdc = tessellations[tesselation]->getNodesGradient(vertices, params, tri);
//
//    VectorXT x0 = tessellations[tesselation]->getNodes(vertices, params, tri);
//
//    {
//        std::random_device rd;
//        std::mt19937 gen(rd());
//        std::uniform_real_distribution<double> dis(-1, 1);
//        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });
//
//        VectorXT xh;
//
//        double eps = 1e-4;
//        double error = 1;
//        for (int i = 0; i < 10; i++) {
//            xh = tessellations[tesselation]->getNodes(vertices + vert_offsets * eps, params, tri);
//
//            VectorXT xfd = x0 + dxdc * vert_offsets * eps;
//
//            std::cout << "dxdc error " << (xfd - xh).norm() << " " << error / (xfd - xh).norm() << std::endl;
//
//            error = (xfd - xh).norm();
//            eps *= 0.5;
//        }
//    }
//
//    VectorXi area_triangles = getAreaTriangles(tessellations[tesselation]->getCells(vertices, tri, x0));
//
//    VectorXT A = evaluate_A(vertices.segment<NCELLS * 2>(0), x0, area_triangles);
//    Eigen::SparseMatrix<double> dAdx = evaluate_dAdx(vertices.segment<NCELLS * 2>(0), x0, area_triangles);
//
//    for (int i = 0; i < A.rows(); i++) {
//        std::cout << "area " << A[i] << std::endl;
//    }
//
//    {
//        std::random_device rd;
//        std::mt19937 gen(rd());
//        std::uniform_real_distribution<double> dis(-1, 1);
//        VectorXT node_offsets = VectorXT::Zero(x0.rows()).unaryExpr([&](float dummy) { return dis(gen); });
//
//        VectorXT Ah;
//
//        double eps = 1e-4;
//        double error = 1;
//        for (int i = 0; i < 10; i++) {
//            Ah = evaluate_A(vertices.segment<NCELLS * 2>(0), x0 + eps * node_offsets, area_triangles);
//
//            VectorXT Afd = A + dAdx * node_offsets * eps;
//
//            std::cout << "dAdx error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;
//
//            error = (Afd - Ah).norm();
//            eps *= 0.5;
//        }
//    }
//
//    Eigen::SparseMatrix<double> dAdc = dAdx * dxdc;
//    {
//        std::random_device rd;
//        std::mt19937 gen(rd());
//        std::uniform_real_distribution<double> dis(-1, 1);
//        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });
//
//        VectorXT xh;
//        VectorXT Ah;
//
//        double eps = 1e-4;
//        double error = 1;
//        for (int i = 0; i < 10; i++) {
//            xh = tessellations[tesselation]->getNodes(vertices + vert_offsets * eps, params, tri);
//            Ah = evaluate_A((vertices + vert_offsets * eps).segment<NCELLS * 2>(0), xh, area_triangles);
//
//            VectorXT Afd = A + dAdc * vert_offsets * eps;
//
//            std::cout << "dAdc error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;
//
//            error = (Afd - Ah).norm();
//            eps *= 0.5;
//        }
//    }
//}

void Foam2D::generateRandomVoronoi() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(40 * 2);
    for (int i = 0; i < 40; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / 40), sin(i * 2 * M_PI / 40));
    }

    vertices = VectorXT::Zero((NFREE + NFIXED) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment<NFIXED * 2>(NFREE * 2) = boundary_points;

    resetVertexParams();
}

void Foam2D::optimize() {
    objective.tessellation = tessellations[tesselation];

    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    objective.c_fixed = c.segment(NFREE * (2 + tessellations[tesselation]->getNumVertexParams()),
                                  NFIXED * (2 + tessellations[tesselation]->getNumVertexParams()));
    VectorXT c_free = c.segment(0,
                                NFREE * (2 + tessellations[tesselation]->getNumVertexParams()));
    minimizers[opttype]->minimize(&objective, c_free);

    c.segment(0, NFREE * (2 + tessellations[tesselation]->getNumVertexParams())) = c_free;

    tessellations[tesselation]->separateVerticesParams(c, vertices, params);
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < NFREE) {
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
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
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
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    C.resize(n_vtx, 3);
    C.setZero();
    for (int i = 0; i < n_vtx; i++) {
        C.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
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
    int n_cells = NFREE;
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
}
