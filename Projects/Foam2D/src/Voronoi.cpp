#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "../include/Voronoi.h"
#include "../include/CodeGen.h"

VectorXi Voronoi::getDualGraph(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 2;

    MatrixXT P;
    P.resize(n_vtx, 2);
    for (int i = 0; i < n_vtx; i++) {
        P.row(i) = vertices.segment<2>(i * 2);
    }

    MatrixXT V;
    MatrixXi F;
    igl::triangle::triangulate(P,
                               MatrixXi(),
                               MatrixXT(),
                               "cQ", // Enclose convex hull with segments
                               V, F);

    VectorXi tri;
    tri.resize(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++)
        tri.segment<3>(i * 3) = F.row(i);
    return tri;
}

VectorXT Voronoi::getNodes(const VectorXT &vertices, const VectorXi &dual) {
    return evaluate_x(vertices, dual);
}

Eigen::SparseMatrix<double> Voronoi::getNodesGradient(const VectorXT &vertices, const VectorXi &dual) {
    return evaluate_dxdc(vertices, dual);
}

