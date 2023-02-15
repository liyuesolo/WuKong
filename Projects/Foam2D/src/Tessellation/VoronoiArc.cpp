#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/CellFunction.h"
#include <iostream>

void Voronoi::getArcBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, const double r,
                                 const int flag, VectorXT &node) {
    assert(0);
}

void
Voronoi::getArcBoundaryNodeGradient(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, const double r,
                                    const int flag, MatrixXT &nodeGrad) {
    assert(0);
}

void
Voronoi::getArcBoundaryNodeHessian(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, const double r,
                                   const int flag, std::vector<MatrixXT> &nodeHess) {
    assert(0);
}
