#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/CellFunction.h"
#include <iostream>

void Voronoi::getBezierBoundaryNode(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1, const double q0,
                                    const double q1,
                                    const int flag, VectorXT &node) {
    assert(0);
}

void
Voronoi::getBezierBoundaryNodeGradient(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1,
                                       const double q0, const double q1,
                                       const int flag, MatrixXT &nodeGrad) {
    assert(0);
}

void
Voronoi::getBezierBoundaryNodeHessian(const VectorXT &v1, const VectorXT &v2, const TV &b0, const TV &b1,
                                      const double q0, const double q1,
                                      const int flag, std::vector<MatrixXT> &nodeHess) {
    assert(0);
}
