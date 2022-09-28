#include "../../include/Objective/AreaLengthObjective2.h"
#include "../../include/CodeGen.h"
#include "../../include/Constants.h"

static VectorXi getAreaTriangles2(std::vector<std::vector<int>> cells) {
    VectorXi area_triangles(NAREA * 3);

    int edge = 0;
    for (size_t i = 0; i < NFREE; i++) {
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
        area_triangles[i * 3 + 0] = NFREE; // TODO: this is a hack, degenerate triangle with area 0...
        area_triangles[i * 3 + 1] = 0;
        area_triangles[i * 3 + 2] = 0;
    }

    return area_triangles;
}


double AreaLengthObjective2::evaluate(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXT x;
    VectorXi e;
    VectorXT p;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    x = tessellation->getNodes(vertices, params, tri);
    e = getAreaTriangles2(tessellation->getCells(vertices, tri, x));

    p.resize(3 + NFREE);
    p << area_weight, length_weight, centroid_weight, area_target * VectorXT::Ones(NFREE);

    return tessellation->getNumVertexParams() == 0 ? evaluate_O_voronoi(c, tri, e, p) : evaluate_O_sectional(c, tri,
                                                                                                             e, p);
}

void AreaLengthObjective2::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd AreaLengthObjective2::get_dOdc(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXT x;
    VectorXi e;
    VectorXT p;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    x = tessellation->getNodes(vertices, params, tri);
    e = getAreaTriangles2(tessellation->getCells(vertices, tri, x));

    p.resize(3 + NFREE);
    p << area_weight, length_weight, centroid_weight, area_target * VectorXT::Ones(NFREE);

    return tessellation->getNumVertexParams() == 0 ? evaluate_dOdc_voronoi(c, tri, e, p).transpose()
                                                   : evaluate_dOdc_sectional(c, tri, e, p).transpose();
}

void AreaLengthObjective2::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> AreaLengthObjective2::get_d2Odc2(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXT x;
    VectorXi e;
    VectorXT p;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    x = tessellation->getNodes(vertices, params, tri);
    e = getAreaTriangles2(tessellation->getCells(vertices, tri, x));

    p.resize(3 + NFREE);
    p << area_weight, length_weight, centroid_weight, area_target * VectorXT::Ones(NFREE);

    return tessellation->getNumVertexParams() == 0 ? evaluate_d2Odc2_voronoi(c, tri, e, p)
                                                   : evaluate_d2Odc2_sectional(c, tri, e, p);
}

