#include "../../include/Objective/AreaLengthObjective.h"
#include "../../include/CodeGen.h"
#include "../../include/Constants.h"

void
AreaLengthObjective::getInputs(const VectorXT &c, const int cellIndex, std::vector<int> cell, VectorXT &c_cell,
                               VectorXT &p_cell,
                               VectorXi &i_cell) const {
    int n_neighbors = cell.size();

    cell.insert(cell.begin(), cellIndex);
    i_cell = VectorXi::Map(cell.data(), cell.size());

    int dims = 2 + tessellation->getNumVertexParams();

    c_cell.resize(20 * dims);
    c_cell.setZero();
    for (int j = 0; j < i_cell.rows(); j++) {
        c_cell.segment(j * dims, dims) = c.segment(i_cell(j) * dims, dims);
    }

    p_cell.resize(8);
    p_cell << area_weight, length_weight, centroid_weight, getAreaTarget(cellIndex), n_neighbors,
            (cellIndex == drag_idx) * drag_target_weight, drag_target_pos(0), drag_target_pos(1);
}

double AreaLengthObjective::evaluate(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighbors(vertices, tri, n_free);

    double O = 0;
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            O += 1e5;
            continue;
        }

        VectorXT c_cell, p_cell;
        VectorXi i_cell;
        getInputs(c, i, cells[i], c_cell, p_cell, i_cell);

        add_O_cell(tessellation, c_cell, p_cell, O);
    }

    return O;
}

void AreaLengthObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd AreaLengthObjective::get_dOdc(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighbors(vertices, tri, n_free);

    int dims = 2 + tessellation->getNumVertexParams();
    VectorXT dOdc = VectorXT::Zero(n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT c_cell, p_cell;
        VectorXi i_cell;
        getInputs(c, i, cells[i], c_cell, p_cell, i_cell);

        add_dOdc_cell(tessellation, c_cell, p_cell, i_cell, dOdc);
    }

    return dOdc;
}

void AreaLengthObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> AreaLengthObjective::get_d2Odc2(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighbors(vertices, tri, n_free);

    int dims = 2 + tessellation->getNumVertexParams();
    Eigen::SparseMatrix<double> d2Odc2(n_free * dims, n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT c_cell, p_cell;
        VectorXi i_cell;
        getInputs(c, i, cells[i], c_cell, p_cell, i_cell);

        add_d2Odc2_cell(tessellation, c_cell, p_cell, i_cell, d2Odc2);
    }

    return d2Odc2;
}

double AreaLengthObjective::getAreaTarget(int cellIndex) const {
    return area_targets[cellIndex % area_targets.size()];
}

