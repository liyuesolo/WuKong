#include "../../include/Energy/EnergyObjective.h"
#include "Projects/Foam2D/include/Energy/CodeGen.h"
#include "../../include/Constants.h"

static void printVectorXT(std::string name, const VectorXT &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

static void printVectorXi(std::string name, const VectorXi &x, int start = 0, int space = 1) {
    std::cout << name << ": [";
    for (int i = start; i < x.rows(); i += space) {
        std::cout << x[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

void
EnergyObjective::getInputs(const VectorXT &c, const int cellIndex, std::vector<int> cell, VectorXT &p_in,
                           VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
                           VectorXi &map) const {
    int n_neighbors = cell.size();

    p_in.resize(8);
    p_in << area_weight, length_weight, ((cellIndex == drag_idx) ? 1 : centroid_weight), getAreaTarget(
            cellIndex), n_neighbors,
            ((cellIndex == drag_idx) ? drag_target_weight : 0), drag_target_pos(0), drag_target_pos(1);

    int n_vtx = n_free + n_fixed;
    int n_bdy = boundary.rows() / 2;
    int dims = 2 + tessellation->getNumVertexParams();

    int maxN = 20;

    n_in.resize((maxN + 1) * 1);
    n_in.setZero();

    c_in.resize((maxN + 1) * dims);
    c_in.setZero();

    b_in.resize((maxN + 1) * 4);
    b_in.setZero();

    cell.insert(cell.begin(), cellIndex);
    map = VectorXi::Map(cell.data(), cell.size());

    for (int i = 0; i < cell.size(); i++) {
        n_in(i) = map(i) >= n_vtx;

        if (cell[i] < n_vtx) {
            c_in.segment(i * dims, dims) = c.segment(map(i) * dims, dims);
        } else {
            b_in.segment<2>(i * 4 + 0) = boundary.segment<2>((map(i) - n_vtx) * 2);
            b_in.segment<2>(i * 4 + 2) = boundary.segment<2>(((map(i) - n_vtx + 1) % n_bdy) * 2);
        }
    }
}

double EnergyObjective::evaluate(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighborsClipped(vertices, params, tri, boundary, n_free);

    double O = 0;
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            O += 1e5;
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_E_cell(tessellation, p_in, n_in, c_in, b_in, O);
    }

    return O;
}

void EnergyObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd EnergyObjective::get_dOdc(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighborsClipped(vertices, params, tri, boundary, n_free);

    int dims = 2 + tessellation->getNumVertexParams();
    VectorXT dOdc = VectorXT::Zero(n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_dEdc_cell(tessellation, p_in, n_in, c_in, b_in, map, dOdc);
    }

    return dOdc;
}

void EnergyObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + c_fixed.size());
    c << c_free, c_fixed;

    VectorXT vertices;
    VectorXT params;
    tessellation->separateVerticesParams(c, vertices, params);

    tri = tessellation->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = tessellation->getNeighborsClipped(vertices, params, tri, boundary, n_free);

    int dims = 2 + tessellation->getNumVertexParams();
    MatrixXT d2Odc2 = MatrixXT::Zero(n_free * dims, n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_d2Edc2_cell(tessellation, p_in, n_in, c_in, b_in, map, d2Odc2);
    }

    return d2Odc2.sparseView();
}

double EnergyObjective::getAreaTarget(int cellIndex) const {
    return area_targets[cellIndex % area_targets.size()];
}

