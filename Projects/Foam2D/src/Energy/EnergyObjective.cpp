#include "../../include/Energy/EnergyObjective.h"
#include "Projects/Foam2D/include/Energy/CodeGen.h"

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
    p_in << info->energy_area_weight, info->energy_length_weight, ((cellIndex == info->selected) ? 1
                                                                                                 : info->energy_centroid_weight), info->energy_area_targets(
            cellIndex), n_neighbors,
            ((cellIndex == info->selected) ? info->energy_drag_target_weight : 0), info->selected_target_pos(
            0), info->selected_target_pos(1);

    int n_vtx = info->n_free + info->n_fixed;
    int n_bdy = info->boundary.rows() / 2;
    int dims = 2 + info->getTessellation()->getNumVertexParams();

    int maxN = 20;

    n_in.resize((maxN + 1) * 1);
    n_in.setZero();

    c_in.resize((maxN + 1) * dims);
    c_in.setZero();

    b_in.resize((maxN + 1) * 4);
    b_in.setZero();

    cell.insert(cell.begin(), cellIndex);
    map.resize(cell.size());
    for (int i = 0; i < cell.size(); i++) {
        map(i) = cell[i];
    }

    for (int i = 0; i < cell.size(); i++) {
        n_in(i) = map(i) >= n_vtx;

        if (cell[i] < n_vtx) {
            c_in.segment(i * dims, dims) = c.segment(map(i) * dims, dims);
        } else {
            b_in.segment<2>(i * 4 + 0) = info->boundary.segment<2>((map(i) - n_vtx) * 2);
            b_in.segment<2>(i * 4 + 2) = info->boundary.segment<2>(((map(i) - n_vtx + 1) % n_bdy) * 2);
        }
    }
}

double EnergyObjective::evaluate(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;

    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);

    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    tri = info->getTessellation()->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = info->getTessellation()->getNeighborsClipped(vertices, params, tri,
                                                                                       info->boundary, info->n_free);

    double O = 0;
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            O += 1e5;
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_E_cell(info->getTessellation(), p_in, n_in, c_in, b_in, O);
    }

    return O;
}

void EnergyObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd EnergyObjective::get_dOdc(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;

    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);

    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    tri = info->getTessellation()->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = info->getTessellation()->getNeighborsClipped(vertices, params, tri,
                                                                                       info->boundary, info->n_free);

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT dOdc = VectorXT::Zero(info->n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_dEdc_cell(info->getTessellation(), p_in, n_in, c_in, b_in, map, dOdc);
    }

    return dOdc;
}

void EnergyObjective::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
    hessian = get_d2Odc2(c_free);
}

Eigen::SparseMatrix<double> EnergyObjective::get_d2Odc2(const VectorXd &c_free) const {
    VectorXi tri;
    VectorXi e;

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;

    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);

    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    tri = info->getTessellation()->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = info->getTessellation()->getNeighborsClipped(vertices, params, tri,
                                                                                       info->boundary, info->n_free);

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    MatrixXT d2Odc2 = MatrixXT::Zero(info->n_free * dims, info->n_free * dims);
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_d2Edc2_cell(info->getTessellation(), p_in, n_in, c_in, b_in, map, d2Odc2);
    }

    return d2Odc2.sparseView();
}

