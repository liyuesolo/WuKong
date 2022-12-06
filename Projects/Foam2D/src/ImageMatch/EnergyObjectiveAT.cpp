#include "../../include/ImageMatch/EnergyObjectiveAT.h"
#include "Projects/Foam2D/include/ImageMatch/EnergyCodeGen.h"

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
EnergyObjectiveAT::getInputs(const VectorXT &c, const VectorXT &area_targets, const int cellIndex,
                             std::vector<int> cell, VectorXT &p_in,
                             VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
                             VectorXi &map) const {
    int n_neighbors = cell.size();

    p_in.resize(7);
    p_in << info->energy_area_weight, info->energy_length_weight, info->energy_centroid_weight, n_neighbors,
            ((cellIndex == info->selected) ? info->energy_drag_target_weight : 0), info->selected_target_pos(
            0), info->selected_target_pos(1);

    int n_vtx = info->n_free + info->n_fixed;
    int n_bdy = info->boundary.rows() / 2;
    int dims = 2 + info->getTessellation()->getNumVertexParams();

    int maxN = 20;

    n_in.resize((maxN + 1) * 1);
    n_in.setZero();

    c_in.resize((maxN + 1) * dims + 1);
    c_in.setZero();
    c_in(0) = area_targets(cellIndex);

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
            c_in.segment(i * dims + 1, dims) = c.segment(map(i) * dims, dims);
        } else {
            b_in.segment<2>(i * 4 + 0) = info->boundary.segment<2>((map(i) - n_vtx) * 2);
            b_in.segment<2>(i * 4 + 2) = info->boundary.segment<2>(((map(i) - n_vtx + 1) % n_bdy) * 2);
        }
    }
}

double EnergyObjectiveAT::evaluate(const VectorXd &x) const {
    VectorXi tri;
    VectorXi e;

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXd c_free = x.segment(0, info->n_free * dims);
    VectorXd area_targets = x.segment(info->n_free * dims, info->n_free);

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
        getInputs(c, area_targets, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_E_cell_AT(info->getTessellation(), p_in, n_in, c_in, b_in, O);
    }

    return O;
}

void EnergyObjectiveAT::addGradientTo(const VectorXd &x, VectorXd &grad) const {
    grad += get_dOdx(x);
}

VectorXd EnergyObjectiveAT::get_dOdx(const VectorXd &x) const {
    VectorXi tri;
    VectorXi e;

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXd c_free = x.segment(0, info->n_free * dims);
    VectorXd area_targets = x.segment(info->n_free * dims, info->n_free);

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;

    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);

    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    tri = info->getTessellation()->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = info->getTessellation()->getNeighborsClipped(vertices, params, tri,
                                                                                       info->boundary, info->n_free);

    VectorXT dOdc = VectorXT::Zero(info->n_free * (dims + 1));
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, area_targets, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_dEdc_cell_AT(info->getTessellation(), p_in, n_in, c_in, b_in, map, dOdc);
    }

    return dOdc;
}

void EnergyObjectiveAT::getHessian(const VectorXd &x, SparseMatrixd &hessian) const {
    hessian = get_d2Odx2(x);
}

Eigen::SparseMatrix<double> EnergyObjectiveAT::get_d2Odx2(const VectorXd &x) const {
    VectorXi tri;
    VectorXi e;

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXd c_free = x.segment(0, info->n_free * dims);
    VectorXd area_targets = x.segment(info->n_free * dims, info->n_free);

    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;

    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);

    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    tri = info->getTessellation()->getDualGraph(vertices, params);
    std::vector<std::vector<int>> cells = info->getTessellation()->getNeighborsClipped(vertices, params, tri,
                                                                                       info->boundary, info->n_free);

    MatrixXT d2Odc2 = MatrixXT::Zero(info->n_free * (dims + 1), info->n_free * (dims + 1));
    for (int i = 0; i < cells.size(); i++) {
        if (cells[i].size() > 18 || cells[i].size() < 3) {
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in;
        VectorXi map;
        getInputs(c, area_targets, i, cells[i], p_in, n_in, c_in, b_in, map);

        add_d2Edc2_cell_AT(info->getTessellation(), p_in, n_in, c_in, b_in, map, d2Odc2);
    }

    return d2Odc2.sparseView();
}

