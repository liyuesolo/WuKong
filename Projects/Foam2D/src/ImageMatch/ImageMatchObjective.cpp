#include "../../include/ImageMatch/ImageMatchObjective.h"
#include "Projects/Foam2D/include/ImageMatch/CodeGen.h"

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
ImageMatchObjective::getInputs(const VectorXT &c, const int cellIndex, std::vector<int> cell, VectorXT &p_in,
                               VectorXT &n_in, VectorXT &c_in, VectorXT &b_in, VectorXT &pix_in,
                               VectorXi &map) const {
    int n_neighbors = cell.size();
    int n_points = pix[cellIndex].rows() / 2;

    p_in.resize(2);
    p_in << n_neighbors, n_points;

    int n_vtx = n_free + n_fixed;
    int n_bdy = boundary.rows() / 2;
    int dims = 2 + tessellation->getNumVertexParams();

    int maxN = 20;
    int maxP = 1000;

    n_in.resize((maxN + 1) * 1);
    n_in.setZero();

    c_in.resize((maxN + 1) * dims);
    c_in.setZero();

    b_in.resize((maxN + 1) * 4);
    b_in.setZero();

    pix_in.resize(maxP * 2);
    pix_in.setZero();
    pix_in.segment(0, 2 * n_points) = pix[cellIndex];

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
            b_in.segment<2>(i * 4 + 0) = boundary.segment<2>((map(i) - n_vtx) * 2);
            b_in.segment<2>(i * 4 + 2) = boundary.segment<2>(((map(i) - n_vtx + 1) % n_bdy) * 2);
        }
    }
}

double ImageMatchObjective::evaluate(const VectorXd &c_free) const {
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
//            O += 1e20;
            continue;
        }

        VectorXT p_in, n_in, c_in, b_in, pix_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, pix_in, map);

        add_value_cell(tessellation, p_in, n_in, c_in, b_in, pix_in, O);
    }

    return O;
}

void ImageMatchObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd ImageMatchObjective::get_dOdc(const VectorXd &c_free) const {
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

        VectorXT p_in, n_in, c_in, b_in, pix_in;
        VectorXi map;
        getInputs(c, i, cells[i], p_in, n_in, c_in, b_in, pix_in, map);

        add_gradient_cell(tessellation, p_in, n_in, c_in, b_in, pix_in, map, dOdc);
    }

    return dOdc;
}
