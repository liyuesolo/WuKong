//#include "../../include/Energy/EnergyObjectiveCasadi.h"
//#include "Projects/Foam2D/include/Energy/CodeGen.h"
//
//static void printVectorXT(std::string name, const VectorXT &x, int start = 0, int space = 1) {
//    std::cout << name << ": [";
//    for (int i = start; i < x.rows(); i += space) {
//        std::cout << x[i] << ", ";
//    }
//    std::cout << "]" << std::endl;
//}
//
//static void printVectorXi(std::string name, const VectorXi &x, int start = 0, int space = 1) {
//    std::cout << name << ": [";
//    for (int i = start; i < x.rows(); i += space) {
//        std::cout << x[i] << ", ";
//    }
//    std::cout << "]" << std::endl;
//}
//
//void
//EnergyObjectiveCasadi::getInputs(const int cellIndex, VectorXT &p_in,
//                           VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
//                           VectorXi &map) const {
//    std::vector<int> neighborhood = info->getTessellation()->todo_neighborhoods[cellIndex];
//    int n_neighbors = neighborhood.size();
//
//    p_in.resize(8);
//    p_in << info->energy_area_weight, info->energy_length_weight, ((cellIndex == info->selected) ? 1
//                                                                                                 : info->energy_centroid_weight), info->energy_area_targets(
//            cellIndex), n_neighbors,
//            ((cellIndex == info->selected) ? info->energy_drag_target_weight : 0), info->selected_target_pos(
//            0), info->selected_target_pos(1);
//
//    int n_vtx = info->n_free + info->n_fixed;
//    int n_bdy = info->boundary.rows() / 2;
//    int dims = 2 + info->getTessellation()->getNumVertexParams();
//
//    int maxN = 20;
//
//    n_in.resize((maxN + 1) * 1);
//    n_in.setZero();
//
//    c_in.resize((maxN + 1) * dims);
//    c_in.setZero();
//
//    b_in.resize((maxN + 1) * 4);
//    b_in.setZero();
//
//    neighborhood.insert(neighborhood.begin(), cellIndex);
//    map.resize(neighborhood.size());
//    for (int i = 0; i < neighborhood.size(); i++) {
//        map(i) = neighborhood[i];
//    }
//
//    for (int i = 0; i < neighborhood.size(); i++) {
//        n_in(i) = map(i) >= n_vtx;
//
//        if (neighborhood[i] < n_vtx) {
//            c_in.segment(i * dims, dims) = info->getTessellation()->c.segment(map(i) * dims, dims);
//        } else {
//            b_in.segment<2>(i * 4 + 0) = info->boundary.segment<2>((map(i) - n_vtx) * 2);
//            b_in.segment<2>(i * 4 + 2) = info->boundary.segment<2>(((map(i) - n_vtx + 1) % n_bdy) * 2);
//        }
//    }
//}
//
//double EnergyObjectiveCasadi::evaluate(const VectorXd &c_free) const {
//    VectorXi tri;
//    VectorXi e;
//
//    VectorXd c(c_free.size() + info->c_fixed.size());
//    c << c_free, info->c_fixed;
//
//    VectorXT vertices;
//    VectorXT params;
//    info->getTessellation()->separateVerticesParams(c, vertices, params);
//
//    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
//
//    double O = 0;
//    for (int i = 0; i < info->n_free; i++) {
//        if (info->getTessellation()->todo_neighborhoods[i].size() > 18 ||
//            info->getTessellation()->todo_neighborhoods[i].size() < 3) {
//            O += 1e5;
//            continue;
//        }
//
//        VectorXT p_in, n_in, c_in, b_in;
//        VectorXi map;
//        getInputs(i, p_in, n_in, c_in, b_in, map);
//
//        add_E_cell(info->getTessellation(), p_in, n_in, c_in, b_in, O);
//    }
//
//    return O;
//}
//
//void EnergyObjectiveCasadi::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
//    grad += get_dOdc(c_free);
//}
//
//VectorXd EnergyObjectiveCasadi::get_dOdc(const VectorXd &c_free) const {
//    VectorXi tri;
//    VectorXi e;
//
//    VectorXd c(c_free.size() + info->c_fixed.size());
//    c << c_free, info->c_fixed;
//
//    VectorXT vertices;
//    VectorXT params;
//    info->getTessellation()->separateVerticesParams(c, vertices, params);
//
//    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
//
//    int dims = 2 + info->getTessellation()->getNumVertexParams();
//    VectorXT dOdc = VectorXT::Zero(info->n_free * dims);
//    for (int i = 0; i < info->n_free; i++) {
//        if (info->getTessellation()->todo_neighborhoods[i].size() > 18 ||
//            info->getTessellation()->todo_neighborhoods[i].size() < 3) {
//            continue;
//        }
//
//        VectorXT p_in, n_in, c_in, b_in;
//        VectorXi map;
//        getInputs(i, p_in, n_in, c_in, b_in, map);
//
//        add_dEdc_cell(info->getTessellation(), p_in, n_in, c_in, b_in, map, dOdc);
//    }
//
//    return dOdc;
//}
//
//void EnergyObjectiveCasadi::getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const {
//    hessian = get_d2Odc2(c_free);
//}
//
//Eigen::SparseMatrix<double> EnergyObjectiveCasadi::get_d2Odc2(const VectorXd &c_free) const {
//    VectorXi tri;
//    VectorXi e;
//
//    VectorXd c(c_free.size() + info->c_fixed.size());
//    c << c_free, info->c_fixed;
//
//    VectorXT vertices;
//    VectorXT params;
//    info->getTessellation()->separateVerticesParams(c, vertices, params);
//
//    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
//
//    int dims = 2 + info->getTessellation()->getNumVertexParams();
//    MatrixXT d2Odc2 = MatrixXT::Zero(info->n_free * dims, info->n_free * dims);
//    for (int i = 0; i < info->n_free; i++) {
//        if (info->getTessellation()->todo_neighborhoods[i].size() > 18 ||
//            info->getTessellation()->todo_neighborhoods[i].size() < 3) {
//            continue;
//        }
//
//        VectorXT p_in, n_in, c_in, b_in;
//        VectorXi map;
//        getInputs(i, p_in, n_in, c_in, b_in, map);
//
//        add_d2Edc2_cell(info->getTessellation(), p_in, n_in, c_in, b_in, map, d2Odc2);
//    }
//
//    return d2Odc2.sparseView();
//}
//
