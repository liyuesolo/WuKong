#include "../../include/ImageMatch/ImageMatchObjective.h"
#include "../../include/ImageMatch/CellFunctionImageMatch2AreaScaled.h"

void ImageMatchObjective::preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const {
    VectorXd c(c_free.size() + info->c_fixed.size());
    c << c_free, info->c_fixed;
    VectorXT vertices;
    VectorXT params;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    cellInfos.resize(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        cellInfos[i].border_pix = pix[i];
    }
}

double ImageMatchObjective::evaluate(const VectorXd &c_free) const {
    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    if (!info->getTessellation()->isValid) {
        return 1e10;
    }

    CellFunctionImageMatch2AreaScaled imageMatchFunction;

    double O = 0;
    info->getTessellation()->addFunctionValue(imageMatchFunction, O, cellInfos);

    return O;
}

void ImageMatchObjective::addGradientTo(const VectorXd &c_free, VectorXd &grad) const {
    grad += get_dOdc(c_free);
}

VectorXd ImageMatchObjective::get_dOdc(const VectorXd &c_free) const {
    std::vector<CellInfo> cellInfos;
    preProcess(c_free, cellInfos);

    VectorXT gradient = VectorXT::Zero(c_free.rows());
    if (!info->getTessellation()->isValid) {
        return gradient;
    }

    CellFunctionImageMatch2AreaScaled imageMatchFunction;
    info->getTessellation()->addFunctionGradient(imageMatchFunction, gradient, cellInfos);

    return gradient;
}
