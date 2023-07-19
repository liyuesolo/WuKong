#include "Projects/Foam3D/include/Boundary/BoundaryEnergyPerNeighborhood.h"

void BoundaryEnergyPerNeighborhood::getValue(Boundary *boundary, double &value) const {
    std::vector<NeighborhoodValue> neighborhoods(boundary->v.size());

    for (int i = 0; i < boundary->v.size(); i++) {
        neighborhoods[i].ic = i;
        neighborhoods[i].c = boundary->v[i].pos;
    }
    for (int i = 0; i < boundary->f.size(); i++) {
        IV3 tri = boundary->f[i].vertices;
        for (int j = 0; j < 3; j++) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            neighborhoods[v0].iv.push_back(v1);
            neighborhoods[v0].v.push_back(boundary->v[v1].pos);
        }
    }

    value = 0;
    for (NeighborhoodValue neighborhood: neighborhoods) {
        internalFunction->getValue(neighborhood);
        value += neighborhood.value;
    }
}

void BoundaryEnergyPerNeighborhood::getGradient(Boundary *boundary, VectorXT &gradient) const {
    std::vector<NeighborhoodValue> neighborhoods(boundary->v.size());

    for (int i = 0; i < boundary->v.size(); i++) {
        neighborhoods[i].ic = i;
        neighborhoods[i].c = boundary->v[i].pos;
    }
    for (int i = 0; i < boundary->f.size(); i++) {
        IV3 tri = boundary->f[i].vertices;
        for (int j = 0; j < 3; j++) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            neighborhoods[v0].iv.push_back(v1);
            neighborhoods[v0].v.push_back(boundary->v[v1].pos);
        }
    }

    gradient = VectorXT::Zero(boundary->v.size() * 3);
    for (NeighborhoodValue neighborhood: neighborhoods) {
        internalFunction->getGradient(neighborhood);
        gradient.segment<3>(neighborhood.ic * 3) += neighborhood.gradient.tail<3>();
        for (int i = 0; i < neighborhood.v.size(); i++) {
            gradient.segment<3>(neighborhood.iv[i] * 3) += neighborhood.gradient.segment<3>(i * 3);
        }
    }
}

void BoundaryEnergyPerNeighborhood::getHessian(Boundary *boundary, MatrixXT &hessian) const {
    std::vector<NeighborhoodValue> neighborhoods(boundary->v.size());

    for (int i = 0; i < boundary->v.size(); i++) {
        neighborhoods[i].ic = i;
        neighborhoods[i].c = boundary->v[i].pos;
    }
    for (int i = 0; i < boundary->f.size(); i++) {
        IV3 tri = boundary->f[i].vertices;
        for (int j = 0; j < 3; j++) {
            int v0 = tri(j);
            int v1 = tri((j + 1) % 3);
            neighborhoods[v0].iv.push_back(v1);
            neighborhoods[v0].v.push_back(boundary->v[v1].pos);
        }
    }

    hessian = MatrixXT::Zero(boundary->v.size() * 3, boundary->v.size() * 3);
    for (NeighborhoodValue neighborhood: neighborhoods) {
        internalFunction->getHessian(neighborhood);
        hessian.block<3, 3>(neighborhood.ic * 3, neighborhood.ic * 3) += neighborhood.hessian.bottomRightCorner<3, 3>();
        for (int i = 0; i < neighborhood.v.size(); i++) {
            hessian.block<3, 3>(neighborhood.ic * 3, neighborhood.iv[i] * 3) += neighborhood.hessian.block<3, 3>(
                    neighborhood.v.size() * 3, i * 3);
            hessian.block<3, 3>(neighborhood.iv[i] * 3, neighborhood.ic * 3) += neighborhood.hessian.block<3, 3>(i * 3,
                                                                                                                 neighborhood.v.size() *
                                                                                                                 3);
            for (int j = 0; j < neighborhood.v.size(); j++) {
                hessian.block<3, 3>(neighborhood.iv[i] * 3, neighborhood.iv[j] * 3) += neighborhood.hessian.block<3, 3>(
                        i * 3, j * 3);
            }
        }
    }
}
