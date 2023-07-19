#include <igl/fast_find_self_intersections.h>
#include <igl/slice.h>
#include "../../include/Boundary/GastrulationBoundary.h"
#include "../../include/Boundary/BoundaryEnergyPerNeighborhood.h"
#include "../../include/Boundary/NeighborhoodLaplacian.h"
#include "../../include/Boundary/BoundaryEnergyVolumeTarget.h"
#include "../../include/Boundary/BoundaryEnergySphericalBarrier.h"

#include "../../include/Globals.h"

GastrulationBoundary::GastrulationBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_) : Boundary(
        Eigen::Map<VectorXT>(MatrixXT(v_.transpose()).data(), v_.size()), free_) {
    initialize(v_.rows(), f_);
}

bool GastrulationBoundary::checkValid() {
    MatrixXT V(v.size(), 3);
    for (int i = 0; i < v.size(); i++) {
        V.row(i) = v[i].pos;
        if (v[i].pos.squaredNorm() > pow(outerRadius, 2.0)) {
            return false;
        }
    }
    MatrixXi F(f.size(), 3);
    for (int i = 0; i < f.size(); i++) {
        F.row(i) = f[i].vertices;
    }
    MatrixXi I;

    return !igl::fast_find_self_intersections(V, F, I);
}

void GastrulationBoundary::computeVertices() {
    TV3 averagePos = TV3::Zero();

    for (int i = 0; i < v.size(); i++) {
        v[i].pos = p.segment<3>(i * 3);
        averagePos += v[i].pos;

        for (int j = 0; j < 3; j++) {
            addGradientEntry(i, j, i * 3 + j, 1.0);
        }
    }

    averagePos /= v.size();
//    std::cout << "Boundary Centroid " << averagePos(0) << " " << averagePos(1) << " " << averagePos(2) << std::endl;
}

double GastrulationBoundary::computeEnergy() {
    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget volumeEnergyFunc(volTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    double laplacianEnergy = 0;
    laplacianEnergyFunc.getValue(this, laplacianEnergy);
    double volumeEnergy = 0;
    volumeEnergyFunc.getValue(this, volumeEnergy);
    double sphericalBarrierEnergy = 0;
    sphericalBarrierEnergyFunc.getValue(this, sphericalBarrierEnergy);

    return kNeighborhood * laplacianEnergy + kVol * volumeEnergy + sphericalBarrierEnergy;
}

VectorXT GastrulationBoundary::computeEnergyGradient() {
    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget volumeEnergyFunc(volTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    VectorXT laplacianGrad;
    laplacianEnergyFunc.getGradient(this, laplacianGrad);
    VectorXT volumeGrad;
    volumeEnergyFunc.getGradient(this, volumeGrad);
    VectorXT sphericalBarrierGrad;
    sphericalBarrierEnergyFunc.getGradient(this, sphericalBarrierGrad);

    return kNeighborhood * laplacianGrad + kVol * volumeGrad + sphericalBarrierGrad;
}

MatrixXT GastrulationBoundary::computeEnergyHessian() {
    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget volumeEnergyFunc(volTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    MatrixXT laplacianHess;
    laplacianEnergyFunc.getHessian(this, laplacianHess);
    MatrixXT volumeHess;
    volumeEnergyFunc.getHessian(this, volumeHess);
    MatrixXT sphericalBarrierHess;
    sphericalBarrierEnergyFunc.getHessian(this, sphericalBarrierHess);

    return kNeighborhood * laplacianHess + kVol * volumeHess + sphericalBarrierHess;
}

//double GastrulationBoundary::computeEnergy() {
//    double energy = 0;
//    double volume = 0;
//
//    PerTriangleVolume volFunc;
//    for (int iF = 0; iF < f.size(); iF++) {
//        BoundaryFace face = f[iF];
//
//        IV3 verts = face.vertices;
//        for (int i = 0; i < 3; i++) {
//            int v0 = verts[i];
//            int v1 = verts[(i + 1) % 3];
//
//            double d2 = (v[v1].pos - v[v0].pos).squaredNorm();
//            energy += 0.5 * kEdge * d2;
//        }
//
//        if (iF < f.size() / 2) continue;
//
//        TriangleValue vol;
//        vol.v0 = v[verts[0]].pos;
//        vol.v1 = v[verts[1]].pos;
//        vol.v2 = v[verts[2]].pos;
//        volFunc.getValue(vol);
//        volume += vol.value;
//    }
//
//    std::cout << "Sphere volume " << volume << std::endl;
//    energy += kVol * pow(volume - volTarget, 2);
//
//    return energy;
//}
//
//VectorXT GastrulationBoundary::computeEnergyGradient() {
//    VectorXT gradient = VectorXT::Zero(nfree);
//
//    double volume = 0;
//    VectorXT volGradient = VectorXT::Zero(nfree);
//
//    PerTriangleVolume volFunc;
//    for (int iF = 0; iF < f.size(); iF++) {
//        BoundaryFace face = f[iF];
//
//        IV3 verts = face.vertices;
//        for (int i = 0; i < 3; i++) {
//            int v0 = verts[i];
//            int v1 = verts[(i + 1) % 3];
//
//            TV3 d = v[v1].pos - v[v0].pos;
//            gradient(v0 * 3 + 0) += -kEdge * d(0);
//            gradient(v0 * 3 + 1) += -kEdge * d(1);
//            gradient(v0 * 3 + 2) += -kEdge * d(2);
//            gradient(v1 * 3 + 0) += kEdge * d(0);
//            gradient(v1 * 3 + 1) += kEdge * d(1);
//            gradient(v1 * 3 + 2) += kEdge * d(2);
//        }
//
//        if (iF < f.size() / 2) continue;
//
//        TriangleValue vol;
//        vol.v0 = v[verts[0]].pos;
//        vol.v1 = v[verts[1]].pos;
//        vol.v2 = v[verts[2]].pos;
//        volFunc.getValue(vol);
//        volFunc.getGradient(vol);
//        volume += vol.value;
//        for (int i = 0; i < 3; i++) {
//            for (int ii = 0; ii < 3; ii++) {
//                volGradient(verts[i] * 3 + ii) += vol.gradient(i * 3 + ii);
//            }
//        }
//    }
//    gradient += kVol * 2 * (volume - volTarget) * volGradient;
//
//    if (nfree < np) {
//        VectorXT gradient2;
//        igl::slice(gradient, free_map, gradient2);
//        return gradient2;
//    } else {
//        return gradient;
//    }
//}
//
//MatrixXT GastrulationBoundary::computeEnergyHessian() {
//    MatrixXT hessian = MatrixXT::Zero(np, np);
//
//    double volume = 0;
//    VectorXT volGradient = VectorXT::Zero(np);
//    MatrixXT volHessian = MatrixXT::Zero(np, np);
//
//    PerTriangleVolume volFunc;
//    for (int iF = 0; iF < f.size(); iF++) {
//        BoundaryFace face = f[iF];
//
//        IV3 verts = face.vertices;
//        for (int i = 0; i < 3; i++) {
//            int v0 = verts[i];
//            int v1 = verts[(i + 1) % 3];
//
//            int i00 = v0 * 3 + 0;
//            int i01 = v0 * 3 + 1;
//            int i02 = v0 * 3 + 2;
//            int i10 = v1 * 3 + 0;
//            int i11 = v1 * 3 + 1;
//            int i12 = v1 * 3 + 2;
//
//            hessian(i00, i00) += kEdge;
//            hessian(i00, i10) += -kEdge;
//            hessian(i10, i00) += -kEdge;
//            hessian(i10, i10) += kEdge;
//            hessian(i01, i01) += kEdge;
//            hessian(i01, i11) += -kEdge;
//            hessian(i11, i01) += -kEdge;
//            hessian(i11, i11) += kEdge;
//            hessian(i02, i02) += kEdge;
//            hessian(i02, i12) += -kEdge;
//            hessian(i12, i02) += -kEdge;
//            hessian(i12, i12) += kEdge;
//        }
//
//        if (iF < f.size() / 2) continue;
//
//        TriangleValue vol;
//        vol.v0 = v[verts[0]].pos;
//        vol.v1 = v[verts[1]].pos;
//        vol.v2 = v[verts[2]].pos;
//        volFunc.getValue(vol);
//        volFunc.getGradient(vol);
//        volFunc.getHessian(vol);
//        volume += vol.value;
//        for (int i = 0; i < 3; i++) {
//            for (int ii = 0; ii < 3; ii++) {
//                volGradient(verts[i] * 3 + ii) = vol.gradient(i * 3 + ii);
//                for (int j = 0; j < 3; j++) {
//                    for (int jj = 0; jj < 3; jj++) {
//                        volHessian(verts[i] * 3 + ii, verts[j] * 3 + jj) +=
//                                vol.hessian(i * 3 + ii, j * 3 + jj);
//                    }
//                }
//            }
//        }
//    }
//    hessian += kVol * (2 * (volume - volTarget) * volHessian + 2 * volGradient * volGradient.transpose());
//
//    if (nfree < np) {
//        MatrixXT hessian2;
//        igl::slice(hessian, free_map, free_map, hessian2);
//        return hessian2;
//    } else {
//        return hessian;
//    }
//}

