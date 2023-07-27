#include <igl/fast_find_self_intersections.h>
#include <igl/slice.h>
#include <tbb/tbb.h>
#include "../../include/Boundary/GastrulationBoundary.h"
#include "../../include/Boundary/BoundaryEnergyPerNeighborhood.h"
#include "../../include/Boundary/NeighborhoodLaplacian.h"
#include "../../include/Boundary/BoundaryEnergyVolumeTarget.h"
#include "../../include/Boundary/BoundaryEnergySphericalBarrier.h"
#include <chrono>

#include "../../include/Globals.h"

#define PRINT_INTERMEDIATE_TIMES false
#define PRINT_TOTAL_TIME false

static void
printTime(std::chrono::high_resolution_clock::time_point tstart, std::string description = "", bool final = false) {
    if (PRINT_INTERMEDIATE_TIMES || (final && PRINT_TOTAL_TIME)) {
        const auto tcurr = std::chrono::high_resolution_clock::now();
        std::cout << description << "Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(tcurr - tstart).count() * 1.0e-6
                  << std::endl;
    }
}


//GastrulationBoundary::GastrulationBoundary(MatrixXT &v_, MatrixXi &f_, const VectorXi &free_) : MeshBoundary(
//        v_, f_, free_) {
//    if (nfree < np) {
//        dvdp_is_identity = false;
//    }
//}

GastrulationBoundary::GastrulationBoundary(MatrixXT &v_, MatrixXi &f_, const VectorXi &free_) : SubdivisionMeshBoundary(
        v_, f_, free_, 1) {
    dvdp_is_identity = false;
}

bool GastrulationBoundary::checkValid() {
    for (int i = 0; i < v.size(); i++) {
        if (v[i].pos.squaredNorm() > pow(outerRadius, 2.0)) {
            return false;
        }
    }

    return MeshBoundary::checkValid();
}

double GastrulationBoundary::computeEnergy() {
    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> outerFaces;
    for (int iF = 0; iF < f.size() / 2; iF++) {
        outerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget outerFluidVolumeEnergyFunc(4.0 / 3.0 * M_PI * pow(outerRadius, 3.0) - outerFluidTarget,
                                                          outerFaces);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget yolkVolumeEnergyFunc(-yolkTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    double laplacianEnergy = 0;
    laplacianEnergyFunc.getValue(this, laplacianEnergy);
    double outerFluidVolumeEnergy = 0;
    outerFluidVolumeEnergyFunc.getValue(this, outerFluidVolumeEnergy);
    double yolkVolumeEnergy = 0;
    yolkVolumeEnergyFunc.getValue(this, yolkVolumeEnergy);
    double sphericalBarrierEnergy = 0;
    sphericalBarrierEnergyFunc.getValue(this, sphericalBarrierEnergy);

    return kNeighborhood * laplacianEnergy + kOuterFluid * outerFluidVolumeEnergy + kYolk * yolkVolumeEnergy +
           sphericalBarrierEnergy;
}

void GastrulationBoundary::computeEnergyGradient(VectorXT &gradient) {
    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> outerFaces;
    for (int iF = 0; iF < f.size() / 2; iF++) {
        outerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget outerFluidVolumeEnergyFunc(4.0 / 3.0 * M_PI * pow(outerRadius, 3.0) - outerFluidTarget,
                                                          outerFaces);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget yolkVolumeEnergyFunc(-yolkTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    VectorXT laplacianGrad;
    laplacianEnergyFunc.getGradient(this, laplacianGrad);
    VectorXT outerFluidVolumeGrad;
    outerFluidVolumeEnergyFunc.getGradient(this, outerFluidVolumeGrad);
    VectorXT yolkVolumeGrad;
    yolkVolumeEnergyFunc.getGradient(this, yolkVolumeGrad);
    VectorXT sphericalBarrierGrad;
    sphericalBarrierEnergyFunc.getGradient(this, sphericalBarrierGrad);

    VectorXT dFdv = kNeighborhood * laplacianGrad + kOuterFluid * outerFluidVolumeGrad + kYolk * yolkVolumeGrad +
                    sphericalBarrierGrad;
    if (dvdp_is_identity) {
        gradient = dFdv;
    } else {
        gradient = dvdp.transpose() * dFdv;
    }
}

void GastrulationBoundary::computeEnergyHessian(MatrixXT &hessian) {
    const auto tstart = std::chrono::high_resolution_clock::now();

    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> outerFaces;
    for (int iF = 0; iF < f.size() / 2; iF++) {
        outerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget outerFluidVolumeEnergyFunc(4.0 / 3.0 * M_PI * pow(outerRadius, 3.0) - outerFluidTarget,
                                                          outerFaces);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget yolkVolumeEnergyFunc(-yolkTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    Eigen::SparseMatrix<double> laplacianHess;
    MatrixXT outerFluidVolumeHess, yolkVolumeHess, sphericalBarrierHess;
    tbb::task_group g;
    g.run([&] {
        laplacianEnergyFunc.getHessian(this, laplacianHess);
    });
    g.run([&] {
        outerFluidVolumeEnergyFunc.getHessian(this, outerFluidVolumeHess);
    });
    g.run([&] {
        yolkVolumeEnergyFunc.getHessian(this, yolkVolumeHess);
    });
    g.run([&] {
        sphericalBarrierEnergyFunc.getHessian(this, sphericalBarrierHess);
    });
    g.wait();

//    laplacianEnergyFunc.getHessian(this, laplacianHess);
//    outerFluidVolumeEnergyFunc.getHessian(this, outerFluidVolumeHess);
//    yolkVolumeEnergyFunc.getHessian(this, yolkVolumeHess);
//    sphericalBarrierEnergyFunc.getHessian(this, sphericalBarrierHess);

    MatrixXT d2Fdv2 = kNeighborhood * laplacianHess + kOuterFluid * outerFluidVolumeHess + kYolk * yolkVolumeHess +
                      sphericalBarrierHess;

    printTime(tstart, "Boundary energy partial hessian");

    if (dvdp_is_identity) {
        hessian = d2Fdv2;
    } else {
        VectorXT laplacianGrad;
        laplacianEnergyFunc.getGradient(this, laplacianGrad);
        VectorXT outerFluidVolumeGrad;
        outerFluidVolumeEnergyFunc.getGradient(this, outerFluidVolumeGrad);
        VectorXT yolkVolumeGrad;
        yolkVolumeEnergyFunc.getGradient(this, yolkVolumeGrad);
        VectorXT sphericalBarrierGrad;
        sphericalBarrierEnergyFunc.getGradient(this, sphericalBarrierGrad);

        VectorXT dFdv = kNeighborhood * laplacianGrad + kOuterFluid * outerFluidVolumeGrad + kYolk * yolkVolumeGrad +
                        sphericalBarrierGrad;

        printTime(tstart, "Boundary energy hessian before sums");

        MatrixXT sum_dFdv_d2vdp2 = MatrixXT::Zero(nfree, nfree);
        for (int i = 0; i < dFdv.rows(); i++) {
            sum_dFdv_d2vdp2 += dFdv(i) * d2vdp2[i];
        }

        printTime(tstart, "Boundary energy hessian after sums");

        hessian = dvdp.transpose() * d2Fdv2 * dvdp + sum_dFdv_d2vdp2;
    }
}

void GastrulationBoundary::computeEnergyHessianWoodbury(Eigen::SparseMatrix<double> &K, MatrixXT &UV) {
    const auto tstart = std::chrono::high_resolution_clock::now();

    NeighborhoodLaplacian laplacian;
    BoundaryEnergyPerNeighborhood laplacianEnergyFunc(&laplacian);
    std::vector<int> outerFaces;
    for (int iF = 0; iF < f.size() / 2; iF++) {
        outerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget outerFluidVolumeEnergyFunc(4.0 / 3.0 * M_PI * pow(outerRadius, 3.0) - outerFluidTarget,
                                                          outerFaces);
    std::vector<int> innerFaces;
    for (int iF = f.size() / 2; iF < f.size(); iF++) {
        innerFaces.push_back(iF);
    }
    BoundaryEnergyVolumeTarget yolkVolumeEnergyFunc(-yolkTarget, innerFaces);
    BoundaryEnergySphericalBarrier sphericalBarrierEnergyFunc(outerRadius, TV3::Zero());

    Eigen::SparseMatrix<double> laplacianHess;
    MatrixXT sphericalBarrierHess;
    Eigen::SparseMatrix<double> outerFluidVolumeK, yolkVolumeK;
    VectorXT outerFluidVolumeUV, yolkVolumeUV;
    tbb::task_group g;
    g.run([&] {
        laplacianEnergyFunc.getHessian(this, laplacianHess);
    });
    g.run([&] {
        outerFluidVolumeEnergyFunc.getHessianWoodbury(this, outerFluidVolumeK, outerFluidVolumeUV);
    });
    g.run([&] {
        yolkVolumeEnergyFunc.getHessianWoodbury(this, yolkVolumeK, yolkVolumeUV);
    });
    g.run([&] {
        sphericalBarrierEnergyFunc.getHessian(this, sphericalBarrierHess);
    });
    g.wait();

//    laplacianEnergyFunc.getHessian(this, laplacianHess);
//    outerFluidVolumeEnergyFunc.getHessian(this, outerFluidVolumeHess);
//    yolkVolumeEnergyFunc.getHessian(this, yolkVolumeHess);
//    sphericalBarrierEnergyFunc.getHessian(this, sphericalBarrierHess);

    MatrixXT d2Fdv2 = kNeighborhood * laplacianHess + kOuterFluid * outerFluidVolumeK + kYolk * yolkVolumeK +
                      sphericalBarrierHess;

    printTime(tstart, "Boundary energy partial hessian");

    if (dvdp_is_identity) {
        K = d2Fdv2.sparseView();
        UV.resize(outerFluidVolumeUV.rows(), 2);
        UV.col(0) = sqrt(kOuterFluid) * outerFluidVolumeUV;
        UV.col(1) = sqrt(kYolk) * yolkVolumeUV;
    } else {
        // TODO: d2vdp2 is zero, this is irrelevant
//        VectorXT laplacianGrad;
//        laplacianEnergyFunc.getGradient(this, laplacianGrad);
//        VectorXT outerFluidVolumeGrad;
//        outerFluidVolumeEnergyFunc.getGradient(this, outerFluidVolumeGrad);
//        VectorXT yolkVolumeGrad;
//        yolkVolumeEnergyFunc.getGradient(this, yolkVolumeGrad);
//        VectorXT sphericalBarrierGrad;
//        sphericalBarrierEnergyFunc.getGradient(this, sphericalBarrierGrad);
//
//        VectorXT dFdv = kNeighborhood * laplacianGrad + kOuterFluid * outerFluidVolumeGrad + kYolk * yolkVolumeGrad +
//                        sphericalBarrierGrad;
//
//        MatrixXT sum_dFdv_d2vdp2 = MatrixXT::Zero(nfree, nfree);
//        for (int i = 0; i < dFdv.rows(); i++) {
//            sum_dFdv_d2vdp2 += dFdv(i) * d2vdp2[i];
//        }
//        hessian = dvdp.transpose() * d2Fdv2 * dvdp + sum_dFdv_d2vdp2;

        K = dvdp.transpose() * d2Fdv2.sparseView() * dvdp;
        UV.resize(outerFluidVolumeUV.rows(), 2);
        UV.col(0) = sqrt(kOuterFluid) * outerFluidVolumeUV;
        UV.col(1) = sqrt(kYolk) * yolkVolumeUV;
        UV = dvdp.transpose() * UV;
    }
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

