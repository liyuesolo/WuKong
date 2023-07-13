#include <igl/fast_find_self_intersections.h>
#include "../../include/Boundary/MeshSpringBoundary.h"
#include "../../include/Energy/PerTriangleVolume.h"

MeshSpringBoundary::MeshSpringBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_) : Boundary(
        Eigen::Map<VectorXT>(MatrixXT(v_.transpose()).data(), v_.size()), free_) {
    initialize(v_.rows(), f_);
}

bool MeshSpringBoundary::checkValid() {
    MatrixXT V(v.size(), 3);
    for (int i = 0; i < v.size(); i++) {
        V.row(i) = v[i].pos;
    }
    MatrixXi F(f.size(), 3);
    for (int i = 0; i < f.size(); i++) {
        F.row(i) = f[i].vertices;
    }
    MatrixXi I;

    return !igl::fast_find_self_intersections(V, F, I);
}

void MeshSpringBoundary::computeVertices() {
    for (int i = 0; i < v.size(); i++) {
        v[i].pos = p.segment<3>(i * 3);

        for (int j = 0; j < 3; j++) {
            addGradientEntry(i, j, i * 3 + j, 1.0);
        }
    }
}

double MeshSpringBoundary::computeEnergy() {
    double energy = 0;
    double volume = 0;

    PerTriangleVolume volFunc;
    for (int iF = 0; iF < f.size(); iF++) {
        BoundaryFace face = f[iF];

        IV3 verts = face.vertices;
        for (int i = 0; i < 3; i++) {
            int v0 = verts[i];
            int v1 = verts[(i + 1) % 3];

            double d2 = (v[v1].pos - v[v0].pos).squaredNorm();
            energy += 0.5 * kEdge * d2;
        }

        if (iF < f.size() / 2) continue;

        TriangleValue vol;
        vol.v0 = v[verts[0]].pos;
        vol.v1 = v[verts[1]].pos;
        vol.v2 = v[verts[2]].pos;
        volFunc.getValue(vol);
        volume += vol.value;
    }

    std::cout << "Sphere volume " << volume << std::endl;
    energy += kVol * pow(volume - volTarget, 2);

    return energy;
}

VectorXT MeshSpringBoundary::computeEnergyGradient() {
    Eigen::SparseVector<double> gradient(nfree);

    double volume = 0;
    Eigen::SparseVector<double> volGradient(nfree);

    PerTriangleVolume volFunc;
    for (int iF = 0; iF < f.size(); iF++) {
        BoundaryFace face = f[iF];

        IV3 verts = face.vertices;
        for (int i = 0; i < 3; i++) {
            int v0 = verts[i];
            int v1 = verts[(i + 1) % 3];

            TV3 d = v[v1].pos - v[v0].pos;
            addEnergyGradientEntry(gradient, v0 * 3 + 0, -kEdge * d(0));
            addEnergyGradientEntry(gradient, v0 * 3 + 1, -kEdge * d(1));
            addEnergyGradientEntry(gradient, v0 * 3 + 2, -kEdge * d(2));
            addEnergyGradientEntry(gradient, v1 * 3 + 0, kEdge * d(0));
            addEnergyGradientEntry(gradient, v1 * 3 + 1, kEdge * d(1));
            addEnergyGradientEntry(gradient, v1 * 3 + 2, kEdge * d(2));
        }

        if (iF < f.size() / 2) continue;

        TriangleValue vol;
        vol.v0 = v[verts[0]].pos;
        vol.v1 = v[verts[1]].pos;
        vol.v2 = v[verts[2]].pos;
        volFunc.getValue(vol);
        volFunc.getGradient(vol);
        volume += vol.value;
        for (int i = 0; i < 3; i++) {
            for (int ii = 0; ii < 3; ii++) {
                addEnergyGradientEntry(volGradient, verts[i] * 3 + ii, vol.gradient(i * 3 + ii));
            }
        }
    }
    gradient += kVol * 2 * (volume - volTarget) * volGradient;

    return gradient;
}

MatrixXT MeshSpringBoundary::computeEnergyHessian() {
    Eigen::SparseMatrix<double> hessian(nfree, nfree);

    double volume = 0;
    Eigen::SparseVector<double> volGradient(nfree);
    Eigen::SparseMatrix<double> volHessian(nfree, nfree);

    PerTriangleVolume volFunc;
    for (int iF = 0; iF < f.size(); iF++) {
        BoundaryFace face = f[iF];

        IV3 verts = face.vertices;
        for (int i = 0; i < 3; i++) {
            int v0 = verts[i];
            int v1 = verts[(i + 1) % 3];

            int i00 = v0 * 3 + 0;
            int i01 = v0 * 3 + 1;
            int i02 = v0 * 3 + 2;
            int i10 = v1 * 3 + 0;
            int i11 = v1 * 3 + 1;
            int i12 = v1 * 3 + 2;

            addEnergyHessianEntry(hessian, i00, i00, kEdge);
            addEnergyHessianEntry(hessian, i00, i10, -kEdge);
            addEnergyHessianEntry(hessian, i10, i00, -kEdge);
            addEnergyHessianEntry(hessian, i10, i10, kEdge);
            addEnergyHessianEntry(hessian, i01, i01, kEdge);
            addEnergyHessianEntry(hessian, i01, i11, -kEdge);
            addEnergyHessianEntry(hessian, i11, i01, -kEdge);
            addEnergyHessianEntry(hessian, i11, i11, kEdge);
            addEnergyHessianEntry(hessian, i02, i02, kEdge);
            addEnergyHessianEntry(hessian, i02, i12, -kEdge);
            addEnergyHessianEntry(hessian, i12, i02, -kEdge);
            addEnergyHessianEntry(hessian, i12, i12, kEdge);
        }

        if (iF < f.size() / 2) continue;

        TriangleValue vol;
        vol.v0 = v[verts[0]].pos;
        vol.v1 = v[verts[1]].pos;
        vol.v2 = v[verts[2]].pos;
        volFunc.getValue(vol);
        volFunc.getGradient(vol);
        volFunc.getHessian(vol);
        volume += vol.value;
        for (int i = 0; i < 3; i++) {
            for (int ii = 0; ii < 3; ii++) {
                addEnergyGradientEntry(volGradient, verts[i] * 3 + ii, vol.gradient(i * 3 + ii));
                for (int j = 0; j < 3; j++) {
                    for (int jj = 0; jj < 3; jj++) {
                        addEnergyHessianEntry(volHessian, verts[i] * 3 + ii, verts[j] * 3 + jj,
                                              vol.hessian(i * 3 + ii, j * 3 + jj));
                    }
                }
            }
        }
    }
    hessian += kVol * (2 * (volume - volTarget) * volHessian + 2 * volGradient * volGradient.transpose());

    return hessian;
}

