#include "Projects/Foam3D/include/Boundary/BoundaryEnergyVolumeTarget.h"
#include "../../include/Energy/PerTriangleVolume.h"

void BoundaryEnergyVolumeTarget::getValue(Boundary *boundary, double &value) const {
    double volume = 0;

    PerTriangleVolume volFunc;
    for (int iF: faces) {
        BoundaryFace face = boundary->f[iF];
        IV3 verts = face.vertices;

        TriangleValue vol;
        vol.v0 = boundary->v[verts[0]].pos;
        vol.v1 = boundary->v[verts[1]].pos;
        vol.v2 = boundary->v[verts[2]].pos;
        volFunc.getValue(vol);
        volume += vol.value;
    }

    std::cout << "Sphere volume " << volume << std::endl;
    value = 0.5 * pow(volume - target, 2.0);
}

void BoundaryEnergyVolumeTarget::getGradient(Boundary *boundary, VectorXT &gradient) const {
    double volume = 0;
    VectorXT volGradient = VectorXT::Zero(boundary->v.size() * 3);

    PerTriangleVolume volFunc;
    for (int iF: faces) {
        BoundaryFace face = boundary->f[iF];
        IV3 verts = face.vertices;

        TriangleValue vol;
        vol.v0 = boundary->v[verts[0]].pos;
        vol.v1 = boundary->v[verts[1]].pos;
        vol.v2 = boundary->v[verts[2]].pos;
        volFunc.getValue(vol);
        volFunc.getGradient(vol);
        volume += vol.value;
        for (int i = 0; i < 3; i++) {
            for (int ii = 0; ii < 3; ii++) {
                volGradient(verts[i] * 3 + ii) += vol.gradient(i * 3 + ii);
            }
        }
    }
    gradient = (volume - target) * volGradient;
}

void BoundaryEnergyVolumeTarget::getHessian(Boundary *boundary, MatrixXT &hessian) const {
    double volume = 0;
    VectorXT volGradient = VectorXT::Zero(boundary->v.size() * 3);
    Eigen::SparseMatrix<double> volHessian(boundary->v.size() * 3, boundary->v.size() * 3);

    PerTriangleVolume volFunc;
    std::vector<Eigen::Triplet<double>> tripletsVolHessian;
    for (int iF: faces) {
        BoundaryFace face = boundary->f[iF];
        IV3 verts = face.vertices;

        TriangleValue vol;
        vol.v0 = boundary->v[verts[0]].pos;
        vol.v1 = boundary->v[verts[1]].pos;
        vol.v2 = boundary->v[verts[2]].pos;
        volFunc.getValue(vol);
        volFunc.getGradient(vol);
        volFunc.getHessian(vol);
        volume += vol.value;
        for (int i = 0; i < 3; i++) {
            for (int ii = 0; ii < 3; ii++) {
                volGradient(verts[i] * 3 + ii) += vol.gradient(i * 3 + ii);
                for (int j = 0; j < 3; j++) {
                    for (int jj = 0; jj < 3; jj++) {
                        tripletsVolHessian.emplace_back(verts[i] * 3 + ii, verts[j] * 3 + jj,
                                                        vol.hessian(i * 3 + ii, j * 3 + jj));
                    }
                }
            }
        }
    }
    volHessian.setFromTriplets(tripletsVolHessian.begin(), tripletsVolHessian.end());

    hessian = (volume - target) * volHessian + volGradient * volGradient.transpose();
}
