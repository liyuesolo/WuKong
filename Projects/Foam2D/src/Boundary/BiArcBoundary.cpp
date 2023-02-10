#include "../../include/Boundary/BiArcBoundary.h"
#include "../../include/Boundary/BiArc.h"

void BiArcBoundary::computeVertices() {
    int ncp = p.rows() / 3;
    v.resize(ncp * 4);
    radii.resize(ncp * 2);

    int n_vtx = v.rows() / 2;
    next.resize(n_vtx);
    if (n_vtx > 0) {
        next << Eigen::VectorXi::LinSpaced(n_vtx - 1, 1, n_vtx - 1), 0;
    }
    r_map = Eigen::VectorXi::LinSpaced(n_vtx, 0, n_vtx - 1);

    for (int i = 0; i < ncp; i++) {
        int j = (i + 1) % ncp;

        v.segment<2>(i * 4) = p.segment<2>(i * 3);

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        VectorXT outputs;
        BiArc::getBiArcValues(inputs, outputs);

        v.segment<2>(i * 4 + 2) = outputs.segment<2>(0);
        radii.segment<2>(i * 2) = outputs.segment<2>(2);
    }
}

void BiArcBoundary::computeGradient() {
    dvdp = MatrixXT::Zero(v.rows(), nfree);
    drdp = MatrixXT::Zero(radii.rows(), nfree);

    int ncp = p.rows() / 3;
    for (int i = 0; i < ncp; i++) {
        int j = (i + 1) % ncp;

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        MatrixXT outputs;
        BiArc::getBiArcGradient(inputs, outputs);

        for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                setGradientEntry(i * 4 + 2 + ii, i * 3 + jj, outputs(ii, jj));
                setGradientEntry(i * 4 + 2 + ii, j * 3 + jj, outputs(ii, jj + 3));
                setRGradientEntry(i * 2 + ii, i * 3 + jj, outputs(ii + 2, jj));
                setRGradientEntry(i * 2 + ii, j * 3 + jj, outputs(ii + 2, jj + 3));
            }
        }

        setGradientEntry(i * 4 + 0, i * 3 + 0, 1);
        setGradientEntry(i * 4 + 1, i * 3 + 1, 1);
    }
}

void BiArcBoundary::computeHessian() {
    d2vdp2.resize(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        d2vdp2[i] = MatrixXT::Zero(nfree, nfree);
    }
    d2rdp2.resize(radii.rows());
    for (int i = 0; i < radii.rows(); i++) {
        d2rdp2[i] = MatrixXT::Zero(nfree, nfree);
    }

    int ncp = p.rows() / 3;
    for (int i = 0; i < ncp; i++) {
        int j = (i + 1) % ncp;

        VectorXT inputs(6);
        inputs << p.segment<3>(i * 3), p.segment<3>(j * 3);
        std::vector<MatrixXT> outputs;
        BiArc::getBiArcHessian(inputs, outputs);

        MatrixXT hess;
        int idx;
        for (int ii = 0; ii < 3; ii++) {
            for (int jj = 0; jj < 3; jj++) {
                hess = outputs[0];
                idx = i * 4 + 2 + 0;
                setHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                setHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                setHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                setHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[1];
                idx = i * 4 + 2 + 1;
                setHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                setHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                setHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                setHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[2];
                idx = i * 2 + 0;
                setRHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                setRHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                setRHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                setRHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));

                hess = outputs[3];
                idx = i * 2 + 1;
                setRHessianEntry(idx, i * 3 + ii, i * 3 + jj, hess(ii, jj));
                setRHessianEntry(idx, i * 3 + ii, j * 3 + jj, hess(ii, jj + 3));
                setRHessianEntry(idx, j * 3 + ii, i * 3 + jj, hess(ii + 3, jj));
                setRHessianEntry(idx, j * 3 + ii, j * 3 + jj, hess(ii + 3, jj + 3));
            }
        }
    }
}

