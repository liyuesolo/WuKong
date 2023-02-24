#include "../../include/Energy/CellFunctionDeformationVolumeIntegral.h"
#include <iostream>

void
CellFunctionDeformationVolumeIntegral::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                const VectorXi &btype,
                                                double &value,
                                                const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double xc = 0;
    xc_function.addValue(site, nodes, next, btype, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, btype, yc, cellInfo);

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        // @formatter:off
        double t4 = x0 * x0;
        double t8 = x1 * x1;
        double t11 = y0 * y0;
        double t15 = xc * xc;
        double t17 = y1 * y1;
        double t20 = yc * yc;
        value += (t4 + (x1 - 0.4e1 * xc) * x0 + t8 - 0.4e1 * x1 * xc + t11 + (y1 - 0.4e1 * yc) * y0 + 0.6e1 * t15 + t17 - 0.4e1 * y1 * yc + 0.6e1 * t20) * (x0 * y1 - x1 * y0) / 0.12e2;
        // @formatter:on
    }
}

void
CellFunctionDeformationVolumeIntegral::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                   const VectorXi &btype,
                                                   VectorXT &gradient_c,
                                                   VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;
    VectorXT gradient_centroid = VectorXT::Zero(2);

    double xc = 0;
    xc_function.addValue(site, nodes, next, btype, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, btype, yc, cellInfo);

    VectorXT temp;
    VectorXT xc_grad = VectorXT::Zero(nodes.rows());
    xc_function.addGradient(site, nodes, next, btype, temp, xc_grad, cellInfo);
    VectorXT yc_grad = VectorXT::Zero(nodes.rows());
    yc_function.addGradient(site, nodes, next, btype, temp, yc_grad, cellInfo);

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        // @formatter:off
        double t1 = x0 * x0;
        double t2 = 0.4e1 * xc;
        double t5 = x1 * x1;
        double t8 = y0 * y0;
        double t9 = 0.4e1 * yc;
        double t12 = xc * xc;
        double t14 = y1 * y1;
        double t17 = yc * yc;
        double t19 = t1 + x0 * (x1 - t2) + t5 - 0.4e1 * x1 * xc + t8 + y0 * (y1 - t9) + 0.6e1 * t12 + t14 - 0.4e1 * y1 * yc + 0.6e1 * t17;
        double t23 = x0 * y1 - x1 * y0;
        gradient_x[x0i] += t19 * y1 / 0.12e2 + (0.2e1 * x0 + x1 - t2) * t23 / 0.12e2;
        gradient_x[y0i] += -t19 * x1 / 0.12e2 + (0.2e1 * y0 + y1 - t9) * t23 / 0.12e2;
        gradient_x[x1i] += -t19 * y0 / 0.12e2 + (x0 + 0.2e1 * x1 - t2) * t23 / 0.12e2;
        gradient_x[y1i] += t19 * x0 / 0.12e2 + (y0 + 0.2e1 * y1 - t9) * t23 / 0.12e2;
        gradient_centroid[0] += (-0.4e1 * x0 - 0.4e1 * x1 + 0.12e2 * xc) * t23 / 0.12e2;
        gradient_centroid[1] += (-0.4e1 * y0 - 0.4e1 * y1 + 0.12e2 * yc) * t23 / 0.12e2;
        // @formatter:on
    }

    MatrixXT d_centroid_d_x(2, nodes.rows());
    d_centroid_d_x.row(0) = xc_grad;
    d_centroid_d_x.row(1) = yc_grad;
    gradient_x += gradient_centroid.transpose() * d_centroid_d_x;
}

void
CellFunctionDeformationVolumeIntegral::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                  const VectorXi &btype,
                                                  MatrixXT &hessian,
                                                  const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;
    VectorXT gradient_centroid = VectorXT::Zero(2);

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    MatrixXT hess_Cx = MatrixXT::Zero(2, nodes.rows());
    MatrixXT hess_CC = MatrixXT::Zero(2, 2);

    double xc = 0;
    xc_function.addValue(site, nodes, next, btype, xc, cellInfo);
    double yc = 0;
    yc_function.addValue(site, nodes, next, btype, yc, cellInfo);

    VectorXT temp;
    VectorXT xc_grad = VectorXT::Zero(nodes.rows());
    xc_function.addGradient(site, nodes, next, btype, temp, xc_grad, cellInfo);
    VectorXT yc_grad = VectorXT::Zero(nodes.rows());
    yc_function.addGradient(site, nodes, next, btype, temp, yc_grad, cellInfo);

    MatrixXT xc_hess = MatrixXT::Zero(hessian.rows(), hessian.cols());
    xc_function.addHessian(site, nodes, next, btype, xc_hess, cellInfo);
    MatrixXT yc_hess = MatrixXT::Zero(hessian.rows(), hessian.cols());
    yc_function.addHessian(site, nodes, next, btype, yc_hess, cellInfo);

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        double unknown[6][6];

        // @formatter:off
        double t2 = 0.4e1 * xc;
        double t3 = 0.2e1 * x0 + x1 - t2;
        double t5 = x0 * y1;
        double t6 = x1 * y0;
        double t9 = 0.4e1 * yc;
        double t10 = 0.2e1 * y0 + y1 - t9;
        double t13 = t10 * y1 - t3 * x1;
        double t15 = x0 + 0.2e1 * x1 - t2;
        double t18 = t15 * y1 - t3 * y0 + t5 - t6;
        double t19 = x0 * x0;
        double t20 = t19 / 0.12e2;
        double t23 = x0 * (x1 - t2) / 0.12e2;
        double t24 = x1 * x1;
        double t25 = t24 / 0.12e2;
        double t27 = x1 * xc / 0.3e1;
        double t28 = y0 * y0;
        double t29 = t28 / 0.12e2;
        double t32 = y0 * (y1 - t9) / 0.12e2;
        double t33 = xc * xc;
        double t34 = t33 / 0.2e1;
        double t35 = y1 * y1;
        double t36 = t35 / 0.12e2;
        double t38 = y1 * yc / 0.3e1;
        double t39 = yc * yc;
        double t40 = t39 / 0.2e1;
        double t42 = y0 + 0.2e1 * y1 - t9;
        double t47 = t20 + t23 + t25 - t27 + t29 + t32 + t34 + t36 - t38 + t40 + t42 * y1 / 0.12e2 + t3 * x0 / 0.12e2;
        double t51 = -0.4e1 * x0 - 0.4e1 * x1 + 0.12e2 * xc;
        double t54 = t5 / 0.3e1;
        double t55 = t6 / 0.3e1;
        double t56 = t51 * y1 / 0.12e2 - t54 + t55;
        double t60 = -0.4e1 * y0 - 0.4e1 * y1 + 0.12e2 * yc;
        double t62 = t60 * y1 / 0.12e2;
        double t69 = -t20 - t23 - t25 + t27 - t29 - t32 - t34 - t36 + t38 - t40 - t15 * x1 / 0.12e2 - t10 * y0 / 0.12e2;
        double t72 = t10 * x0 - t42 * x1 + t5 - t6;
        double t74 = t51 * x1 / 0.12e2;
        double t77 = -t60 * x1 / 0.12e2 - t54 + t55;
        double t82 = t15 * x0 - t42 * y0;
        double t85 = -t51 * y0 / 0.12e2 - t54 + t55;
        double t87 = t60 * y0 / 0.12e2;
        double t91 = t51 * x0 / 0.12e2;
        double t94 = t60 * x0 / 0.12e2 - t54 + t55;
        double t95 = t5 - t6;
        unknown[0][0] = t3 * y1 / 0.6e1 + t5 / 0.6e1 - t6 / 0.6e1;
        unknown[0][1] = t13 / 0.12e2;
        unknown[0][2] = t18 / 0.12e2;
        unknown[0][3] = t47;
        unknown[0][4] = t56;
        unknown[0][5] = t62;
        unknown[1][0] = t13 / 0.12e2;
        unknown[1][1] = -t10 * x1 / 0.6e1 + t5 / 0.6e1 - t6 / 0.6e1;
        unknown[1][2] = t69;
        unknown[1][3] = t72 / 0.12e2;
        unknown[1][4] = -t74;
        unknown[1][5] = t77;
        unknown[2][0] = t18 / 0.12e2;
        unknown[2][1] = t69;
        unknown[2][2] = -t15 * y0 / 0.6e1 + t5 / 0.6e1 - t6 / 0.6e1;
        unknown[2][3] = t82 / 0.12e2;
        unknown[2][4] = t85;
        unknown[2][5] = -t87;
        unknown[3][0] = t47;
        unknown[3][1] = t72 / 0.12e2;
        unknown[3][2] = t82 / 0.12e2;
        unknown[3][3] = t42 * x0 / 0.6e1 + t5 / 0.6e1 - t6 / 0.6e1;
        unknown[3][4] = t91;
        unknown[3][5] = t94;
        unknown[4][0] = t56;
        unknown[4][1] = -t74;
        unknown[4][2] = t85;
        unknown[4][3] = t91;
        unknown[4][4] = t95;
        unknown[4][5] = 0.0e0;
        unknown[5][0] = t62;
        unknown[5][1] = t77;
        unknown[5][2] = -t87;
        unknown[5][3] = t94;
        unknown[5][4] = 0.0e0;
        unknown[5][5] = t95;
        // @formatter:on

        MatrixXT hessian_maple = Eigen::Map<Eigen::MatrixXd>(&unknown[0][0], 6, 6);
        hess_CC += hessian_maple.bottomRightCorner(2, 2);
        MatrixXT hess_Cx_undistributed = hessian_maple.bottomLeftCorner(2, 4);
        MatrixXT hess_xx_undistributed = hessian_maple.topLeftCorner(4, 4);

        VectorXi idx(4);
        idx << x0i, y0i, x1i, y1i;
        for (int j = 0; j < 4; j++) {
            hess_Cx(0, idx(j)) += hess_Cx_undistributed(0, j);
            hess_Cx(1, idx(j)) += hess_Cx_undistributed(1, j);
            for (int k = 0; k < 4; k++) {
                hess_xx(idx(j), idx(k)) += hess_xx_undistributed(j, k);
            }
        }
    }

    MatrixXT d_centroid_d_x(2, nodes.rows());
    d_centroid_d_x.row(0) = xc_grad;
    d_centroid_d_x.row(1) = yc_grad;
    hess_xx += d_centroid_d_x.transpose() * hess_CC * d_centroid_d_x + d_centroid_d_x.transpose() * hess_Cx +
               hess_Cx.transpose() * d_centroid_d_x +
               gradient_centroid(0) * xc_hess.bottomRightCorner(nodes.rows(), nodes.rows()) +
               gradient_centroid(1) * yc_hess.bottomRightCorner(nodes.rows(), nodes.rows());


//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, next, btype, temp, grad, cellInfo);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, next, temp, gradp, cellInfo);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "deformation hess_xx(" << j << "," << i << ") " << (gradp[j] - grad[j]) / eps << " "
//                      << hess_xx(j, i) << std::endl;
//        }
//    }

//    VectorXT gradc = VectorXT::Zero(site.rows());
//    VectorXT gradx = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, next, btype, gradc, gradx, cellInfo);
//    VectorXT grad(gradc.rows() + gradx.rows());
//    grad << gradc, gradx;
//    double eps = 1e-6;
//    VectorXT y(site.rows() + nodes.rows());
//    y << site, nodes;
//    for (int i = 0; i < hessian.rows(); i++) {
//        VectorXT yp = y;
//        yp(i) += eps;
//        VectorXT gradcp = VectorXT::Zero(site.rows());
//        VectorXT gradxp = VectorXT::Zero(nodes.rows());
//        addGradient(yp.segment(0, site.rows()), yp.segment(site.rows(), nodes.rows()), next, gradcp, gradxp, cellInfo);
//        VectorXT gradp(gradc.rows() + gradx.rows());
//        gradp << gradcp, gradxp;
//        for (int j = 0; j < hessian.rows(); j++) {
//            std::cout << "deformation hessian(" << j << "," << i << ") " << (gradp[j] - grad[j]) / eps << " "
//                      << hessian(j, i) << std::endl;
//        }
//    }
}
