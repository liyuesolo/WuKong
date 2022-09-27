#include "../include/CodeGen.h"

#include "../codegen/ca_x_voronoi.h"
#include "../codegen/ca_dxdc_voronoi.h"
#include "../codegen/ca_x_sectional.h"
#include "../codegen/ca_dxdc_sectional.h"
#include "../codegen/ca_A.h"
#include "../codegen/ca_dAdx.h"
#include "../codegen/ca_L.h"
#include "../codegen/ca_dLdx.h"

#include <iostream>

VectorXT evaluate_x_voronoi(const VectorXT &c, const VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_x_voronoi_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    int n_faces = tri.rows() / 3;
    casadi_real x[n_faces * 2];
    res[0] = x;

    ca_x_voronoi(arg, res, iw, w, 0);

    return Eigen::Map<VectorXT>(x, n_faces * 2);
}

Eigen::SparseMatrix<double> evaluate_dxdc_voronoi(const VectorXT &c, const VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dxdc_voronoi_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    const casadi_int *sp_i = ca_dxdc_voronoi_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dxdc[nnz];
    res[0] = dxdc;
    ca_dxdc_voronoi(arg, res, iw, w, 0); /* Actual function evaluation */

    std::vector<Eigen::Triplet<double>> triplets(nnz);
    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            triplets[nzidx] = Eigen::Triplet<double>(rr, cc, dxdc[nzidx]);
            nzidx++;
        }
    }

    Eigen::SparseMatrix<double> DXDC(nrow, ncol);
    DXDC.setFromTriplets(triplets.begin(), triplets.end());

    return DXDC;
}

VectorXT evaluate_x_sectional(const VectorXT &c, const VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_x_sectional_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    int n_faces = tri.rows() / 3;
    casadi_real x[n_faces * 2];
    res[0] = x;

    ca_x_sectional(arg, res, iw, w, 0);

    return Eigen::Map<VectorXT>(x, n_faces * 2);
}

Eigen::SparseMatrix<double> evaluate_dxdc_sectional(const VectorXT &c, const VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dxdc_sectional_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    const casadi_int *sp_i = ca_dxdc_sectional_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dxdc[nnz];
    res[0] = dxdc;
    ca_dxdc_sectional(arg, res, iw, w, 0); /* Actual function evaluation */

    std::vector<Eigen::Triplet<double>> triplets(nnz);
    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            triplets[nzidx] = Eigen::Triplet<double>(rr, cc, dxdc[nzidx]);
            nzidx++;
        }
    }

    Eigen::SparseMatrix<double> DXDC(nrow, ncol);
    DXDC.setFromTriplets(triplets.begin(), triplets.end());

    return DXDC;
}

VectorXT evaluate_A(const VectorXT &c, const VectorXT &x, const VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_A_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = x.data();
    VectorXT e_d = e.cast<double>();
    arg[2] = e_d.data();

    int n_sites = c.rows() / 2;
    casadi_real A[n_sites];
    res[0] = A;

    ca_A(arg, res, iw, w, 0);

    return Eigen::Map<VectorXT>(A, n_sites);
}

Eigen::SparseMatrix<double> evaluate_dAdx(const VectorXT &c, const VectorXT &x, const VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dAdx_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = x.data();
    VectorXT e_d = e.cast<double>();
    arg[2] = e_d.data();

    const casadi_int *sp_i = ca_dAdx_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dAdx[nnz];
    res[0] = dAdx;
    ca_dAdx(arg, res, iw, w, 0); /* Actual function evaluation */

    std::vector<Eigen::Triplet<double>> triplets(nnz);
    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            triplets[nzidx] = Eigen::Triplet<double>(rr, cc, dAdx[nzidx]);
            nzidx++;
        }
    }

    Eigen::SparseMatrix<double> DADX(nrow, ncol);
    DADX.setFromTriplets(triplets.begin(), triplets.end());

    return DADX;
}

double evaluate_L(const VectorXT &x, const VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_L_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = x.data();
    VectorXT e_d = e.cast<double>();
    arg[1] = e_d.data();

    casadi_real L[1];
    res[0] = L;

    ca_L(arg, res, iw, w, 0);

    return L[0];
}

Eigen::SparseMatrix<double> evaluate_dLdx(const VectorXT &x, const VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dLdx_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = x.data();
    VectorXT e_d = e.cast<double>();
    arg[1] = e_d.data();

    const casadi_int *sp_i = ca_dLdx_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dLdx[nnz];
    res[0] = dLdx;
    ca_dLdx(arg, res, iw, w, 0); /* Actual function evaluation */

    std::vector<Eigen::Triplet<double>> triplets(nnz);
    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            triplets[nzidx] = Eigen::Triplet<double>(rr, cc, dLdx[nzidx]);
            nzidx++;
        }
    }

    Eigen::SparseMatrix<double> DLDX(nrow, ncol);
    DLDX.setFromTriplets(triplets.begin(), triplets.end());

    return DLDX;
}
