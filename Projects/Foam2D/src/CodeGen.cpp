#include "../include/CodeGen.h"

#include "../codegen/ca_O_voronoi_cell.h"
#include "../codegen/ca_dOdc_voronoi_cell.h"
#include "../codegen/ca_d2Odc2_voronoi_cell.h"
#include "../codegen/ca_O_sectional_cell.h"
#include "../codegen/ca_dOdc_sectional_cell.h"
#include "../codegen/ca_d2Odc2_sectional_cell.h"

#include <iostream>

void add_O_voronoi_cell(const VectorXT &c, const VectorXT &p, double &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_O_voronoi_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    casadi_real Obj[1];
    res[0] = Obj;

    ca_O_voronoi_cell(arg, res, iw, w, 0);

    out += Obj[0];
}

void add_dOdc_voronoi_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, VectorXT &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dOdc_voronoi_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    const casadi_int *sp_i = ca_dOdc_voronoi_cell_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dOdc[nnz];
    res[0] = dOdc;
    ca_dOdc_voronoi_cell(arg, res, iw, w, 0); /* Actual function evaluation */

    for (int rr = 0; rr < map.rows() * 2; rr++) {
        int ir = map(rr / 2) * 2 + (rr % 2);
        if (ir < out.rows()) {
            out(ir) += dOdc[rr];
        }
    }
}

void
add_d2Odc2_voronoi_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, Eigen::SparseMatrix<double> &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_d2Odc2_voronoi_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    const casadi_int *sp_i = ca_d2Odc2_voronoi_cell_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real d2Odc2[nnz];
    res[0] = d2Odc2;
    ca_d2Odc2_voronoi_cell(arg, res, iw, w, 0); /* Actual function evaluation */

    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            if (rr < map.rows() * 2 && cc < map.rows() * 2) {
                int ir = map(rr / 2) * 2 + (rr % 2);
                int ic = map(cc / 2) * 2 + (cc % 2);
                if (ir < out.rows() && ic < out.cols()) {
                    out.coeffRef(ir, ic) += d2Odc2[nzidx];
                }
            }
            nzidx++;
        }
    }
}

void add_O_sectional_cell(const VectorXT &c, const VectorXT &p, double &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_O_sectional_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    casadi_real Obj[1];
    res[0] = Obj;

    ca_O_sectional_cell(arg, res, iw, w, 0);

    out += Obj[0];
}

void add_dOdc_sectional_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, VectorXT &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dOdc_sectional_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    const casadi_int *sp_i = ca_dOdc_sectional_cell_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dOdc[nnz];
    res[0] = dOdc;
    ca_dOdc_sectional_cell(arg, res, iw, w, 0); /* Actual function evaluation */

    for (int rr = 0; rr < map.rows() * 3; rr++) {
        int ir = map(rr / 3) * 3 + (rr % 3);
        if (ir < out.rows()) {
            out(ir) += dOdc[rr];
        }
    }
}

void
add_d2Odc2_sectional_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, Eigen::SparseMatrix<double> &out) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_d2Odc2_sectional_cell_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = p.data();

    const casadi_int *sp_i = ca_d2Odc2_sectional_cell_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real d2Odc2[nnz];
    res[0] = d2Odc2;
    ca_d2Odc2_sectional_cell(arg, res, iw, w, 0); /* Actual function evaluation */

    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; ++cc) {                    /* loop over columns */
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            if (rr < map.rows() * 3 && cc < map.rows() * 3) {
                int ir = map(rr / 3) * 3 + (rr % 3);
                int ic = map(cc / 3) * 3 + (cc % 3);
                if (ir < out.rows() && ic < out.cols()) {
                    out.coeffRef(ir, ic) += d2Odc2[nzidx];
                }
            }
            nzidx++;
        }
    }
}
