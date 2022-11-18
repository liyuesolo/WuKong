#include "../include/CodeGen.h"

#include "../codegen/ca_O_voronoi_cell_10.h"
#include "../codegen/ca_dOdc_voronoi_cell_10.h"
#include "../codegen/ca_d2Odc2_voronoi_cell_10.h"
#include "../codegen/ca_O_power_cell_10.h"
#include "../codegen/ca_dOdc_power_cell_10.h"
#include "../codegen/ca_d2Odc2_power_cell_10.h"
#include "../codegen/ca_O_voronoi_cell_20.h"
#include "../codegen/ca_dOdc_voronoi_cell_20.h"
#include "../codegen/ca_d2Odc2_voronoi_cell_20.h"
#include "../codegen/ca_O_power_cell_20.h"
#include "../codegen/ca_dOdc_power_cell_20.h"
#include "../codegen/ca_d2Odc2_power_cell_20.h"

#include <iostream>

struct CasadiFunctions {
    int (*work)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);

    const casadi_int *(*sparsity)(casadi_int);

    int (*evaluate)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
};

CasadiFunctions getCasadiFunctions(Tessellation *tessellation, double order, int num_neighbors) {
    CasadiFunctions casadiFunctions;
    if (order == 0) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_O_voronoi_cell_10_work;
                    casadiFunctions.sparsity = &ca_O_voronoi_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_O_voronoi_cell_10;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_O_power_cell_10_work;
                    casadiFunctions.sparsity = &ca_O_power_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_O_power_cell_10;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_O_voronoi_cell_20_work;
                    casadiFunctions.sparsity = &ca_O_voronoi_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_O_voronoi_cell_20;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_O_power_cell_20_work;
                    casadiFunctions.sparsity = &ca_O_power_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_O_power_cell_20;
                    break;
                default:
                    break;
            }
        }
    } else if (order == 1) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_dOdc_voronoi_cell_10_work;
                    casadiFunctions.sparsity = &ca_dOdc_voronoi_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_dOdc_voronoi_cell_10;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_dOdc_power_cell_10_work;
                    casadiFunctions.sparsity = &ca_dOdc_power_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_dOdc_power_cell_10;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_dOdc_voronoi_cell_20_work;
                    casadiFunctions.sparsity = &ca_dOdc_voronoi_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_dOdc_voronoi_cell_20;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_dOdc_power_cell_20_work;
                    casadiFunctions.sparsity = &ca_dOdc_power_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_dOdc_power_cell_20;
                    break;
                default:
                    break;
            }
        }
    } else if (order == 2) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_d2Odc2_voronoi_cell_10_work;
                    casadiFunctions.sparsity = &ca_d2Odc2_voronoi_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_d2Odc2_voronoi_cell_10;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_d2Odc2_power_cell_10_work;
                    casadiFunctions.sparsity = &ca_d2Odc2_power_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_d2Odc2_power_cell_10;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_d2Odc2_voronoi_cell_20_work;
                    casadiFunctions.sparsity = &ca_d2Odc2_voronoi_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_d2Odc2_voronoi_cell_20;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_d2Odc2_power_cell_20_work;
                    casadiFunctions.sparsity = &ca_d2Odc2_power_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_d2Odc2_power_cell_20;
                    break;
                default:
                    break;
            }
        }
    }

    return casadiFunctions;
}

void add_O_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
                double &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 0, p(4));

    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    casadiFunctions.work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = p.data();
    arg[1] = n.data();
    arg[2] = c.data();
    arg[3] = b.data();

    casadi_real Obj[1];
    res[0] = Obj;
    casadiFunctions.evaluate(arg, res, iw, w, 0);

    out += Obj[0];
}

void
add_dOdc_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
              const VectorXi &map,
              VectorXT &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 1, p(4));

    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    casadiFunctions.work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = p.data();
    arg[1] = n.data();
    arg[2] = c.data();
    arg[3] = b.data();

    const casadi_int *sp_i = casadiFunctions.sparsity(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dOdc[nnz];
    res[0] = dOdc;
    casadiFunctions.evaluate(arg, res, iw, w, 0);

    int dims = 2 + tessellation->getNumVertexParams();
    for (int rr = 0; rr < map.rows() * dims; rr++) {
        int ir = map(rr / dims) * dims + (rr % dims);
        if (ir < out.rows()) {
            out(ir) += dOdc[rr];
        }
    }
}

void
add_d2Odc2_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
                const VectorXi &map,
                MatrixXT &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 2, p(4));

    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    casadiFunctions.work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = p.data();
    arg[1] = n.data();
    arg[2] = c.data();
    arg[3] = b.data();

    const casadi_int *sp_i = casadiFunctions.sparsity(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real d2Odc2[nnz];
    res[0] = d2Odc2;
    casadiFunctions.evaluate(arg, res, iw, w, 0);

    int dims = 2 + tessellation->getNumVertexParams();
    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < map.rows() * dims; ++cc) {                    /* loop over columns */
        int ic = map(cc / dims) * dims + (cc % dims);
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            if (rr < map.rows() * dims) {
                int ir = map(rr / dims) * dims + (rr % dims);
                if (ir < out.rows() && ic < out.cols()) {
                    out(ir, ic) += d2Odc2[nzidx];
                }
            }
            nzidx++;
        }
    }
}
