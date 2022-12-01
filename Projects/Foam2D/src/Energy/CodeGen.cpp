#include "Projects/Foam2D/include/Energy/CodeGen.h"

#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_10.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_10_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_10_hessian.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_10.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_10_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_10_hessian.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_20.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_20_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_voronoi_cell_20_hessian.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_20.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_20_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy/ca_energy_power_cell_20_hessian.h"

#include <iostream>

struct CasadiFunctions {
    int (*work)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);

    const casadi_int *(*sparsity)(casadi_int);

    int (*evaluate)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
};

static CasadiFunctions getCasadiFunctions(Tessellation *tessellation, int order, int num_neighbors) {
    CasadiFunctions casadiFunctions;
    if (order == 0) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_10_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_10;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_10_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_10_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_10;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_20_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_20;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_20_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_20_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_20;
                    break;
                default:
                    break;
            }
        }
    } else if (order == 1) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_10_gradient_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_10_gradient_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_10_gradient;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_10_gradient_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_10_gradient_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_10_gradient;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_20_gradient_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_20_gradient_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_20_gradient;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_20_gradient_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_20_gradient_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_20_gradient;
                    break;
                default:
                    break;
            }
        }
    } else if (order == 2) {
        if (num_neighbors < 9) {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_10_hessian_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_10_hessian_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_10_hessian;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_10_hessian_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_10_hessian_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_10_hessian;
                    break;
                default:
                    break;
            }
        } else {
            switch (tessellation->getTessellationType()) {
                case VORONOI:
                    casadiFunctions.work = &ca_energy_voronoi_cell_20_hessian_work;
                    casadiFunctions.sparsity = &ca_energy_voronoi_cell_20_hessian_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_voronoi_cell_20_hessian;
                    break;
                case POWER:
                    casadiFunctions.work = &ca_energy_power_cell_20_hessian_work;
                    casadiFunctions.sparsity = &ca_energy_power_cell_20_hessian_sparsity_out;
                    casadiFunctions.evaluate = &ca_energy_power_cell_20_hessian;
                    break;
                default:
                    break;
            }
        }
    }

    return casadiFunctions;
}

void add_E_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
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
add_dEdc_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
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
    int n_cells = out.rows() / dims;

    for (int rr = 0; rr < map.rows() * dims; rr++) {
        int ir = map(rr / dims) * dims + (rr % dims);
        if (map(rr / dims) < n_cells) {
            out(ir) += dOdc[rr];
        }
    }
}

void
add_d2Edc2_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
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
    int n_cells = out.rows() / dims;

    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < map.rows() * dims; ++cc) {                    /* loop over columns */
        int ic = map(cc / dims) * dims + (cc % dims);
        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            rr = row[el];
            if (rr < map.rows() * dims) {
                int ir = map(rr / dims) * dims + (rr % dims);
                if (map(rr / dims) < n_cells && map(cc / dims) < n_cells) {
                    out(ir, ic) += d2Odc2[nzidx];
                }
            }
            nzidx++;
        }
    }
}
