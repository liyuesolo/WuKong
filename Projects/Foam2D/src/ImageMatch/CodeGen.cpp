#include "Projects/Foam2D/include/ImageMatch/CodeGen.h"

#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_power_cell_20.h"
#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_power_cell_20_gradient.h"

#include <iostream>

struct CasadiFunctions {
    int (*work)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);

    const casadi_int *(*sparsity)(casadi_int);

    int (*evaluate)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
};

static CasadiFunctions getCasadiFunctions(Tessellation *tessellation, double order) {
    CasadiFunctions casadiFunctions;
    if (order == 0) {
        casadiFunctions.work = &ca_imagematch_power_cell_20_work;
        casadiFunctions.sparsity = &ca_imagematch_power_cell_20_sparsity_out;
        casadiFunctions.evaluate = &ca_imagematch_power_cell_20;
    } else if (order == 1) {
        casadiFunctions.work = &ca_imagematch_power_cell_20_gradient_work;
        casadiFunctions.sparsity = &ca_imagematch_power_cell_20_gradient_sparsity_out;
        casadiFunctions.evaluate = &ca_imagematch_power_cell_20_gradient;
    }

    return casadiFunctions;
}

void
add_value_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
               const VectorXT &pix,
               double &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 0);

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
    arg[4] = pix.data();

    casadi_real Obj[1];
    res[0] = Obj;
    casadiFunctions.evaluate(arg, res, iw, w, 0);

    out += Obj[0];
}

void
add_gradient_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c,
                  const VectorXT &b, const VectorXT &pix,
                  const VectorXi &map,
                  VectorXT &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 1);

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
    arg[4] = pix.data();

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
