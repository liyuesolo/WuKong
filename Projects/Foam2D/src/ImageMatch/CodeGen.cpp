#include "Projects/Foam2D/include/ImageMatch/CodeGen.h"

#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_power_cell_20.h"
#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_power_cell_20_gradient.h"
#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_voronoi_cell_20.h"
#include "Projects/Foam2D/codegen/codegen_imagematch/ca_imagematch_voronoi_cell_20_gradient.h"


#include <iostream>

struct CasadiFunctions {
    int (*work)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);

    const casadi_int *(*sparsity)(casadi_int);

    int (*evaluate)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
};

static CasadiFunctions getCasadiFunctions(Tessellation *tessellation, int order) {
    CasadiFunctions casadiFunctions;
    if (order == 0) {
        switch (tessellation->getTessellationType()) {
            case VORONOI:
                casadiFunctions.work = &ca_imagematch_voronoi_cell_20_work;
                casadiFunctions.sparsity = &ca_imagematch_voronoi_cell_20_sparsity_out;
                casadiFunctions.evaluate = &ca_imagematch_voronoi_cell_20;
                break;
            case POWER:
                casadiFunctions.work = &ca_imagematch_power_cell_20_work;
                casadiFunctions.sparsity = &ca_imagematch_power_cell_20_sparsity_out;
                casadiFunctions.evaluate = &ca_imagematch_power_cell_20;
                break;
            default:
                break;
        }
    } else if (order == 1) {
        switch (tessellation->getTessellationType()) {
            case VORONOI:
                casadiFunctions.work = &ca_imagematch_voronoi_cell_20_gradient_work;
                casadiFunctions.sparsity = &ca_imagematch_voronoi_cell_20_gradient_sparsity_out;
                casadiFunctions.evaluate = &ca_imagematch_voronoi_cell_20_gradient;
                break;
            case POWER:
                casadiFunctions.work = &ca_imagematch_power_cell_20_gradient_work;
                casadiFunctions.sparsity = &ca_imagematch_power_cell_20_gradient_sparsity_out;
                casadiFunctions.evaluate = &ca_imagematch_power_cell_20_gradient;
                break;
            default:
                break;
        }
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

//    std::cout << "cell vcalue " << out << " " << Obj[0] << std::endl;

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
    for (int rr = 0; rr < map.rows() * dims && rr < nnz; rr++) {
        int ir = map(rr / dims) * dims + (rr % dims);
        if (ir < out.rows()) {
            out(ir) += dOdc[rr];

//            double eps = 1e-6;
//            VectorXT dc = VectorXT::Zero(c.rows());
//            dc(rr) += eps;
//            double f = 0, fp = 0;
//            add_value_cell(tessellation, p, n, c, b, pix, f);
//            add_value_cell(tessellation, p, n, c + dc, b, pix, fp);
//            std::cout << "im code gen " << nnz << " " << p(0) << " " << p(1) << " " << rr << " wow " << fp << " " << f
//                      << " " << (fp - f) / eps
//                      << " " << dOdc[rr]
//                      << std::endl;
        }
    }
}
