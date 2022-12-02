#include "Projects/Foam2D/include/ImageMatch/CodeGen.h"

#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_power_cell_areatarget_20.h"
#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_power_cell_areatarget_20_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_power_cell_areatarget_20_hessian.h"
#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_voronoi_cell_areatarget_20.h"
#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_voronoi_cell_areatarget_20_gradient.h"
#include "Projects/Foam2D/codegen/codegen_energy_areatarget/ca_energy_voronoi_cell_areatarget_20_hessian.h"

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
                casadiFunctions.work = &ca_energy_voronoi_cell_areatarget_20_work;
                casadiFunctions.sparsity = &ca_energy_voronoi_cell_areatarget_20_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_voronoi_cell_areatarget_20;
                break;
            case POWER:
                casadiFunctions.work = &ca_energy_power_cell_areatarget_20_work;
                casadiFunctions.sparsity = &ca_energy_power_cell_areatarget_20_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_power_cell_areatarget_20;
                break;
            default:
                break;
        }
    } else if (order == 1) {
        switch (tessellation->getTessellationType()) {
            case VORONOI:
                casadiFunctions.work = &ca_energy_voronoi_cell_areatarget_20_gradient_work;
                casadiFunctions.sparsity = &ca_energy_voronoi_cell_areatarget_20_gradient_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_voronoi_cell_areatarget_20_gradient;
                break;
            case POWER:
                casadiFunctions.work = &ca_energy_power_cell_areatarget_20_gradient_work;
                casadiFunctions.sparsity = &ca_energy_power_cell_areatarget_20_gradient_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_power_cell_areatarget_20_gradient;
                break;
            default:
                break;
        }
    } else if (order == 2) {
        switch (tessellation->getTessellationType()) {
            case VORONOI:
                casadiFunctions.work = &ca_energy_voronoi_cell_areatarget_20_hessian_work;
                casadiFunctions.sparsity = &ca_energy_voronoi_cell_areatarget_20_hessian_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_voronoi_cell_areatarget_20_hessian;
                break;
            case POWER:
                casadiFunctions.work = &ca_energy_power_cell_areatarget_20_hessian_work;
                casadiFunctions.sparsity = &ca_energy_power_cell_areatarget_20_hessian_sparsity_out;
                casadiFunctions.evaluate = &ca_energy_power_cell_areatarget_20_hessian;
                break;
            default:
                break;
        }
    }

    return casadiFunctions;
}

void
add_E_cell_AT(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
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

    casadi_real Obj[1];
    res[0] = Obj;
    casadiFunctions.evaluate(arg, res, iw, w, 0);

    out += Obj[0];
}

void
add_dEdc_cell_AT(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
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
    int n_cells = out.rows() / (dims + 1);
    int idx_at = n_cells * dims + map(0);

    if (nrow != 1) {
        std::cout << "Failed to read casadi output. More than one row when only one expected." << std::endl;
    }
    for (int cc = 0; cc < ncol; cc++) {
        if (colind[cc] < colind[cc + 1]) {
            bool c_in_range = true;

            int ic;
            if (cc == 0) {
                ic = idx_at;
            } else {
                if ((cc - 1) / dims >= map.rows() || map((cc - 1) / dims) >= n_cells) {
                    c_in_range = false;
                } else {
                    ic = map((cc - 1) / dims) * dims + ((cc - 1) % dims);
                }
            }

            if (c_in_range) {
                out(ic) += dOdc[cc];
            }
        }
    }
//
//    for (int rr = 1; rr < map.rows() * dims + 1; rr++) {
//        if (map((rr - 1) / dims) >= n_cells) {
//            continue;
//        }
//        int ir = map((rr - 1) / dims) * dims + ((rr - 1) % dims);
//        if (fabs(dOdc[rr]) > 1e-3) {
//            std::cout << rr << " " << map(0) << " " << ir << " " << dOdc[rr] << " " << 2 * c(1 + ((rr - 1) % dims))
//                      << " " << 2 * c(rr) << " " << nnz
//                      << std::endl;
//            std::cout << row[0] << row[1] << row[2] << std::endl;
//            std::cout << colind[0] << colind[1] << colind[2] << std::endl;
//        }
//        out(ir) += dOdc[rr];
//
////        double eps = 1e-6;
////        VectorXT dc = VectorXT::Zero(c.rows());
////        dc(rr) += eps;
////        double f = 0, fp = 0;
////        add_E_cell_AT(tessellation, p, n, c, b, f);
////        add_E_cell_AT(tessellation, p, n, c + dc, b, fp);
////        std::cout << "energy code gen " << p(3) << " " << rr << " " << (fp - f) / eps << " " << dOdc[rr] << std::endl;
//    }
}

void
add_d2Edc2_cell_AT(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c,
                   const VectorXT &b,
                   const VectorXi &map,
                   MatrixXT &out) {
    CasadiFunctions casadiFunctions = getCasadiFunctions(tessellation, 2);

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
    int n_cells = out.rows() / (dims + 1);
    int idx_at = n_cells * dims + map(0);

    casadi_int rr, cc, el;
    int nzidx = 0;
    for (cc = 0; cc < ncol; cc++) {                    /* loop over columns */
        bool c_in_range = true;

        int ic = -1;
        if (cc == 0) {
            ic = idx_at;
        } else {
            if ((cc - 1) / dims >= map.rows() || map((cc - 1) / dims) >= n_cells) {
                c_in_range = false;
            } else {
                ic = map((cc - 1) / dims) * dims + ((cc - 1) % dims);
            }
        }

        for (el = colind[cc]; el < colind[cc + 1]; ++el) { /* loop over the nonzeros entries of the column */
            bool r_in_range = true;

            rr = row[el];

            int ir = -1;
            if (rr == 0) {
                ir = idx_at;
            } else {
                if ((rr - 1) / dims >= map.rows() || map((rr - 1) / dims) >= n_cells) {
                    r_in_range = false;
                } else {
                    ir = map((rr - 1) / dims) * dims + ((rr - 1) % dims);
                }
            }

            if (c_in_range && r_in_range) {
//                std::cout << "[" << ir << ", " << ic << "] " << d2Odc2[nzidx] << std::endl;
                out(ir, ic) += d2Odc2[nzidx];
            }

            nzidx++;
        }
    }
}
