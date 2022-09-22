#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "../include/Foam2D.h"
#include "../codegen/ca_x.h"
#include "../codegen/ca_dxdc.h"
#include "../codegen/ca_A.h"
#include "../codegen/ca_dAdx.h"
#include "../src/optLib/GradientDescentMinimizer.h"
#include <random>

#define NCELLS 40
#define NAREA 300
#define OBJWEIGHT 2

static Foam2D::VectorXT evaluate_x(const Foam2D::VectorXT &c, const Foam2D::VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_x_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    Foam2D::VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    int n_faces = tri.rows() / 3;
    casadi_real x[n_faces * 2];
    res[0] = x;

    ca_x(arg, res, iw, w, 0);

    return Eigen::Map<Foam2D::VectorXT>(x, n_faces * 2);
}

static Eigen::SparseMatrix<double> evaluate_dxdc(const Foam2D::VectorXT &c, const Foam2D::VectorXi &tri) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dxdc_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    Foam2D::VectorXT tri_d = tri.cast<double>();
    arg[1] = tri_d.data();

    const casadi_int *sp_i = ca_dxdc_sparsity_out(0);
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol + 1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    casadi_real dxdc[nnz];
    res[0] = dxdc;
    ca_dxdc(arg, res, iw, w, 0); /* Actual function evaluation */

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

static Foam2D::VectorXT evaluate_A(const Foam2D::VectorXT &c, const Foam2D::VectorXT &x, const Foam2D::VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_A_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = x.data();
    Foam2D::VectorXT e_d = e.cast<double>();
    arg[2] = e_d.data();

    int n_sites = c.rows() / 2;
    casadi_real A[n_sites];
    res[0] = A;

    ca_A(arg, res, iw, w, 0);

    return Eigen::Map<Foam2D::VectorXT>(A, n_sites);
}

static Eigen::SparseMatrix<double>
evaluate_dAdx(const Foam2D::VectorXT &c, const Foam2D::VectorXT &x, const Foam2D::VectorXi &e) {
    casadi_int sz_arg, sz_res, sz_iw, sz_w;
    ca_dAdx_work(&sz_arg, &sz_res, &sz_iw, &sz_w);

    const casadi_real *arg[sz_arg];
    casadi_real *res[sz_res];
    casadi_int iw[sz_iw];
    casadi_real w[sz_w];

    arg[0] = c.data();
    arg[1] = x.data();
    Foam2D::VectorXT e_d = e.cast<double>();
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

void Foam2D::testCasadiCode() {
    Eigen::SparseMatrix<double> dxdc = evaluate_dxdc(vertices, tri_face_indices);

    VectorXT x0 = evaluate_x(vertices, tri_face_indices);

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT xh;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            xh = evaluate_x(vertices + vert_offsets * eps, tri_face_indices);

            VectorXT xfd = x0 + dxdc * vert_offsets * eps;

            std::cout << "dxdc error " << (xfd - xh).norm() << " " << error / (xfd - xh).norm() << std::endl;

            error = (xfd - xh).norm();
            eps *= 0.5;
        }
    }

    // Test A
    int n_faces = tri_face_indices.rows() / 3;

    MatrixXi F;
    F.resize(n_faces, 3);
    for (int i = 0; i < n_faces; i++) {
        F.row(i) = tri_face_indices.segment<3>(i * 3);
    }

    VectorXi area_triangles(NAREA * 3);

    int edge = 0;
    for (int i = 0; i < n_faces; i++) {
        for (int j = i + 1; j < n_faces; j++) {
            int num_shared_vertices = 0;
            bool shared[3] = {false, false, false};
            for (int ii = 0; ii < 3; ii++) {
                if (F(i, ii) == F(j, 0) || F(i, ii) == F(j, 1) || F(i, ii) == F(j, 2)) {
                    num_shared_vertices++;
                    shared[ii] = true;
                }
            }
            if (num_shared_vertices == 2) {
                if (shared[0] && shared[1]) {
                    if (F(i, 0) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 0);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (F(i, 1) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 1);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else if (shared[1] && shared[2]) {
                    if (F(i, 1) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 1);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (F(i, 2) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 2);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else if (shared[2] && shared[0]) {
                    if (F(i, 2) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 2);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (F(i, 0) < NCELLS) {
                        area_triangles[edge * 3 + 0] = F(i, 0);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else {
                    assert(false);
                }
            }
        }
    }
    std::cout << "NUM AREA TRIS " << edge << std::endl;

    for (int i = edge; i < NAREA; i++) {
        area_triangles[i * 3 + 0] = 0; // TODO: this is a hack, degenerate triangle with area 0...
        area_triangles[i * 3 + 1] = 0;
        area_triangles[i * 3 + 2] = 0;
    }

    VectorXT A = evaluate_A(vertices.segment<NCELLS * 2>(0), x0, area_triangles);
    Eigen::SparseMatrix<double> dAdx = evaluate_dAdx(vertices.segment<NCELLS * 2>(0), x0, area_triangles);

    for (int i = 0; i < A.rows(); i++) {
        std::cout << "area " << A[i] << std::endl;
    }

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT node_offsets = VectorXT::Zero(x0.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT Ah;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            Ah = evaluate_A(vertices.segment<NCELLS * 2>(0), x0 + eps * node_offsets, area_triangles);

            VectorXT Afd = A + dAdx * node_offsets * eps;

            std::cout << "dAdx error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;

            error = (Afd - Ah).norm();
            eps *= 0.5;
        }
    }

    Eigen::SparseMatrix<double> dAdc = dAdx * dxdc;
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);
        VectorXT vert_offsets = VectorXT::Zero(vertices.rows()).unaryExpr([&](float dummy) { return dis(gen); });

        VectorXT xh;
        VectorXT Ah;

        double eps = 1e-4;
        double error = 1;
        for (int i = 0; i < 10; i++) {
            xh = evaluate_x(vertices + vert_offsets * eps, tri_face_indices);
            Ah = evaluate_A((vertices + vert_offsets * eps).segment<NCELLS * 2>(0), xh, area_triangles);

            VectorXT Afd = A + dAdc * vert_offsets * eps;

            std::cout << "dAdc error " << (Afd - Ah).norm() << " " << error / (Afd - Ah).norm() << std::endl;

            error = (Afd - Ah).norm();
            eps *= 0.5;
        }
    }
}

static void triangulateWrapper(Foam2D::VectorXT verticesIn, Foam2D::VectorXT &verticesOut, Foam2D::VectorXi &triOut) {
    int n_vtx = verticesIn.rows() / 2;

    Foam2D::MatrixXT P;
    P.resize(n_vtx, 2);
    for (int i = 0; i < n_vtx; i++) {
        P.row(i) = verticesIn.segment<2>(i * 2);
    }

    Foam2D::MatrixXT V;
    Foam2D::MatrixXi F;
    igl::triangle::triangulate(P,
                               Foam2D::MatrixXi(),
                               Foam2D::MatrixXT(),
                               "cQ", // Enclose convex hull with segments
                               V, F);

    verticesOut.resize(V.rows() * 2);
    for (int i = 0; i < V.rows(); i++)
        verticesOut.segment<2>(i * 2) = V.row(i);
    triOut.resize(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++)
        triOut.segment<3>(i * 3) = F.row(i);
}


void Foam2D::generateRandomVoronoi() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(40 * 2);
    for (int i = 0; i < 40; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / 40), sin(i * 2 * M_PI / 40));
    }

    VectorXT voronoi_points = VectorXT::Zero((40 + NCELLS) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    voronoi_points.segment<40 * 2>(NCELLS * 2) = boundary_points;

    triangulateWrapper(voronoi_points, vertices, tri_face_indices);
}

void Foam2D::retriangulate() {
    triangulateWrapper(vertices, vertices, tri_face_indices);
}

static Foam2D::VectorXi getAreaTriangles(Foam2D::VectorXi tri) {
    Foam2D::VectorXi area_triangles(NAREA * 3);

    int n_faces = tri.rows() / 3;
    int edge = 0;
    for (int i = 0; i < n_faces; i++) {
        for (int j = i + 1; j < n_faces; j++) {
            int num_shared_vertices = 0;
            bool shared[3] = {false, false, false};
            for (int ii = 0; ii < 3; ii++) {
                if (tri(i * 3 + ii) == tri(j * 3 + 0) || tri(i * 3 + ii) == tri(j * 3 + 1) ||
                    tri(i * 3 + ii) == tri(j * 3 + 2)) {
                    num_shared_vertices++;
                    shared[ii] = true;
                }
            }
            if (num_shared_vertices == 2) {
                if (shared[0] && shared[1]) {
                    if (tri(i * 3 + 0) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 0);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (tri(i * 3 + 1) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 1);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else if (shared[1] && shared[2]) {
                    if (tri(i * 3 + 1) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 1);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (tri(i * 3 + 2) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 2);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else if (shared[2] && shared[0]) {
                    if (tri(i * 3 + 2) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 2);
                        area_triangles[edge * 3 + 1] = j;
                        area_triangles[edge * 3 + 2] = i;
                        edge++;
                    }
                    if (tri(i * 3 + 0) < NCELLS) {
                        area_triangles[edge * 3 + 0] = tri(i * 3 + 0);
                        area_triangles[edge * 3 + 1] = i;
                        area_triangles[edge * 3 + 2] = j;
                        edge++;
                    }
                } else {
                    assert(false);
                }
            }
        }
    }

    for (int i = edge; i < NAREA; i++) {
        area_triangles[i * 3 + 0] = 0; // TODO: this is a hack, degenerate triangle with area 0...
        area_triangles[i * 3 + 1] = 0;
        area_triangles[i * 3 + 2] = 0;
    }

    return area_triangles;
}


class AreaObjective : public ObjectiveFunction {

public:
    double area_target = 0.1;

public:
    virtual double evaluate(const VectorXd &c) const {
        Foam2D::VectorXi tri;
        Foam2D::VectorXT x;
        Foam2D::VectorXi e;
        Foam2D::VectorXT A;

        Foam2D::VectorXT c_temp;
        triangulateWrapper(c, c_temp, tri);

        x = evaluate_x(c, tri);

        e = getAreaTriangles(tri);

        A = evaluate_A(c.segment<NCELLS * 2>(0), x, e);

        double O = 0;
        for (int i = 0; i < A.rows(); i++) {
            O += (A(i) - area_target) * (A(i) - area_target);
        }

        return OBJWEIGHT * O;
    }

    virtual void addGradientTo(const VectorXd &c, VectorXd &grad) const {
        grad += get_DODc(c);
    }

    VectorXd get_DODc(const VectorXd &c) const {
        Foam2D::VectorXi tri;
        Foam2D::VectorXT x;
        Eigen::SparseMatrix<double> dxdc;
        Foam2D::VectorXi e;
        Foam2D::VectorXT A;
        Eigen::SparseMatrix<double> dAdx;
        Foam2D::MatrixXT dOdA;

        Foam2D::VectorXT c_temp;
        triangulateWrapper(c, c_temp, tri);

        x = evaluate_x(c, tri);
        dxdc = evaluate_dxdc(c, tri);

        e = getAreaTriangles(tri);

        A = evaluate_A(c.segment<NCELLS * 2>(0), x, e);
        dAdx = evaluate_dAdx(c.segment<NCELLS * 2>(0), x, e);

        Foam2D::VectorXT targets;
        targets.resize(A.rows());
        targets.setOnes();
        targets = targets * area_target;
        dOdA = 2 * (A - targets).transpose();

        Foam2D::VectorXT dOdc = (dOdA * dAdx * dxdc).transpose();
        dOdc.segment<40 * 2>(NCELLS * 2) *= 0; // Fix boundary sites...

        return OBJWEIGHT * dOdc.transpose();
    }
};

void Foam2D::optimize(double area_target) {
    AreaObjective objective;
    objective.area_target = area_target;

    GradientDescentLineSearch minimizer(1, 1e-6, 15);
    minimizer.minimize(&objective, vertices);

    retriangulate();
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < NCELLS) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Foam2D::moveVertex(int idx, const Foam2D::TV &pos) {
    vertices.segment<2>(idx * 2) = pos;
    retriangulate();
}

static Foam2D::TV getCircumcentre(const Foam2D::TV &v1, const Foam2D::TV &v2, const Foam2D::TV &v3) {
    double x1 = v1(0), x2 = v2(0), x3 = v3(0);
    double y1 = v1(1), y2 = v2(1), y3 = v3(1);

    double m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1));

    double xc = 0.5 * (x1 + x3) - m * (y3 - y1);
    double yc = 0.5 * (y1 + y3) + m * (x3 - x1);

    return {xc, yc};
}

static Foam2D::TV3 getRandomColorFromVertex(const Foam2D::TV &v) {
    double r = (double) ((int) (0.5 * (v(0) + 1) * RAND_MAX) % 255) / 255;
    double g = (double) ((int) (0.5 * (v(1) + 1) * RAND_MAX) % 255) / 255;
    double b = (double) ((int) (0.25 * (v(0) + v(1) + 2) * RAND_MAX) % 255) / 255;
    return {r, g, b};
}

void Foam2D::generateVoronoiDiagramForVisualization(MatrixXT &C, MatrixXT &X, MatrixXi &E) {
    long n_vtx = vertices.rows() / 2, n_faces = tri_face_indices.rows() / 3;

    C.resize(n_vtx, 3);
    C.setZero();
    for (int i = 0; i < n_vtx; i++) {
        C.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    MatrixXi F;
    F.resize(n_faces, 3);
    for (int i = 0; i < n_faces; i++) {
        F.row(i) = tri_face_indices.segment<3>(i * 3);
    }

    int num_voronoi_nodes = n_faces;
    X.resize(num_voronoi_nodes, 3);
    X.setZero();
    for (int i = 0; i < n_faces; i++) {
        TV vc = getCircumcentre(C.block<1, 2>(F(i, 0), 0), C.block<1, 2>(F(i, 1), 0), C.block<1, 2>(F(i, 2), 0));
        X.row(i).segment<2>(0) = vc;
    }

    int num_voronoi_edges = 2 * n_faces - n_vtx + 1;
    E.resize(num_voronoi_edges, 2);

    int edge = 0;
    for (int i = 0; i < n_faces; i++) {
        for (int j = i + 1; j < n_faces; j++) {
            int num_shared_vertices = 0;
            for (int ii = 0; ii < 3; ii++) {
                if (F(i, ii) == F(j, 0) || F(i, ii) == F(j, 1) || F(i, ii) == F(j, 2)) {
                    num_shared_vertices++;
                }
            }
            if (num_shared_vertices == 2) {
                E.row(edge) = IV(i, j);
                edge++;
            }
        }
    }
}
