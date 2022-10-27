#include "../include/Foam2D.h"
#include "../include/CodeGen.h"
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/Sectional.h"
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../include/Constants.h"
#include <random>
#include "../include/TrajectoryOpt/IpoptSolver.h"

Foam2D::Foam2D() {
    tessellations.push_back(new Voronoi());
    tessellations.push_back(new Sectional());
    tessellations.push_back(new Power());
    minimizers.push_back(new GradientDescentLineSearch(1, 1e-6, 15));
    minimizers.push_back(new NewtonFunctionMinimizer(1, 1e-6, 15));

    nlp.energy = &energyObjective;
    nlp.dynamics = &dynamicObjective;
}

void Foam2D::resetVertexParams() {
    params = tessellations[tesselation]->getDefaultVertexParams(vertices);
}

void Foam2D::initRandomSitesInCircle(int n_free_in, int n_fixed_in) {
    n_free = n_free_in;
    n_fixed = n_fixed_in;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(n_fixed * 2);
    for (int i = 0; i < n_fixed; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / n_fixed), sin(i * 2 * M_PI / n_fixed));
    }

    vertices = VectorXT::Zero((n_free + n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(n_free * 2, n_fixed * 2) = boundary_points;

    resetVertexParams();
}

void Foam2D::initBasicTestCase() {
    n_free = 4;
    n_fixed = 8;

    vertices.resize((n_free + n_fixed) * 2);

    vertices << -0.5, 0, 0.5, 0, 0, -0.5, 0, 0.5, -1, -1, 1, -1, 1, 1, -1, 1, -1, 0, 1, 0, 0, -1, 0, 1;

    resetVertexParams();
}

void Foam2D::dynamicsInit(double dt, double m) {
    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tesselation]->getNumVertexParams()));
    dynamicObjective.init(c_free, dt, m, &energyObjective);
}

void Foam2D::dynamicsNewStep() {
    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tesselation]->getNumVertexParams()));
    dynamicObjective.newStep(c_free);
}

void Foam2D::optimize(bool dynamic) {
    energyObjective.tessellation = tessellations[tesselation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;

    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * (2 + tessellations[tesselation]->getNumVertexParams()),
                                        n_fixed * (2 + tessellations[tesselation]->getNumVertexParams()));
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tesselation]->getNumVertexParams()));

    if (dynamic) {
        minimizers[opttype]->minimize(&dynamicObjective, c_free);
    } else {
        minimizers[opttype]->minimize(&energyObjective, c_free);
    }

    c.segment(0, n_free * (2 + tessellations[tesselation]->getNumVertexParams())) = c_free;

    tessellations[tesselation]->separateVerticesParams(c, vertices, params);
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < n_free) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Foam2D::moveVertex(int idx, const TV &pos) {
    vertices.segment<2>(idx * 2) = pos;
}

void Foam2D::trajectoryOptSetInit() {
    int dims = energyObjective.tessellation->getNumVertexParams() + 2;
    nlp.c0 = energyObjective.tessellation->combineVerticesParams(vertices, params).segment(0, n_free * dims);
    nlp.v0 = VectorXd::Zero(n_free * dims);
}

void Foam2D::trajectoryOptGenerateExampleSol(int N) {
    nlp.N = N;
    nlp.agent = energyObjective.drag_idx;
    nlp.target_pos = energyObjective.drag_target_pos;

    nlp.x_guess.resize(N * (nlp.c0.rows() + 2));
    VectorXd u_guess = VectorXT::Zero(2 * N);

    // x format is [c1 ... cN u1 ... uN]
    nlp.x_guess << nlp.c0.replicate(N, 1), u_guess;
    nlp.x_sol = nlp.x_guess;

    int dims = energyObjective.tessellation->getNumVertexParams() + 2;
    TV pos0 = nlp.c0.segment<2>(dims * nlp.agent);
    TV posN = nlp.target_pos;
    for (int i = 0; i < N; i++) {
        nlp.x_sol.segment<2>(i * n_free * dims + dims * nlp.agent) = pos0 + (posN - pos0) * (i + 1.0) / N;
    }
}

bool Foam2D::trajectoryOptOptimizeIPOPT(int N) {
    /** NLP INITIALIZATION **/
    nlp.N = N;
    nlp.agent = energyObjective.drag_idx;
    nlp.target_pos = energyObjective.drag_target_pos;

    nlp.x_guess.resize(N * (nlp.c0.rows() + 2));
    VectorXd u_guess = VectorXT::Zero(2 * N);

    // x format is [c1 ... cN u1 ... uN]
    nlp.x_guess << nlp.c0.replicate(N, 1), u_guess;
    nlp.x_sol = nlp.x_guess;

    /** IPOPT SOLVE **/
    int b = 0;
    std::cout << b++ << std::endl;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    std::cout << b++ << std::endl;
    app->RethrowNonIpoptException(true);

    std::cout << b++ << std::endl;
    app->Options()->SetNumericValue("tol", 1e-5);
//    app->Options()->SetStringValue("mu_strategy", "monotone");
    // app->Options()->SetStringValue("mu_strategy", "adaptive");

//    app->Options()->SetStringValue("output_file", data_folder + "/ipopt.out");
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    // app->Options()->SetIntegerValue("limited_memory_max_history", 50);
//    app->Options()->SetIntegerValue("accept_after_max_steps", 20);
    //        app->Options()->SetNumericValue("mu_max", 0.0001);
    //        app->Options()->SetNumericValue("constr_viol_tol", T(1e-7));
    //        app->Options()->SetNumericValue("acceptable_constr_viol_tol", T(1e-7));
    //        bound_relax_factor
    //        app->Options()->SetStringValue("derivative_test", "first-order");
    // The following overwrites the default name (ipopt.opt) of the
    // options file
    // app->Options()->SetStringValue("option_file_name", "hs071.opt");

    // Initialize the IpoptApplication and process the options
    std::cout << b++ << std::endl;
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded) {
        std::cout << std::endl
                  << std::endl
                  << "*** Error during initialization!" << std::endl;
        return (int) status;
    }
    std::cout << b++ << std::endl;
    // Ask Ipopt to solve the problem
    std::cout << "Solving problem using IPOPT" << std::endl;
    std::cout << b++ << std::endl;
    // objective.bound[0] = 1e-5;
    // objective.bound[1] = 12.0 * simulation.cells.unit;
    std::cout << b++ << std::endl;
    Ipopt::SmartPtr<IpoptSolver> mynlp = new IpoptSolver(nlp);
    std::cout << b++ << std::endl;
    status = app->OptimizeTNLP(mynlp);
    std::cout << b++ << std::endl;
    if (status == Ipopt::Solve_Succeeded) {
        std::cout << std::endl
                  << std::endl
                  << "*** The problem solved!" << std::endl;
    } else {
        std::cout << std::endl
                  << std::endl
                  << "*** The problem FAILED!" << std::endl;
    }
    std::cout << b++ << status << std::endl;
    return (int) status;
}

void Foam2D::trajectoryOptGetFrame(int frame) {
    int dims = energyObjective.tessellation->getNumVertexParams() + 2;
    VectorXT c_frame = frame == 0 ? nlp.c0 : nlp.x_sol.segment((frame - 1) * n_free * dims, n_free * dims);

    VectorXT verts_free;
    VectorXT params_free;
    energyObjective.tessellation->separateVerticesParams(c_frame, verts_free, params_free);
    vertices.segment(0, n_free * 2) = verts_free;
    params.segment(0, n_free * (dims - 2)) = params_free;
}

static TV3 getColor(double area, double target) {
    double r, g, b;
    double q = (area / target);
    if (q >= 1) q = sqrt(q - 1); else q = -sqrt(1 - q);
    r = 1 - q;
    g = 1;
    b = 1 + q;

    r = fmin(fmax(r, 0), 1);
    g = fmin(fmax(g, 0), 1);
    b = fmin(fmax(b, 0), 1);
    return {r, g, b};
}

static VectorXT getCellAreas(VectorXT x, std::vector<std::vector<int>> cells, int n_cells) {
    VectorXT A(n_cells);
    for (int i = 0; i < n_cells; i++) {
        if (cells[i].size() < 3) {
            A(i) = 0;
        } else {
            vector<int> cell = cells[i];
            double x1 = x(cell[0] * 2 + 0);
            double y1 = x(cell[0] * 2 + 1);

            double area = 0;
            for (int j = 1; j < cell.size() - 1; j++) {
                double x2 = x(cell[j] * 2 + 0);
                double y2 = x(cell[j] * 2 + 1);
                double x3 = x(cell[j + 1] * 2 + 0);
                double y3 = x(cell[j + 1] * 2 + 1);
                area += 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
            }
            A(i) = area;
        }
    }
    return A;
}

void Foam2D::getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    X = S;

    int num_edges = n_faces * 3;
    E.resize(num_edges, 2);
    E.setZero();

    for (int i = 0; i < n_faces; i++) {
        int v0 = tri(i * 3 + 0);
        int v1 = tri(i * 3 + 1);
        int v2 = tri(i * 3 + 2);

        E.row(i * 3 + 0) = IV(v0, v1);
        E.row(i * 3 + 1) = IV(v1, v2);
        E.row(i * 3 + 2) = IV(v2, v0);
    }

    // Mesh points and faces
    V.resize(n_vtx, 3);
    V << S;

    F.resize(n_faces, 3);
    F.setZero();
    C.resize(n_faces, 3);
    C.setZero();
    for (int i = 0; i < n_faces; i++) {
        F.row(i) = tri.segment<3>(i * 3);
        C.row(i) = TV3(0.5, 0.5, 0.5);
    }
}

void Foam2D::getTessellationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &V, MatrixXi &F, MatrixXT &C,
                                       int selected) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3;

    // Overlay points and edges
    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    VectorXT x = tessellations[tesselation]->getNodes(vertices, params, tri);
    X.resize(n_faces, 3);
    X.setZero();
    for (int i = 0; i < n_faces; i++) {
        X.row(i).segment<2>(0) = x.segment<2>(i * 2);
    }

    int num_voronoi_edges = 2 * n_faces - n_vtx + 1;
    E.resize(num_voronoi_edges * 2, 2);
    E.setZero();

    std::vector<std::vector<int>> cells = tessellations[tesselation]->getCells(vertices, tri, x);
    int edge = 0;
    int n_cells = n_free;
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &cell = cells[i];
        size_t degree = cell.size();

        for (size_t j = 0; j < degree; j++) {
            int v1 = cell[j];
            int v2 = cell[(j + 1) % degree];

            // TODO: This adds most edges twice (once per adjacent cell), fine for now.
            E.row(edge) = IV(v1, v2);
            edge++;
        }
    }

    // Mesh points and faces
    V.resize(n_vtx + n_faces, 3);
    V << S, X;

    F.resize(num_voronoi_edges * 2, 3);
    F.setZero();
    C.resize(num_voronoi_edges * 2, 3);
    C.setZero();
    VectorXT areas = getCellAreas(x, cells, n_free);
    edge = 0;
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &cell = cells[i];
        for (size_t j = 1; j < cells[i].size() - 1; j++) {
            int v1 = cell[0];
            int v2 = cell[j];
            int v3 = cell[j + 1];

            F.row(edge) = IV3(v1 + n_vtx, v2 + n_vtx, v3 + n_vtx);
            C.row(edge) = (i == selected ? TV3(0.1, 0.1, 0.1) : getColor(areas(i), energyObjective.getAreaTarget(i)));
            edge++;
        }
    }
}

void Foam2D::getPlotAreaHistogram(VectorXT &areas) {
    VectorXi tri = tessellations[tesselation]->getDualGraph(vertices, params);
    VectorXT x = tessellations[tesselation]->getNodes(vertices, params, tri);
    std::vector<std::vector<int>> cells = tessellations[tesselation]->getCells(vertices, tri, x);

    areas = getCellAreas(x, cells, n_free);
    for (int i = 0; i < areas.rows(); i++) {
        areas(i) /= energyObjective.getAreaTarget(i);
    }
}

bool Foam2D::isConvergedDynamic(double tol) {
    energyObjective.tessellation = tessellations[tesselation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;

    int dims = 2 + tessellations[tesselation]->getNumVertexParams();
    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * dims, n_fixed * dims);
    VectorXT c_free = c.segment(0, n_free * dims);

    return dynamicObjective.getGradient(c_free).norm() < tol;
}

void Foam2D::getPlotObjectiveStats(bool dynamics, double &obj_value, double &gradient_norm, bool &hessian_pd) {
    energyObjective.tessellation = tessellations[tesselation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;

    int dims = 2 + tessellations[tesselation]->getNumVertexParams();
    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * dims, n_fixed * dims);
    VectorXT c_free = c.segment(0, n_free * dims);

    Eigen::SparseMatrix<double> hessian;
    if (dynamics) {
        obj_value = dynamicObjective.evaluate(c_free);
        gradient_norm = dynamicObjective.getGradient(c_free).norm();
        dynamicObjective.getHessian(c_free, hessian);
    } else {
        obj_value = energyObjective.evaluate(c_free);
        gradient_norm = energyObjective.getGradient(c_free).norm();
        hessian = energyObjective.get_d2Odc2(c_free);
    }
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver(hessian);
    hessian_pd = solver.info() != Eigen::ComputationInfo::NumericalIssue;
}

void
Foam2D::getPlotObjectiveFunctionLandscape(int selected_vertex, int type, int image_size, double range, VectorXf &obj,
                                          double &obj_min, double &obj_max) {
    energyObjective.tessellation = tessellations[tesselation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;

    int dims = 2 + tessellations[tesselation]->getNumVertexParams();
    VectorXT c = tessellations[tesselation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * dims, n_fixed * dims);
    VectorXT c_free = c.segment(0, n_free * dims);

    obj.resize(image_size * image_size * 3);

    obj_max = 0;
    obj_min = INFINITY;

    int xindex = selected_vertex * dims + 0;
    int yindex = selected_vertex * dims + 1;
    VectorXT DX = VectorXT::Zero(c_free.rows());
    DX(xindex) = 1;
    VectorXT DY = VectorXT::Zero(c_free.rows());
    DY(yindex) = 1;
    for (int i = 0; i < image_size; i++) {
        for (int j = 0; j < image_size; j++) {
            double dx = (double) (j - image_size / 2) / image_size * range;
            double dy = (double) -(i - image_size / 2) / image_size * range;
            double o;
            if (type == 0) {
                o = energyObjective.evaluate(c_free + dx * DX + dy * DY);
            } else if (type == 1) {
                o = energyObjective.get_dOdc(c_free + dx * DX + dy * DY)(xindex);
            } else {
                // type == 2
                o = energyObjective.get_dOdc(c_free + dx * DX + dy * DY)(yindex);
            }

            if (o > obj_max) obj_max = o;
            if (o < obj_min) obj_min = o;

            obj(i * image_size * 3 + j * 3 + 0) = o;
            obj(i * image_size * 3 + j * 3 + 1) = o;
            obj(i * image_size * 3 + j * 3 + 2) = o;
        }
    }

    obj = (obj.array() - obj_min) / (obj_max - obj_min);
}
