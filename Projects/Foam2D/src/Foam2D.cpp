#include "../include/Foam2D.h"
#include "../include/CodeGen.h"
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../include/Constants.h"
#include <random>
#include "../include/TrajectoryOpt/IpoptSolver.h"
#include <thread>

Foam2D::Foam2D() {
    tessellations.push_back(new Voronoi());
    tessellations.push_back(new Power());
    minimizers.push_back(new GradientDescentLineSearch(1, 1e-6, 15));
    minimizers.push_back(new NewtonFunctionMinimizer(1, 1e-6, 15));

    nlp.energy = &energyObjective;
    nlp.dynamics = &dynamicObjective;
}

void Foam2D::resetVertexParams() {
    params = tessellations[tessellation]->getDefaultVertexParams(vertices);
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

    boundary.resize(0);
    resetVertexParams();
}

void Foam2D::initBasicTestCase() {
    n_free = 4;
    n_fixed = 8;

    vertices.resize((n_free + n_fixed) * 2);

    vertices << -0.5, 0, 0.5, 0, 0, -0.5, 0, 0.5, -1, -1, 1, -1, 1, 1, -1, 1, -1, 0, 1, 0, 0, -1, 0, 1;

    boundary.resize(0);
    resetVertexParams();
}

void Foam2D::initRandomCellsInBox(int n_free_in) {
    n_free = n_free_in;
    n_fixed = 8;

    double hw = 0.75;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-hw, hw);

    VectorXT inf_points(n_fixed * 2);
    double inf = 100;
    inf_points << -inf, -inf, inf, -inf, inf, inf, -inf, inf, -inf, 0, inf, 0, 0, -inf, 0, inf;

    vertices = VectorXT::Zero((n_free + n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(n_free * 2, n_fixed * 2) = inf_points;

    boundary.resize(4 * 2);
    boundary << -hw, -hw, hw, -hw, hw, hw, -hw, hw;

    resetVertexParams();
}

void Foam2D::dynamicsInit(double dt, double m, double mu) {
    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tessellation]->getNumVertexParams()));
    dynamicObjective.init(c_free, dt, m, mu, &energyObjective);
}

void Foam2D::dynamicsNewStep() {
    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tessellation]->getNumVertexParams()));
    dynamicObjective.newStep(c_free);
}

void Foam2D::optimize(bool dynamic) {
    energyObjective.tessellation = tessellations[tessellation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;
    energyObjective.boundary = boundary;

    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * (2 + tessellations[tessellation]->getNumVertexParams()),
                                        n_fixed * (2 + tessellations[tessellation]->getNumVertexParams()));
    VectorXT c_free = c.segment(0,
                                n_free * (2 + tessellations[tessellation]->getNumVertexParams()));

    if (dynamic) {
        minimizers[opttype]->minimize(&dynamicObjective, c_free);
    } else {
        minimizers[opttype]->minimize(&energyObjective, c_free);
    }

    c.segment(0, n_free * (2 + tessellations[tessellation]->getNumVertexParams())) = c_free;

    tessellations[tessellation]->separateVerticesParams(c, vertices, params);
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

//    std::cout << "c_fixed" << std::endl;
//    for (int i = 0; i < nlp.energy->c_fixed.rows(); i++) {
//        std::cout << nlp.energy->c_fixed(i) << ", ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "c0" << std::endl;
//    for (int i = 0; i < nlp.c0.rows(); i++) {
//        std::cout << nlp.c0(i) << ", ";
//    }
//    std::cout << std::endl;
}

static void threadIPOPT(TrajectoryOptNLP *nlp) {
    /** IPOPT SOLVE **/
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

    app->RethrowNonIpoptException(true);

    app->Options()->SetNumericValue("tol", 1e-5);
//    app->Options()->SetStringValue("mu_strategy", "monotone");
    // app->Options()->SetStringValue("mu_strategy", "adaptive");

//    app->Options()->SetStringValue("output_file", data_folder + "/ipopt.out");
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetStringValue("linear_solver", "ma57");
//    app->Options()->SetStringValue("linear_solver", "pardisomkl");
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
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded) {
        std::cout << std::endl
                  << std::endl
                  << "*** Error during initialization!" << std::endl;
        return;
    }

    // Ask Ipopt to solve the problem
    std::cout << "Solving problem using IPOPT" << std::endl;

    // objective.bound[0] = 1e-5;
    // objective.bound[1] = 12.0 * simulation.cells.unit;

    Ipopt::SmartPtr<IpoptSolver> mynlp = new IpoptSolver(nlp);

    status = app->OptimizeTNLP(mynlp);
    if (status == Ipopt::Solve_Succeeded) {
        std::cout << std::endl
                  << std::endl
                  << "*** The problem solved!" << std::endl;
    } else {
        std::cout << std::endl
                  << std::endl
                  << "*** The problem FAILED!" << std::endl;
    }
}

void Foam2D::trajectoryOptOptimizeIPOPT(int N) {
    /** NLP INITIALIZATION **/
    nlp.N = N;
    nlp.agent = energyObjective.drag_idx;
    nlp.target_pos = energyObjective.drag_target_pos;

    nlp.x_guess.resize(N * (nlp.c0.rows() + 2));
    VectorXd u_guess = VectorXT::Zero(2 * N);

    // x format is [c1 ... cN u1 ... uN]
    nlp.x_guess << nlp.c0.replicate(N, 1), u_guess;
    nlp.x_sol = nlp.x_guess;

    nlp.early_stop = false;

    /** IPOPT SOLVE **/
    std::thread t1(threadIPOPT, &nlp);
    t1.detach();
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

void Foam2D::trajectoryOptGetForces(VectorXd &forceX, VectorXd &forceY) {
    int dims = energyObjective.tessellation->getNumVertexParams() + 2;
    VectorXT u = nlp.x_sol.segment(nlp.N * n_free * dims, nlp.N * 2);
    forceX.resize(u.rows() / 2);
    forceY.resize(u.rows() / 2);

    for (int i = 0; i < forceX.rows(); i++) {
        forceX(i) = u(i * 2 + 0);
        forceY(i) = u(i * 2 + 1);
    }
}

void Foam2D::trajectoryOptStop() {
    nlp.early_stop = true;
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

void Foam2D::getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                        MatrixXi &F, MatrixXT &Fc) {
    VectorXi tri = tessellations[tessellation]->getDualGraph(vertices, params);
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
    Fc.resize(n_faces, 3);
    Fc.setZero();
    for (int i = 0; i < n_faces; i++) {
        F.row(i) = tri.segment<3>(i * 3);
        Fc.row(i) = TV3(0.5, 0.5, 0.5);
    }

    Sc.resize(S.rows(), 3);
    Sc.setZero();

    Ec.resize(E.rows(), 3);
    Ec.setZero();

    if (energyObjective.drag_idx >= 0) {
        MatrixXT V_target;
        V_target.resize(3, 3);
        V_target.col(2) = TV3(0.01, 0.01, 0.01);

        V_target.row(0).segment<2>(0) = energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI_2), sin(M_PI_2));
        V_target.row(1).segment<2>(0) =
                energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI * 7.0 / 6.0), sin(M_PI * 7.0 / 6.0));
        V_target.row(2).segment<2>(0) =
                energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI * 11.0 / 6.0), sin(M_PI * 11.0 / 6.0));

        MatrixXT V_temp = V;
        V.resize(V_temp.rows() + V_target.rows(), 3);
        V << V_temp, V_target;

        MatrixXi F_target = IV3(V_temp.rows() + 0, V_temp.rows() + 1, V_temp.rows() + 2).transpose();
        MatrixXi F_temp = F;
        F.resize(F_temp.rows() + F_target.rows(), 3);
        F << F_temp, F_target;

        MatrixXT Fc_target = TV3(0, 0.4, 0).transpose();
        MatrixXT Fc_temp = Fc;
        Fc.resize(Fc_temp.rows() + Fc_target.rows(), 3);
        Fc << Fc_temp, Fc_target;
    }
}

void Foam2D::getTessellationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                       MatrixXi &F, MatrixXT &Fc) {
    VectorXi tri = tessellations[tessellation]->getDualGraph(vertices, params);
    long n_vtx = vertices.rows() / 2, n_faces = tri.rows() / 3, n_bdy = boundary.rows() / 2;

    // Overlay points and edges
    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    std::vector<TV> face0;
    std::vector<TV> face1;
    std::vector<TV> face2;

    int n_cells = n_free;
    VectorXT areas = VectorXT::Zero(n_cells);

    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    int dims = 2 + tessellations[tessellation]->getNumVertexParams();

//    std::vector<std::vector<int>> neighborLists = tessellations[tessellation]->getNeighbors(vertices, tri, n_free);
    std::vector<std::vector<int>> neighborLists = tessellations[tessellation]->getNeighborsClipped(vertices, params,
                                                                                                   tri, boundary,
                                                                                                   n_cells);
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &neighbors = neighborLists[i];
        size_t degree = neighbors.size();

        TV v0 = vertices.segment<2>(i * 2);
        VectorXT c0 = c.segment(i * dims, dims);

        for (size_t j = 0; j < degree; j++) {
            TV v1, v2;

            int n1 = neighbors[j];
            int n2 = neighbors[(j + 1) % degree];
            int n3 = neighbors[(j + 2) % degree];

            if (n1 < n_vtx && n2 < n_vtx) {
                // Normal node.
                v1 = tessellations[tessellation]->getNode(c0, c.segment(n1 * dims, dims), c.segment(n2 * dims, dims));
            } else if (n1 < n_vtx && n2 >= n_vtx) {
                // Boundary node with n2 a boundary edge.
                v1 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n1 * dims, dims),
                                                                  boundary.segment<2>((n2 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n2 - n_vtx + 1) % n_bdy) * 2));
            } else if (n1 >= n_vtx && n2 < n_vtx) {
                // Boundary node with n1 a boundary edge.
                v1 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n2 * dims, dims),
                                                                  boundary.segment<2>((n1 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n1 - n_vtx + 1) % n_bdy) * 2));
            } else {
                // Boundary vertex.
                assert(n1 >= n_vtx && n2 >= n_vtx);
                v1 = boundary.segment<2>((n2 - n_vtx) * 2);
            }

            if (n2 < n_vtx && n3 < n_vtx) {
                // Normal node.
                v2 = tessellations[tessellation]->getNode(c0, c.segment(n2 * dims, dims), c.segment(n3 * dims, dims));
            } else if (n2 < n_vtx && n3 >= n_vtx) {
                // Boundary node with n3 a boundary edge.
                v2 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n2 * dims, dims),
                                                                  boundary.segment<2>((n3 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n3 - n_vtx + 1) % n_bdy) * 2));
            } else if (n2 >= n_vtx && n3 < n_vtx) {
                // Boundary node with n2 a boundary edge.
                v2 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n3 * dims, dims),
                                                                  boundary.segment<2>((n2 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n2 - n_vtx + 1) % n_bdy) * 2));
            } else {
                // Boundary vertex.
                assert(n2 >= n_vtx && n3 >= n_vtx);
                v2 = boundary.segment<2>((n3 - n_vtx) * 2);
            }

            face0.push_back(v0);
            face1.push_back(v1);
            face2.push_back(v2);

            areas(i) += 0.5 * ((v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1)));
        }
    }

    V.resize(face0.size() * 3, 3);
    V.setZero();
    E.resize(face0.size(), 2);
    E.setZero();
    F.resize(face0.size(), 3);
    F.setZero();
    Fc.resize(face0.size(), 3);
    Fc.setZero();

    int currentCell = 0;
    int currentIdxInCell = 0;

    for (int i = 0; i < face0.size(); i++) {
        TV v0 = face0[i];
        TV v1 = face1[i];
        TV v2 = face2[i];

        V.row(i * 3 + 0).segment<2>(0) = v0;
        V.row(i * 3 + 1).segment<2>(0) = v1;
        V.row(i * 3 + 2).segment<2>(0) = v2;

        E.row(i) = IV(i * 3 + 1, i * 3 + 2);
        F.row(i) = IV3(i * 3 + 0, i * 3 + 1, i * 3 + 2);
        Fc.row(i) = (currentCell == energyObjective.drag_idx ? TV3(0.1, 0.1, 0.1) : getColor(areas(currentCell),
                                                                                             energyObjective.getAreaTarget(
                                                                                                     currentCell)));

        currentIdxInCell++;
        if (currentIdxInCell == neighborLists[currentCell].size()) {
            currentIdxInCell = 0;
            currentCell++;
        }
    }

    X = V;

    Sc.resize(S.rows(), 3);
    Sc.setZero();

    Ec.resize(E.rows(), 3);
    Ec.setZero();

    if (energyObjective.drag_idx >= 0) {
        MatrixXT V_target;
        V_target.resize(3, 3);
        V_target.col(2) = TV3(0.01, 0.01, 0.01);

        V_target.row(0).segment<2>(0) = energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI_2), sin(M_PI_2));
        V_target.row(1).segment<2>(0) =
                energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI * 7.0 / 6.0), sin(M_PI * 7.0 / 6.0));
        V_target.row(2).segment<2>(0) =
                energyObjective.drag_target_pos + 0.03 * TV(cos(M_PI * 11.0 / 6.0), sin(M_PI * 11.0 / 6.0));

        MatrixXT V_temp = V;
        V.resize(V_temp.rows() + V_target.rows(), 3);
        V << V_temp, V_target;

        MatrixXi F_target = IV3(V_temp.rows() + 0, V_temp.rows() + 1, V_temp.rows() + 2).transpose();
        MatrixXi F_temp = F;
        F.resize(F_temp.rows() + F_target.rows(), 3);
        F << F_temp, F_target;

        MatrixXT Fc_target = TV3(0, 0.4, 0).transpose();
        MatrixXT Fc_temp = Fc;
        Fc.resize(Fc_temp.rows() + Fc_target.rows(), 3);
        Fc << Fc_temp, Fc_target;
    }
}

void Foam2D::addTrajectoryOptViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                        MatrixXi &F, MatrixXT &Fc) {
    MatrixXT S_traj(nlp.N + 1, 3);
    S_traj.setZero();

    int dims = energyObjective.tessellation->getNumVertexParams() + 2;
    S_traj(0, 0) = nlp.c0(nlp.agent * dims + 0);
    S_traj(0, 1) = nlp.c0(nlp.agent * dims + 1);
    for (int k = 0; k < nlp.N; k++) {
        S_traj(k + 1, 0) = nlp.x_sol(k * nlp.c0.rows() + nlp.agent * dims + 0);
        S_traj(k + 1, 1) = nlp.x_sol(k * nlp.c0.rows() + nlp.agent * dims + 1);
    }

    MatrixXT S_temp = S;
    S.resize(S_temp.rows() + S_traj.rows(), 3);
    S << S_temp, S_traj;

    MatrixXT X_temp = X;
    X.resize(X_temp.rows() + S_traj.rows(), 3);
    X << X_temp, S_traj;

    MatrixXT Sc_traj = TV3(1, 0, 0).transpose().replicate(nlp.N + 1, 1);
    MatrixXT Sc_temp = Sc;
    Sc.resize(Sc_temp.rows() + Sc_traj.rows(), 3);
    Sc << Sc_temp, Sc_traj;

    MatrixXi E_traj(nlp.N, 2);
    for (int k = 0; k < nlp.N; k++) {
        E_traj(k, 0) = X_temp.rows() + k + 0;
        E_traj(k, 1) = X_temp.rows() + k + 1;
    }

    MatrixXi E_temp = E;
    E.resize(E_temp.rows() + E_traj.rows(), 2);
    E << E_temp, E_traj;

    MatrixXT Ec_traj = TV3(1, 0, 0).transpose().replicate(nlp.N, 1);
    MatrixXT Ec_temp = Ec;
    Ec.resize(Ec_temp.rows() + Ec_traj.rows(), 3);
    Ec << Ec_temp, Ec_traj;
}

void Foam2D::getPlotAreaHistogram(VectorXT &areas) {
    VectorXi tri = tessellations[tessellation]->getDualGraph(vertices, params);

    int n_cells = n_free;
    areas.resize(n_cells);
    areas.setZero();

    int n_vtx = vertices.rows() / 2, n_bdy = boundary.rows() / 2;

    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    int dims = 2 + tessellations[tessellation]->getNumVertexParams();

//    std::vector<std::vector<int>> neighborLists = tessellations[tessellation]->getNeighbors(vertices, tri, n_free);
    std::vector<std::vector<int>> neighborLists = tessellations[tessellation]->getNeighborsClipped(vertices, params,
                                                                                                   tri, boundary,
                                                                                                   n_cells);
    for (int i = 0; i < n_cells; i++) {
        std::vector<int> &neighbors = neighborLists[i];
        size_t degree = neighbors.size();

        TV v0 = vertices.segment<2>(i * 2);
        VectorXT c0 = c.segment(i * dims, dims);

        for (size_t j = 0; j < degree; j++) {
            TV v1, v2;

            int n1 = neighbors[j];
            int n2 = neighbors[(j + 1) % degree];
            int n3 = neighbors[(j + 2) % degree];

            if (n1 < n_vtx && n2 < n_vtx) {
                // Normal node.
                v1 = tessellations[tessellation]->getNode(c0, c.segment(n1 * dims, dims), c.segment(n2 * dims, dims));
            } else if (n1 < n_vtx && n2 >= n_vtx) {
                // Boundary node with n2 a boundary edge.
                v1 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n1 * dims, dims),
                                                                  boundary.segment<2>((n2 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n2 - n_vtx + 1) % n_bdy) * 2));
            } else if (n1 >= n_vtx && n2 < n_vtx) {
                // Boundary node with n1 a boundary edge.
                v1 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n2 * dims, dims),
                                                                  boundary.segment<2>((n1 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n1 - n_vtx + 1) % n_bdy) * 2));
            } else {
                // Boundary vertex.
                assert(n1 >= n_vtx && n2 >= n_vtx);
                v1 = boundary.segment<2>((n2 - n_vtx) * 2);
            }

            if (n2 < n_vtx && n3 < n_vtx) {
                // Normal node.
                v2 = tessellations[tessellation]->getNode(c0, c.segment(n2 * dims, dims), c.segment(n3 * dims, dims));
            } else if (n2 < n_vtx && n3 >= n_vtx) {
                // Boundary node with n3 a boundary edge.
                v2 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n2 * dims, dims),
                                                                  boundary.segment<2>((n3 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n3 - n_vtx + 1) % n_bdy) * 2));
            } else if (n2 >= n_vtx && n3 < n_vtx) {
                // Boundary node with n2 a boundary edge.
                v2 = tessellations[tessellation]->getBoundaryNode(c0, c.segment(n3 * dims, dims),
                                                                  boundary.segment<2>((n2 - n_vtx) * 2),
                                                                  boundary.segment<2>(((n2 - n_vtx + 1) % n_bdy) * 2));
            } else {
                // Boundary vertex.
                assert(n2 >= n_vtx && n3 >= n_vtx);
                v2 = boundary.segment<2>((n3 - n_vtx) * 2);
            }

            areas(i) += 0.5 * ((v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1)));
        }
    }

    for (int i = 0; i < areas.rows(); i++) {
        areas(i) /= energyObjective.getAreaTarget(i);
    }
}

bool Foam2D::isConvergedDynamic(double tol) {
    energyObjective.tessellation = tessellations[tessellation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;
    energyObjective.boundary = boundary;

    int dims = 2 + tessellations[tessellation]->getNumVertexParams();
    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
    energyObjective.c_fixed = c.segment(n_free * dims, n_fixed * dims);
    VectorXT c_free = c.segment(0, n_free * dims);

    return dynamicObjective.getGradient(c_free).norm() < tol;
}

void Foam2D::getPlotObjectiveStats(bool dynamics, double &obj_value, double &gradient_norm, bool &hessian_pd) {
    energyObjective.tessellation = tessellations[tessellation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;
    energyObjective.boundary = boundary;

    int dims = 2 + tessellations[tessellation]->getNumVertexParams();
    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
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
        energyObjective.getHessian(c_free, hessian);
    }
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver(hessian);
    hessian_pd = solver.info() != Eigen::ComputationInfo::NumericalIssue;
}

void
Foam2D::getPlotObjectiveFunctionLandscape(int selected_vertex, int type, int image_size, double range, VectorXf &obj,
                                          double &obj_min, double &obj_max) {
    energyObjective.tessellation = tessellations[tessellation];
    energyObjective.n_free = n_free;
    energyObjective.n_fixed = n_fixed;
    energyObjective.boundary = boundary;

    int dims = 2 + tessellations[tessellation]->getNumVertexParams();
    VectorXT c = tessellations[tessellation]->combineVerticesParams(vertices, params);
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
