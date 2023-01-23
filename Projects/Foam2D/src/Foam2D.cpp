#include "../include/Foam2D.h"
#include "Projects/Foam2D/include/Tessellation/Voronoi.h"
#include "Projects/Foam2D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"
#include <random>
#include "../include/TrajectoryOpt/TrajectoryOptSolver.h"
#include <thread>

#include "Projects/Foam2D/include/Energy/CellFunctionArea.h"
#include "../src/optLib/ParallelLineSearchMinimizers.h"

#include "Projects/Foam2D/include/Boundary/SimpleBoundary.h"
#include "Projects/Foam2D/include/Boundary/CircleBoundary.h"
#include "Projects/Foam2D/include/Boundary/RigidBodyAgentBoundary.h"

Foam2D::Foam2D() {
    info = new Foam2DInfo();
    info->tessellations.push_back(new Voronoi());
    info->tessellations.push_back(new Power());

    energyObjective.info = info;
    dynamicObjective.info = info;
    trajOptNLP.info = info;
    imageMatchObjective.info = info;
    imageMatchSAObjective.info = info;

    dynamicObjective.energyObjective = &energyObjective;
    trajOptNLP.energy = &energyObjective;
}

void Foam2D::resetVertexParams() {
    params = info->getTessellation()->getDefaultVertexParams(vertices);

    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXd c = info->getTessellation()->combineVerticesParams(vertices, params);
    info->c_fixed = c.segment(info->n_free * dims, info->n_fixed * dims);

    dynamicsInit();
}

void Foam2D::initRandomSitesInCircle(int n_free_in, int n_fixed_in) {
    info->n_free = n_free_in;
    info->n_fixed = n_fixed_in;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, 0.5);

    VectorXT boundary_points;
    boundary_points.resize(info->n_fixed * 2);
    for (int i = 0; i < info->n_fixed; i++) {
        boundary_points.segment<2>(i * 2) = TV(cos(i * 2 * M_PI / info->n_fixed), sin(i * 2 * M_PI / info->n_fixed));
    }

    vertices = VectorXT::Zero((info->n_free + info->n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(info->n_free * 2, info->n_fixed * 2) = boundary_points;

    info->boundary = new SimpleBoundary({}, {});

    resetVertexParams();
}

void Foam2D::initBasicTestCase() {
    info->n_free = 4;
    info->n_fixed = 8;

    vertices.resize((info->n_free + info->n_fixed) * 2);
    vertices << -0.6, 0, 0.6, 0, 0, -0.5, 0, 0.5, -1, -1, 1, -1, 1, 1, -1, 1, -1, 0, 1, 0, 0, -1, 0, 1;

    info->boundary = new SimpleBoundary({}, {});

    resetVertexParams();
}

void Foam2D::initRandomCellsInBox(int n_free_in) {
    info->n_free = n_free_in;
    info->n_fixed = 8;

    VectorXT inf_points(info->n_fixed * 2);
    double inf = 100;
    inf_points << -inf, -inf, inf, -inf, inf, inf, -inf, inf, -inf, 0, inf, 0, 0, -inf, 0, inf;

    double dx = 0.75;
    double dy = 0.75;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-dx, dx);

    vertices = VectorXT::Zero((info->n_free + info->n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(info->n_free * 2, info->n_fixed * 2) = inf_points;

    VectorXT v(4 * 2);
    v << -dx, -dy, dx, -dy, dx, dy, -dx, dy;
    info->boundary = new SimpleBoundary(v, {});

    resetVertexParams();
}

void Foam2D::initDynamicBox(int n_free_in) {
    info->n_free = n_free_in;
    info->n_fixed = 8;

    VectorXT inf_points(info->n_fixed * 2);
    double inf = 100;
    inf_points << -inf, -inf, inf, -inf, inf, inf, -inf, inf, -inf, 0, inf, 0, 0, -inf, 0, inf;

    double dx = 0.75;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-dx, dx);

    vertices = VectorXT::Zero((info->n_free + info->n_fixed) * 2).unaryExpr([&](float dummy) { return dis(gen); });
    vertices.segment(info->n_free * 2, info->n_fixed * 2) = inf_points;

    TV p(0.75 * sqrt(2.0), M_PI_4);
    IV free_idx(0, 1);
    info->boundary = new CircleBoundary(p, free_idx, 4);

    resetVertexParams();
}

void Foam2D::initRigidBodyAgent(int n_free_in) {
    info->n_free = n_free_in;
    info->n_fixed = 8;

    VectorXT inf_points(info->n_fixed * 2);
    double inf = 100;
    inf_points << -inf, -inf, inf, -inf, inf, inf, -inf, inf, -inf, 0, inf, 0, 0, -inf, 0, inf;

//    VectorXT agent(8);
//    double a = 0.15;
//    agent << -a, -a, -a, a, a, a, a, -a;

    int nsides = 100;
    VectorXT agent(nsides * 2);
    double a = 0.2;
    double b = 0.1;
    for (int i = 0; i < nsides; i++) {
        agent(i * 2 + 0) = a * cos(-i * 2 * M_PI / nsides);
        agent(i * 2 + 1) = b * sin(-i * 2 * M_PI / nsides);
    }

    double dx = 0.75;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-dx, dx);
    vertices.resize((info->n_free + info->n_fixed) * 2);
    for (int i = 0; i < info->n_free; i++) {
        double x = dis(gen), y = dis(gen);
        while (fabs(x) < a && fabs(y) < a) {
            x = dis(gen);
            y = dis(gen);
        }
        vertices(i * 2 + 0) = x;
        vertices(i * 2 + 1) = y;
    }

    vertices.segment(info->n_free * 2, info->n_fixed * 2) = inf_points;

    TV3 p(0, 0, 0);
//    VectorXi free_idx(1);
//    free_idx(0) = 2;
    IV3 free_idx(0, 1, 2);
    info->boundary = new RigidBodyAgentBoundary(p, free_idx, agent);

    resetVertexParams();
}

void Foam2D::initImageMatch(MatrixXi &markers) {
    markers = markers.unaryExpr([](int x) { return x - 1; });
    int threshold = 5;
    for (int i = 0; i < markers.maxCoeff(); i++) {
        if ((markers.array() == i).count() < threshold) {
            markers = markers.unaryExpr([i](int x) { return x > i ? x - 1 : x; });
            i--;
        }
    }

    info->n_free = markers.maxCoeff() + 1;
    info->n_fixed = 8;

    VectorXT inf_points(info->n_fixed * 2);
    double inf = 100;
    inf_points << -inf, -inf, inf, -inf, inf, inf, -inf, inf, -inf, 0, inf, 0, 0, -inf, 0, inf;

    vertices = VectorXT::Zero((info->n_free + info->n_fixed) * 2);
    vertices.segment(info->n_free * 2, info->n_fixed * 2) = inf_points;

    imageMatchObjective.dx = markers.cols() * 0.8 / std::max(markers.rows(), markers.cols());
    imageMatchObjective.dy = markers.rows() * 0.8 / std::max(markers.rows(), markers.cols());
    double dx = imageMatchObjective.dx;
    double dy = imageMatchObjective.dy;

    VectorXT v(4 * 2);
    v << -dx, -dy, dx, -dy, dx, dy, -dx, dy;
    info->boundary = new SimpleBoundary(v, {});

    // Get only boundary pixel info in order to set objective function pixels.
    VectorXi countOuter = VectorXi::Zero(info->n_free);
    for (int i = 0; i < markers.rows(); i++) {
        for (int j = 0; j < markers.cols(); j++) {
            int mark = markers(i, j);
            bool outer = false;
            if (i == 0 || markers(i - 1, j) != mark) outer = true;
            if (i == markers.rows() - 1 || markers(i + 1, j) != mark) outer = true;
            if (j == 0 || markers(i, j - 1) != mark) outer = true;
            if (j == markers.cols() - 1 || markers(i, j + 1) != mark) outer = true;
            if (mark >= 0 && outer) {
                countOuter(mark) += 1;
            }
        }
    }
    // Allocate objective function pixel arrays.
    std::vector<VectorXd> pix(info->n_free);
    for (int i = 0; i < info->n_free; i++) {
        if (countOuter(i) > 1000) {
            std::cout << "oh no" << std::endl;
            assert(0);
        }
        pix[i].resize(countOuter(i) * 2);
        pix[i].setZero();
    }
    // Assign objective function pixel values.
    VectorXi idx = VectorXi::Zero(info->n_free);
    for (int i = 0; i < markers.rows(); i++) {
        for (int j = 0; j < markers.cols(); j++) {
            int mark = markers(i, j);
            bool outer = false;
            if (i == 0 || markers(i - 1, j) != mark) outer = true;
            if (i == markers.rows() - 1 || markers(i + 1, j) != mark) outer = true;
            if (j == 0 || markers(i, j - 1) != mark) outer = true;
            if (j == markers.cols() - 1 || markers(i, j + 1) != mark) outer = true;
            if (mark >= 0 && outer) {
                pix[mark].segment<2>(idx(mark) * 2) = TV((j * 2.0 / markers.cols() - 1.0) * dx,
                                                         -(i * 2.0 / markers.rows() - 1.0) * dy);
                idx(mark)++;
            }
        }
    }
    imageMatchObjective.pix = pix;
    imageMatchSAObjective.pix = pix;

    // Get info for all pixels in order to set site positions and area targets.
    VectorXi count = VectorXi::Zero(info->n_free);
    VectorXT sumX = VectorXT::Zero(info->n_free);
    VectorXT sumY = VectorXT::Zero(info->n_free);
    for (int i = 0; i < markers.rows(); i++) {
        for (int j = 0; j < markers.cols(); j++) {
            int mark = markers(i, j);
            if (mark >= 0) {
                count(mark) += 1;
                sumX(mark) += (j * 2.0 / markers.cols() - 1.0) * dx;
                sumY(mark) -= (i * 2.0 / markers.rows() - 1.0) * dy;
            }
        }
    }

    // Set area targets
    VectorXT areas(info->n_free);
    double totalArea = 4 * dx * dy;
    int totalPixels = markers.cols() * markers.rows();
    for (int i = 0; i < info->n_free; i++) {
        areas(i) = count(i) * totalArea / totalPixels;
    }
    info->energy_area_targets = areas;

    // Set site positions
    for (int i = 0; i < info->n_free; i++) {
        double x = sumX(i) / count(i);
        double y = sumY(i) / count(i);
        vertices.segment<2>(i * 2) = TV(x, y);
    }

    resetVertexParams();
}

void Foam2D::imageMatchGetInfo(double &obj_value, std::vector<VectorXd> &pix) {
    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXd c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0, info->n_free * dims);

    obj_value = imageMatchObjective.evaluate(c_free);
    pix = imageMatchObjective.pix;
}

void Foam2D::dynamicsInit() {
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                info->n_free * (2 + info->getTessellation()->getNumVertexParams()));
    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    dynamicObjective.y_prev = y;
    dynamicObjective.v_prev = VectorXd::Zero(y.rows());
    trajOptNLP.c0 = c_free;
    trajOptNLP.v0 = VectorXd::Zero(c_free.rows());
}

void Foam2D::dynamicsNewStep() {
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                info->n_free * (2 + info->getTessellation()->getNumVertexParams()));
    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    dynamicObjective.newStep(y);
}

void Foam2D::optimize(int mode) {
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0,
                                info->n_free * (2 + info->getTessellation()->getNumVertexParams()));

    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    GradientDescentLineSearch *minimizer;
    switch (opttype) {
        case 0:
            minimizer = &minimizerGradientDescent;
            break;
        case 1:
            minimizer = &minimizerNewton;
            break;
        case 2:
            minimizer = &minimizerBFGS;
            break;
        default:
            std::cout << "Invalid minimizer!!!" << std::endl;
    }

    switch (mode) {
        case 0:
//            energyObjective.check_gradients(y);
            minimizer->minimize(&energyObjective, y);
            c_free = y.segment(0, c_free.rows());
            p_free = y.segment(c_free.rows(), p_free.rows());
            break;
        case 1:
//            dynamicObjective.check_gradients(y);
            minimizer->minimize(&dynamicObjective, y);
            c_free = y.segment(0, c_free.rows());
            p_free = y.segment(c_free.rows(), p_free.rows());
            break;
        case 2:
            if (info->boundary->nfree > 0) {
                std::cout << "WARNING: Image match with free boundaries not supported!" << std::endl;
                assert(0);
            }

            imageMatchSAObjective.c0 = c_free;
            imageMatchSAObjective.tau0 = info->energy_area_targets;
            imageMatchSAObjective.dcdtau = SparseMatrixd(1, 1);

//            imageMatchSAObjective.check_gradients(info->energy_area_targets);
//            imageMatchSAObjective.sols.clear();

            minimizerImageMatch.minimize(&imageMatchSAObjective, info->energy_area_targets);

            c_free = (imageMatchSAObjective.sols.begin())->second;
            imageMatchSAObjective.sols.clear();

//            if(!imageMatchSAObjective.getC(info->energy_area_targets, info->getTessellation(), c_free)){
//                std::cout << "Output invalid wowzers!!!" << std::endl << std::endl;
//            }
            break;
        default:
            std::cout << "Invalid optimization mode?" << std::endl;
    }

    info->boundary->compute(p_free);

    c.segment(0, info->n_free * (2 + info->getTessellation()->getNumVertexParams())) = c_free;
    info->getTessellation()->separateVerticesParams(c, vertices, params);
}

int Foam2D::getClosestMovablePointThreshold(const TV &p, double threshold) {
    int n_vtx = vertices.rows() / 2;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV p2 = vertices.segment<2>(i * 2);
        double d = (p2 - p).norm();
        if (d < threshold && d < dmin && i < info->n_free) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Foam2D::moveSelectedVertex(const TV &pos) {
    vertices.segment<2>(info->selected * 2) = pos;
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
//    app->Options()->SetStringValue("derivative_test", "first-order");
//    app->Options()->SetStringValue("derivative_test_print_all", "yes");
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

    Ipopt::SmartPtr<TrajectoryOptSolver> mynlp = new TrajectoryOptSolver(nlp);

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

void Foam2D::trajectoryOptOptimizeIPOPT() {
    /** NLP INITIALIZATION **/
    trajOptNLP.x_guess.resize(info->trajOpt_N * (trajOptNLP.c0.rows() + 2));
    VectorXd u_guess = VectorXT::Zero(2 * info->trajOpt_N);

    // x format is [c1 ... cN u1 ... uN]
    trajOptNLP.x_guess << trajOptNLP.c0.replicate(info->trajOpt_N, 1), u_guess;
//    for (int i = 0; i < info->trajOpt_N; i++) {
//        trajOptNLP.x_guess.segment<2>((i + 1) * trajOptNLP.c0.rows() + 2 * info->selected) =
//                trajOptNLP.c0.segment<2>(2 * info->selected) + (i + 1) * 1.0 / info->trajOpt_N *
//                                                               (info->selected_target_pos -
//                                                                trajOptNLP.c0.segment<2>(2 * info->selected));
//    }
    trajOptNLP.x_sol = trajOptNLP.x_guess;
    trajOptNLP.early_stop = false;

    // TODO: This is kind of a hack, to separate the tessellation used in the IPOPT thread from the one used in the visualization.
    Foam2DInfo *info_ = new Foam2DInfo(*info);
    switch (info->getTessellation()->getTessellationType()) {
        case VORONOI:
            info_->tessellations[0] = new Voronoi();
            break;
        case POWER:
            info_->tessellations[1] = new Power();
            break;
        default:
            assert(0);
            break;
    }
    trajOptNLP.info = info_;
    trajOptNLP.energy = new EnergyObjective(*trajOptNLP.energy);
    trajOptNLP.energy->info = info_;

//    VectorXT gradientTest_x = trajOptNLP.x_guess;
//    gradientTest_x = gradientTest_x.unaryExpr([](double x) {
//        std::random_device rd;
//        std::mt19937 gen(rd());
//        std::uniform_real_distribution<double> dis(-1e-3, 1e-3);
//        return x + dis(gen);
//    });
//    trajOptNLP.check_gradients(gradientTest_x);

    /** IPOPT SOLVE **/
    std::thread t1(threadIPOPT, &trajOptNLP);
    t1.detach();
}

void Foam2D::trajectoryOptGetFrame(int frame) {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    VectorXT c_frame =
            frame == 0 ? trajOptNLP.c0 : trajOptNLP.x_sol.segment((frame - 1) * info->n_free * dims,
                                                                  info->n_free * dims);

    VectorXT verts_free;
    VectorXT params_free;
    info->getTessellation()->separateVerticesParams(c_frame, verts_free, params_free);
    vertices.segment(0, info->n_free * 2) = verts_free;
    params.segment(0, info->n_free * (dims - 2)) = params_free;
}

void Foam2D::trajectoryOptGetForces(VectorXd &forceX, VectorXd &forceY) {
    int dims = info->getTessellation()->getNumVertexParams() + 2;
    VectorXT u = trajOptNLP.x_sol.segment(info->trajOpt_N * info->n_free * dims, info->trajOpt_N * 2);
    forceX.resize(u.rows() / 2);
    forceY.resize(u.rows() / 2);

    for (int i = 0; i < forceX.rows(); i++) {
        forceX(i) = u(i * 2 + 0);
        forceY(i) = u(i * 2 + 1);
    }
}

void Foam2D::trajectoryOptStop() {
    trajOptNLP.early_stop = true;
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

void Foam2D::getTriangulationViewerData(MatrixXT &S, MatrixXT &X, MatrixXi &E, MatrixXT &Sc, MatrixXT &Ec, MatrixXT &V,
                                        MatrixXi &F, MatrixXT &Fc) {
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    VectorXi tri = info->getTessellation()->getDualGraph(vertices, params);
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

    if (info->selected >= 0) {
        MatrixXT V_target;
        V_target.resize(3, 3);
        V_target.col(2) = TV3(0.01, 0.01, 0.01);

        V_target.row(0).segment<2>(0) = info->selected_target_pos + 0.03 * TV(cos(M_PI_2), sin(M_PI_2));
        V_target.row(1).segment<2>(0) =
                info->selected_target_pos + 0.03 * TV(cos(M_PI * 7.0 / 6.0), sin(M_PI * 7.0 / 6.0));
        V_target.row(2).segment<2>(0) =
                info->selected_target_pos + 0.03 * TV(cos(M_PI * 11.0 / 6.0), sin(M_PI * 11.0 / 6.0));

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
                                       MatrixXi &F, MatrixXT &Fc, int colormode = 0) {
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);
    long n_vtx = vertices.rows() / 2;

    // Overlay points and edges
    S.resize(n_vtx, 3);
    S.setZero();
    for (int i = 0; i < n_vtx; i++) {
        S.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }

    std::vector<TV> face0;
    std::vector<TV> face1;
    std::vector<TV> face2;

    int n_cells = info->n_free;
    VectorXT areas = VectorXT::Zero(n_cells);

    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    int dims = 2 + info->getTessellation()->getNumVertexParams();

    for (int i = 0; i < n_cells; i++) {
        Cell cell = info->getTessellation()->cells[i];
        size_t degree = cell.edges.size();

        TV v0 = vertices.segment<2>(i * 2);

        for (size_t j = 0; j < degree; j++) {
            TV v1 = info->getTessellation()->x.segment<2>(cell.edges[j].startNode * 2);
            TV v2 = info->getTessellation()->x.segment<2>(cell.edges[cell.edges[j].nextEdge].startNode * 2);

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

        if (colormode == 0) {
            Fc.row(i) = (currentCell == info->selected ? TV3(0.4, 0.4, 0.4) : getColor(areas(currentCell),
                                                                                       info->energy_area_targets(
                                                                                               currentCell)));
        } else {
            Fc.row(i) = currentCell % 2 == 0 ? TV3(1.0, 0.6, 0.6) : TV3(0.6, 1.0, 0.6);
        }

//        double cc = currentCell * 1.0 / info->n_free;
//        Fc.row(i) = TV3(cc, cc, cc);

        currentIdxInCell++;
        if (currentIdxInCell == info->getTessellation()->cells[currentCell].edges.size()) {
            currentIdxInCell = 0;
            currentCell++;
        }
    }

    X = V;
    S.col(2).setConstant(2e-6);
    X.col(2).setConstant(2e-6);

    Sc.resize(S.rows(), 3);
    Sc.setZero();

    Ec.resize(E.rows(), 3);
    Ec.setZero();

    if (info->selected >= 0) {
        MatrixXT V_target;
        V_target.resize(3, 3);
        V_target.col(2) = TV3(0.01, 0.01, 0.01);

        V_target.row(0).segment<2>(0) = info->selected_target_pos + 0.03 * TV(cos(M_PI_2), sin(M_PI_2));
        V_target.row(1).segment<2>(0) =
                info->selected_target_pos + 0.03 * TV(cos(M_PI * 7.0 / 6.0), sin(M_PI * 7.0 / 6.0));
        V_target.row(2).segment<2>(0) =
                info->selected_target_pos + 0.03 * TV(cos(M_PI * 11.0 / 6.0), sin(M_PI * 11.0 / 6.0));

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
    MatrixXT S_traj(info->trajOpt_N + 1, 3);
    S_traj.setZero();

    int dims = info->getTessellation()->getNumVertexParams() + 2;
    S_traj(0, 0) = trajOptNLP.c0(info->selected * dims + 0);
    S_traj(0, 1) = trajOptNLP.c0(info->selected * dims + 1);
    for (int k = 0; k < info->trajOpt_N; k++) {
        S_traj(k + 1, 0) = trajOptNLP.x_sol(k * trajOptNLP.c0.rows() + info->selected * dims + 0);
        S_traj(k + 1, 1) = trajOptNLP.x_sol(k * trajOptNLP.c0.rows() + info->selected * dims + 1);
    }

    MatrixXT S_temp = S;
    S.resize(S_temp.rows() + S_traj.rows(), 3);
    S << S_temp, S_traj;

    MatrixXT X_temp = X;
    X.resize(X_temp.rows() + S_traj.rows(), 3);
    X << X_temp, S_traj;

    MatrixXT Sc_traj = TV3(1, 0, 0).transpose().replicate(info->trajOpt_N + 1, 1);
    MatrixXT Sc_temp = Sc;
    Sc.resize(Sc_temp.rows() + Sc_traj.rows(), 3);
    Sc << Sc_temp, Sc_traj;

    MatrixXi E_traj(info->trajOpt_N, 2);
    for (int k = 0; k < info->trajOpt_N; k++) {
        E_traj(k, 0) = X_temp.rows() + k + 0;
        E_traj(k, 1) = X_temp.rows() + k + 1;
    }

    MatrixXi E_temp = E;
    E.resize(E_temp.rows() + E_traj.rows(), 2);
    E << E_temp, E_traj;

    MatrixXT Ec_traj = TV3(1, 0, 0).transpose().replicate(info->trajOpt_N, 1);
    MatrixXT Ec_temp = Ec;
    Ec.resize(Ec_temp.rows() + Ec_traj.rows(), 3);
    Ec << Ec_temp, Ec_traj;
}

void Foam2D::getPlotAreaHistogram(VectorXT &areas) {
    info->getTessellation()->tessellate(vertices, params, info->boundary, info->n_free);

    int n_cells = info->n_free;
    areas.resize(n_cells);
    areas.setZero();

    int n_vtx = vertices.rows() / 2, n_bdy = info->boundary->v.rows() / 2;

    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    int dims = 2 + info->getTessellation()->getNumVertexParams();

    for (int i = 0; i < n_cells; i++) {
        Cell cell = info->getTessellation()->cells[i];
        size_t degree = cell.edges.size();

        TV v0 = vertices.segment<2>(i * 2);

        for (size_t j = 0; j < degree; j++) {
            TV v1 = info->getTessellation()->x.segment<2>(cell.edges[j].startNode * 2);
            TV v2 = info->getTessellation()->x.segment<2>(cell.edges[cell.edges[j].nextEdge].startNode * 2);

            areas(i) += 0.5 * ((v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1)));
        }
    }

    for (int i = 0; i < areas.rows(); i++) {
        areas(i) /= info->energy_area_targets(i);
    }
}

bool Foam2D::isConvergedDynamic(double tol) {
    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0, info->n_free * dims);
    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    return dynamicObjective.getGradient(y).norm() < tol;
}

void Foam2D::getPlotObjectiveStats(bool dynamics, double &obj_value, double &gradient_norm, bool &hessian_pd) {
    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0, info->n_free * dims);
    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    Eigen::SparseMatrix<double> hessian;
    if (dynamics) {
        obj_value = dynamicObjective.evaluate(y);
        gradient_norm = dynamicObjective.getGradient(y).norm();
        dynamicObjective.getHessian(y, hessian);
    } else {
        obj_value = energyObjective.evaluate(y);
        gradient_norm = energyObjective.getGradient(y).norm();
        energyObjective.getHessian(y, hessian);
    }
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> solver(hessian);
    hessian_pd = solver.info() != Eigen::ComputationInfo::NumericalIssue;
}

void
Foam2D::getPlotObjectiveFunctionLandscape(int selected_vertex, int type, int image_size, double range, VectorXf &obj,
                                          double &obj_min, double &obj_max) {
    int dims = 2 + info->getTessellation()->getNumVertexParams();
    VectorXT c = info->getTessellation()->combineVerticesParams(vertices, params);
    VectorXT c_free = c.segment(0, info->n_free * dims);
    VectorXT p_free = info->boundary->get_p_free();
    VectorXT y(c_free.rows() + p_free.rows());
    y << c_free, p_free;

    obj.resize(image_size * image_size * 3);

    obj_max = 0;
    obj_min = INFINITY;

    int xindex = selected_vertex * dims + 0;
    int yindex = selected_vertex * dims + 1;
    VectorXT DX = VectorXT::Zero(y.rows());
    DX(xindex) = 1;
    VectorXT DY = VectorXT::Zero(y.rows());
    DY(yindex) = 1;
    for (int i = 0; i < image_size; i++) {
        for (int j = 0; j < image_size; j++) {
            double dx = (double) (j - image_size / 2) / image_size * range;
            double dy = (double) -(i - image_size / 2) / image_size * range;
            double o;

            switch (type) {
                case 0:
                    o = energyObjective.evaluate(y + dx * DX + dy * DY);
                    break;
                case 1:
                    o = energyObjective.get_dOdc(y + dx * DX + dy * DY)(xindex);
                    break;
                case 2:
                    o = energyObjective.get_dOdc(y + dx * DX + dy * DY)(yindex);
                    break;
                case 3:
                    o = imageMatchObjective.evaluate(y + dx * DX + dy * DY);
                    break;
                case 4:
                    o = imageMatchObjective.get_dOdc(y + dx * DX + dy * DY)(xindex);
                    break;
                case 5:
                    o = imageMatchObjective.get_dOdc(y + dx * DX + dy * DY)(yindex);
                    break;
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
