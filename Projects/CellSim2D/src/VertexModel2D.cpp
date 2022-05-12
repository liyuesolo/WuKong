#include <Eigen/PardisoSupport>
#include <igl/readOBJ.h>
#include <iomanip>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include "../include/IMLS.h"

#include "../include/VertexModel2D.h"
#include "../include/autodiff/EdgeEnergy.h"
#include "../include/autodiff/AreaEnergy.h"
#include "../include/autodiff/MembraneEnergy.h"

void VertexModel2D::initializeScene()
{
    
    T unit = 1e-2;
    // 80 cells 200 µm diameter, 35 µm height, 7.85 µm width
    T r = 100 * unit; // 2 pi r = 200 
    radius = r + 0.1 * unit;
    int n_div = 80;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 2);
    for(int i = 0; i < n_div; i++)
        points.segment<2>(i * 2) = TV(r * std::cos(theta * T(i)), 
                                        r * std::sin(theta*T(i)));
    undeformed = points;
    // apical edges
    int n_pt = points.rows() / 2;
    basal_vtx_start = n_pt;
    for (int i = 0; i < n_pt; i++)
    {
        int j = (i + 1) % n_pt;
        edges.push_back(Edge(i, j));
    }
    basal_edge_start = edges.size();

    T edge_length = 35 * unit;
    
    undeformed.conservativeResize(n_pt * 2 * 2);

    for (int i = 0; i < n_pt; i++)
    {
        TV pt = points.segment<2>(i * 2);
        TV pt_inner = pt - pt.normalized() * edge_length;
        undeformed.segment<2>(basal_vtx_start * 2 + i * 2) = pt_inner;
    }

    for (int i = 0; i < n_pt; i++)
    {
        int j = (i + 1) % n_pt;
        edges.push_back(Edge(i + basal_vtx_start, j + basal_vtx_start));
        inner_edges.push_back(Edge(i, j + basal_vtx_start));
        inner_edges.push_back(Edge(j, i + basal_vtx_start));
    }
    lateral_edge_start = edges.size();
    for (int i = 0; i < n_pt; i++)
    {
        edges.push_back(Edge(i, i + basal_vtx_start));
    }

    deformed = undeformed;
    n_cells = basal_edge_start;
    u = VectorXT::Zero(deformed.rows());
    u0 = VectorXT::Zero(deformed.rows());
    num_nodes = deformed.rows() / 2;
    mesh_centroid = TV::Zero();

    for (int i = basal_vtx_start * 2; i < basal_vtx_start * 2 + 4; i++)
    {
        dirichlet_data[i] = 0;
    }
    
    // mild buckling n_div == 50
    int scene = 1;
    has_rest_state = true;
    if (has_rest_state)
    {
        w_ea = 1.0;
        w_eb = 0.2;
        w_el = 4.0;
        w_c = 20.0;
    }
    else
    {
        w_ea = 20.0;
        w_eb = 0.2;
        w_el = 4.0;
        w_c = 20.0;
    }
    
    w_a = 1e7;
    w_mb = 1e5;
    w_yolk = 1e5;
    barrier_weight = 1e-22;

    yolk_area_rest = computeYolkArea();
    
    
    use_ipc = true;
    add_ipc_friction = false;
    if (use_ipc)
    {
        ipc_barrier_weight = 1e7;
        ipc_barrier_distance = 1e-3;
        buildIPCRestData();
    }
    if (has_rest_state)
        computeRestLength();
    add_inner_edges = false;

    configContractingWeights();
}


void VertexModel2D::computeRestLength()
{
    rest_edge_length.resize(edges.size());
    for (int i = 0; i < edges.size(); i++)
    {
        TV vi = deformed.segment<2>(edges[i][0] * 2);
        TV vj = deformed.segment<2>(edges[i][1] * 2);
        rest_edge_length[i] = 1.0 * (vi - vj).norm();
    }
}

void VertexModel2D::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool show_current, bool show_rest)
{
    V.resize(0, 0); F.resize(0, 0); C.resize(0, 0);
    
    std::vector<std::pair<TV, TV>> edge_pairs;
    for (int i = 0; i < edges.size(); i++)
    {
        TV vi = deformed.segment<2>(edges[i][0] * 2);
        TV vj = deformed.segment<2>(edges[i][1] * 2);
        edge_pairs.push_back(std::make_pair(vi, vj));
    }

    TV v0 = deformed.segment<2>(edges[0][0] * 2);
    TV v1 = deformed.segment<2>(edges[0][1] * 2);
    
    T edge_length = (v1 - v0).norm();

    appendCylindersToEdges(edge_pairs, TV3(0, 0.3, 1), 0.05 * edge_length, V, F, C);
}

void VertexModel2D::appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV3(radius * std::cos(theta * T(i)), 
        0.0, radius * std::sin(theta*T(i)));

    int offset_v = n_div * 2;
    int offset_f = n_div * 2;

    int n_row_V = _V.rows();
    int n_row_F = _F.rows();

    int n_edge = edge_pairs.size();

    _V.conservativeResize(n_row_V + offset_v * n_edge, 3);
    _F.conservativeResize(n_row_F + offset_f * n_edge, 3);
    _C.conservativeResize(n_row_F + offset_f * n_edge, 3);

    tbb::parallel_for(0, n_edge, [&](int ei)
    {
        
        TV3 v0(edge_pairs[ei].first[0], edge_pairs[ei].first[1], 0);
        TV3 v1(edge_pairs[ei].second[0], edge_pairs[ei].second[1], 0);

        TV3 axis_world = v1 - v0;
        TV3 axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                _V(n_row_V + ei * offset_v + i, d) = points[i * 3 + d];
                _V(n_row_V + ei * offset_v + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    _V(n_row_V + ei * offset_v + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            _V.row(n_row_V + ei * offset_v + i) = (_V.row(n_row_V + ei * offset_v + i) * R).transpose() + v0;
            _V.row(n_row_V + ei * offset_v + i + n_div) = (_V.row(n_row_V + ei * offset_v + i + n_div) * R).transpose() + v0;

            _F.row(n_row_F + ei * offset_f + i*2 ) = IV3(n_row_V + ei * offset_v + i, 
                                    n_row_V + ei * offset_v + i+n_div, 
                                    n_row_V + ei * offset_v + (i+1)%(n_div));

            _F.row(n_row_F + ei * offset_f + i*2 + 1) = IV3(n_row_V + ei * offset_v + (i+1)%(n_div), 
                                        n_row_V + ei * offset_v + i+n_div, 
                                        n_row_V + + ei * offset_v + (i+1)%(n_div) + n_div);

            _C.row(n_row_F + ei * offset_f + i*2 ) = color[ei];
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color[ei];
        }
    });
}

void VertexModel2D::appendCylindersToEdges(const std::vector<std::pair<TV, TV>>& edge_pairs, 
        const TV3& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C)
{
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV3(radius * std::cos(theta * T(i)), 
        0.0, radius * std::sin(theta*T(i)));

    int offset_v = n_div * 2;
    int offset_f = n_div * 2;

    int n_row_V = _V.rows();
    int n_row_F = _F.rows();

    int n_edge = edge_pairs.size();

    _V.conservativeResize(n_row_V + offset_v * n_edge, 3);
    _F.conservativeResize(n_row_F + offset_f * n_edge, 3);
    _C.conservativeResize(n_row_F + offset_f * n_edge, 3);

    tbb::parallel_for(0, n_edge, [&](int ei)
    {
        
        TV3 v0(edge_pairs[ei].first[0], edge_pairs[ei].first[1], 0);
        TV3 v1(edge_pairs[ei].second[0], edge_pairs[ei].second[1], 0);

        TV3 axis_world = v1 - v0;
        TV3 axis_local(0, axis_world.norm(), 0);

        Matrix<T, 3, 3> R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();

        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                _V(n_row_V + ei * offset_v + i, d) = points[i * 3 + d];
                _V(n_row_V + ei * offset_v + i+n_div, d) = points[i * 3 + d];
                if (d == 1)
                    _V(n_row_V + ei * offset_v + i+n_div, d) += axis_world.norm();
            }

            // central vertex of the top and bottom face
            _V.row(n_row_V + ei * offset_v + i) = (_V.row(n_row_V + ei * offset_v + i) * R).transpose() + v0;
            _V.row(n_row_V + ei * offset_v + i + n_div) = (_V.row(n_row_V + ei * offset_v + i + n_div) * R).transpose() + v0;

            _F.row(n_row_F + ei * offset_f + i*2 ) = IV3(n_row_V + ei * offset_v + i, 
                                    n_row_V + ei * offset_v + i+n_div, 
                                    n_row_V + ei * offset_v + (i+1)%(n_div));

            _F.row(n_row_F + ei * offset_f + i*2 + 1) = IV3(n_row_V + ei * offset_v + (i+1)%(n_div), 
                                        n_row_V + ei * offset_v + i+n_div, 
                                        n_row_V + + ei * offset_v + (i+1)%(n_div) + n_div);

            _C.row(n_row_F + ei * offset_f + i*2 ) = color;
            _C.row(n_row_F + ei * offset_f + i*2 + 1) = color;
        }
    });
}

void VertexModel2D::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    std::vector<Entry> entries;
    
    addEdgeHessianEntries(Apical, w_ea, entries);
    addEdgeHessianEntries(Basal, w_eb, entries);
    addEdgeHessianEntries(Lateral, w_el, entries);

    addAreaPreservationHessianEntries(w_a, entries);

    addMembraneBoundHessianEntries(w_mb, entries);

    addYolkPreservationHessianEntries(w_yolk, entries);

    addContractingHessianEntries(entries);

    addAreaBarrierHessianEntries(barrier_weight, entries);

    if (use_ipc)
        addIPCHessianEntries(entries);

    if (add_inner_edges)
        addEdgeHessianEntries(INNER, w_ei, entries);
    if (add_yolk_pressure)
        addPressureHessianEntries(entries);

    K.resize(num_nodes * 2, num_nodes * 2);
    K.setFromTriplets(entries.begin(), entries.end());
    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    K.makeCompressed();
}

void VertexModel2D::projectDirichletDoFMatrix(StiffnessMatrix& A, 
    const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

T VertexModel2D::computeTotalEnergy(const VectorXT& _u)
{
    T energy = 0.0;
    
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    
    deformed = undeformed + projected;

    T edge_term = 0.0, area_term = 0.0, mem_term = 0.0,
        yolk_term = 0.0, contracting_term = 0.0, area_barrier_term = 0.0;

    addEdgeEnergy(Apical, w_ea, edge_term);
    addEdgeEnergy(Basal, w_eb, edge_term);
    addEdgeEnergy(Lateral, w_el, edge_term);
    energy += edge_term;

    addAreaPreservationEnergy(w_a, area_term);
    energy += area_term;

    addMembraneBoundTerm(w_mb, mem_term);
    energy += mem_term;

    addYolkPreservationEnergy(w_yolk, yolk_term);
    energy += yolk_term;
    
    addContractingEnergy(contracting_term);
    energy += contracting_term;

    addAreaBarrierEnergy(barrier_weight, area_barrier_term);
    energy += area_barrier_term;

    if (use_ipc)
    {
        T ipc_energy = 0.0;
        addIPCEnergy(ipc_energy);
        energy += ipc_energy;
    }

    if (add_inner_edges)
    {
        T inner_edge_energy = 0.0;
        addEdgeEnergy(INNER, w_ei, inner_edge_energy);
        energy += inner_edge_energy;
    }

    if (add_yolk_pressure)
    {
        T pV = 0.0;
        addPressureEnergy(pV);
        energy += pV;
    }

    return energy;
}

T VertexModel2D::computeResidual(const VectorXT& _u,  VectorXT& residual)
{
    VectorXT residual_temp = residual;
    VectorXT projected = _u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
        
    deformed = undeformed + projected;

    addEdgeForceEntries(Apical, w_ea, residual);
    addEdgeForceEntries(Basal, w_eb, residual);
    addEdgeForceEntries(Lateral, w_el, residual);

    if (add_inner_edges)
    {
        addEdgeForceEntries(INNER, w_ei, residual);
    }

    if (verbose)
        std::cout << "edge force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    addAreaPreservationForceEntries(w_a, residual);

    if (verbose)
        std::cout << "area force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    addMembraneBoundForceEntries(w_mb, residual);

    if (verbose)
        std::cout << "mem force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    addYolkPreservationForceEntries(w_yolk, residual);

    if (verbose)
        std::cout << "yolk force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    addContractingForceEntries(residual);

    if (verbose)
        std::cout << "contract force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;
    
    addAreaBarrierForceEntries(barrier_weight, residual);
    if (verbose)
        std::cout << "barrier force norm: " << (residual - residual_temp).norm() << std::endl;
    residual_temp = residual;

    if (use_ipc)
    {
        addIPCForceEntries(residual);
        if (verbose)
            std::cout << "ipc force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (add_yolk_pressure)
    {
        addPressureForceEntries(residual);
        if (verbose)
            std::cout << "pressure force norm: " << (residual - residual_temp).norm() << std::endl;
        residual_temp = residual;
    }

    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });

    return residual.norm();
}

void VertexModel2D::loadEdgeWeights(const std::string& filename, VectorXT& weights)
{
    std::ifstream in(filename);
    std::vector<T> weights_std_vec;
    T w;
    while (in >> w)
        weights_std_vec.push_back(w);
    weights = Eigen::Map<VectorXT>(weights_std_vec.data(), weights_std_vec.size());
    in.close();
}

T VertexModel2D::computeAreaInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    int cnt = 0;
    
    auto unsignedArea = [&](TV& a, TV& b, TV& c)->T
    {
        TV3 v0(a[0], a[1], 0.0);
        TV3 v1(b[0], b[1], 0.0);
        TV3 v2(c[0], c[1], 0.0);

        return 0.5 * (v1 - v0).cross(v2 - v0)[2];
    };

    auto computeCellSubArea = [&]()
    {
        std::vector<Vector<T, 4>> cell_sub_area;
        iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            VectorXT positions;
            positionsFromIndices(positions, indices);
            int n_pt = indices.size();
            TV centroid;
            computeCellCentroid(cell_idx, centroid);
            Vector<T, 4> sub_area;
            for (int i = 0; i < n_pt; i++)
            {
                int j = (i + 1) % n_pt;
                TV r0 = positions.segment<2>(i * 2);
                TV r1 = positions.segment<2>(j * 2);
                T area = unsignedArea(r0, r1, centroid);
                sub_area[i] = area;
            }
            cell_sub_area.push_back(sub_area);
        });
        for (auto sub_area : cell_sub_area)
        {
            std::cout << sub_area.transpose() << std::endl;
        }
        
    };
    // computeCellSubArea();
    while (true)
    {
        deformed = undeformed + _u + step_size * du;
        bool constraint_violated = false;
        if (cnt > 60)
        {
            std::cout << "unable to find inversion free state " << std::endl;
            std::exit(0);
        }
        iterateCellSerial([&](VtxList& indices, int cell_idx)
        {
            if (constraint_violated)
                return;
            VectorXT positions;
            positionsFromIndices(positions, indices);
            int n_pt = indices.size();
            TV centroid;
            computeCellCentroid(cell_idx, centroid);
            for (int i = 0; i < n_pt; i++)
            {
                int j = (i + 1) % n_pt;
                TV r0 = positions.segment<2>(i * 2);
                TV r1 = positions.segment<2>(j * 2);
                T area = unsignedArea(r0, r1, centroid);
                // std::cout << computeSignedTriangleArea(r0, r1, centroid) << std::endl;
                // std::cout << "area " << area << std::endl;
                if (area < 1e-9)
                {
                    constraint_violated = true;
                    break;
                }
            }
        });

        if (constraint_violated)
        {
            step_size *= 0.8;
            cnt++;
        }
        else
            return step_size;
    }
}

T VertexModel2D::computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du)
{
    T step_size = 1.0;
    step_size = std::min(computeAreaInversionFreeStepsize(_u, du), step_size);
    
    if (use_ipc)
        step_size = std::min(computeCollisionFreeStepsize(_u, du), step_size);

    if (verbose)
        std::cout << "step size: " << step_size << std::endl;
    return step_size;
}

T VertexModel2D::lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());

    buildSystemMatrix(_u, K);

    bool success = linearSolve(K, residual, du); 

    T norm = du.norm();
    T alpha = computeLineSearchInitStepsize(_u, du);
    T E0 = computeTotalEnergy(_u);
    
    int cnt = 1;

    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > ls_max)
        {
            _u = u_ls;
            if (verbose)
                std::cout << "\tls " << cnt << "/" << ls_max << std::endl;
            if (cnt > ls_max)
            {
                // saveStates("ls_failure.obj");
                // checkTotalGradientScale();
                // checkTotalHessianScale();
                // // checkTotalHessian();
                // std::getchar();
            }
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }

    return norm;
}

bool VertexModel2D::advanceOneStep(int step)
{
    VectorXT residual(deformed.rows());
    residual.setZero();
    
    if (use_ipc)
        updateIPCVertices(u);

    T residual_norm = computeResidual(u, residual);
    std::cout << "[Newton] iter " << step << "/" << max_newton_iter << " |g|: " << residual_norm << std::endl;
    if (residual_norm < newton_tol)
        return true;

    T dq_norm = lineSearchNewton(u, residual);
    checkFunctionPerIteration(step);
    if (save_mesh)
        saveStates(data_folder + "iter_" + std::to_string(step) + ".obj");
    if(step == max_newton_iter || dq_norm > 1e10)
    {
        return true;
    }
    
    return false;    
}

bool VertexModel2D::linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du)
{
    Eigen::PardisoLLT<StiffnessMatrix> solver;

    T alpha = 10e-6;

    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha; 
            alpha *= 10;
            continue;
        }
        
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\t# regularization step " << i << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha; 
            alpha *= 10;
        }
    }
    return false;
}

void VertexModel2D::addContractingEnergy(T& energy)
{
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        T ei;
        // if (has_rest_state)
        // {
        //     computeEdgeSpringEnergy2D(contracting_weight, contracting_percentage * rest_edge_length[edge_id], vi, vj, ei);
        //     energy += contracting_mask[cnt] * ei;
        // }
        // else
        {
            computeEdgeSquaredNorm2D(vi, vj, ei);
            energy += apical_edge_contracting_weights[cnt] * ei;
        }
        cnt++;
    });
}

void VertexModel2D::addContractingForceEntries(VectorXT& residual)
{
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Vector<T, 4> dedx;
        // if (has_rest_state)
        // {
        //     computeEdgeSpringEnergy2DGradient(contracting_weight, contracting_percentage * rest_edge_length[edge_id], vi, vj, dedx);
        //     dedx *= -contracting_mask[cnt];
        // }
        // else
        {
            computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
            dedx *= -apical_edge_contracting_weights[cnt];
        }
        addForceEntry<4>(residual, {e[0], e[1]}, dedx);
        cnt++;
    });
}

void VertexModel2D::addContractingHessianEntries(std::vector<Entry>& entries)
{
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Matrix<T, 4, 4> hessian;
        // if (has_rest_state)
        // {
        //     computeEdgeSpringEnergy2DHessian(contracting_weight, contracting_percentage * rest_edge_length[edge_id], vi, vj, hessian);
        //     hessian *= contracting_mask[cnt];
        // }
        // else
        {
            computeEdgeSquaredNorm2DHessian(vi, vj, hessian);
            hessian *= apical_edge_contracting_weights[cnt];
        }
        addHessianEntry<4>(entries, {e[0], e[1]}, hessian);
        cnt++;
    });
}

void VertexModel2D::addEdgeEnergy(Region region, T w, T& energy)
{
    if (region == Apical)
    {
        iterateApicalEdgeSerial([&](Edge& edge, int edge_id){
            TV vi = deformed.segment<2>(edge[0] * 2);
            TV vj = deformed.segment<2>(edge[1] * 2);
            T ei;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2D(w, rest_edge_length[edge_id], vi, vj, ei);
                energy += ei;
            }
            else
            {
                computeEdgeSquaredNorm2D(vi, vj, ei);
                energy += w * ei;
            }
        });
    }
    else if (region == Basal)
    {
        iterateBasalEdgeSerial([&](Edge& edge, int edge_id){
            TV vi = deformed.segment<2>(edge[0] * 2);
            TV vj = deformed.segment<2>(edge[1] * 2);
            T ei;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2D(w, rest_edge_length[edge_id], vi, vj, ei);
                energy += ei;
            }
            else
            {
                computeEdgeSquaredNorm2D(vi, vj, ei);
                energy += w * ei;
            }
        });
    }
    else if (region == Lateral)
    {
        iterateLateralEdgeSerial([&](Edge& edge, int edge_id){
            TV vi = deformed.segment<2>(edge[0] * 2);
            TV vj = deformed.segment<2>(edge[1] * 2);
            T ei;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2D(w, rest_edge_length[edge_id], vi, vj, ei);
                energy += ei;
            }
            else
            {
                computeEdgeSquaredNorm2D(vi, vj, ei);
                energy += w * ei;
            }
        });
    }
}

void VertexModel2D::addEdgeForceEntries(Region region, T w, VectorXT& residual)
{
    if (region == Apical)
    {
        iterateApicalEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Vector<T, 4> dedx;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DGradient(w, rest_edge_length[edge_id], vi, vj, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<4>(residual, {e[0], e[1]}, dedx);
        });
    }
    else if (region == Basal)
    {
        iterateBasalEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Vector<T, 4> dedx;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DGradient(w, rest_edge_length[edge_id], vi, vj, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<4>(residual, {e[0], e[1]}, dedx);
        });
    }
    else if (region == Lateral)
    {
        iterateLateralEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Vector<T, 4> dedx;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DGradient(w, rest_edge_length[edge_id], vi, vj, dedx);
                dedx *= -1.0;
            }
            else
            {
                computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
                dedx *= -w;
            }
            addForceEntry<4>(residual, {e[0], e[1]}, dedx);
        });
    }
}

void VertexModel2D::addEdgeHessianEntries(Region region, T w, std::vector<Entry>& entries)
{
    if (region == Apical)
    {
        iterateApicalEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Matrix<T, 4, 4> hessian;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DHessian(w, rest_edge_length[edge_id], vi, vj, hessian);
            }
            else
            {
                computeEdgeSquaredNorm2DHessian(vi, vj, hessian);
                hessian *= w;
            }
            addHessianEntry<4>(entries, {e[0], e[1]}, hessian);
        });
    }
    else if (region == Basal)
    {
        iterateBasalEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Matrix<T, 4, 4> hessian;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DHessian(w, rest_edge_length[edge_id], vi, vj, hessian);
            }
            else
            {
                computeEdgeSquaredNorm2DHessian(vi, vj, hessian);
                hessian *= w;
            }
            addHessianEntry<4>(entries, {e[0], e[1]}, hessian);
        });
    }
    else if (region == Lateral)
    {
        iterateLateralEdgeSerial([&](Edge& e, int edge_id){
            TV vi = deformed.segment<2>(e[0] * 2);
            TV vj = deformed.segment<2>(e[1] * 2);
            Matrix<T, 4, 4> hessian;
            if (has_rest_state)
            {
                computeEdgeSpringEnergy2DHessian(w, rest_edge_length[edge_id], vi, vj, hessian);
            }
            else
            {
                computeEdgeSquaredNorm2DHessian(vi, vj, hessian);
                hessian *= w;
            }
            addHessianEntry<4>(entries, {e[0], e[1]}, hessian);
        });
    }
}

void VertexModel2D::positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state)
{
    positions = VectorXT::Zero(indices.size() * 2);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<2>(i * 2) = rest_state ? undeformed.segment<2>(indices[i] * 2) : deformed.segment<2>(indices[i] * 2);
    }
}

T VertexModel2D::computeCellArea(int cell_idx, bool rest)
{
    VtxList indices;
    getCellVtxIndices(indices, cell_idx);
    VectorXT positions;
    positionsFromIndices(positions, indices, rest);
    T area = 0.0;
    TV centroid;
    computeCellCentroid(cell_idx, centroid, rest);
    for (int i = 0; i < indices.size(); i++)
    {
        int j = (i + 1) % indices.size();
        TV vi = positions.segment<2>(i * 2);
        TV vj = positions.segment<2>(j * 2);
        T ai;
        computeTriangleArea(vi, vj, centroid, ai);
        area += ai;
    }
    return area;
}

void VertexModel2D::checkFinalState()
{
    T total_area = computeYolkArea();
    std::cout << "initial yolk area: " << yolk_area_rest << " final yolk area: " << total_area << std::endl;
    
    VectorXT cell_areas_final(n_cells), cell_areas_init(n_cells);
    int cnt = 0;
    T avg_compression = 0.0;
    iterateCellParallel([&](VtxList& indices, int cell_idx)
    {
        cell_areas_final[cell_idx] = computeCellArea(cell_idx);
        cell_areas_init[cell_idx] = computeCellArea(cell_idx, /*rest_state = */ true);
        if (cell_areas_final[cell_idx] < cell_areas_init[cell_idx])
        {
            cnt++;
            avg_compression += cell_areas_final[cell_idx] - cell_areas_init[cell_idx];
        }
    });
    avg_compression /= T(cnt);
    std::cout << "initial cell area: " << cell_areas_init.sum() << " final cell area: " << cell_areas_final.sum() << std::endl;
    std::cout << cnt << "/" << n_cells << " are compressed "
        << "avg cell compression: " << avg_compression << std::endl;
}

void VertexModel2D::addAreaPreservationEnergy(T w, T& energy)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions, positions_rest;
        positionsFromIndices(positions, indices);
        positionsFromIndices(positions_rest, indices, true);
        T ei;
        // computeArea4PointsSquared2D(w, positions, positions_rest, ei);
        computeArea4PointsLineIntegral(w, positions, positions_rest, ei);
        energy += ei;
    });
}

void VertexModel2D::addAreaPreservationForceEntries(T w, VectorXT& residual)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions, positions_rest;
        positionsFromIndices(positions, indices);
        positionsFromIndices(positions_rest, indices, true);
        Vector<T, 8> dedx;
        // computeArea4PointsSquared2DGradient(w, positions, positions_rest, dedx);
        computeArea4PointsLineIntegralGradient(w, positions, positions_rest, dedx);
        dedx *= -1;
        addForceEntry<8>(residual, indices, dedx);
    });
}

void VertexModel2D::addAreaPreservationHessianEntries(T w, std::vector<Entry>& entries)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions, positions_rest;
        positionsFromIndices(positions, indices);
        positionsFromIndices(positions_rest, indices, true);
        Matrix<T, 8, 8> hessian;
        // computeArea4PointsSquared2DHessian(w, positions, positions_rest, hessian);
        computeArea4PointsLineIntegralHessian(w, positions, positions_rest, hessian);
        addHessianEntry<8>(entries, indices, hessian);
    });
}

void VertexModel2D::addMembraneBoundTerm(T w, T& energy)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<2>(i * 2);
        T dis = (xi - mesh_centroid).norm();
        if (dis <= radius)
            continue;
        T ei;
        computeMembraneQubicPenalty(w, radius, xi, mesh_centroid, ei);
        energy += ei;
    }
    
}

void VertexModel2D::addMembraneBoundForceEntries(T w, VectorXT& residual)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<2>(i * 2);
        T dis = (xi - mesh_centroid).norm();
        if (dis <= radius)
            continue;
        Vector<T, 2> dedx;
        computeMembraneQubicPenaltyGradient(w, radius, xi, mesh_centroid, dedx);
        addForceEntry<2>(residual, {i}, -dedx);
    }
}

void VertexModel2D::addMembraneBoundHessianEntries(T w, std::vector<Entry>& entries)
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV xi = deformed.segment<2>(i * 2);
        T dis = (xi - mesh_centroid).norm();
        if (dis <= radius)
            continue;
        Matrix<T, 2, 2> hessian;
        computeMembraneQubicPenaltyHessian(w, radius, xi, mesh_centroid, hessian);
        addHessianEntry<2>(entries, {i}, hessian);
    }
}

T VertexModel2D::computeYolkArea()
{
    T total_area = 0.0;
    int n_nodes_basal = num_nodes / 2;
    TV3 center_3D = TV3::Zero();
    center_3D.segment<2>(0) = mesh_centroid;
    for (int i = 0; i < n_nodes_basal; i++)
    {
        int j = (i + 1) % n_nodes_basal;
        TV vi = deformed.segment<2>(basal_vtx_start * 2 + i * 2);
        TV vj = deformed.segment<2>(basal_vtx_start * 2 + j * 2);

        // TV3 vi_3D = TV3::Zero(), vj_3D = TV3::Zero();
        // vi_3D.segment<2>(0) = vi; vj_3D.segment<2>(0) = vj;
        // T area = 0.5 * (vj_3D - center_3D).cross(vi_3D - center_3D).norm();
        T area;
        computeSignedTriangleArea(vi, vj, mesh_centroid, area);
        total_area += area;
    }
    return total_area;
}

void VertexModel2D::addPressureEnergy(T& energy)
{   
    T total_area = computeYolkArea();
    energy += -pressure * total_area;
}

void VertexModel2D::addPressureForceEntries(VectorXT& residual)
{
    T total_area = computeYolkArea();
    int n_nodes_basal = num_nodes / 2;

    for (int i = 0; i < n_nodes_basal; i++)
    {
        int j = (i + 1) % n_nodes_basal;
        TV vi = deformed.segment<2>(basal_vtx_start * 2 + i * 2);
        TV vj = deformed.segment<2>(basal_vtx_start * 2 + j * 2);

        Vector<T, 4> dedx;
        // computeTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        computeSignedTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        addForceEntry<4>(residual, {basal_vtx_start + i, basal_vtx_start + j}, pressure * dedx);
    }
}

void VertexModel2D::addPressureHessianEntries(std::vector<Entry>& entries)
{
    int n_nodes_basal = num_nodes / 2;

    for (int i = 0; i < n_nodes_basal; i++)
    {
        int j = (i + 1) % n_nodes_basal;
        TV vi = deformed.segment<2>(basal_vtx_start * 2 + i * 2);
        TV vj = deformed.segment<2>(basal_vtx_start * 2 + j * 2);

        Matrix<T, 4, 4> d2edx2;
        computeSignedTriangleAreaHessian(vi, vj, mesh_centroid, d2edx2);

        Matrix<T, 4, 4> hessian = -pressure * d2edx2;
        addHessianEntry(entries, {basal_vtx_start + i, basal_vtx_start + j}, hessian);
    }
}

void VertexModel2D::addYolkPreservationEnergy(T w, T& energy)
{
    T total_area = computeYolkArea();
    energy += 0.5 * w * (total_area - yolk_area_rest) * (total_area - yolk_area_rest);
}

void VertexModel2D::addYolkPreservationForceEntries(T w, VectorXT& residual)
{
    T total_area = computeYolkArea();
    int n_nodes_basal = num_nodes / 2;
    T de_dsum = w * (total_area - yolk_area_rest);
    for (int i = 0; i < n_nodes_basal; i++)
    {
        int j = (i + 1) % n_nodes_basal;
        TV vi = deformed.segment<2>(basal_vtx_start * 2 + i * 2);
        TV vj = deformed.segment<2>(basal_vtx_start * 2 + j * 2);

        Vector<T, 4> dedx;
        // computeTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        computeSignedTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        addForceEntry<4>(residual, {basal_vtx_start + i, basal_vtx_start + j}, -de_dsum * dedx);
    }
}

void VertexModel2D::addYolkPreservationHessianEntries(T w, std::vector<Entry>& entries)
{
    T total_area = computeYolkArea();
    T de_dsum = w * (total_area - yolk_area_rest);
    int n_nodes_basal = num_nodes / 2;
    VectorXT dAdx_full = VectorXT::Zero(deformed.rows());

    for (int i = 0; i < n_nodes_basal; i++)
    {
        int j = (i + 1) % n_nodes_basal;
        TV vi = deformed.segment<2>(basal_vtx_start * 2 + i * 2);
        TV vj = deformed.segment<2>(basal_vtx_start * 2 + j * 2);

        Vector<T, 4> dedx;
        // computeTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        computeSignedTriangleAreaGradient(vi, vj, mesh_centroid, dedx);
        
        addForceEntry<4>(dAdx_full, {basal_vtx_start + i, basal_vtx_start + j}, dedx);

        Matrix<T, 4, 4> d2edx2;
        computeSignedTriangleAreaHessian(vi, vj, mesh_centroid, d2edx2);
        // computeTriangleAreaHessian(vi, vj, mesh_centroid, d2edx2);

        Matrix<T, 4, 4> hessian = de_dsum * d2edx2;
        addHessianEntry(entries, {basal_vtx_start + i, basal_vtx_start + j}, hessian);
    }

    for (int i = n_nodes_basal; i < num_nodes; i++)
    {
        for (int j = n_nodes_basal; j < num_nodes; j++)
        {
            Vector<T, 4> dAdx;
            getSubVector<4>(dAdx_full, {i, j}, dAdx);
            TV dAdxi = dAdx.segment<2>(0);
            TV dAdxj = dAdx.segment<2>(2);
            Matrix<T, 2, 2> hessian_partial = w * dAdxi * dAdxj.transpose();
            // if (hessian_partial.nonZeros() > 0)
            addHessianBlock<2>(entries, {i, j}, hessian_partial);
        }    
    }
    
}

void VertexModel2D::configContractingWeights()
{
    apical_edge_contracting_weights.resize(basal_edge_start);
    apical_edge_contracting_weights.setZero();
    contracting_mask = apical_edge_contracting_weights;

    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id)
    {
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        // if (vi[1] > 0.85 * radius && vj[1] > 0.85 * radius)
        if (vi[1] > 0.95 * radius && vj[1] > 0.95 * radius)
        {
            apical_edge_contracting_weights[cnt] = w_c;
            contracting_mask[cnt] = 1.0;
        }
        cnt++;
    });
}

void VertexModel2D::getCellVtxIndices(VtxList& indices, int cell_idx)
{
    indices.resize(0);
    indices.push_back(edges[cell_idx][0]);
    indices.push_back(edges[cell_idx][1]);
    indices.push_back(edges[cell_idx][1] + basal_vtx_start);
    indices.push_back(edges[cell_idx][0] + basal_vtx_start);
}

void VertexModel2D::computeCellCentroid(int cell_idx, TV& centroid, bool rest)
{
    VtxList edge_vtx_list;
    getCellVtxIndices(edge_vtx_list, cell_idx);
    VectorXT positions;
    positionsFromIndices(positions, edge_vtx_list, rest);
    centroid = TV::Zero();
    for (int i = 0; i < edge_vtx_list.size(); i++)
    {
        centroid += positions.segment<2>(i * 2);
    }
    centroid /= T(edge_vtx_list.size());
}

void VertexModel2D::computeAllCellCentroids(VectorXT& cell_centroids)
{
    cell_centroids.resize(n_cells * 2);
    iterateCellParallel([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, indices);
        TV centroid = TV::Zero();
        for (int i = 0; i < indices.size(); i++)
        {
            centroid += positions.segment<2>(i * 2);
        }
        centroid /= T(indices.size());
        cell_centroids.segment<2>(cell_idx * 2) = centroid;
    });
}

void VertexModel2D::saveCellCentroidsToFile(const std::string& filename, T perburbance)
{
    std::ofstream out(filename);
    VectorXT cell_centroids;
    computeAllCellCentroids(cell_centroids);

    TV c0 = cell_centroids.segment<2>(0), c1 = cell_centroids.segment<2>(2);
    
    T length = (c0 - c1).norm();

    for (int i = 0; i < n_cells; i++)
    {
        TV random_dir = TV::Random().normalized();
        TV perburbed = cell_centroids.segment<2>(i * 2) + perburbance * length * random_dir;
        out << i << " " << perburbed.transpose() << std::endl;
    }
    out.close();
}

void VertexModel2D::savePerturbedCellCentroidsToFile(const std::string& filename, T perturbance, int n_pt_per_cell)
{
    std::ofstream out(filename);
    VectorXT cell_centroids;
    computeAllCellCentroids(cell_centroids);

    TV c0 = cell_centroids.segment<2>(0), c1 = cell_centroids.segment<2>(2);
    
    T length = (c0 - c1).norm();
    for (int i = 0; i < n_cells; i++)
    {
        for (int j = 0; j < n_pt_per_cell; j++)
        {
            TV random_dir = TV::Random().normalized();
            TV perburbed = cell_centroids.segment<2>(i * 2) + perturbance * length * random_dir;
            out << i << " " << perburbed.transpose() << std::endl;
        }
    }
    out.close();
}

void VertexModel2D::reset()
{
    deformed = undeformed;
    u.setZero();
    if (use_ipc)
    {
        for (int i = 0; i < num_nodes; i++)
        {
            ipc_vertices.row(i).segment<2>(0) = undeformed.segment<2>(i * 2);
        }
    }
}

void VertexModel2D::checkFunctionPerIteration(int step)
{
    VectorXT edge_length(edges.size());
    for (int i = 0; i < edges.size(); i++)
    {
        TV vi = deformed.segment<2>(edges[i][0] * 2);
        TV vj = deformed.segment<2>(edges[i][1] * 2);
        edge_length[i] = (vj - vi).norm();
    }
    std::cout << "min edge length " << edge_length.minCoeff() << std::endl;
    // std::getchar();
}

bool VertexModel2D::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;
    T residual_norm_init = 0.0;
    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();
        
        if (use_ipc)
            updateIPCVertices(u);
        
        residual_norm = computeResidual(u, residual);
        if (cnt == 0)
            residual_norm_init = residual_norm;
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;


        if (residual_norm < newton_tol)
            break;

        dq_norm = lineSearchNewton(u, residual);

        if(cnt == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
            break;
        cnt++;
    }

    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });

    deformed = undeformed + u;

    if (cnt == max_newton_iter)
    {
        // std::ofstream out("troubled_weights.txt");
        // out << std::setprecision(20) << apical_edge_contracting_weights << std::endl;
        // out.close();
        // saveStates("troubled_mesh.obj");
        // out.open("u0.txt");
        // out << std::setprecision(20) << u0 << std::endl;
        // out.close();
        // std::cout << "NEWTON REACHED MAX ITERATION" << std::endl;
        // std::exit(0);
    }

    std::cout << "# of newton iter: " << cnt << " exited with |g|: " 
            << residual_norm << " |ddu|: " << dq_norm  
            << " |g_init|: " << residual_norm_init << std::endl;
    return true;
}

void VertexModel2D::dOdpEdgeWeightsFromLambda(const VectorXT& lambda, VectorXT& dOdp)
{
    dOdp = VectorXT::Zero(apical_edge_contracting_weights.rows());
    
    std::vector<int> dirichlet_idx;
    for (auto data : dirichlet_data)
    {
        dirichlet_idx.push_back(data.first);
    }

    auto maskDirichletDof = [&](Vector<T, 4>& vec, int node_i, int node_j)
    {
        for (int d = 0; d < 2; d++)
        {
            bool find_node_i = std::find(dirichlet_idx.begin(), dirichlet_idx.end(), node_i * 2 + d) != dirichlet_idx.end();
            bool find_node_j = std::find(dirichlet_idx.begin(), dirichlet_idx.end(), node_j * 2 + d) != dirichlet_idx.end();
            if (find_node_i) vec[d] = 0;
            if (find_node_j) vec[2 + d] = 0;
        }    
    };

    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Vector<T, 4> dedx;
        computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
        dedx *= -1.0;
        maskDirichletDof(dedx, e[0], e[1]);
        dOdp[cnt] += lambda.segment<2>(e[0] * 2).dot(dedx.segment<2>(0));
        dOdp[cnt] += lambda.segment<2>(e[1] * 2).dot(dedx.segment<2>(2));
        cnt++;
    });
}

void VertexModel2D::computededp(VectorXT& dedp)
{
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        T edge_length_squared;
        computeEdgeSquaredNorm2D(vi, vj, edge_length_squared);
        dedp[cnt] = edge_length_squared;
        cnt++;
    });
}

void VertexModel2D::dfdpWeightsSparse(StiffnessMatrix& dfdp)
{
    dfdp.resize(num_nodes * 2, apical_edge_contracting_weights.rows());
    std::vector<Entry> entries;
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_id)
    {
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Vector<T, 4> dedx;
        computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
        dedx *= -1.0;
        for (int i = 0; i < 2; i++)
        {
            entries.push_back(Entry(e[0] * 2 + i, cnt, dedx[i]));
            entries.push_back(Entry(e[1] * 2 + i, cnt, dedx[i+2]));
        }
        cnt++;
    });

    dfdp.setFromTriplets(entries.begin(), entries.end());

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp.coeffRef(data.first, i) = 0;
}

void VertexModel2D::dfdpWeightsDense(MatrixXT& dfdp)
{
    dfdp.resize(num_nodes * 2, apical_edge_contracting_weights.rows());
    dfdp.setZero();
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_idx){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Vector<T, 4> dedx;
        computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
        dedx *= -1.0;
        dfdp.col(cnt).segment<2>(e[0] * 2) += dedx.segment<2>(0);
        dfdp.col(cnt).segment<2>(e[1] * 2) += dedx.segment<2>(2);
        cnt++;
    });

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp(data.first, i) = 0;
}

void VertexModel2D::dxdpFromdxdpEdgeWeights(MatrixXT& dxdp)
{
    MatrixXT dfdp(num_nodes * 2, apical_edge_contracting_weights.rows());
    dfdp.setZero();
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e, int edge_idx){
        TV vi = deformed.segment<2>(e[0] * 2);
        TV vj = deformed.segment<2>(e[1] * 2);
        Vector<T, 4> dedx;
        computeEdgeSquaredNorm2DGradient(vi, vj, dedx);
        dedx *= -1.0;
        dfdp.col(cnt).segment<2>(e[0] * 2) += dedx.segment<2>(0);
        dfdp.col(cnt).segment<2>(e[1] * 2) += dedx.segment<2>(2);
        cnt++;
    });

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp(data.first, i) = 0;
    
    dxdp.resize(num_nodes * 2, apical_edge_contracting_weights.rows());
    dxdp.setZero();
    StiffnessMatrix d2edx2(num_nodes*2, num_nodes*2);
    buildSystemMatrix(u, d2edx2);
    // Timer ttt(true);
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    for (int i = 0; i < cnt; i++)
    {
        dxdp.col(i) = solver.solve(dfdp.col(i));
    }
}

void VertexModel2D::appendSphereToPositionVector(const VectorXT& position, T radius, const TV3& color,
    Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    int n_pt = position.rows() / 3;

    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);
    
    Eigen::MatrixXd c_sphere(f_sphere.rows(), f_sphere.cols());
    
    v_sphere = v_sphere * radius;

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    V.conservativeResize(V.rows() + v_sphere.rows() * n_pt, 3);
    F.conservativeResize(F.rows() + f_sphere.rows() * n_pt, 3);
    C.conservativeResize(C.rows() + f_sphere.rows() * n_pt, 3);

    tbb::parallel_for(0, n_pt, [&](int i)
    {
        Eigen::MatrixXd v_sphere_i = v_sphere;
        Eigen::MatrixXi f_sphere_i = f_sphere;
        Eigen::MatrixXd c_sphere_i = c_sphere;

        tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
            v_sphere_i.row(row_idx) += position.segment<3>(i * 3);
        });


        int offset_v = n_vtx_prev + i * v_sphere.rows();
        int offset_f = n_face_prev + i * f_sphere.rows();

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            f_sphere_i.row(row_idx) += Eigen::Vector3i(offset_v, offset_v, offset_v);
        });

        tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
            c_sphere_i.row(row_idx) = color;
        });

        V.block(offset_v, 0, v_sphere.rows(), 3) = v_sphere_i;
        F.block(offset_f, 0, f_sphere.rows(), 3) = f_sphere_i;
        C.block(offset_f, 0, f_sphere.rows(), 3) = c_sphere_i;
    });
}

void VertexModel2D::checkHessianPD(bool save_txt)
{
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    bool use_Spectra = true;

    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2); 
    solver.factorize(d2edx2);
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
        // saveStates("indefinite_state.obj");
        // std::exit(0);
    }

    if (use_Spectra)
    {
        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(d2edx2);

        //0 cannot cannot be used as a shift
        T shift = indefinite ? -1e2 : -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values.transpose() << std::endl;
            if (save_txt)
            {
                std::ofstream out("cell_eigen_vectors_2d.txt");
                out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
                for (int i = 0; i < eigen_vectors.cols(); i++)
                    out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
                out << std::endl;
                for (int i = 0; i < eigen_vectors.rows(); i++)
                {
                    // for (int j = 0; j < eigen_vectors.cols(); j++)
                    for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                        out << eigen_vectors(i, j) << " ";
                    out << std::endl;
                }       
                out << std::endl;
                out.close();
            }
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
}

bool VertexModel2D::fetchNegativeEigenVectorIfAny(T& negative_eigen_value, VectorXT& negative_eigen_vector)
{
    int n_dof_sim = deformed.rows();
    VectorXT residual(n_dof_sim); residual.setZero();
    computeResidual(u, residual);
    if (residual.norm() > newton_tol)
        return false;
    int nmodes = 10;
    
    StiffnessMatrix d2edx2(n_dof_sim, n_dof_sim);
    buildSystemMatrix(u, d2edx2);
    
    Eigen::PardisoLLT<StiffnessMatrix> solver;
    solver.analyzePattern(d2edx2); 
    solver.factorize(d2edx2);
    
    if (solver.info() != Eigen::NumericalIssue)
    {
        return false;
    }

    Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(d2edx2);

        //0 cannot cannot be used as a shift
    T shift = -10;
    Spectra::SymEigsShiftSolver<T, 
        Spectra::LARGEST_MAGN, 
        Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
        eigs(&op, nmodes, 2 * nmodes, shift);

    eigs.init();

    int nconv = eigs.compute();

    if (eigs.info() == Spectra::SUCCESSFUL)
    {
        // std::cout << "Spectra successful" << std::endl;
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
        Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
        int last_col = eigen_values.rows() - 1;
        // std::cout << eigen_values.transpose() << std::endl;
        if (eigen_values[last_col] < 0.0)
        {
            negative_eigen_vector = eigen_vectors.col(last_col);
            negative_eigen_value = eigen_values[last_col]; 
            return true;
        }   
        return false;
    }
    else
    {
        std::cout << "Spectra failed" << std::endl;
        return false;
    }
}

void VertexModel2D::addAreaBarrierEnergy(T w, T& energy)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, indices);
        T ei;
        computeAreaBarrier4Points(w, positions, ei);
        energy += ei;
    });
}

void VertexModel2D::addAreaBarrierForceEntries(T w, VectorXT& residual)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, indices);
        Vector<T, 8> dedx;
        computeAreaBarrier4PointsGradient(w, positions, dedx);
        addForceEntry<8>(residual, indices, -dedx);
    });
}

void VertexModel2D::addAreaBarrierHessianEntries(T w, std::vector<Entry>& entries)
{
    iterateCellSerial([&](VtxList& indices, int cell_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, indices);
        Matrix<T, 8, 8> hessian;
        computeAreaBarrier4PointsHessian(w, positions, hessian);
        addHessianEntry<8>(entries, indices, hessian);
    });
}

void VertexModel2D::saveStates(const std::string& filename)
{
    std::ofstream out(filename);
    for (int i = 0; i < num_nodes; i++)
    {
        out << std::setprecision(20) << "v " << deformed.segment<2>(i * 2).transpose() << " 0" << std::endl;
    }
    for (Edge& e : edges)
    {
        out << "l " << (e + IV::Ones()).transpose() << std::endl;
    }
    out.close();
}

void VertexModel2D::loadStates(const std::string& filename)
{
    
    std::ifstream in(filename);
    char v; T x, y, z;
    int cnt = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        in >> v >> x >> y >> z;
        deformed.segment<2>(i * 2) = TV(x, y);
    }
    in.close();
    // Eigen::MatrixXd V;
    // Eigen::MatrixXi F;
    // igl::readOBJ(filename, V, F);
    // for (int i = 0; i < V.rows(); i++)
    // {
    //     deformed.segment<2>(i * 2) = V.row(i).segment<2>(0);
    // }
    u = deformed - undeformed;
    
}
