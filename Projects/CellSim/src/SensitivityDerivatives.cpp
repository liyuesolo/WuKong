#include "../include/VertexModel.h"
#include "../include/LinearSolver.h"
#include "../include/autodiff/EdgeEnergy.h"
#include "../../../Solver/CHOLMODSolver.hpp"
#include <Eigen/PardisoSupport>
#include <Eigen/CholmodSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

void VertexModel::edgeWeightsSGNMatrix(StiffnessMatrix& mat_SGN, std::vector<Entry>& d2Odx2_entries)
{
    int n_dof_design = edge_weights.rows();
    int n_dof_sim = num_nodes * 3;
    mat_SGN.resize(n_dof_sim * 2 + n_dof_design, n_dof_sim * 2 + n_dof_design);
    StiffnessMatrix hessian;
    MatrixXT UV;
    buildSystemMatrixWoodbury(u, hessian, UV);
    Eigen::MatrixXd UVT  = UV * UV.transpose();
    UVT += hessian;
    hessian = UVT.sparseView();

    std::vector<Entry> entries = entriesFromSparseMatrix(hessian);
    std::vector<Entry> mat_SGN_entries;
    for (auto entry : entries)
    {
        mat_SGN_entries.push_back(Entry(entry.row() + n_dof_sim + n_dof_design,
            entry.col(), entry.value()));
        mat_SGN_entries.push_back(Entry(entry.row(),
            entry.col() + n_dof_sim + n_dof_design, entry.value()));
    }
    mat_SGN_entries.insert(mat_SGN_entries.end(), d2Odx2_entries.begin(), d2Odx2_entries.end());

    for (int i = 0; i < n_dof_sim * 2 + n_dof_design; i++)
    {
        mat_SGN_entries.push_back(Entry(i, i, 1e-6));
    }
    
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e)
    {
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -1.0;
        for (int i = 0; i < 3; i++)
        {
            mat_SGN_entries.push_back(Entry(e[0] * 3 + i + n_dof_sim, cnt + n_dof_sim + n_dof_design, dedx[i]));
            mat_SGN_entries.push_back(Entry(e[1] * 3 + i + n_dof_sim, cnt + n_dof_sim + n_dof_design, dedx[i+3]));

            mat_SGN_entries.push_back(Entry(cnt + n_dof_sim + n_dof_design, e[0] * 3 + i + n_dof_sim, dedx[i]));
            mat_SGN_entries.push_back(Entry(cnt + n_dof_sim + n_dof_design, e[1] * 3 + i + n_dof_sim, dedx[i+3]));
        }
        cnt++;
    });
    mat_SGN.setFromTriplets(mat_SGN_entries.begin(), mat_SGN_entries.end());
    mat_SGN.makeCompressed();

    // std::ofstream out("hessian.txt");
    // out << hessian << std::endl;
    // out.close();
    // std::exit(0);
}

void VertexModel::dfdpWeightsSparse(StiffnessMatrix& dfdp)
{
    dfdp.resize(num_nodes * 3, edge_weights.rows());
    std::vector<Entry> entries;
    int cnt = 0;
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -1.0;
            for (int i = 0; i < 3; i++)
            {
                entries.push_back(Entry(e[0] * 3 + i, cnt, dedx[i]));
                entries.push_back(Entry(e[1] * 3 + i, cnt, dedx[i+3]));
            }
            cnt++;
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if ((apical || basal) || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -1.0;
                for (int i = 0; i < 3; i++)
                {
                    entries.push_back(Entry(e[0] * 3 + i, cnt, dedx[i]));
                    entries.push_back(Entry(e[1] * 3 + i, cnt, dedx[i+3]));
                }
                cnt++;
            }
        }
    }

    
    dfdp.setFromTriplets(entries.begin(), entries.end());

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp.coeffRef(data.first, i) = 0;
}


void VertexModel::dfdpWeightsDense(MatrixXT& dfdp)
{
    dfdp.resize(num_nodes * 3, edge_weights.rows());
    dfdp.setZero();
    int cnt = 0;
        
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -1.0;
            dfdp.col(cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
            dfdp.col(cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
            cnt++;
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if ((apical || basal) || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -1.0;
                dfdp.col(cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
                dfdp.col(cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
                cnt++;
            }
        }
    }

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp(data.first, i) = 0;
}


void VertexModel::dxdpFromdxdpEdgeWeights(MatrixXT& dxdp)
{
    MatrixXT dfdp(num_nodes * 3, edge_weights.rows());
    dfdp.setZero();
    int cnt = 0;
    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            dedx *= -1.0;
            dfdp.col(cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
            dfdp.col(cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
            cnt++;
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if ((apical || basal) || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                dedx *= -1.0;
                dfdp.col(cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
                dfdp.col(cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
                cnt++;
            }
        }
    }
        

    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
            dfdp(data.first, i) = 0;
    
    dxdp.resize(num_nodes * 3, edge_weights.rows());
    // dxdp.setZero(); 
    StiffnessMatrix d2edx2(num_nodes*3, num_nodes*3);
    // buildSystemMatrix(u, d2edx2);
    // Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // solver.compute(d2edx2);
    // dxdp.noalias() = solver.solve(dfdp);
    // StiffnessMatrix full_hessian = d2edx2;   
    // if (lower_triangular) 
    //     full_hessian = d2edx2.selfadjointView<Eigen::Lower>();
    MatrixXT UV;
    buildSystemMatrixWoodbury(u, d2edx2, UV);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "forward hessian indefinite when computing dxdp" << std::endl;
    }
    
    // solver.compute(d2edx2);
    
    dxdp.noalias() = solver.solve(dfdp);
    VectorXT A_inv_u = solver.solve(UV.col(0));
    // std::cout << "hree" << std::endl;
    tbb::parallel_for(0, (int)dxdp.cols(), [&](int i)
    {
        T dem = 1.0 + UV.col(0).dot(A_inv_u);
        dxdp.col(i) = dxdp.col(i) - (dxdp.col(i).dot(UV.col(0))) * A_inv_u / dem;
    });

    
    
    for (int i = 0; i < cnt; i++)
        for (auto data : dirichlet_data)
        {
            dxdp(data.first, i) = 0;
        }
}

void VertexModel::computededp(VectorXT& dedp)
{
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length_squared = computeEdgeSquaredNorm(vi, vj);
        dedp[cnt] = edge_length_squared;
        cnt++;
    });
}

void VertexModel::dOdpFromdxdpEdgeWeights(const VectorXT& dOdu, VectorXT& dOdp)
{
    MatrixXT dfdp(num_nodes * 3, edge_weights.rows());
    dfdp.setZero();
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -1.0;
        dfdp.col(cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
        dfdp.col(cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
        cnt++;
    });

    StiffnessMatrix d2edx2(num_nodes*3, num_nodes*3);
    if (woodbury)
    {
        MatrixXT UV;
        buildSystemMatrixWoodbury(u, d2edx2, UV);
    }
    else
    {
        buildSystemMatrix(u, d2edx2);
    }
    
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    solver.analyzePattern(d2edx2);
    solver.factorize(d2edx2);
    if (solver.info() == Eigen::NumericalIssue)
        std::cout << "Forward simulation hessian indefinite" << std::endl;
    MatrixXT dxdp(num_nodes * 3, edge_weights.rows());
    for (int i = 0; i < cnt; i++)
    {
        dxdp.col(i) = -solver.solve(dfdp.col(i));
    }
    dOdp = dOdu.transpose() * dxdp;
}

void VertexModel::multiplyDpWithDfdp(VectorXT& result, const VectorXT& dp)
{
    result = VectorXT::Zero(num_nodes * 3);
    int cnt = 0;
    iterateApicalEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeSquaredNormGradient(vi, vj, dedx);
        dedx *= -1.0;
        addForceEntry<6>(result, {e[0], e[1]}, dp[cnt] * dedx);
        cnt++;
    });
}
void VertexModel::dOdpEdgeWeightsFromLambda(const VectorXT& lambda, VectorXT& dOdp)
{
    dOdp = VectorXT::Zero(edge_weights.rows());
    // MatrixXT dfdp(num_nodes * 3, edge_weights.rows());
    // dfdp.setZero();
    int cnt = 0;
    std::vector<int> dirichlet_idx;
    for (auto data : dirichlet_data)
    {
        dirichlet_idx.push_back(data.first);
    }

    auto maskDirichletDof = [&](Vector<T, 6>& vec, int node_i, int node_j)
    {
        for (int d = 0; d < 3; d++)
        {
            bool find_node_i = std::find(dirichlet_idx.begin(), dirichlet_idx.end(), node_i * 3 + d) != dirichlet_idx.end();
            bool find_node_j = std::find(dirichlet_idx.begin(), dirichlet_idx.end(), node_j * 3 + d) != dirichlet_idx.end();
            if (find_node_i) vec[d] = 0;
            if (find_node_j) vec[3 + d] = 0;
        }    
    };

    if (contracting_type == ApicalOnly)
    {
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            
            dedx *= -1.0;
            maskDirichletDof(dedx, e[0], e[1]);
            dOdp[cnt] += lambda.segment<3>(e[0] * 3).dot(dedx.segment<3>(0));
            dOdp[cnt] += lambda.segment<3>(e[1] * 3).dot(dedx.segment<3>(3));
            cnt++;
        });
    }
    else
    {
        for (Edge& e : edges)
        {
            bool apical = e[0] < basal_vtx_start && e[1] < basal_vtx_start;
            bool basal = e[0] >= basal_vtx_start && e[1] >= basal_vtx_start;
            if ((apical || basal) || contracting_type == ALLEdges)
            {
                TV vi = deformed.segment<3>(e[0] * 3);
                TV vj = deformed.segment<3>(e[1] * 3);
                Vector<T, 6> dedx;
                computeEdgeSquaredNormGradient(vi, vj, dedx);
                
                dedx *= -1.0;
                maskDirichletDof(dedx, e[0], e[1]);
                dOdp[cnt] += lambda.segment<3>(e[0] * 3).dot(dedx.segment<3>(0));
                dOdp[cnt] += lambda.segment<3>(e[1] * 3).dot(dedx.segment<3>(3));
                cnt++;
            }
        }
    }
    
    T epsilon = 1e-6;
    auto edgeForce = [&](VectorXT& force)
    {
        cnt = 0;
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            addForceEntry<6>(force, {e[0], e[1]}, -dedx * edge_weights[cnt++]);
        });
    };

    auto testdfdp = [&]()
    {
        MatrixXT dfdp(num_nodes * 3, edge_weights.rows());
        dfdp.setZero();
        int _cnt =  0;
        iterateApicalEdgeSerial([&](Edge& e){
            TV vi = deformed.segment<3>(e[0] * 3);
            TV vj = deformed.segment<3>(e[1] * 3);
            Vector<T, 6> dedx;
            computeEdgeSquaredNormGradient(vi, vj, dedx);
            
            dedx *= -1.0;
            maskDirichletDof(dedx, e[0], e[1]);
            dfdp.col(_cnt).segment<3>(e[0] * 3) += dedx.segment<3>(0);
            dfdp.col(_cnt).segment<3>(e[1] * 3) += dedx.segment<3>(3);
            _cnt++;
        });
        int n_edge = cnt;
        MatrixXT dfdp_fd(num_nodes * 3, n_edge);
        for (int i = 0; i < n_edge; i++)
        {
            VectorXT f0 = VectorXT::Zero(num_nodes * 3); 
            VectorXT f1 = f0;
            edge_weights[i] += epsilon;
            edgeForce(f1); 
            edge_weights[i] -= 2.0 * epsilon;
            edgeForce(f0);
            edge_weights[i] += epsilon;
            dfdp_fd.col(i) = (f1 - f0) / (2.0 * epsilon);
        }

        for (int i = 0; i < num_nodes * 3; i++)
        {
            for (int j = 0; j < cnt; j++)
            {
                if (std::abs(dfdp_fd(i ,j)) < 1e-6 && std::abs(dfdp(i, j)) < 1e-6)
                    continue;
                if (std::abs(dfdp_fd(i ,j) - dfdp(i, j)) < 1e-3 * std::abs(dfdp_fd(i ,j)))
                    continue;
                std::cout << " " << dfdp_fd(i ,j) << " " << dfdp(i, j) << std::endl;
                std::getchar();
            }
            
        }
        std::exit(0);
    };
    // testdfdp();
    // VectorXT dOdp2 = lambda.transpose() * dfdp;
    // std::cout << (dOdp2 - dOdp).norm() << std::endl;
    // std::getchar();
    
    // std::cout << "dfdp" << std::endl;
   
}

void VertexModel::dfdpWeights(MatrixXT& dfdp)
{
    // if (dfdp.rows() != num_nodes * 3 || dfdp.cols() != 8)
        // std::cout << "wrong dfdp input dimension" << std::endl;
    
    // alpha gamma sigma Gamma we B By Bp
    
    VectorXT df_dalpha = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Lateral, 1.0, df_dalpha);
    VectorXT df_dgamma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Basal, 1.0, df_dgamma);
    VectorXT df_dsigma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Apical, 1.0, df_dsigma);

    VectorXT df_dGamma = VectorXT::Zero(num_nodes * 3);
    addEdgeContractionForceEntries(1.0, df_dGamma);

    VectorXT df_dwe = VectorXT::Zero(num_nodes * 3);
    addEdgeForceEntries(ALL, 1.0, df_dwe);

    VectorXT df_dB = VectorXT::Zero(num_nodes * 3);
    T _B = B;
    B = 1.0;
    addCellVolumePreservationForceEntries(df_dB);
    B = _B;

    VectorXT df_dBy = VectorXT::Zero(num_nodes * 3);
    T _By = By;
    By = 1.0;
    addYolkVolumePreservationForceEntries(df_dBy);
    By = _By;

    VectorXT df_dBp = VectorXT::Zero(num_nodes * 3);
    T _Bp = Bp;
    Bp = 1.0;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    Bp = _Bp;

    dfdp.col(0) = df_dalpha * alpha;
    dfdp.col(1) = df_dgamma * gamma;
    dfdp.col(2) = df_dsigma * sigma;
    dfdp.col(3) = df_dGamma * Gamma;
    dfdp.col(4) = df_dwe * weights_all_edges;
    dfdp.col(5) = df_dB * B;
    dfdp.col(6) = df_dBy * By;
    dfdp.col(7) = df_dBp * Bp;

    for (int i = 0; i < 8; i++)
    {
        for (auto data : dirichlet_data)
            dfdp.col(i)[data.first] = 0;
    }

    // dfdp.col(0) = df_dB;
    // for (int i = 0; i < 1; i++)
    // {
    //     for (auto data : dirichlet_data)
    //         dfdp.col(i)[data.first] = 0;
    // }
}

void VertexModel::dfdpWeightsFD(MatrixXT& dfdp)
{
    if (dfdp.rows() != num_nodes * 3 || dfdp.cols() != 8)
        std::cout << "wrong dfdp input dimension" << std::endl;
    
    // alpha gamma sigma Gamma we B By Bp
    T epsilon = 1e-6;

    VectorXT df_dalpha = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Lateral, alpha - epsilon, df_dalpha);
    df_dalpha *= -1.0;
    addFaceAreaForceEntries(Lateral, alpha + epsilon, df_dalpha);
    df_dalpha /= (2.0 * epsilon);

    VectorXT df_dgamma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Basal, gamma - epsilon, df_dgamma);
    df_dgamma *= -1.0;
    addFaceAreaForceEntries(Basal, gamma + epsilon, df_dgamma);
    df_dgamma /= (2.0 * epsilon);

    VectorXT df_dsigma = VectorXT::Zero(num_nodes * 3);
    addFaceAreaForceEntries(Apical, sigma - epsilon, df_dsigma);
    df_dsigma *= -1.0;
    addFaceAreaForceEntries(Apical, sigma + epsilon, df_dsigma);
    df_dsigma /= (2.0 * epsilon);

    VectorXT df_dGamma = VectorXT::Zero(num_nodes * 3);
    addEdgeContractionForceEntries(Gamma - epsilon, df_dGamma);
    df_dGamma *= -1.0;
    addEdgeContractionForceEntries(Gamma + epsilon, df_dGamma);
    df_dGamma /= (2.0 * epsilon);

    VectorXT df_dwe = VectorXT::Zero(num_nodes * 3);
    addEdgeForceEntries(ALL, weights_all_edges - epsilon, df_dwe);
    df_dwe *= -1.0;
    addEdgeForceEntries(ALL, weights_all_edges + epsilon, df_dwe);
    df_dwe /= (2.0 * epsilon);

    VectorXT df_dB = VectorXT::Zero(num_nodes * 3);
    B -= epsilon * B;
    addCellVolumePreservationForceEntries(df_dB);
    df_dB *= -1.0;
    B += 2.0 * epsilon * B;
    addCellVolumePreservationForceEntries(df_dB);
    B -= epsilon * B;
    df_dB  /= (2.0 * epsilon * B);

    VectorXT df_dBy = VectorXT::Zero(num_nodes * 3);
    By -= epsilon * By;
    addYolkVolumePreservationForceEntries(df_dBy);
    df_dBy *= -1.0;
    By += 2.0 * epsilon * By;
    addYolkVolumePreservationForceEntries(df_dBy);
    By -= epsilon * By;
    df_dBy /= (2.0 * epsilon * By);

    VectorXT df_dBp = VectorXT::Zero(num_nodes * 3);
    Bp -= epsilon * Bp;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    df_dBp *= -1.0;
    Bp += 2.0 * epsilon * Bp;
    addPerivitellineVolumePreservationForceEntries(df_dBp);
    Bp -= epsilon * Bp;
    df_dBp /= (2.0 * epsilon * Bp);

    dfdp.col(0) = df_dalpha;
    dfdp.col(1) = df_dgamma;
    dfdp.col(2) = df_dsigma;
    dfdp.col(3) = df_dGamma;
    dfdp.col(4) = df_dwe;
    dfdp.col(5) = df_dB;
    dfdp.col(6) = df_dBy;
    dfdp.col(7) = df_dBp;

    for (int i = 0; i < 8; i++)
    {
        for (auto data : dirichlet_data)
            dfdp.col(i)[data.first] = 0;
    }
}