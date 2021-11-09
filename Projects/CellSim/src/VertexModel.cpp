#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <unordered_set>
#include "../include/VertexModel.h"
#include "../include/autodiff/VertexModelEnergy.h"

void VertexModel::vertexModelFromMesh(const std::string& filename)
{
    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;
    igl::readOBJ(filename, V, F);

    // face centroids corresponds to the vertices of the dual mesh 
    std::vector<TV> face_centroids(F.rows());
    deformed.resize(F.rows() * 3);

    tbb::parallel_for(0, (int)F.rows(), [&](int i)
    {
        TV centroid = 1.0/3.0*(V.row(F.row(i)[0]) + V.row(F.row(i)[1]) + V.row(F.row(i)[2]));
        face_centroids[i] = centroid;
        deformed.segment<3>(i * 3) = centroid;
    });

    std::vector<std::vector<int>> dummy;

    igl::vertex_triangle_adjacency(V.rows(), F, faces, dummy);
    igl::per_face_normals(V, F, N);

    // re-order so that the faces around one vertex is clockwise
    tbb::parallel_for(0, (int)faces.size(), [&](int vi)
    {
        std::vector<int>& one_ring_face = faces[vi];
        TV avg_normal = N.row(one_ring_face[0]);
        for (int i = 1; i < one_ring_face.size(); i++)
        {
            avg_normal += N.row(one_ring_face[i]);
        }
        avg_normal /= one_ring_face.size();

        TV vtx = V.row(vi);
        TV centroid0 = face_centroids[one_ring_face[0]];
        std::sort(one_ring_face.begin(), one_ring_face.end(), [&](int a, int b){
            TV E0 = (face_centroids[a] - vtx).normalized();
            TV E1 = (face_centroids[b] - vtx).normalized();
            TV ref = (centroid0 - vtx).normalized();
            T dot_sign0 = E0.dot(ref);
            T dot_sign1 = E1.dot(ref);
            TV cross_sin0 = E0.cross(ref);
            TV cross_sin1 = E1.cross(ref);
            // use normal and cross product to check if it's larger than 180 degree
            T angle_a = cross_sin0.dot(avg_normal) > 0 ? std::acos(dot_sign0) : 2.0 * M_PI - std::acos(dot_sign0);
            T angle_b = cross_sin1.dot(avg_normal) > 0 ? std::acos(dot_sign1) : 2.0 * M_PI - std::acos(dot_sign1);
            
            return angle_a < angle_b;
        });
    });

    // extrude 
    TV mesh_center = TV::Zero();
    for (int i = 0; i < V.rows(); i++)
        mesh_center += V.row(i);
    mesh_center /= T(V.rows());

    basal_vtx_start = deformed.size() / 3;
    deformed.conservativeResize(deformed.rows() * 2);

    T e0_norm = (V.row(F.row(0)[1]) - V.row(F.row(0)[0])).norm();
    T cell_height = 0.8 * e0_norm;

    tbb::parallel_for(0, (int)basal_vtx_start, [&](int i){
        TV apex = deformed.segment<3>(i * 3);
        deformed.segment<3>(basal_vtx_start * 3 + i * 3) = mesh_center + 
            (apex - mesh_center) * ((apex - mesh_center).norm() - cell_height);
    });

    // add apical edges
    for (auto one_ring_face : faces)
    {
        for (int i = 0; i < one_ring_face.size(); i++)
        {
            int next = (i + 1) % one_ring_face.size();
            Edge e(one_ring_face[i], one_ring_face[next]);
            auto find_iter = std::find_if(edges.begin(), edges.end(), [&e](Edge& ei){
                return (e[0] == ei[0] && e[1] == ei[1]) || (e[0] == ei[1] && e[1] == ei[0]);
            });
            if (find_iter == edges.end())
            {
                edges.push_back(e);
            }
        }
    }
    
    basal_face_start = faces.size();
    for (int i = 0; i < basal_face_start; i++)
    {
        VtxList new_face = faces[i];
        for (int& idx : new_face)
            idx += basal_vtx_start;
        std::reverse(new_face.begin(), new_face.end());
        faces.push_back(new_face);
    }
    
    lateral_face_start = faces.size();

    std::vector<Edge> basal_and_lateral_edges;

    for (Edge edge : edges)
    {
        Edge basal_edge(edge[0] + basal_vtx_start, edge[1] + basal_vtx_start);
        Edge lateral0(edge[0], basal_edge[0]);
        Edge lateral1(edge[1], basal_edge[1]);
        basal_and_lateral_edges.push_back(basal_edge);
        basal_and_lateral_edges.push_back(lateral0);
        basal_and_lateral_edges.push_back(lateral1);

        VtxList lateral_face = {edge[0], edge[1], basal_edge[1], basal_edge[0]};
        faces.push_back(lateral_face);
    }
    edges.insert(edges.end(), basal_and_lateral_edges.begin(), basal_and_lateral_edges.end());

    num_nodes = deformed.rows() / 3;
    u = VectorXT::Zero(deformed.rows());
    undeformed = deformed;
}



void VertexModel::generateMeshForRendering(Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    // compute polygon face centroid
    std::vector<TV> face_centroid(faces.size());
    tbb::parallel_for(0, (int)faces.size(), [&](int i){
        TV centroid = deformed.segment<3>(faces[i][0] * 3);
        for (int j = 1; j < faces[i].size(); j++)
            centroid += deformed.segment<3>(faces[i][j] * 3);
        face_centroid[i] = centroid / T(faces[i].size());
    });

    V.resize(deformed.rows() / 3 + face_centroid.size(), 3);

    int centroids_start = deformed.rows() / 3;
    
    for (int i = 0; i < deformed.rows()/ 3; i++)
        V.row(i) = deformed.segment<3>(i * 3);
    
    for (int i = 0; i < face_centroid.size(); i++)
        V.row(deformed.rows() / 3 + i) = face_centroid[i];
    
    int face_start = 0;
    // std::cout << basal_face_start << " " << faces.size() << std::endl;
    int face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
        face_cnt += faces[i].size();
    F.resize(face_cnt, 3);

    face_cnt = 0;
    for (int i = face_start; i < faces.size(); i++)
    {
        for (int j = 0; j < faces[i].size(); j++)
        {
            int next = (j + 1) % faces[i].size();
            F.row(face_cnt++) = Eigen::Vector3i(centroids_start + i, faces[i][next], faces[i][j]);
        }       
    }

    // std::cout << face_cnt << " " << F.rows() << std::endl;
    
    C.resize(F.rows(), F.cols());

    tbb::parallel_for(0, (int)F.rows(), [&](int i)
    {
        C.row(i) = Eigen::Vector3d(0, 0.3, 1.0);
    });

}

void VertexModel::computeCellCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
    {
        centroid += deformed.segment<3>(vtx_idx * 3);
        centroid += deformed.segment<3>((vtx_idx + basal_vtx_start) * 3);
    }
    centroid /= T(face_vtx_list.size() * 2);
}

void VertexModel::computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid)
{
    centroid = TV::Zero();
    for (int vtx_idx : face_vtx_list)
        centroid += deformed.segment<3>(vtx_idx * 3);

    centroid /= T(face_vtx_list.size());
}

void VertexModel::computeCellInitialVolume(VectorXT& cell_volume_list)
{
    // each apical face corresponds to one cell
    cell_volume_list = VectorXT::Ones(basal_face_start);

    // use apical face to iterate other faces within this cell for now
    iterateFaceParallel([&](VtxList& face_vtx_list, int face_idx){
        if (face_idx < basal_face_start)
        {
            TV cell_centroid;
            TV apical_face_centroid, basal_face_centroid;
            T cell_volume = 0.0;

            computeCellCentroid(face_vtx_list, cell_centroid);
            computeFaceCentroid(face_vtx_list, apical_face_centroid);
            VtxList basal_face_vtx_list = face_vtx_list;
            for (int& idx : basal_face_vtx_list)
                idx += basal_vtx_start;
            computeFaceCentroid(basal_face_vtx_list, basal_face_centroid);

            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);
                T Vij = computeVolume(vi, vj, apical_face_centroid, cell_centroid);
                cell_volume += Vij; // apical

                vi = deformed.segment<3>(basal_face_vtx_list[i] * 3);
                vj = deformed.segment<3>(basal_face_vtx_list[j] * 3);
                Vij = computeVolume(vi, vj, basal_face_centroid, cell_centroid);
                cell_volume += Vij; // basal

                VtxList lateral_vtx = { face_vtx_list[i], face_vtx_list[j],
                    basal_face_vtx_list[j], basal_face_vtx_list[i] };

                TV lateral_face_centroid;
                computeFaceCentroid(lateral_vtx, lateral_face_centroid);
                for (int k = 0; k < lateral_vtx.size(); k++)
                {
                    int l = (k + 1) % lateral_vtx.size();
                    vi = deformed.segment<3>(lateral_vtx[k] * 3);
                    vj = deformed.segment<3>(lateral_vtx[l] * 3);
                    Vij = computeVolume(vi, vj, lateral_face_centroid, cell_centroid);
                    cell_volume += Vij; // lateral
                }
            }
            cell_volume_list[face_idx] = cell_volume;
        }
    });
    // std::cout << cell_volume_init << std::endl;
}

T VertexModel::computeTotalEnergy(const VectorXT& _u)
{
    T energy = 0.0;
    deformed = undeformed  + _u;
    // edge length term
    iterateEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeLength(vi, vj);
        energy += sigma * edge_length;

    });

    VectorXT current_cell_volume;
    computeCellInitialVolume(current_cell_volume);

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            energy += 0.5 * B * std::pow(cell_volume_init[face_idx] - current_cell_volume[face_idx], 2);
        }
        else // basal and lateral faces area term
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);
                T area = computeArea(vi, vj, centroid);
                energy += coeff * area;
            }
        }
        // yolk volume preservation term
        
    });

    return energy;
}

T VertexModel::computeResidual(const VectorXT& _u,  VectorXT& residual)
{
    deformed = undeformed  + _u;

    // edge length term
    iterateEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Vector<T, 6> dedx;
        computeEdgeLengthGradient(vi, vj, dedx);
        residual.segment<3>(e[0] * 3) += -sigma * dedx.segment<3>(0);
        residual.segment<3>(e[1] * 3) += -sigma * dedx.segment<3>(3);
    });

    VectorXT current_cell_volume;
    computeCellInitialVolume(current_cell_volume);

    

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            TV cell_centroid;
            TV apical_face_centroid, basal_face_centroid;
            

            computeCellCentroid(face_vtx_list, cell_centroid);
            computeFaceCentroid(face_vtx_list, apical_face_centroid);
            VtxList basal_face_vtx_list = face_vtx_list;
            for (int& idx : basal_face_vtx_list)
                idx += basal_vtx_start;
            computeFaceCentroid(basal_face_vtx_list, basal_face_centroid);

            TV dedcell_centroid = TV::Zero();
            TV dedapical = TV::Zero();
            TV dedbasal = TV::Zero();
            T coeff = 0.5 * B * 2.0 * (current_cell_volume[face_idx] - cell_volume_init[face_idx]);

            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);

                Vector<T, 12> dedx;
                computeVolumeGradient(vi, vj, apical_face_centroid, cell_centroid, dedx);
                addForceEntry<6>(residual, {face_vtx_list[i], face_vtx_list[j]}, -coeff * dedx.segment<6>(0));
                dedapical += -coeff * dedx.segment<3>(6);
                dedcell_centroid += -coeff * dedx.segment<3>(9);
                
                vi = deformed.segment<3>(basal_face_vtx_list[i] * 3);
                vj = deformed.segment<3>(basal_face_vtx_list[j] * 3);

                computeVolumeGradient(vi, vj, basal_face_centroid, cell_centroid, dedx);

                addForceEntry<6>(residual, {basal_face_vtx_list[i], basal_face_vtx_list[j]}, -coeff * dedx.segment<6>(0));
                dedbasal += -coeff * dedx.segment<3>(6);
                dedcell_centroid += -coeff * dedx.segment<3>(9);

                VtxList lateral_vtx = { face_vtx_list[i], face_vtx_list[j],
                    basal_face_vtx_list[j], basal_face_vtx_list[i] };

                TV lateral_face_centroid;
                computeFaceCentroid(lateral_vtx, lateral_face_centroid);
                TV dedlateral = TV::Zero();
                for (int k = 0; k < lateral_vtx.size(); k++)
                {
                    int l = (k + 1) % lateral_vtx.size();
                    vi = deformed.segment<3>(lateral_vtx[k] * 3);
                    vj = deformed.segment<3>(lateral_vtx[l] * 3);

                    computeVolumeGradient(vi, vj, lateral_face_centroid, cell_centroid, dedx);

                    addForceEntry<6>(residual, {lateral_vtx[k], lateral_vtx[l]}, -coeff * dedx.segment<6>(0));
                    dedlateral += -coeff * dedx.segment<3>(6);
                    dedcell_centroid += -coeff * dedx.segment<3>(9);
                }
                dedlateral /= lateral_vtx.size();
                for (int vtx_idx : lateral_vtx)
                    addForceEntry<3>(residual, {vtx_idx}, dedlateral );
            }
            dedcell_centroid /= (face_vtx_list.size() * 2.0);
            dedapical /= face_vtx_list.size();
            dedbasal /= basal_face_vtx_list.size();

            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                addForceEntry<3>(residual, {face_vtx_list[i]}, dedcell_centroid );
                addForceEntry<3>(residual, {basal_face_vtx_list[i]}, dedcell_centroid );
                addForceEntry<3>(residual, {face_vtx_list[i]}, dedapical );
                addForceEntry<3>(residual, {basal_face_vtx_list[i]}, dedbasal );
            }
        }
        else // basal and lateral faces area term
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            TV dedc = TV::Zero();
            
            // std::cout << face_vtx_list.size() << std::endl;
            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);
                Vector<T, 9> dedx;
                computeAreaGradient(vi, vj, centroid, dedx);
                
                // dedvij and dedcentroid/dcentroid dvij
                dedc += -coeff * dedx.segment<3>(6);
                addForceEntry<6>(residual, {face_vtx_list[i], face_vtx_list[j]}, -coeff * dedx.segment<6>(0));   
            }
            dedc /= face_vtx_list.size();
            for (int vtx_id : face_vtx_list)
                addForceEntry<3>(residual, {vtx_id}, dedc );
        }
        // yolk volume preservation term
        
    });

    return residual.norm();
}


void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{
    deformed = undeformed  + _u;
    std::vector<Entry> entries;
    
    // edge length term
    iterateEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        Matrix<T, 6, 6> hessian;
        computeEdgeLengthHessian(vi, vj, hessian);
        addHessianEntry<6>(entries, {e[0], e[1]}, hessian);
    });

    std::unordered_set<int> n_edges;

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        n_edges.insert(face_vtx_list.size());
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            return;
            TV cell_centroid;
            TV apical_face_centroid, basal_face_centroid;
            
            computeCellCentroid(face_vtx_list, cell_centroid);
            computeFaceCentroid(face_vtx_list, apical_face_centroid);
            VtxList basal_face_vtx_list = face_vtx_list;
            for (int& idx : basal_face_vtx_list)
                idx += basal_vtx_start;
            computeFaceCentroid(basal_face_vtx_list, basal_face_centroid);

            TM3 dcdx = TM3::Identity() / face_vtx_list.size();

            T coeff = 0.5 * B * 2.0;

            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);

                Matrix<T, 12, 12> sub_hessian;
                computeVolumeHessian(vi, vj, apical_face_centroid, cell_centroid, sub_hessian);
                sub_hessian *= coeff;
                
                vi = deformed.segment<3>(basal_face_vtx_list[i] * 3);
                vj = deformed.segment<3>(basal_face_vtx_list[j] * 3);

                computeVolumeHessian(vi, vj, basal_face_centroid, cell_centroid, sub_hessian);
                sub_hessian *= coeff;
             
                VtxList lateral_vtx = { face_vtx_list[i], face_vtx_list[j],
                    basal_face_vtx_list[j], basal_face_vtx_list[i] };

                TV lateral_face_centroid;
                computeFaceCentroid(lateral_vtx, lateral_face_centroid);
                TV dedlateral = TV::Zero();
                for (int k = 0; k < lateral_vtx.size(); k++)
                {
                    int l = (k + 1) % lateral_vtx.size();
                    vi = deformed.segment<3>(lateral_vtx[k] * 3);
                    vj = deformed.segment<3>(lateral_vtx[l] * 3);

                    computeVolumeHessian(vi, vj, lateral_face_centroid, cell_centroid, sub_hessian);                    
                    sub_hessian *= coeff;

                }   
            }
        }
        else // basal and lateral faces area term
        {
            TV centroid;
            computeFaceCentroid(face_vtx_list, centroid);
            T coeff = face_idx >= lateral_face_start ? alpha : gamma;
            TM3 d2edc2 = TM3::Zero();
            std::vector<TM3> d2ecdx_list(face_vtx_list.size(), TM3::Zero());
            TM3 dcdx = TM3::Identity() / face_vtx_list.size();
            std::vector<std::vector<TM3>> deidcidv_list(face_vtx_list.size(), 
                std::vector<TM3>(face_vtx_list.size(), TM3::Zero()));
            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                int j = (i + 1) % face_vtx_list.size();
                TV vi = deformed.segment<3>(face_vtx_list[i] * 3);
                TV vj = deformed.segment<3>(face_vtx_list[j] * 3);
                Matrix<T, 9, 9> sub_hessian;
                computeAreaHessian(vi, vj, centroid, sub_hessian);
                sub_hessian *= coeff;

                d2edc2 += sub_hessian.block(6, 6, 3, 3);
                TM3 d2edcdx0 = sub_hessian.block(0, 6, 3, 3);
                TM3 d2edcdx1 = sub_hessian.block(3, 6, 3, 3);

                TM3 d2edcdx0_dcdx = d2edcdx0 * dcdx;
                TM3 d2edcdx1_dcdx = d2edcdx1 * dcdx;
                deidcidv_list[i][i] += d2edcdx0_dcdx;
                deidcidv_list[i][j] += d2edcdx1_dcdx;

                addHessianEntry<6>(entries, {face_vtx_list[i], face_vtx_list[j]}, sub_hessian.block(0, 0, 6, 6));
            }
            for (int i = 0; i < face_vtx_list.size(); i++)
            {
                for (int j = 0; j < face_vtx_list.size(); j++)
                {
                    // TM3 chain_rule_term = (d2ecdx_list[j] + d2edc2 * dcdx) * dcdx;
                    TM3 chain_rule_term = d2edc2 * dcdx * dcdx;
                    // chain_rule_term.setZero();
                    for (int k = 0; k < face_vtx_list.size(); k++)
                        chain_rule_term += 2.0 * deidcidv_list[k][j];
                    
                    addHessianBlock<3>(entries, {face_vtx_list[i], face_vtx_list[j]}, chain_rule_term);
                }
            }
        }
    });
    for (auto num : n_edges)
        std::cout << num << std::endl;
        
    K.resize(num_nodes * 3, num_nodes * 3);
    K.setFromTriplets(entries.begin(), entries.end());
    K.makeCompressed();
}

void VertexModel::faceHessianChainRuleTest()
{
    TV v0(0, 0, 0), v1(1, 0, 0), v2(1, 1, 0), v3(0, 1, 0);
    VectorXT positions(4 * 3);
    positions << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0;
    TV centroid = 0.25 * (v0 + v1 + v2 + v3);

    auto updateCentroid = [&]()
    {
        centroid.setZero();

        for (int i = 0; i < 4; i++)
        {
            centroid += 0.25 * positions.segment<3>(i * 3);
        }
    };

    VtxList face_vtx_list = {0 ,1, 2, 3};
    auto computeEnergy = [&]()->T
    {
        updateCentroid();
        T energy  = 0.0;
        // for (int i = 0; i < face_vtx_list.size(); i++)
        // {
        //     int j = (i + 1) % face_vtx_list.size();
        //     TV vi = positions.segment<3>(face_vtx_list[i] * 3);
        //     TV vj = positions.segment<3>(face_vtx_list[j] * 3);

        //     T area = computeArea(vi, vj, centroid);
        //     energy += area;
        // }
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        energy = computeAreaFourPoints(r0, r1, r2, r3);
        return energy;
    };

    auto computeGradient = [&](VectorXT& residual)
    {
        updateCentroid();
        residual = VectorXT::Zero(positions.rows());
        TV dedc = TV::Zero();
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            int j = (i + 1) % face_vtx_list.size();
            TV vi = positions.segment<3>(face_vtx_list[i] * 3);
            TV vj = positions.segment<3>(face_vtx_list[j] * 3);
            Vector<T, 9> dedx;
            computeAreaGradient(vi, vj, centroid, dedx);
            
            // dedvij and dedcentroid/dcentroid dvij
            dedc += -dedx.segment<3>(6);
            addForceEntry<6>(residual, {face_vtx_list[i], face_vtx_list[j]}, -dedx.segment<6>(0));   
        }
        TM3 dcdx = TM3::Identity() / face_vtx_list.size();
        for (int vtx_id : face_vtx_list)
            addForceEntry<3>(residual, {vtx_id}, dedc.transpose() * dcdx );
    };

    auto computeGradientADFUll =[&](VectorXT& residual)
    {
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        Vector<T, 12> dedx;
        computeAreaFourPointsGradient(r0, r1, r2, r3, dedx);
        addForceEntry<12>(residual, face_vtx_list, -dedx);
    };

    auto hessianFullAutoDiff =[&](StiffnessMatrix& K)
    {
        TV r0 = positions.segment<3>(0), r1 = positions.segment<3>(3), 
            r2 = positions.segment<3>(6), r3 = positions.segment<3>(9);
        Matrix<T, 12, 12> d2edx2;
        computeAreaFourPointsHessian(r0, r1, r2, r3, d2edx2);
        std::vector<Entry> entries;
        addHessianEntry<12>(entries, face_vtx_list, d2edx2);
        K.resize(4 * 3, 4 * 3);
        K.setFromTriplets(entries.begin(), entries.end());
        K.makeCompressed();
    };

    auto computeHessian = [&](StiffnessMatrix& K)
    {
        updateCentroid();
        std::vector<Entry> entries;
        TM3 dcdx = TM3::Identity() / face_vtx_list.size();

        TM3 d2edc2_dcdvi_dcdvj = TM3::Zero();
        std::vector<std::vector<TM3>> deidcidv_list(face_vtx_list.size(), 
            std::vector<TM3>(face_vtx_list.size(), TM3::Zero()));
        
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            int j = (i + 1) % face_vtx_list.size();
            TV vi = positions.segment<3>(face_vtx_list[i] * 3);
            TV vj = positions.segment<3>(face_vtx_list[j] * 3);
            Matrix<T, 9, 9> sub_hessian;
            computeAreaHessian(vi, vj, centroid, sub_hessian);

            addHessianEntry<6>(entries, {face_vtx_list[i], face_vtx_list[j]}, sub_hessian.block(0, 0, 6, 6));

            // std::cout << sub_hessian << std::endl;
            // std::cout << "--------------------------------" << std::endl;
            TM3 d2edcdx0 = sub_hessian.block(0, 6, 3, 3);
            TM3 d2edcdx1 = sub_hessian.block(3, 6, 3, 3);
            TM3 d2edc2 = sub_hessian.block(6, 6, 3, 3);

            // std::cout << d2edc2 << std::endl;
            // std::cout << "--------------------------------" << std::endl;
            
            d2edc2_dcdvi_dcdvj += d2edc2 * dcdx * dcdx;

            TM3 d2edcdx0_dcdx = d2edcdx0 * dcdx;
            TM3 d2edcdx1_dcdx = d2edcdx1 * dcdx;

            deidcidv_list[i][i] += d2edcdx0_dcdx;
            deidcidv_list[i][j] += d2edcdx1_dcdx;
        }
        // std::cout << d2edc2_dcdvi_dcdvj << std::endl;
        // std::getchar();
        for (int i = 0; i < face_vtx_list.size(); i++)
        {
            for (int j = 0; j < face_vtx_list.size(); j++)
            {
                TM3 chain_rule_term = d2edc2_dcdvi_dcdvj;
                // chain_rule_term.setZero();
                
                for (int k = 0; k < face_vtx_list.size(); k++)
                {
                    // d2e_k/dcdv_j
                    // std::cout <<deidcidv_list[k][j] << std::endl;
                    chain_rule_term += 1.0 * deidcidv_list[k][j];
                }
                // std::cout << d2edc2 * dcdx << std::endl;
                // std::cout << d2ecdx_list[i] << std::endl;
                // std::cout << chain_rule_term << std::endl;
                // std::cout << "--------------------------------" << std::endl;
                // std::getchar();
                addHessianBlock<3>(entries, {face_vtx_list[i], face_vtx_list[j]}, chain_rule_term);
            }
        }
        K.resize(4 * 3, 4 * 3);
        K.setFromTriplets(entries.begin(), entries.end());
        K.makeCompressed();
    };

    auto checkGradient = [&]()
    {
        int n_dof = 4 * 3;
        T epsilon = 1e-6;
        VectorXT gradient_FD(n_dof);
        gradient_FD.setZero();
        VectorXT gradient(n_dof);
        gradient.setZero();

        computeGradient(gradient);
        
        int cnt = 0;
        for(int dof_i = 0; dof_i < n_dof; dof_i++)
        {
            positions(dof_i) += epsilon;
            // std::cout << W * dq << std::endl;
            T E0 = computeEnergy();
            
            positions(dof_i) -= 2.0 * epsilon;
            T E1 = computeEnergy();
            positions(dof_i) += epsilon;
            // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
            gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
            if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
                continue;
            // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
            //     continue;
            std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
            std::getchar();
            cnt++;   
        }
    };

    auto checkHessian = [&]()
    {
        T epsilon = 1e-6;
        int n_dof = 4 * 3;
        StiffnessMatrix A(n_dof, n_dof);
        computeHessian(A);

        StiffnessMatrix B = A;
        hessianFullAutoDiff(B);

        std::cout << A << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << B << std::endl;
        std::getchar();

        for(int dof_i = 0; dof_i < n_dof; dof_i++)
        {
            positions(dof_i) += epsilon;
            VectorXT g0(n_dof), g1(n_dof);
            g0.setZero(); g1.setZero();
            computeGradient(g0);

            positions(dof_i) -= 2.0 * epsilon;
            computeGradient(g1);
            positions(dof_i) += epsilon;
            VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

            for(int i = 0; i < n_dof; i++)
            {
                if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                    continue;
                std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
                std::getchar();
            }
        }
    };

    // VectorXT g0, g1;
    // computeGradient(g0);
    // computeGradient(g1);
    // std::cout << g0.transpose() << std::endl;
    // std::cout << "--------------------------------" << std::endl;
    // std::cout << g1.transpose() << std::endl;
    // std::getchar();

    // checkGradient();
    checkHessian();
}

void VertexModel::checkTotalGradient()
{
    // B = 0;
    run_diff_test = true;
    std::cout << "======================== CHECK GRADIENT ========================" << std::endl;
    int n_dof = num_nodes * 3;
    T epsilon = 1e-6;
    VectorXT gradient(n_dof);
    gradient.setZero();

    computeResidual(u, gradient);

    // std::cout << gradient.transpose() << std::endl;
    
    VectorXT gradient_FD(n_dof);
    gradient_FD.setZero();

    int cnt = 0;
    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        // std::cout << W * dq << std::endl;
        T E0 = computeTotalEnergy(u);
        
        u(dof_i) -= 2.0 * epsilon;
        T E1 = computeTotalEnergy(u);
        u(dof_i) += epsilon;
        // std::cout << "E1 " << E1 << " E0 " << E0 << std::endl;
        gradient_FD(dof_i) = (E1 - E0) / (2.0 *epsilon);
        if( gradient_FD(dof_i) == 0 && gradient(dof_i) == 0)
            continue;
        // if (std::abs( gradient_FD(d, n_node) - gradient(d, n_node)) < 1e-4)
        //     continue;
        std::cout << " dof " << dof_i << " " << gradient_FD(dof_i) << " " << gradient(dof_i) << std::endl;
        std::getchar();
        cnt++;   
    }
    run_diff_test = false;
}

void VertexModel::checkTotalHessian()
{
    // alpha = 0;
    // gamma = 0;
    B = 0;
    std::cout << "======================== CHECK HESSIAN ========================" << std::endl;
    run_diff_test = true;
    T epsilon = 1e-6;
    int n_dof = num_nodes * 3;
    StiffnessMatrix A(n_dof, n_dof);
    buildSystemMatrix(u, A);

    for(int dof_i = 0; dof_i < n_dof; dof_i++)
    {
        u(dof_i) += epsilon;
        VectorXT g0(n_dof), g1(n_dof);
        g0.setZero(); g1.setZero();
        computeResidual(u, g0);

        u(dof_i) -= 2.0 * epsilon;
        computeResidual(u, g1);
        u(dof_i) += epsilon;
        VectorXT row_FD = (g1 - g0) / (2.0 * epsilon);

        for(int i = 0; i < n_dof; i++)
        {
            if(A.coeff(dof_i, i) == 0 && row_FD(i) == 0)
                continue;
            // if (std::abs( A.coeff(dof_i, i) - row_FD(i)) < 1e-4)
            //     continue;
            // std::cout << "node i: "  << std::floor(dof_i / T(dof)) << " dof " << dof_i%dof 
            //     << " node j: " << std::floor(i / T(dof)) << " dof " << i%dof 
            //     << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::cout << "H(" << i << ", " << dof_i << ") " << " FD: " <<  row_FD(i) << " symbolic: " << A.coeff(i, dof_i) << std::endl;
            std::getchar();
        }
    }
    run_diff_test = false;
}