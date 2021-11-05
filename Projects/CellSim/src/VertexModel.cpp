#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>

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


void VertexModel::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
{

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

void VertexModel::computeCellInitialVolume()
{
    // each apical face corresponds to one cell
    cell_volume_init = VectorXT::Ones(basal_face_start);

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
            cell_volume_init[face_idx] = cell_volume;
        }
    });
    // std::cout << cell_volume_init << std::endl;
}

T VertexModel::computeTotalEnergy(const VectorXT& _u)
{
    T energy = 0.0;

    // edge length term
    iterateEdgeSerial([&](Edge& e){
        TV vi = deformed.segment<3>(e[0] * 3);
        TV vj = deformed.segment<3>(e[1] * 3);
        T edge_length = computeEdgeLength(vi, vj);
        energy += sigma * edge_length;

    });

    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {

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
    return 0.0;
}