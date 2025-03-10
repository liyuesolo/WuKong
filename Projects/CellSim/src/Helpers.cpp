
#include "../include/VertexModel.h"

void VertexModel::removeAllTerms()
{
    alpha = 0.0;
    sigma = 0.0;
    gamma = 0.0;
    weights_all_edges = 0.0;

    B = 0.0;

    add_contraction_term = false;
    Gamma = 0.0;

    add_yolk_volume = false;
    By = 0.0;

    add_perivitelline_liquid_volume = false;
    Bp = 0.0;

    add_tet_vol_barrier = false;
    dynamics = false;

    use_sphere_radius_bound = false;
    use_ipc_contact = false;

    woodbury = false;
}

void VertexModel::saveSingleCellEdges(const std::string& filename, 
        const VtxList& indices, const VectorXT& positions, bool save_tets) const
{
    std::ofstream out(filename);
    for (int i = 0; i < indices.size(); i++)
    {
        out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
    }
    int half = indices.size() / 2;
    
    for (int i = 0; i < half; i++)
    {
        int j = (i + 1) % half;
        out << "l " << i + 1 << " " << j + 1 << std::endl;
    }
    for (int i = 0; i < half; i++)
    {
        out << "l " << i + 1 << " " << i + 1 + half << std::endl;
    }
    for (int i = 0; i < half; i++)
    {
        int j = (i + 1) % half;
        out << "l " << i + 1 + half << " " << j + 1 + half << std::endl;
    }
    out.close();
}

void VertexModel::normalizeToUnit(MatrixXT& V)
{
    TV min_corner, max_corner;

    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);
    TV center = TV::Zero();
    for (int i = 0; i < V.rows(); i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], V(i, d));
            min_corner[d] = std::min(min_corner[d], V(i, d));
            center[d] += V(i, d);
        }
    }
    center /= T(V.rows());

    tbb::parallel_for(0, int(V.rows()), [&](int i){
        V.row(i) = V.row(i) - center.transpose();
    });

    T max_length = std::max(max_corner[2] - min_corner[2], 
        std::max(max_corner[1] - min_corner[1], max_corner[0] - min_corner[0]));
    
    for (int i = 0; i < V.rows(); i++)
    {
        for (int d = 0; d < 3; d++)
        {
            V(i, d) = 2.0 * V(i, d) / max_length;
        }
    }
}

void VertexModel::getInitialApicalSurface(VectorXT& positions, VectorXi& indices)
{
    int offset = basal_vtx_start * 3;
    positions = undeformed.segment(0, basal_vtx_start * 3);
    
    std::vector<TV> face_centroid(basal_face_start);
    tbb::parallel_for(0, basal_face_start, [&](int i){
        TV centroid = undeformed.segment<3>(faces[i][0] * 3);
        for (int j = 1; j < faces[i].size(); j++)
            centroid += undeformed.segment<3>(faces[i][j] * 3);
        face_centroid[i] = centroid / T(faces[i].size());
    });

    positions.conservativeResize(offset + face_centroid.size() * 3);

    for (int i = 0; i < face_centroid.size(); i++)
        positions.segment<3>(offset + i * 3) = face_centroid[i];

    int face_cnt = 0;
    for (int i = 0; i < basal_face_start; i++)
        face_cnt += faces[i].size();
    indices.resize(face_cnt * 3);
    
    face_cnt = 0;
    for (int i = 0; i < basal_face_start; i++)
    {
        for (int j = 0; j < faces[i].size(); j++)
        {
            int next = (j + 1) % faces[i].size();

            indices.segment<3>(face_cnt * 3) = Eigen::Vector3i(basal_vtx_start + i, faces[i][next], faces[i][j]);
            face_cnt++;
        }   
    }
    // saveMeshVector("apical_surface.obj", positions, indices);
    
}

void VertexModel::saveMeshVector(const std::string& filename,
    const VectorXT& positions, const VectorXi& indices) const
{
    std::ofstream out(filename);
    for (int i = 0; i < (positions.rows() / 3); i ++)
        out << "v " << positions.segment<3>(i * 3).transpose() << std::endl;
    for (int i = 0; i < indices.rows() / 3; i++)
        out << "f " << (indices.segment<3>(i * 3) + IV::Ones()).transpose() << std::endl;
    out.close();
}

void VertexModel::updateFixedCentroids()
{
    fixed_cell_centroids = VectorXT::Zero(basal_face_start * 3);
    fixed_face_centroids = VectorXT::Zero(faces.size() * 3);
    iterateFaceSerial([&](VtxList& face_vtx_list, int face_idx)
    {
        VectorXT positions;
        positionsFromIndices(positions, face_vtx_list);
        
        // cell-wise volume preservation term
        if (face_idx < basal_face_start)
        {
            TV centroid;
            computeCellCentroid(face_vtx_list, centroid);
            fixed_cell_centroids.segment<3>(face_idx * 3) = centroid;
        }
        TV face_centroid;
        computeFaceCentroid(face_vtx_list, face_centroid);
        fixed_face_centroids.segment<3>(face_idx * 3) = face_centroid;
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

void VertexModel::positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state)
{
    positions = VectorXT::Zero(indices.size() * 3);
    for (int i = 0; i < indices.size(); i++)
    {
        positions.segment<3>(i * 3) = rest_state ? undeformed.segment<3>(indices[i] * 3) : deformed.segment<3>(indices[i] * 3);
    }
}

void VertexModel::getVFCellIds(VtxList& indices)
{
    indices.resize(0);
    TV min_corner, max_corner;
    computeBoundingBox(min_corner, max_corner);
    TV mid_point = 0.5 * (min_corner + max_corner);
    TV delta = max_corner - min_corner;
    iterateCellSerial([&](VtxList& face_vtx_list, int cell_idx)
    {
        TV centroid;
        computeFaceCentroid(face_vtx_list, centroid);
        T percent_y = 0.4;
        bool bottom_y = centroid[1] < min_corner[1] + (max_corner[1] - min_corner[1]) * percent_y;
        T percent_x = 0.5, percent_z = 0.2;
        bool middle_z = centroid[2] > mid_point[2] - 0.5 * delta[2] * percent_z && 
                centroid[2] < mid_point[2] + 0.5 * delta[2] * percent_z;
        bool middle_x = centroid[0] > mid_point[0] - 0.5 * delta[0] * percent_x &&
            centroid[0] < mid_point[0] + 0.5 * delta[0] * percent_x;
        if (bottom_y && middle_z && middle_x)
            indices.push_back(cell_idx);
    });
}

void VertexModel::getCellVtxAndIdx(int cell_idx, VectorXT& positions, VtxList& indices, bool rest_state)
{
    VtxList face_vtx_list = faces[cell_idx];
    indices = face_vtx_list;
    for (int idx : face_vtx_list)
        indices.push_back(idx + basal_vtx_start);
    positionsFromIndices(positions, indices, rest_state);
}

void VertexModel::getAllCellCentroids(VectorXT& cell_centroids)
{
    cell_centroids.resize(n_cells * 3);
    iterateCellParallel([&](VtxList& face_vtx_list, int cell_idx){
        TV centroid;
        computeCellCentroid(face_vtx_list, centroid);
        cell_centroids.segment<3>(cell_idx * 3) = centroid;
    });
}