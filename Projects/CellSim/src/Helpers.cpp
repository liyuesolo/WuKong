
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