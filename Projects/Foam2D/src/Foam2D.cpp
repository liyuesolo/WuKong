#include <igl/triangle/triangulate.h>
// libigl libirary must be included first
#include "../include/Foam2D.h"


void Foam2D::createRectangleScene()
{
    MatrixXT boundary_vertices;
    MatrixXi boundary_edges;
    MatrixXT points_inside_a_hole;

    boundary_vertices.resize(4,2);
    boundary_edges.resize(4,2);
    boundary_vertices << -1,-1, 1,-1, 1,1, -1, 1;
    //    -2,-2, 2,-2, 2,2, -2, 2;

    // add the edges of the squares
    boundary_edges << 0,1, 1,2, 2,3, 3,0;
        // 4,5, 5,6, 6,7, 7,4;
    
    // points_inside_a_hole.resize(1,2);
    // points_inside_a_hole << 0,0;

    MatrixXT V;
    MatrixXi F;
    std::string cmd = "pY";
    igl::triangle::triangulate(boundary_vertices,
        boundary_edges,
        points_inside_a_hole, 
        // cmd, 
        "a0.005q",
        V, F);
    
    vertices.resize(V.rows() * 2);
    for (int i = 0; i < V.rows(); i++)
        vertices.segment<2>(i * 2) = V.row(i);
    tri_face_indices.resize(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++)
        tri_face_indices.segment<3>(i * 3) = F.row(i);
}


void Foam2D::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    int n_vtx = vertices.rows() / 2, n_faces = tri_face_indices.rows() / 3;
    V.resize(n_vtx, 3); F.resize(n_faces, 3); C.resize(n_faces, 3);
    V.setZero();
    for (int i = 0; i < n_vtx; i++)
    {
        V.row(i).segment<2>(0) = vertices.segment<2>(i * 2);
    }
    for (int i = 0; i < n_faces; i++)
    {
        F.row(i) = tri_face_indices.segment<3>(i * 3);
        C.row(i) = TV3(0, 0.3, 1.0);
    }
    
}