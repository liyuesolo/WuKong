#ifndef VERTEXMODEL_H
#define VERTEXMODEL_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

class VertexModel
{
public:
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using IV = Vector<int, 3>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    
public:
    T sigma = 1.0;
    T alpha = 2.13;
    T gamma = 0.98;
    T B = 100.0;

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            f(cell_face, cnt);
            cnt++;
        }
    }

    template <typename OP>
    void iterateFaceParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)faces.size(), [&](int i){
            f(faces[i], i);
        });
    }

    template <typename OP>
    void iterateCellFaces(const OP& f)
    {

    }

    template <typename OP>
    void iterateEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            f(e);
        }   
    }

    template <typename OP>
    void iterateEdgeParallel(const OP& f)
    {
        tbb::parallel_for(0, (int)edges.size(), [&](int i){
            f(edges[i]);
        });
    }

public:
    // deformed and undeformed location of all vertices, u are the displacements
    VectorXT undeformed, deformed, u;

    std::vector<VtxList> faces; // all faces
    std::vector<FaceList> cell_faces; // face id list for each cell
    VectorXT cell_volume_init;
    std::vector<Edge> edges; // all edges

    int basal_vtx_start;
    int basal_face_start;
    int lateral_face_start;

    void computeCellCentroid(const VtxList& face_vtx_list, TV& centroid);
    void computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid);

    void computeCellInitialVolume();
    void vertexModelFromMesh(const std::string& filename);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeTotalEnergy(const VectorXT& _u);
    T computeResidual(const VectorXT& _u,  VectorXT& residual);

public:
    VertexModel() {}
    ~VertexModel() {}
};

#endif