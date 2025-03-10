#ifndef INTRINSIC_SIMULATION_H
#define INTRINSIC_SIMULATION_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>
#include <iomanip>

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/trace_geodesic.h"
#include "geometrycentral/surface/exact_geodesics.h"
#include "geometrycentral/surface/vector_heat_method.h"

#include "geometrycentral/surface/poisson_disk_sampler.h"

#include "VecMatDef.h"
#include "Util.h"
#include "Timer.h"

namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;

#define PARALLEL_GEODESIC

template <int dim>
struct VectorHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const IV& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a(d)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        }
        return h;
    }
};

template <int dim>
struct VectorPairHash
{
    typedef Vector<int, dim> IV;
    size_t operator()(const std::pair<IV, IV>& a) const{
        std::size_t h = 0;
        for (int d = 0; d < dim; ++d) {
            h ^= std::hash<int>{}(a.first(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(a.second(d)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};  

class IntrinsicSimulation
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using VtxList = std::vector<int>;
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TV3 = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using TM = Matrix<T, 3, 3>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using Face = Vector<int, 3>;
    using Triangle = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using FacePoint = std::pair<int, TV>;

    
    using gcEdge = geometrycentral::surface::Edge;
    using gcFace = geometrycentral::surface::Face;
    using gcVertex = geometrycentral::surface::Vertex;
    using Vector3 = geometrycentral::Vector3;
    using SurfacePoint = geometrycentral::surface::SurfacePoint;

    struct IxnData
    {
        TV start, end;
        T t;
        int start_vtx_idx, end_vtx_idx;
        IxnData(const TV& _start, const TV& _end, T _t, int idx0, int idx1) 
            : start(_start), end(_end), t(_t), start_vtx_idx(idx0),
            end_vtx_idx(idx1) {};
        IxnData(const TV& _start, const TV& _end, T _t) 
            : start(_start), end(_end), t(_t), start_vtx_idx(-1),
            end_vtx_idx(-1) {};
    };
    

public:
    VectorXT extrinsic_vertices;
    VectorXi extrinsic_indices;

    VectorXT intrinsic_vertices_barycentric_coords;

    VectorXT deformed, undeformed;
    VectorXT delta_u;
    VectorXT u;

    // used to interface to applications
    VectorXT deformed_temp;

    std::vector<int> dirichlet_vertices;
    std::unordered_map<int, T> dirichlet_data;

    bool run_diff_test = false;
    int max_newton_iter = 500;
    bool use_Newton = true;
    T newton_tol = 1e-6;

    bool two_way_coupling = false;

    std::vector<std::pair<SurfacePoint, gcFace>> mass_surface_points;
    std::vector<std::pair<SurfacePoint, gcFace>> mass_surface_points_undeformed;

    // used to interface to applications
    std::vector<std::pair<SurfacePoint, gcFace>> mass_surface_points_temp;

    std::vector<Edge> spring_edges;
    std::vector<T> rest_length;
    VectorXT undeformed_length;
    bool verbose = false;
    T we = 1.0;
    T ref_dis = 1.0;
    T wa = 1.0;
    bool add_area_term = false;
    bool retrace = true;
    T IRREGULAR_EPSILON = 1e-6;
    T w_reg = 0;

    std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<gcs::VertexPositionGeometry> geometry;

    std::vector<std::pair<TV, TV>> all_intrinsic_edges;

    std::vector<T> current_length;
    std::vector<std::vector<SurfacePoint>> paths;
	std::vector<std::vector<IxnData>> ixn_data_list;

    std::vector<Triangle> triangles;
    VectorXT rest_area;
    VectorXT undeformed_area;
    std::unordered_map<Edge, int, VectorHash<2>> edge_map;

    bool Euclidean = false;

    // geodesic triangle 
    bool add_geo_elasticity = false;
    std::vector<TM2> X_inv_geodesic;
    bool use_t_wrapper = true;
    
    // volume term
    T wv = 1e3;
    T rest_volume;
    bool add_volume = false;
    bool woodbury = false;

    // ====================== discrete shell ==========================
    using Hinges = Matrix<int, Eigen::Dynamic, 4>;
    
    using FaceVtx = Matrix<T, 3, 3>;
    using FaceIdx = Vector<int, 3>;
    using HingeIdx = Vector<int, 4>;
    using HingeVtx = Matrix<T, 4, 3>;
    T E = 1e3;
    T nu = 0.48;

    T E_shell = 1e3;
    T nu_shell = 0.48;
    T lambda, mu;

    T lambda_shell, mu_shell;
    int shell_dof_start = 0;
    Hinges hinges;
    std::vector<TM2> Xinv;
    VectorXT shell_rest_area;
    VectorXT thickness;
    VectorXi faces;
    
private:
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }


    Vector3 toVec3(const TV& vec) const
    {
        return Vector3{vec[0], vec[1], vec[2]};
    }

    void formEdgesFromConnection(const MatrixXi& F, MatrixXi& igl_edges);

    TV toTV(const Vector3& vec) const
    {
        return TV(vec.x, vec.y, vec.z);
    }

    void printVec3(const Vector3& vec) const
    {
        std::cout << vec.x << " " << vec.y << " " << vec.z << std::endl;
    }

    template<int dim = 2>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const VectorXT& gradent, int shift = 0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
            residual.template segment<dim>(vtx_idx[i] * dim + shift) += gradent.template segment<dim>(i * dim);
    }

    template<int dim = 2>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        VectorXT& sub_vec, int shift = 0)
    {
        sub_vec.resize(vtx_idx.size() * dim);
        sub_vec.setZero();
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<dim>(i * dim) = _vector.template segment<dim>(vtx_idx[i] * dim + shift);
        }
    }

    
    template<int dim_row=2, int dim_col=2>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.push_back(
                            Entry(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                hessian(i * dim_row + k, j * dim_col + l)
                            ));                
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addJacobianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx,
        const std::vector<int>& vtx_idx2, 
        const MatrixXT& jacobian, 
        int shift_row = 0, int shift_col=0)
    {
        
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx2.size(); j++)
            {
                int dof_j = vtx_idx2[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                        triplets.push_back(
                            Entry(
                                dof_i * dim_row + k + shift_row, 
                                dof_j * dim_col + l + shift_col, 
                                jacobian(i * dim_row + k, j * dim_col + l)
                            ));                
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addHessianMatrixEntry(
        MatrixXT& matrix_global,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian,
        int shift_row = 0, int shift_col=0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                    {
                        matrix_global(dof_i * dim_row + k + shift_row, 
                        dof_j * dim_col + l + shift_col) 
                            += hessian(i * dim_row + k, j * dim_col + l);
                    }
            }
        }
    }

    template<int dim_row=2, int dim_col=2>
    void addJacobianMatrixEntry(
        MatrixXT& matrix_global,
        const std::vector<int>& vtx_idx, 
        const std::vector<int>& vtx_idx2, 
        const MatrixXT& hessian,
        int shift_row = 0, int shift_col=0)
    {
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx2.size(); j++)
            {
                int dof_j = vtx_idx2[j];
                for (int k = 0; k < dim_row; k++)
                    for (int l = 0; l < dim_col; l++)
                    {
                        matrix_global(dof_i * dim_row + k + shift_row, 
                        dof_j * dim_col + l + shift_col) 
                            += hessian(i * dim_row + k, j * dim_col + l);
                    }
            }
        }
    }

    template<int dim0=2, int dim1=2>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const MatrixXT& hessian_block,
        int shift_row = 0, int shift_col=0)
    {

        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];
        
        for (int k = 0; k < dim0; k++)
            for (int l = 0; l < dim1; l++)
            {
                triplets.push_back(Entry(dof_i * dim0 + k + shift_row, 
                    dof_j * dim1 + l + shift_col, 
                    hessian_block(k, l)));
            }
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    void edgeLengthHessian(const TV& v0, const TV& v1, Matrix<T, 6, 6>& hess)
    {
        TV dir = (v1-v0).normalized();
        hess.setZero();
        hess.block(0, 0, 3, 3) = (TM::Identity() - dir * dir.transpose())/(v1 - v0).norm();
        hess.block(3, 3, 3, 3) = hess.block(0, 0, 3, 3);
        hess.block(0, 3, 3, 3) = -hess.block(0, 0, 3, 3);
        hess.block(3, 0, 3, 3) = -hess.block(0, 0, 3, 3);
    }

    bool hasSmallSegment(const std::vector<SurfacePoint>& path);

    bool closeToIrregular(const SurfacePoint& point);

    void areaLengthFormula(const Eigen::Matrix<double,3,1> & l, double& energy);
    void areaLengthFormulaGradient(const Eigen::Matrix<double,3,1> & l, Eigen::Matrix<double, 3, 1>& energygradient);
    void areaLengthFormulaHessian(const Eigen::Matrix<double,3,1> & l, Eigen::Matrix<double, 3, 3>& energyhessian);
    
    void getTriangleEdges(const Triangle& tri, Edge& e0, Edge& e1, Edge& e2)
    {
        Edge _e0(tri[0], tri[1]), _e1(tri[1], tri[2]), _e2(tri[0], tri[2]);
        if ((spring_edges[edge_map[_e0]] - _e0).norm() < 1e-6)
            e0 = _e0;
        else
            e0 = Edge(_e0[1], _e0[0]);
        if ((spring_edges[edge_map[_e1]] - _e1).norm() < 1e-6)
            e1 = _e1;
        else
            e1 = Edge(_e1[1], _e1[0]);
        if ((spring_edges[edge_map[_e2]] - _e2).norm() < 1e-6)
            e2 = _e2;
        else
            e2 = Edge(_e2[1], _e2[0]);
    }

    void getTriangleIndex(const gcFace& fi, IV& tri)
    {
        tri[0] = fi.halfedge().vertex().getIndex();
        tri[1] = fi.halfedge().next().vertex().getIndex();
        tri[2] = fi.halfedge().next().next().vertex().getIndex();
    }

    template<int order>
    T wrapper_t(T t, T eps = 1e-6)
    {
        if constexpr (order == 0)
        {
            if (t < eps && t >= 0)
            {
                return -1.0/std::pow(eps, 2) * std::pow(t, 3) + 2.0 / eps * std::pow(t, 2);
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return t;
            }
            else 
            {
                return -1.0/std::pow(eps, 2) * std::pow(t-1, 3) - 2.0 / eps * std::pow(t-1.0, 2) + 1.0;
            }
        }
        else if constexpr (order == 1)
        {
            if (t < eps && t >= 0)
            {
                return -3.0/std::pow(eps, 2) * std::pow(t, 2) + 4.0 / eps * t;
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return 1.0;
            }
            else 
            {
                return -3.0/std::pow(eps, 2) * std::pow(t-1, 2) - 4.0 / eps * (t-1);
            }
        }
        else if constexpr (order == 2)
        {
            if (t < eps && t >= 0)
            {
                return -6.0/std::pow(eps, 2) * (t - 1.0) + 4.0 / eps;
            }
            else if (t >= eps && t <= 1.0 - eps)
            {
                return 0.0;
            }
            else 
            {
                return -6.0/std::pow(eps, 2) * (t - 1.0) - 4.0 / eps;
            }
        }
        return 0.0;
    }

    template<int order>
    T wrapper_cubic(T t)
    {
        if constexpr (order == 0)
        {
            return -2.0 * std::pow(t, 3) + 3.0 * std::pow(t, 2);
        }
        else if constexpr (order == 1)
        {
            return -6.0 * std::pow(t, 2) + 6.0 * t;
        }
        else if constexpr (order == 2)
        {
            return -12.0 * t + 6.0;
        }
        return 0.0;
    }

    // ====================== discrete shell ==========================
    void updateLameParameters()
    {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }

    void updateShellLameParameters()
    {
        lambda_shell = E_shell * nu_shell / (1.0 + nu_shell) / (1.0 - 2.0 * nu_shell);
        mu_shell = E_shell / 2.0 / (1.0 + nu_shell);
    }
    HingeVtx getHingeVtxDeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = deformed.segment<3>(hi[i]*3 + shell_dof_start);
        }
        return cellx;
    }

    HingeVtx getHingeVtxUndeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = undeformed.segment<3>(hi[i]*3 + shell_dof_start);
        }
        return cellx;
    }

    FaceVtx getFaceVtxDeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = deformed.segment<3>(nodal_indices[i]*3 + shell_dof_start);
        }
        return cellx;
    }

    FaceVtx getFaceVtxUndeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = undeformed.segment<3>(nodal_indices[i]*3 + shell_dof_start);
        }
        return cellx;
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.rows()/3; i++)
            f(i);
    }

    template <typename OP>
    void iterateTriangleSerial(const OP& f)
    {
        for (int i = 0; i < triangles.size(); i++)
            f(triangles[i], i);
    }

    template <typename OP>
    void iterateHingeSerial(const OP& f)
    {
        for (int i = 0; i < hinges.rows(); i++)
        {
            const Vector<int, 4> nodes = hinges.row(i);
            f(nodes);   
        }
    }
    // ================================================================

public:
    void computeExactGeodesicEdgeFlip(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);
    void computeExactGeodesic(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);
    void computeGeodesicHeatMethod(const SurfacePoint& va, const SurfacePoint& vb, 
        T& dis, std::vector<SurfacePoint>& path, 
        std::vector<IxnData>& ixn_data, 
        bool trace_path = false);
    
    T computeTotalEnergy();
    T computeResidual(VectorXT& residual);
    void buildSystemMatrix(StiffnessMatrix& K);
    void buildSystemMatrixWoodbury(StiffnessMatrix& K, MatrixXT& UV);
    T lineSearchNewton(const VectorXT& residual);
    void updateCurrentState(bool trace = true);
    void traceGeodesics();
    void reset();

    bool linearSolve(StiffnessMatrix& K, const VectorXT& residual, VectorXT& du);
    bool linearSolveWoodbury(StiffnessMatrix& K, const MatrixXT& UV,
         const VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    bool advanceOneStep(int step);

    void updateVisualization(bool all_edges = false);
    void checkHessianPD(bool save_result = true);
    void checkInformation();

    // EdgeTerms.cpp
    void computeGeodesicLengthGradient(const Edge& edge, Vector<T, 4>& dldw);
    void computeGeodesicLengthHessian(const Edge& edge, Matrix<T, 4, 4>& d2ldw2);
    void computeGeodesicLengthGradientAndHessian(const Edge& edge, 
        Vector<T, 4>& dldw, Matrix<T, 4, 4>& d2ldw2);

    void computeGeodesicLengthGradientCoupled(const Edge& edge, VectorXT& dldq, 
        std::vector<int>& dof_indices);
    void computeGeodesicLengthGradientAndHessianCoupled(const Edge& edge, 
        VectorXT& dldq, MatrixXT& d2qdw2, std::vector<int>& dof_indices);

    void addEdgeLengthEnergy(T w, T& energy);
    void addEdgeLengthForceEntries(T w, VectorXT& residual);
    void addEdgeLengthHessianEntries(T w, std::vector<Entry>& entries);

    // AreaTerms.cpp
    void computeAllTriangleArea(VectorXT& area);
    void addTriangleAreaEnergy(T w, T& energy);
    void addTriangleAreaForceEntries(T w, VectorXT& residual);
    void addTriangleAreaHessianEntries(T w, std::vector<Entry>& entries);

    // DerivativeTest.cpp    
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalGradient(bool perturb = false);
    void checkTotalHessian(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    // Scene.cpp
    void initializeDiscreteShell();
    void expandBaseMesh(T increment);
    void initializeNetworkData(const std::vector<Edge>& edges);
    void initializeSceneCheckingSmoothness();
    void initializeMassSpringSceneExactGeodesic();
    void initializeMassSpringDebugScene();
    void initializeTriangleDebugScene();
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);
    void movePointsPlotEnergy();

    void massPointPosition(int idx, TV& pos);
    void moveMassPoint(int idx, int bc);
    void getAllPointsPosition(VectorXT& positions);
    void getMarkerPointsPosition(VectorXT& positions);
    
    void getCurrentMassPointConfiguration(std::vector<std::pair<SurfacePoint, gcFace>>& configuration);


    // GeoElasticity.cpp
    void computeGeodesicTriangleRestShape();
    void addGeodesicNHEnergy(T& energy);
    void addGeodesicNHForceEntry(VectorXT& residual);
    void addGeodesicNHHessianEntry(std::vector<Entry>& entries);

    // ====================== DiscreteShell.cpp ==========================
    void buildHingeStructure();
    void computeRestShape();
    int nFaces () { return faces.rows() / 3; }
    void updateRestshape() { computeRestShape(); }
    void addShellEnergy(T& energy);
    void addShellForceEntry(VectorXT& residual);
    void addShellHessianEntries(std::vector<Entry>& entries);

    // Volume.cpp
    void addVolumePreservationEnergy(T w, T& energy);
    void addVolumePreservationForceEntries(T w, VectorXT& residual);
    void addVolumePreservationHessianEntries(T w, std::vector<Entry>& entries,
        MatrixXT& WoodBuryMatrix);
    T computeVolume(bool use_rest_shape = false);


public:
    IntrinsicSimulation() {}
    ~IntrinsicSimulation() {}
};

#endif