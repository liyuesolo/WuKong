#ifndef VERTEXMODEL_H
#define VERTEXMODEL_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <tgmath.h>
#include "VecMatDef.h"
#include "SDF.h"
#include "Timer.h"

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

enum Region
{
    Apical, Basal, Lateral, ALL
};

enum ContractionType
{
    ApicalOnly, ApicalBasal, ALLEdges 
};

class VertexModel
{
public:
    using TV = Vector<double, 3>;
    using TV2 = Vector<double, 2>;
    using TM2 = Matrix<double, 2, 2>;
    using TM3 = Matrix<double, 3, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TetVtx = Matrix<T, 3, 4>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;
    
public:
    // iterators
    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

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
    void iterateContractingFaceSerial(const OP& f)
    {
        for (int idx : contracting_faces)
        {
            VtxList cell_face = faces[idx];
            f(cell_face, idx);
        }
    }

    

    template <typename OP>
    void iterateCellCentroidDoFSerial(const OP& f)
    {
        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            if (cnt < basal_face_start)
            {
                VtxList face_vtx_list = cell_face;
                VtxList cell_vtx_list = face_vtx_list;
                for (int idx : face_vtx_list)
                    cell_vtx_list.push_back(idx + basal_vtx_start);
                
                VectorXT positions;
                positionsFromIndices(positions, cell_vtx_list);       

                int n_point = cell_face.size();
                VectorXT centroids((n_point + 3) * 3);
                centroids.segment<3>(0) = fixed_cell_centroids.segment<3>(cnt * 3);
                centroids.segment<3>(3) = fixed_face_centroids.segment<3>(cnt * 3);
                centroids.segment<3>(6) = fixed_face_centroids.segment<3>((cnt + basal_face_start) * 3);
                // lateral tets
                for (int i = 0; i < n_point; i++)
                {
                    int j = (i + 1) % n_point;
                    int lateral_face_idx = lateral_edge_face_map[Edge(cell_face[i], cell_face[j])];
                    centroids.segment<3>((3 + i) * 3) = fixed_face_centroids.segment<3>(lateral_face_idx * 3);
                }
                f(positions, centroids, cell_vtx_list, cnt);
            }
            cnt++;
        }
    }

    template <typename OP>
    void iterateCentroidTetsSerial(const OP& f)
    {

        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            if (cnt < basal_face_start)
            {
                VtxList face_vtx_list = cell_face;
                VtxList cell_vtx_list = face_vtx_list;
                for (int idx : face_vtx_list)
                    cell_vtx_list.push_back(idx + basal_vtx_start);
                
                VectorXT positions;
                positionsFromIndices(positions, cell_vtx_list);       

                std::vector<TetVtx> cell_tets;
                std::vector<VtxList> vtx_ids;

                int n_point = cell_face.size();
                // apical tets
                for (int i = 0; i < n_point; i++)
                {
                    int j = (i + 1) % n_point;
                    TetVtx tet_vtx;
                    VtxList global_indices(2);
                    tet_vtx.col(0) = positions.segment<3>(j * 3);
                    tet_vtx.col(1) = positions.segment<3>(i * 3);
                    tet_vtx.col(2) = fixed_face_centroids.segment<3>(cnt * 3);
                    tet_vtx.col(3) = fixed_cell_centroids.segment<3>(cnt * 3);
                    cell_tets.push_back(tet_vtx);
                    global_indices[0] = cell_vtx_list[j];
                    global_indices[1] = cell_vtx_list[i];
                    vtx_ids.push_back(global_indices);
                }

                // basal tets
                for (int i = 0; i < n_point; i++)
                {
                    int j = (i + 1) % n_point;
                    TetVtx tet_vtx;
                    VtxList global_indices(2);
                    tet_vtx.col(0) = positions.segment<3>((i + n_point) * 3);
                    tet_vtx.col(1) = positions.segment<3>((j + n_point) * 3);
                    tet_vtx.col(2) = fixed_face_centroids.segment<3>((cnt + basal_face_start) * 3);
                    tet_vtx.col(3) = fixed_cell_centroids.segment<3>(cnt * 3);
                    cell_tets.push_back(tet_vtx);
                    global_indices[0] = cell_vtx_list[i + n_point];
                    global_indices[1] = cell_vtx_list[j + n_point];
                    vtx_ids.push_back(global_indices);
                }

                // lateral tets
                for (int i = 0; i < n_point; i++)
                {
                    int j = (i + 1) % n_point;

                    int lateral_face_idx = lateral_edge_face_map[Edge(cell_face[i], cell_face[j])];

                    std::vector<int> lateral_vtx = {i, j, j + n_point, i + n_point};
                    for (int k = 0; k < 4; k++)
                    {
                        int l = (k + 1) % 4;
                        TetVtx tet_vtx;
                        VtxList global_indices(2);
                        tet_vtx.col(0) = positions.segment<3>(lateral_vtx[k] * 3);
                        tet_vtx.col(1) = positions.segment<3>(lateral_vtx[l] * 3);
                        tet_vtx.col(2) = fixed_face_centroids.segment<3>(lateral_face_idx * 3);
                        tet_vtx.col(3) = fixed_cell_centroids.segment<3>(cnt * 3);
                        cell_tets.push_back(tet_vtx);
                        global_indices[0] = cell_vtx_list[lateral_vtx[k]];
                        global_indices[1] = cell_vtx_list[lateral_vtx[l]];
                        vtx_ids.push_back(global_indices);
                    }
                    
                }
                
                f(cell_tets, vtx_ids);
            }
            cnt++;
        }
    }

    template <typename OP>
    void iterateFixedTetsSerial(const OP& f)
    {
        // this shift is because of a stupid squencing that I haven't fixed
        auto shiftIndex = [&](VtxList& indices)
        {
            VtxList indices_cp = indices;
            int n_pt = indices.size() / 2;
            for (int i = 0; i < n_pt; i++)
            {
                indices_cp[i] = indices[2 * n_pt - 1 - i];	
                indices_cp[i + n_pt] = indices[n_pt - 1 - i];	
            }
            indices = indices_cp;
        };

        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            
            if (cnt < basal_face_start)
            {
                VtxList face_vtx_list = cell_face;
                VtxList cell_vtx_list = face_vtx_list;
                for (int idx : face_vtx_list)
                    cell_vtx_list.push_back(idx + basal_vtx_start);
                shiftIndex(cell_vtx_list);
                
                VectorXT positions, postions_undeformed;
                positionsFromIndices(positions, cell_vtx_list);
                positionsFromIndices(postions_undeformed, cell_vtx_list, true);
                
                std::vector<std::vector<int>> prism_tet_indexing;

                if (cell_face.size() == 4)
                    continue;
                else if (cell_face.size() == 5)
                    prism_tet_indexing = tet_index_penta_prism;
                else if (cell_face.size() == 6)
                    prism_tet_indexing = tet_index_hexa_prism;

                for (auto tet : prism_tet_indexing)
                {
                    
                    Matrix<T, 3, 4> x_deformed, x_undeformed;
                    VtxList global_idx(4);
                    for (int i = 0; i < 4; i++)
                    {
                        global_idx[i] = cell_vtx_list[tet[i]];
                        x_deformed.col(i) = positions.segment<3>(tet[i] * 3);
                        x_undeformed.col(i) = postions_undeformed.segment<3>(tet[i] * 3);
                    }
                    f(x_deformed, x_undeformed, global_idx);
                }
            }
            cnt++;
        }
    }

    template <typename OP>
    void iterateFixedYolkTetsSerial(const OP& f)
    {
        int cnt = 0;
        for (VtxList& cell_face : faces)
        {
            if (cnt < lateral_face_start && cnt >= basal_face_start)
            {
                VectorXT positions;
                positionsFromIndices(positions, cell_face);
                std::vector<VtxList> tet_indices;
                if (cell_face.size() == 4)
                    continue;
                else if (cell_face.size() == 5)
                    tet_indices = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}};
                else if (cell_face.size() == 6)
                    tet_indices = {{0, 1, 2}, {0, 2, 3}, {0, 3, 5}, {5, 3, 4}};
                for (auto tet : tet_indices)
                {
                    Matrix<T, 3, 3> tet_vtx;
                    VtxList global_idx(3);
                    for (int i = 0; i < 3; i++)
                    {
                        tet_vtx.col(i) = positions.segment<3>(tet[i] * 3);
                        global_idx[i] = cell_face[tet[i]];
                    }   
                    f(tet_vtx, global_idx);
                }
            }
            cnt++;
        }
    }

    template <typename OP>
    void iterateBasalFaceSerial(const OP& f)
    {
        int cnt = -1;
        for (VtxList& cell_face : faces)
        {
            cnt++;
            if (cnt < basal_face_start || cnt >= lateral_face_start)
                continue;
            f(cell_face, cnt);
        }
    }

    template <typename OP>
    void iterateApicalFaceSerial(const OP& f)
    {
        int cnt = -1;
        for (VtxList& cell_face : faces)
        {
            cnt++;
            if (cnt < basal_face_start)
                f(cell_face, cnt);
            else
                break;
        }
    }

    template <typename OP>
    void iterateCellParallel(const OP& f)
    {
        tbb::parallel_for(0, basal_face_start, [&](int i)
        {
            VtxList face_vtx_list = faces[i];
            f(face_vtx_list, i);
        });
    }

    template <typename OP>
    void iterateCellSerial(const OP& f)
    {
        for (int i = 0; i < basal_face_start; i++)
        {
            VtxList face_vtx_list = faces[i];
            f(face_vtx_list, i);
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
    void iterateEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            f(e);
        }   
    }

    template <typename OP>
    void iterateContractingEdgeSerial(const OP& f)
    {
        for (Edge& e : contracting_edges)
        {
            f(e);
        }   
    }

    template <typename OP>
    void iterateLateralEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            bool case1 = e[0] < basal_vtx_start && e[1] >= basal_vtx_start;
            bool case2 = e[1] < basal_vtx_start && e[0] >= basal_vtx_start;
            if (case1 || case2)
                f(e);
        }   
    }

    template <typename OP>
    void iterateBasalEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            if (e[0] >= basal_vtx_start && e[1] >= basal_vtx_start)
                f(e);
        }   
    }

    template <typename OP>
    void iterateApicalEdgeSerial(const OP& f)
    {
        for (Edge& e : edges)
        {
            if (e[0] < basal_vtx_start && e[1] < basal_vtx_start)
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

    std::vector<std::vector<int>> tet_index_penta_prism = 
    {
        {3, 8, 9, 0},
        {3, 9, 4, 0},
        {7, 0, 2, 1},
        {9, 8, 5, 0},
        {6, 0, 7, 1},
        {5, 0, 7, 6},
        {5, 8, 7, 0},
        {8, 0, 2, 7},
        {8, 3, 2, 0}
    };

    std::vector<std::vector<int>> tet_index_hexa_prism = 
    {
        {9, 2, 10, 3},
        {2, 10, 3, 4},
        {1, 11, 0, 7},
        {9, 2, 8, 10},
        {10, 1, 7, 11},
        {0, 11, 6, 7},
        {1, 11, 4, 5},
        {2, 10, 1, 8},
        {2, 10, 4, 1},
        {1, 11, 5, 0},
        {10, 1, 8, 7},
        {10, 1, 11, 4}
    };

    int scene_type = 0;

    //dynamics
    bool dynamics = false;
    bool add_mass = false;
    T eta = 1.0;
    T dt = 1.0;
    VectorXT vtx_vel;

    VectorXT vtx_mass;
    T density = 1.0;


    // deformed and undeformed location of all vertices, u are the displacements
    VectorXT undeformed, deformed, u;

    VectorXT f;

    T TET_VOL_MIN = 1e-8;

    std::vector<VtxList> faces; // all faces
    std::vector<FaceList> cell_faces; // face id list for each cell
    VectorXT cell_volume_init;
    std::vector<Edge> edges; // all edges
    std::vector<Edge> contracting_edges;
    std::vector<int> contracting_faces;
    VectorXT fixed_cell_centroids;
    VectorXT fixed_face_centroids;
    std::vector<VtxList> cell_face_indices;
    VectorXT tet_vol_init;

    std::unordered_map<Edge, int, VectorHash<2>> lateral_edge_face_map;
    
    bool use_pre_build_structure = false;
    std::unordered_map<IV2, std::ptrdiff_t, VectorHash<2>> ij_to_value_array;
    std::unordered_map<uint64_t, std::ptrdiff_t> ijv;
    std::vector<StorageIndex> inner_indices, outer_indices;
    std::vector<T> val;
    StiffnessMatrix Hessian_copy;

    int num_nodes;
    int basal_vtx_start;
    int basal_face_start;
    int lateral_face_start;

    int n_cells;
    int n_edges;

    T unit = 1.0;
    int resolution = 0;

    // vertex model
    T sigma = 1.0;
    T alpha = 2.13;
    T gamma = 0.98;
    T Gamma = 5.0;
    T B = 100.0;
    T By = 100.0;
    T pressure_constant = 1e1;
    T Rc = 1.2;
    T bound_coeff = 10e-10;
    int bound_power = 4;
    T total_volume = 0.0;
    T tet_vol_penalty = 1e6;
    
    T membrane_dhat = 1e-3;

    // ALM
    T kappa = 1.0;
    T kappa_max = 1e9;
    VectorXT lambda_cell_vol;

    // IPC
    T barrier_distance = 1e-5;
    T barrier_weight = 1e6;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_faces;
    bool add_basal_faces_ipc = false;
    T add_friction = false;
    T friction_mu = 0.5;
    T epsv_times_h = 1e-5;
    T nu;
    T E;

    // single tet barrier
    bool add_tet_vol_barrier = false;
    T tet_vol_barrier_dhat = 1e-6;
    T tet_vol_barrier_w = 1e6;
    T tet_vol_qubic_w = 1e3;
    T log_active_percentage = 0.01;
    
    T add_qubic_unilateral_term = false;
    T add_log_tet_barrier = false;
    T qubic_active_percentage = 0.5;


    // yolk tet barrier
    bool add_yolk_tet_barrier = false;
    T yolk_tet_vol_barrier_dhat = 1e-3;
    T yolk_tet_vol_barrier_w = 1e6;

    bool single_prism = false;
    bool woodbury = false;
    bool use_alm_on_cell_volume = false;

    bool run_diff_test = false;
    bool add_yolk_volume = true;
    bool use_cell_centroid = false;
    bool use_face_centroid = false;
    bool add_contraction_term = false;
    bool use_yolk_pressure = false;
    bool use_sphere_radius_bound = false;
    bool sphere_bound_penalty = false;
    bool sphere_bound_barrier = false;
    bool use_ipc_contact = false;
    bool use_fixed_centroid = false;
    bool add_perivitelline_liquid_volume = false;
    bool use_perivitelline_liquid_pressure = false;
    bool contract_apical_face = true;
    bool preserve_tet_vol = false;
    bool fixed_tet = false;
    bool use_elastic_potential = false;
    bool project_block_hessian_PD = false;
    bool has_rest_shape = false;
    bool add_centroid_points = false;
    bool check_all_vtx_membrane = false;
    bool add_area_term = true;
    bool print_force_norm = false;
    bool profile = false;
    bool lower_triangular = false;

    bool use_test_mesh = false;

    Timer profile_timer;

    VectorXT rest_length;
    TV mesh_centroid;
    T yolk_vol_init;
    T perivitelline_vol_init;
    T Bp = 1e4;
    T perivitelline_pressure = 0.1;
    T weights_all_edges = 0.1;
    ContractionType contracting_type = ApicalBasal;
    // VdbLevelSetSDF sdf;
    IMLS sdf;

    bool use_sdf_boundary = false;

    //for inverse problems
    bool assign_per_edge_weight = false;
    VectorXT edge_weights;
    VectorXT edge_weight_mask;

    std::unordered_map<int, T> dirichlet_data;

    // VertexModel.cpp
    T computeLineSearchInitStepsize(const VectorXT& _u, const VectorXT& du, bool verbose = false);
    void computeCellInfo();

    void updateALMData(const VectorXT& _u);
    void computeLinearModes();
    
    T computeTotalVolumeFromApicalSurface();
    void projectDirichletDoFMatrix(StiffnessMatrix& A, 
        const std::unordered_map<int, T>& data);
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    void buildSystemMatrixWoodbury(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV);
    void buildSystemMatrixWoodburyFaster(const VectorXT& _u, 
        StiffnessMatrix& K, MatrixXT& UV);
    T computeTotalEnergy(const VectorXT& _u, bool verbose = false, bool add_to_deform = true);
    T computeResidual(const VectorXT& _u,  VectorXT& residual, bool verbose = false);
    void sdfFromHighResDualMesh(const std::string& filename);

    T computeInsideMembraneStepSize(const VectorXT& _u, const VectorXT& du);

    //EdgeTerms.cpp
    void computeRestLength();
    void addEdgeEnergy(Region region, T w, T& energy);
    void addEdgeForceEntries(Region region, T w, VectorXT& residual);
    void addEdgeHessianEntries(Region region, T w, 
        std::vector<Entry>& entries, bool projectPD = false);
    
    void addPerEdgeEnergy(T& energy);
    void addPerEdgeForceEntries(VectorXT& residual);
    void addPerEdgeHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

    void addEdgeContractionEnergy(T w, T& energy);
    void addEdgeContractionForceEntries(T w, VectorXT& residual);
    void addEdgeContractionHessianEntries(T w, std::vector<Entry>& entries, bool projectPD = false);

    // AreaTerms.cpp
    void addFaceContractionEnergy(T w, T& energy);
    void addFaceContractionForceEntries(T w, VectorXT& residual);
    void addFaceContractionHessianEntries(T w, std::vector<Entry>& entries, bool projectPD = false);

    T computeAreaEnergy(const VectorXT& _u);
    void addFaceAreaEnergy(Region region, T w, T& energy);
    void addFaceAreaForceEntries(Region region, T w, VectorXT& residual);
    void addFaceAreaHessianEntries(Region region, T w, 
        std::vector<Entry>& entries, bool projectPD = false);

    void addFaceAreaEnergyWithRestShape(Region region, T w, T& energy);
    void addFaceAreaForceEntriesWithRestShape(Region region, T w, VectorXT& residual);
    void addFaceAreaHessianEntriesWithRestShape(Region region, T w, 
        std::vector<Entry>& entries, bool projectPD = false);

    // CellVolume.cpp
    void computeTetVolInitial();
    void computeCellVolumeHessianEigenValues(VectorXT& cell_hessian_evs);
    void computeTetVolCurent(VectorXT& tet_vol_current);
    void computeVolumeAllCells(VectorXT& cell_volume_list);    
    void addCellVolumePreservationEnergy(T& energy);
    void addCellVolumePreservationForceEntries(VectorXT& residual);
    void addCellVolumePreservationHessianEntries(std::vector<Entry>& entries, 
        bool projectPD = false);
    void addCellVolumePreservationEnergyFixedCentroid(T& energy);
    void addCellVolumePreservationForceEntriesFixedCentroid(VectorXT& residual);
    void addCellVolumePreservationHessianEntriesFixedCentroid(std::vector<Entry>& entries, 
        bool projectPD = false);

    void addTetVolumePreservationEnergy(T& energy);
    void addTetVolumePreservationForceEntries(VectorXT& residual);
    void addTetVolumePreservationHessianEntries(std::vector<Entry>& entries, 
        bool projectPD = false);

    // IPC.cpp
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void computeIPCRestData();
    void updateIPCVertices(const VectorXT& _u);
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries,
        bool projectPD = false);

    // Yolk.cpp
    T computeYolkVolume(bool verbose = false);
    void addYolkVolumePreservationEnergy(T& energy);
    void addYolkVolumePreservationForceEntries(VectorXT& residual);
    void addYolkVolumePreservationHessianEntries(std::vector<Entry>& entries,
        MatrixXT& WoodBuryMatrix, bool projectPD = false);
    T computeYolkInversionFreeStepSize(const VectorXT& _u, const VectorXT& du);
    void addYolkTetLogBarrierEnergy(T& energy);
    void addYolkTetLogBarrierForceEneries(VectorXT& residual);
    void addYolkTetLogBarrierHessianEneries(std::vector<Entry>& entries, bool projectPD = false);

    // Perivitelline.cpp
    T computeInitialApicalVolumeWithOffset(const VectorXT& normals, T epsilon);
    void addPerivitellineVolumePreservationEnergy(T& energy);
    void addPerivitellineVolumePreservationForceEntries(VectorXT& residual);
    void addPerivitellineVolumePreservationHessianEntries(std::vector<Entry>& entries,
    MatrixXT& WoodBuryMatrix, bool projectPD = false);

    //MembraneTerms.cpp
    void addMembraneBoundEnergy(T& energy);
    void addMembraneBoundForceEntries(VectorXT& residual);
    void addMembraneBoundHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

    void addMembraneSDFBoundEnergy(T& energy);
    void addMembraneSDFBoundForceEntries(VectorXT& residual);
    void addMembraneSDFBoundHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

    // ElasticityTerms.cpp
    void addElasticityEnergy(T& energy);
    void addElasticityForceEntries(VectorXT& residual);
    void addElasticityHessianEntries(std::vector<Entry>& entries, bool projectPD = false);

    // VolumeBarrier.cpp
    void computeTetBarrierWeightMask(const VectorXT& positions, 
        const VtxList& face_vtx_list, VectorXT& mask_log_term, 
        VectorXT& mask_qubic_term, T cell_volume);
    void computeCentroidTetVolume(const VectorXT& positions, 
        const VtxList& face_vtx_list, VectorXT& tets_volume,
        std::vector<TetVtx>& tets);
    T computeInversionFreeStepSize(const VectorXT& _u, const VectorXT& du);
    void addFixedTetLogBarrierEnergy(T& energy);
    void addFixedTetLogBarrierForceEneries(VectorXT& residual);
    void addFixedTetLogBarrierHessianEneries(std::vector<Entry>& entries, bool projectPD = false);
    void addSingleTetVolBarrierEnergy(T& energy);
    void addSingleTetVolBarrierForceEntries(VectorXT& residual);
    void addSingleTetVolBarrierHessianEntries(std::vector<Entry>& entries, 
        bool projectPD = false);

    // Dynamics.cpp
    void computeNodalMass();
    void addInertialEnergy(T& energy);
    void addInertialForceEntries(VectorXT& residual);
    void addInertialHessianEntries(std::vector<Entry>& entires);

    // Helpers.cpp
    void getCellVtxAndIdx(int cell_idx, VectorXT& positions, VtxList& indices, bool rest_state = false);
    void getVFCellIds(VtxList& indices);
    void saveSingleCellEdges(const std::string& filename, 
        const VtxList& indices, const VectorXT& positions, bool save_tets = false) const;

    void normalizeToUnit(MatrixXT& V);
    void saveMeshVector(const std::string& filename,
        const VectorXT& positions, const VectorXi& indices) const;
    void getInitialApicalSurface(VectorXT& positions, VectorXi& indices);
    void removeAllTerms();
    void positionsFromIndices(VectorXT& positions, const VtxList& indices, bool rest_state = false);
    void computeCellCentroid(const VtxList& face_vtx_list, TV& centroid);
    void computeFaceCentroid(const VtxList& face_vtx_list, TV& centroid);
    void updateFixedCentroids();
    void getAllCellCentroids(VectorXT& cell_centroids);

    // DerivativeTest.cpp
    void checkTotalGradient(bool perturb = false);
    void checkTotalHessian(bool perturb = false);

    void checkTotalHessianScale(bool perturb = false);
    void checkTotalGradientScale(bool perturb = false);

    // scene.cpp
    
    void computeBoundingBox(TV& min_corner, TV& max_corner);    
    void vertexModelFromMesh(const std::string& filename);
    void addTestPrism(int edge);
    void addTestPrismGrid(int n_row, int n_col);
    void initializeContractionData();
    void saveIPCData(const std::string& folder, int iter = 0, bool save_edges = false);
    void saveCellMesh(int iter = 0);
    void approximateMembraneThickness();
    void constructAnnulusScene();


    //Visualization.cpp
    void getOutsideVtx(Eigen::MatrixXd& points, Eigen::MatrixXd& color, int sdf_test_sample_idx_offset);
    void loadMeshAndSaveCentroid(const std::string& folder, int start, int end);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool rest_state = false);
    void sampleBoundingSurface(Eigen::MatrixXd& V);
    void splitCellsForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool a_bit = false);
    void splitYolkForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool a_bit = false);

    void appendCylinderOnContractingEdges(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C);
    void appendCylinderOnApicalEdges(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C);

    void getYolkForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::MatrixXd& C, bool rest_shape = false);
    void saveIndividualCellsWithOffset();

    void saveBasalSurfaceMesh(const std::string& filename, bool invert_normal = true);
    void saveAPrism(const std::string& filename, const VtxList& face_vtx_list);
    void saveLowVolumeTets(const std::string& filename);

    // Misc.cpp
    void saveHexTetsStep(int iteration);
    void computeCubeVolumeFromTet(const Vector<T, 24>& prism_vertices, T& volume);
    void computePentaPrismVolumeFromTet(const Vector<T, 30>& prism_vertices, T& volume);
    void computeHexPrismVolumeFromTet(const Vector<T, 36>& prism_vertices, T& volume, int iter = 0);
    void computeCubeVolumeCentroid(const Vector<T, 24>& prism_vertices, T& volume);

    // void computePentaPrismTetVol(const Vector<T, 30>& prism_vertices, Vector<T, > tet_vol);
    void computePentaPrismTetVol(const Vector<T, 30>& prism_vertices, Vector<T, 9>& tet_vol);
    void computeHexPrismTetVol(const Vector<T, 36>& prism_vertices, Vector<T, 12>& tet_vol);


    // SensitivityDerivatives.cpp
    void dfdpWeights(MatrixXT& dfdp);
    void dfdpWeightsSparse(StiffnessMatrix& dfdp);
    void dfdpWeightsDense(MatrixXT& dfdp);
    void edgeWeightsSGNMatrix(StiffnessMatrix& mat_SGN, std::vector<Entry>& d2Odx2_entries);

    void dOdpEdgeWeightsFromLambda(const VectorXT& lambda, VectorXT& dOdp);
    void dOdpFromdxdpEdgeWeights(const VectorXT& dOdu, VectorXT& dOdp);
    void dxdpFromdxdpEdgeWeights(MatrixXT& dxdp);
    void multiplyDpWithDfdp(VectorXT& result, const VectorXT& dp);
    void dfdpWeightsFD(MatrixXT& dfdp);
    void computededp(VectorXT& dedp);

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A, bool lower_tri_only = false)
    {
        std::vector<Entry> triplets;

        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
            {
                if (!lower_tri_only)
                    triplets.push_back(Entry(it.row(), it.col(), it.value()));
                else
                {
                    if (it.row() <= it.col())
                        triplets.push_back(Entry(it.row(), it.col(), it.value()));
                }
            }
        return triplets;
    }
    
    template<int dim>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong hessian block size" << std::endl;
        int n_curr = triplets.size();
        int cnt = 0;
        
        int n_entry = (1 + vtx_idx.size() * 3) * vtx_idx.size() * 3 / 2;
        
        // triplets.resize(n_curr + cnt);

        std::vector<uint64_t> keys(vtx_idx.size() * vtx_idx.size() * 3 * 3);
        std::vector<T> values(vtx_idx.size() * vtx_idx.size() * 3 * 3);
        std::vector<Entry> local_triplets(vtx_idx.size() * vtx_idx.size() * 3 * 3);

        auto offset = [&](int i, int j, int k, int l) -> int
        {
            return i * vtx_idx.size() * 3 * 3 + j * 3 * 3 + k * 3 + l;
        };

        if (use_pre_build_structure)
        {
            // tbb::parallel_for(0, (int)vtx_idx.size(), [&](int i){
            for (int i = 0; i < vtx_idx.size(); i++){
                int dof_i = vtx_idx[i];
                for (int j = 0; j < vtx_idx.size(); j++)
                {
                    int dof_j = vtx_idx[j];
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                        {
                            // if (std::abs(hessian(i * 3 + k, j * 3 + l)) > 1e-8)
                            {
                                // if (use_pre_build_structure)
                                {
                                    // IV2 idx(dof_i * 3 + k, dof_j * 3 + l);
                                    // auto idx = std::make_pair(dof_i * 3 + k, dof_j * 3 + l);
                                    uint64_t lower_bit = dof_i * 3 + k;
                                    uint64_t upper_bit = dof_j * 3 + l;
                                    uint64_t whole = upper_bit << 32 | lower_bit;
                                    // if (ij_to_value_array.find(idx) != ij_to_value_array.end())
                                    {
                                        // std::ptrdiff_t ptr_diff = ij_to_value_array[idx];
                                        // std::ptrdiff_t ptr_diff = ijv[whole];
                                        
                                        // *(&val[0] + ptr_diff) += hessian(i * 3 + k, j * 3 + l);
                                        keys[offset(i, j, k, l)] = whole;
                                        values[offset(i, j, k, l)] = hessian(i * 3 + k, j * 3 + l);
                                        local_triplets[offset(i, j, k, l)] = Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l));
                                    }
                                    // else
                                    // {
                                    //     use_pre_build_structure = false;
                                    // }
                                }
                                // triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l)));                
                                // local_triplets[i * 3 + k, j * 3 + l] = Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l));
                            }
                        }
                }
            }
            
            tbb::parallel_for(0, (int)values.size(), [&](int i){
                *(&val[0] + ijv[keys[i]]) += values[i];
            });
            // triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
        else
        {
            // cnt = 0;
            for (int i = 0; i < vtx_idx.size(); i++)
            {
                int dof_i = vtx_idx[i];
                for (int j = 0; j < vtx_idx.size(); j++)
                {
                    int dof_j = vtx_idx[j];
                    if (lower_triangular)
                        if (dof_j > dof_i) continue;
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                        {
                            if (lower_triangular && (dof_i == dof_j))
                                if (l > k) continue;
                            // if (lower_triangular)
                            //     if (dof_j * 3 + l > dof_i * 3 + k) continue;
                            // if (std::abs(hessian(i * 3 + k, j * 3 + l)) > 1e-8)
                            {
                                // triplets[n_curr + cnt] = Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l));
                                // cnt++;
                                triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l)));                
                            }
                        }
                }
            }
        }
    }

private:

    inline T computeTetVolume(const TV& a, const TV& b, const TV& c, const TV& d)
    {
        return 1.0 / 6.0 * (b - a).cross(c - a).dot(d - a);
    };

    template<int dim>
    bool isHessianBlockPD(const Matrix<T, dim, dim> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int dim>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, dim, dim> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    template <int size>
    void projectBlockPD(Eigen::Matrix<T, size, size>& symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            }
            else {
                break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }

    

    template<int dim>
    void addHessianBlock(
        std::vector<Entry>& triplets,
        const std::vector<int>& vtx_idx, 
        const Matrix<T, dim, dim>& hessian_block)
    {

        int dof_i = vtx_idx[0];
        int dof_j = vtx_idx[1];
        if (lower_triangular)
            if (dof_j > dof_i) return;
        for (int k = 0; k < dim; k++)
            for (int l = 0; l < dim; l++)
            {
                if (lower_triangular && (dof_i == dof_j))
                        if (l > k) continue;
                if (std::abs(hessian_block(k, l)) > 1e-8)
                    triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian_block(k, l)));
            }
    }

    template<int dim>
    void addForceEntry(VectorXT& residual, 
        const std::vector<int>& vtx_idx, 
        const Vector<T, dim>& gradent)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<3>(vtx_idx[i] * 3) += gradent.template segment<3>(i * 3);
    }

    template<int dim>
    void getSubVector(const VectorXT& _vector, 
        const std::vector<int>& vtx_idx, 
        Vector<T, dim>& sub_vec)
    {
        if (vtx_idx.size() * 3 != dim)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, dim>::Zero(vtx_idx.size() * 3);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<3>(i * 3) = _vector.segment<3>(vtx_idx[i] * 3);
        }
    }

    bool validFaceIdx(Region region, int idx)
    {
        if (region == Apical)
            return idx < basal_face_start;
        else if (region == Basal)
            return idx >= basal_face_start && idx < lateral_face_start;
        else if (region == Lateral)
            return idx >= lateral_face_start;
        else
            return false;
    }

public:
    VertexModel() 
    {
        
    }
    ~VertexModel() {}
};

#endif