#ifndef FEM_SOLVER_H
#define FEM_SOLVER_H

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
#include <algorithm>
#include <cppad/cppad.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_poly.h>

#include "VecMatDef.h"
#include "MortarMethod.h"
#include "MortarMethodDiff.h"
#include "SpatialHash.h"
#include "../include/IMLSdiff.h"
#include "../include/RIMLSdiff.h"
#include "pallas/basinhopping.h"
#include "glog/logging.h"
#include "bayesopt/bayesopt.hpp"
#include <ipc/ipc.hpp>

#include <CGAL/Epick_d.h>
#include <CGAL/point_generators_d.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Fuzzy_iso_box.h>
#include <CGAL/Search_traits_d.h>

typedef CGAL::Epick_d<CGAL::Dimension_tag<3> > K;
typedef K::Point_d Point_d;
typedef std::tuple<Point_d, int> Point_and_int;
typedef CGAL::Search_traits_d<K>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int, CGAL::Nth_of_tuple_property_map<0, Point_and_int>, Traits_base>       Traits;
typedef CGAL::Kd_tree<Traits> Tree;
typedef CGAL::Fuzzy_sphere<Traits> Fuzzy_sphere;

#define USE_QUAD_ELEMENT 0

#define WIDTH_1 4.
#define HEIGHT_1 2

#define HEIGHT_2 0.5

#define SCALAR 2

#define NO_SAMPLING 1

extern float DISPLAYSMENT;
extern int RES;
extern int RES_2;
extern double FORCE;
extern float WIDTH_2;
extern bool BILATERAL;
extern bool use_NTS_AR;
extern float SPRING_SCALE;
extern int SPRING_BASE;
extern bool USE_NONUNIFORM_MESH;
extern int MODE;
extern float GAP;
extern bool USE_IMLS;
extern bool IMLS_BOTH;
extern bool USE_MORE_POINTS;
extern bool USE_FROM_STL;
extern bool USE_MORTAR_METHOD;
extern bool TEST;
extern int TEST_CASE;
extern bool USE_VIRTUAL_NODE;
extern bool CALCULATE_IMLS_PROJECTION;
extern bool USE_IPC_3D;
extern bool USE_TRUE_IPC_2D;
extern bool IMLS_3D_VIRTUAL;
extern bool USE_NEW_FORMULATION;
extern bool SLIDING_TEST;
extern bool PULL_TEST;
extern bool BARRIER_ENERGY;
extern bool USE_DYNAMICS; 
extern bool USE_SHELL;
extern bool USE_RIMLS;
extern bool USE_FRICTION;
extern VectorXa xz;


using Eigen::MatrixXd;
using Eigen::MatrixXi;


template<int dim>
class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using TM2 = Matrix<T, 2, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    

    using Entry = Eigen::Triplet<T>;

    using EleNodes = Matrix<T, dim + 1, dim>;
    using EleIdx = Vector<int, dim + 1>;
    using EleNodesQuad = Matrix<T, dim + 2, dim>;
    using EleIdxQuad = Vector<int, dim + 2>;

    using Face = Vector<int, 3>;
    using FaceQuad = Vector<int, 4>;
    using Edge = Vector<int, 2>;

public:
    
    VectorXT u;
    VectorXT f;
    VectorXi faces;
    VectorXT deformed, undeformed;

    // For adding additional dof 1 (should be resolved to original dof eventually)
    VectorXT deformed_all, undeformed_all;

    VectorXT deformed_ad, undeformed_ad;

    VectorXi indices, indices_quad;
    VectorXi surface_indices, surface_indices_quad;

    Eigen::MatrixXd VS;
    Eigen::MatrixXi FS;
    std::vector<Eigen::MatrixXd> Vs;
    std::vector<Eigen::MatrixXi> Fs;

    VectorXT residual_step;

    std::unordered_map<int, T> dirichlet_data;

    std::vector<std::pair<int, T>> penalty_pairs;

    int num_nodes;   
    int num_ele, num_ele_quad;
    int num_surface_faces, num_surface_faces_quad;

    bool verbose = true;
    bool run_diff_test = false;

    T vol = 1.0;
    T E = 2.6 * 1e8;
    T nu = 0.48;
    
    T penalty_weight = 1e6;
    bool use_penalty = true;

    T newton_tol = 1e-6;
    int max_newton_iter = 2000;

    TV min_corner, max_corner;
    TV center;
    
    bool project_block_PD = false;

    // IPC
    bool add_friction = false;
    T max_barrier_weight = 1e8;
    T friction_mu = 0.5;
    T epsv_times_h = 1e-5;
    bool self_contact = false;
    bool use_ipc = false;
    int num_ipc_vtx = 0;
    T barrier_distance = 1e-5;
    T barrier_weight = 1e6;
    T ipc_min_dis = 1e-6;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_faces;

    // Contact Information
    Eigen::VectorXd is_boundary;
    Eigen::MatrixXd collision_candidates;
    bool use_ipc_2d = true;
    struct point_segment_pair
    {
        int slave_index;
        int master_index_1;
        int master_index_2;
        int index;
        double scale;
        double dist;
        Eigen::VectorXd dist_grad;
        Eigen::MatrixXd dist_hess;
        std::vector<int> results;
        StiffnessMatrix dist_hess_s;
    };
    std::vector<point_segment_pair> boundary_info;
    std::vector<std::pair<int,int>> slave_segments;
    std::vector<int> slave_ele_indices;
    std::vector<int> slave_nodes;
    std::vector<std::pair<int,int>> master_segments;
    std::vector<int> master_nodes;
    std::vector<int> useful_master_nodes;
    Eigen::MatrixXd V_all;
    Eigen::MatrixXi F_all;
    Eigen::MatrixXi F_all_Quad;
    Eigen::VectorXd OnePKStress;
    std::vector<Eigen::Matrix<double,2,2>> CauchyStressTensor;
    Eigen::MatrixXd ContactForce;
    Eigen::VectorXd ContactTraction;
    Eigen::VectorXd ContactPressure;
    Eigen::VectorXd ContactPenetration;
    Eigen::VectorXd ContactLength;

    // Penalty on CoM
    std::vector<Eigen::VectorXd> Object_indices;
    std::vector<Eigen::VectorXd> Object_indices_rot;
    bool use_pos_penalty = false;
    bool use_elasticity = true;

    // PBC Information and penalty
    std::vector<std::pair<int,int>> pbc_pairs;
    bool use_PBC_penalty = false;

    // add virtual spring
    bool use_virtual_spring = false;
    std::vector<int> left_boundaries;
    std::vector<int> right_boundaries;
    std::vector<int> left_boundaries_master;
    std::vector<int> right_boundaries_master;
    double k1 = 1e3;
    double k2 = SPRING_SCALE*k1;

    bool use_rotational_penalty = false;


    // Inverse Design
    double R;
    Eigen::MatrixXd g; //energy gradient
    StiffnessMatrix H; //energy hessian
    Eigen::MatrixXd dRdx;
    Eigen::MatrixXd dgdp;
    Eigen::MatrixXd dxdp;
    Eigen::MatrixXd dRdp;


    // Implicit Moving Least Square
    bool use_IMLS = true;
    bool use_Kernel_Regression = true;
    int sample_res = 16;
    int object_num = 2;
    Eigen::MatrixXi F_cur;
    Eigen::MatrixXd V_cur;
    std::vector<Eigen::VectorXd> face_indices;
    std::vector<Eigen::VectorXd> boundary_segments;
    std::vector<Eigen::VectorXd> boundary_normals;
    std::vector<std::vector<Eigen::VectorXd>> sample_points;
    std::vector<std::vector<Eigen::VectorXd>> sample_normals;
    std::vector<Eigen::MatrixXd> constrained_points;
    std::vector<Eigen::MatrixXd> constrained_values;
    std::vector<Eigen::MatrixXd> constrained_normals;
    std::vector<Eigen::MatrixXd> result_values;
    std::vector<Eigen::VectorXd> result_grad;
    std::vector<Eigen::MatrixXd> result_hess;
    std::vector<Eigen::VectorXd> samples_grad;
    std::vector<Eigen::MatrixXd> samples_hess;
    std::vector<std::vector<int>> map_sample_to_deform;
    std::vector<std::vector<std::pair<int, double>>> map_sample_to_deform_with_scale;
    std::vector<std::vector<Eigen::VectorXd>> corners;
    Eigen::VectorXd slave_center;

    std::vector<int> force_nodes;

    bool use_yue_code = true;
    
    // MortarMethod Data
    MortarMethodDiff mortar;

    // Virtual Node Information
    struct VirtualNodeInfo
    {
        int left_index;
        int right_index;
        double eta;
        double w;
    };
    std::vector<VirtualNodeInfo> virtual_slave_nodes;
    std::vector<VirtualNodeInfo> virtual_master_nodes;
    std::unordered_map<int,int> map_boundary_virtual;

    // Allow disjoint slave_master pairs
    bool use_multiple_pairs = false;
    std::vector<std::vector<int>> multiple_slave_nodes;
    std::vector<std::vector<int>> multiple_master_nodes;
    std::vector<std::vector<int>> boundary_info_start;

    // Data for projected points on IMLS
    Eigen::MatrixXd projectedPts;
    std::unordered_map<int, double> extendedAoC;

    // Data for 3D meshes slave_surfaces and master_surfaces
    std::vector<std::unordered_map<int,int>> slave_nodes_3d;
    std::vector<std::vector<double>> slave_nodes_area_3d;
    std::vector<std::vector<Eigen::VectorXi>> slave_surfaces_3d;
    std::vector<std::vector<int>> slave_surfaces_global_index;

    std::vector<std::unordered_map<int,int>> master_nodes_3d;
    std::vector<std::vector<double>> master_nodes_area_3d;
    std::vector<std::vector<Eigen::VectorXi>> master_surfaces_3d;
    std::vector<std::vector<int>> master_surfaces_global_index;

    std::vector<std::unordered_map<int,int>> force_nodes_3d;
    std::vector<std::vector<double>> force_nodes_area_3d;
    std::vector<std::vector<Eigen::VectorXi>> force_surfaces_3d;
    

    std::vector<std::vector<int>> vertex_triangle_indices;
    Eigen::SparseMatrix<double> MassMatrix;


    std::vector<std::vector<int>> boundary_info_start_3d;
    Eigen::VectorXd doublearea;

public:

    // ###################### iterators ######################
    template <typename OP>
    void iterateElementSerial(const OP& f)
    {
        for (int i = 0; i < int(indices.size()/(dim + 1)); i++)
        {
            EleIdx tet_idx = indices.segment<dim + 1>(i * (dim + 1));
            EleNodes tet_deformed = getEleNodesDeformed(tet_idx);
            EleNodes tet_undeformed = getEleNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateElementSerialQuad(const OP& f)
    {
        for (int i = 0; i < int(indices_quad.size()/(dim + 2)); i++)
        {
            EleIdxQuad tet_idx = indices_quad.segment<dim + 2>(i * (dim + 2));
            EleNodesQuad tet_deformed = getEleNodesDeformedQuad(tet_idx);
            EleNodesQuad tet_undeformed = getEleNodesUndeformedQuad(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateElementParallel(const OP& f)
    {
        tbb::parallel_for(0, int(indices.size()/(dim + 1)), [&](int i)
        {
            EleIdx tet_idx = indices.segment<dim + 1>(i * (dim + 1));
            EleNodes tet_deformed = getEleNodesDeformed(tet_idx);
            EleNodes tet_undeformed = getEleNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        });
    }

    template <typename OP>
    void iterateElementParallelQuad(const OP& f)
    {
        tbb::parallel_for(0, int(indices_quad.size()/(dim + 2)), [&](int i)
        {
            EleIdxQuad tet_idx = indices_quad.segment<dim + 2>(i * (dim + 2));
            EleNodesQuad tet_deformed = getEleNodesDeformedQuad(tet_idx);
            EleNodesQuad tet_undeformed = getEleNodesUndeformedQuad(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        });
    }

    template <class OP>
    void iterateBCPenaltyPairs(const OP& f)
    {
        for (auto pair : penalty_pairs)
        {
            f(pair.first, pair.second);
        }
    }

    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

private:
    template<int size>
    bool isHessianBlockPD(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
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

    template<int size>
    void addForceEntry(VectorXT& residual, 
        const VectorXi& vtx_idx, 
        const Vector<T, size>& gradent)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.template segment<dim>(vtx_idx[i] * dim) += gradent.template segment<dim>(i * dim);
    }

    template<int size>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const VectorXi& vtx_idx, 
        const Matrix<T, size, size>& hessian)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim; k++)
                    for (int l = 0; l < dim; l++)
                    {
                        if (std::abs(hessian(i * dim + k, j * dim + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * dim + k, dof_j * dim + l, hessian(i * dim + k, j * dim + l)));                
                    }
            }
        }
    }

    EleNodes getEleNodesDeformed(const EleIdx& nodal_indices)
    {
        if (nodal_indices.size() != dim + 1)
            std::cout << "getEleNodesDeformed() not a tet" << std::endl; 
        EleNodes tet_x;
        for (int i = 0; i < dim + 1; i++)
        {
            tet_x.row(i) = deformed.segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    EleNodesQuad getEleNodesDeformedQuad(const EleIdxQuad& nodal_indices)
    {
        if (nodal_indices.size() != dim + 2)
            std::cout << "getEleNodesDeformed() not a tet" << std::endl; 
        EleNodesQuad tet_x;
        for (int i = 0; i < dim + 2; i++)
        {
            tet_x.row(i) = deformed.segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }


    EleNodes getEleNodesUndeformed(const EleIdx& nodal_indices)
    {
        if (nodal_indices.size() != dim + 1)
            std::cout << "getEleNodesUndeformed() not a tet" << std::endl;
        EleNodes tet_x;
        for (int i = 0; i < dim + 1; i++)
        {
            tet_x.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }
    
    EleNodesQuad getEleNodesUndeformedQuad(const EleIdxQuad& nodal_indices)
    {
        if (nodal_indices.size() != dim + 2)
            std::cout << "getEleNodesUndeformed() not a tet" << std::endl;
        EleNodesQuad tet_x;
        for (int i = 0; i < dim + 2; i++)
        {
            tet_x.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;

        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
        return triplets;
    }
    inline T getSmallestPositiveRealQuadRoot(T a, T b, T c, T tol)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::sqrt;
        T t;
        if (abs(a) <= tol) {
            if (abs(b) <= tol) // f(x) = c > 0 for all x
                t = -1;
            else
                t = -c / b;
        }
        else {
            T desc = b * b - 4 * a * c;
            if (desc > 0) {
                t = (-b - sqrt(desc)) / (2 * a);
                if (t < 0)
                    t = (-b + sqrt(desc)) / (2 * a);
            }
            else // desv<0 ==> imag
                t = -1;
        }
        return t;
    }

    inline T getSmallestPositiveRealCubicRoot(T a, T b, T c, T d, T tol = 1e-10)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::complex;
        using std::pow;
        using std::sqrt;
        T t = -1;
        if (abs(a) <= tol)
            t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
        else {
            complex<T> i(0, 1);
            complex<T> delta0(b * b - 3 * a * c, 0);
            complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<T> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            if (abs(C) < tol)
                C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            complex<T> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
            complex<T> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;
            complex<T> t1 = (b + C + delta0 / C) / (-3.0 * a);
            complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
            complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        return t;
    }

public:

    // Scene.cpp
    void initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    void initializeElementData(Eigen::MatrixXd& TV, const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT, const Eigen::MatrixXi& TF_quad, const Eigen::MatrixXi& TT_quad);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void generateMeshForRenderingStress(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, double max_force);
    void computeBoundingBox();
    
    // FEMSolver.cpp
    void reset();
    void clearAll();
    
    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);

    T computeTotalEnergy(const VectorXT& _u);
    T computeInteralEnergy(const VectorXT& _u);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);

    T computeResidual(const VectorXT& _u, VectorXT& residual, double re = false);

    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);

    bool staticSolve();

    bool staticSolveStep(int step);

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    // DerivativeTest.cpp
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);
    void checkTotalGradient(bool perturb = false);
    void checkTotalHessian(bool perturb = false);

    //Helper.cpp
    void computeBBox(const Eigen::MatrixXd& V, TV& bbox_min_corner, TV& bbox_max_corner);

    
    void saveTetOBJ(const std::string& filename, const EleNodes& tet_vtx);
    void saveToOBJ(const std::string& filename);
    void saveIPCMesh(const std::string& filename);

    //BoundaryCondition.cpp
    void dragMiddle();
    void addDirichletBC();
    void addNeumannBC();
    void addDirichletBCFromSTL();
    void addRotationalDirichletBC(double theta);
    void addNeumannBCFromSTL();

    //Penalty.cpp
    void addBCPenaltyEnergy(T& energy);
    void addBCPenaltyForceEntries(VectorXT& residual);
    void addBCPenaltyHessianEntries(std::vector<Entry>& entries);

    // Elasticity.cpp
    T computeNeoHookeanStrainEnergy(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed);
    void computeNeoHookeanStrainEnergyGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Vector<T, 12>& gradient);
    T computeVolume(const EleNodes& x_undeformed);
    void computeDeformationGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, TM& F);
    void computeNeoHookeanStrainEnergyHessian(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Matrix<T, 12, 12>& hessian);
    void polarSVD(TM& F, TM& U, TV& Sigma, TM& VT);

    void addElastsicPotential(T& energy);
    void addElasticForceEntries(VectorXT& residual);
    void addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    //IPC2D.cpp
    T computeCollisionFreeStepsize2D(const VectorXT& _u, const VectorXT& du);
    void computeIPC2DRestData(bool original = false);
    void updateIPC2DVertices(const VectorXT& _u);
    void addIPC2DEnergy(T& energy);
    void addIPC2DForceEntries(VectorXT& residual, double re = false);
    void findProjection(Eigen::MatrixXd& ipc_vertices_deformed, bool re = false, bool stay = true);
    double compute_barrier_potential2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);
    Eigen::VectorXd compute_barrier_potential_gradient2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);
    void addIPC2DHessianEntries(std::vector<Entry>& entries, bool project_PD = false);
    Eigen::SparseMatrix<double> compute_barrier_potential_hessian2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);
    void initializeBoundaryInfo();
    void compute1PKStress();
    void computeCauchyStress();
    void computeContactPressure();
    void displayBoundaryInfo();
    void findProjectionIMLS(Eigen::MatrixXd& ipc_vertices_deformed, bool re = false, bool stay = true);
    void findProjectionIMLSMultiple(Eigen::MatrixXd& ipc_vertices_deformed, bool re = false, bool stay = true);
    void findProjectionIMLSMultiple3D(Eigen::MatrixXd& ipc_vertices_deformed, bool re = false, bool stay = true);
    void checkDeformationGradient();
    void findContactMasterIPC(std::vector<double>& master_contact_nodes);
    double checkMasterVariance();

    //PosPenalty.cpp
    void getCenter();
    void addPosPenaltyEnergy(T& energy);
    void addPosPenaltyForceEntries(VectorXT& residual);
    void addPosPenaltyHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    void addPBCPenaltyEnergy(T& energy);
    void addPBCPenaltyForceEntries(VectorXT& residual);
    void addPBCPenaltyHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    void addRotationalPenaltyEnergy(T& energy);
    void addRotationalPenaltyForceEntries(VectorXT& residual);
    void addRotationalPenaltyHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    void addVirtualSpringEnergy(T& energy);
    void addVirtualSpringForceEntries(VectorXT& residual);
    void addVirtualSpringHessianEntries(std::vector<Entry>& entries, bool project_PD = false);
    void showLeftRight();
    bool checkLeftRight(Eigen::MatrixXd& next_pos);

    //InverseDesign.cpp
    void ForwardSim();
    void calculateR();
    void calculatedfdp();
    void checkDerivative();
    void InverseDesign(double& tar, double& opt_p);
    void ResetSim();

    void eigenAnalysis(StiffnessMatrix& K);
    void checkHessianPD(bool save_txt);

    //ImplicitMLS.cpp
    void samplePoints();
    void samplePointsFromSTL(int pair_id = -1);
    void buildConstraintsPointSet();
    void find_neighbor_pts(Eigen::VectorXd& query_pt, double radius, std::vector<int>& results, std::vector<double>& res_dists, int index = 0);
    int brute_force_NN(Eigen::VectorXd& query_pt, int index = 0);
    void evaluateImplicitPotential(Eigen::MatrixXd& xs, int index = 0);
    void evaluateImplicitPotentialKRGradient(Eigen::MatrixXd& ps, std::vector<Eigen::VectorXd>&dfsdps, Eigen::VectorXd& fs, bool calculate_grad, int index = 1);
    void evaluateImplicitPotentialKR(Eigen::MatrixXd& xs, bool update = false, int index = 0, int pair_id = -1, bool eval_same_side = false);
    void testIMLS(Eigen::MatrixXd& xs, int index);
    void computeSlaveCenter();
    void yueEvaluate(Eigen::MatrixXd& xs, bool update = false);
    void testDerivativeIMLS();
    bool checkInsideRectangle(Eigen::VectorXd q, int index = 0);
    bool checkInsideRectangleFromSTL(Eigen::VectorXd q, int index = 0);
    Eigen::VectorXd convertToGlobalGradient(Eigen::VectorXd& grad,std::vector<int>& useful_index,int index);
    Eigen::MatrixXd convertToGlobalHessian(Eigen::MatrixXd& hess,std::vector<int>& useful_index,int index);

    // cppad functions
    double evaulateIMLSCPPAD(const Eigen::VectorXd& s, std::vector<std::vector<int>>& normal_pairs, std::vector<int>& results_index, double radius, Eigen::VectorXd& grad, Eigen::MatrixXd& hess, bool update = false, bool only_grad = false, std::vector<std::vector<Eigen::VectorXd>> normal_pairs_3d = {}, int index = 0);
    double computeUpperCenter();

    // Mortar Method Energy
    void initializeMortarInformation();
    void updateMortarInformation();
    void addMortarEnergy(T& energy);
    void addMortarForceEntries(VectorXT& residual);
    void addMortarHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    // Virtual Node Functions
    void GenerateVirtualPoints();
    double compute_vts_potential2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);
    Eigen::VectorXd compute_vts_potential_gradient2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);
    Eigen::SparseMatrix<double> compute_vts_potential_hessian2D(Eigen::MatrixXd& ipc_vertices_deformed, bool eval_same_side = false);

    // Find projection point on IMLS surface
    void CalculateProjectionIMLS();
    Eigen::VectorXd getKKTgrad(std::vector<Eigen::VectorXd>& lambdas, Eigen::MatrixXd ps, Eigen::MatrixXd qs, bool test = false);
    Eigen::MatrixXd getQ(Eigen::MatrixXd qs, bool test = false);
    Eigen::VectorXd getc(Eigen::MatrixXd qs, bool test = false);
    double getcf(Eigen::MatrixXd qs, bool test = false);
    Eigen::MatrixXd getgradphi(Eigen::MatrixXd p0s, bool test = false, int index = 1);
    Eigen::VectorXd getphi(Eigen::MatrixXd p0s, bool test = false, int index = 1);
    Eigen::MatrixXd SinglePointProjection(Eigen::MatrixXd ps, bool test = false, int index = 1);
    void IMLSProjectionTest();
    void SortedProjectionPoints2D();

    // Additional functions for 3D
    void CalculateAreaPerVertex(Eigen::MatrixXd& V, Eigen::MatrixXi& faces);

    // True IPC 3D
    void computeIPC3DRestData();
    T computeCollisionFreeStepsize3D(const VectorXT& _u, const VectorXT& du);
    void addIPC3DEnergy(T& energy);
    void addIPC3DForceEntries(VectorXT& residual);
    void addIPC3DHessianEntries(std::vector<Entry>& entries,bool project_PD);
    bool checkInsideMeshFromSTL3D(Eigen::VectorXd& q, int index = 0, int pair_id = 0);

    // True IPC 2D
    void computeIPC2DtrueRestData();
    T computeCollisionFreeStepsize2Dtrue(const VectorXT& _u, const VectorXT& du);
    void addIPC2DtrueEnergy(T& energy);
    void addIPC2DtrueForceEntries(VectorXT& residual);
    void addIPC2DtrueHessianEntries(std::vector<Entry>& entries,bool project_PD);
    void updateBarrierInfo(bool first_step);

    // Penalty terms on new DOFs
    double L2D_param = 1e5;
    void addL2DistanceEnergy(T& energy);
    void addL2DistanceForceEntries(VectorXT& residual);
    void addL2DistanceHessianEntries(std::vector<Entry>& entries,bool project_PD);
    
    double IMLS_param = 1;
    std::vector<point_segment_pair> boundary_info_same_side;
    std::unordered_map<int,int> map_boundary_virtual_same_side;
    void findProjectionIMLSSameSide(Eigen::MatrixXd& ipc_vertices_deformed, bool re = false, bool stay = true);
    void addIMLSPenEnergy(T& energy);
    void addIMLSPenForceEntries(VectorXT& residual);
    void addIMLSPenHessianEntries(std::vector<Entry>& entries,bool project_PD);

    // SpatialHash Acceleration
    std::vector<SpatialHash<dim>> SH_data;
    void updateHashDataStructure(double radius, int index);
    void find_neighbor_pts_SH(Eigen::VectorXd& query_pt, double radius, std::vector<int>& results, std::vector<double>& res_dists, int index);

    // Data and functions for the sliding test
    double r1 = 1;
    double r2 = 1.5;
    double sliding_res = 10;
    double theta1 = 0.12;
    double theta2 = 0.17;
    double sliding_stiffness = 1e-1;

    void addSlidingSpringEnergy(T& energy);
    void addSlidingSpringForceEntries(VectorXT& residual);
    void addSlidingSpringHessianEntries(std::vector<Entry>& entries,bool project_PD);

    // Virtual Springs
    double spring_length = 3;
    std::vector<int> spring_indices;
    std::vector<Eigen::VectorXd> spring_ends;
    double virtual_spring_stiffness = 5;
    void addVirtualSpringEnergy2(T& energy);
    void addVirtualSpringForceEntries2(VectorXT& residual);
    void addVirtualSpringHessianEntries2(std::vector<Entry>& entries,bool project_PD);

    // Dynamics
    VectorXT x_prev;
    VectorXT v_prev;
    double h = 0.1;
    double rou = 10;
    double simulation_time = 10;
    StiffnessMatrix M;
    void constructMassMatrix(std::vector<Entry>& entries, bool project_PD = false, bool add_hessian = false);
    VectorXT get_a(VectorXT& x);

    bool dynamicSolve();

    // Solid Shell
    // If you use solid shells, faces contains information for solid shells and indices contains information for normal tets
    using Hinges = Matrix<int, Eigen::Dynamic, 4>;
    using FaceVtx = Matrix<T, 3, dim>;
    using FaceIdx = Vector<int, 3>;
    using HingeIdx = Vector<int, 4>;
    using HingeVtx = Matrix<T, 4, dim>;

    Hinges hinges;
    bool add_bending = false;
    bool add_stretching = false;
    bool gravitional_energy = false;

    std::vector<TM2> Xinv;
    VectorXT rest_area;
    VectorXT thickness;
    int shell_start_index;

    TV gravity = TV::Zero();

    T lambda, mu;
    void updateLameParameters()
    {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }

    HingeVtx getHingeVtxDeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = deformed.segment<dim>(hi[i]*dim);
        }
        return cellx;
    }
    
    HingeVtx getHingeVtxUndeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = undeformed.segment<dim>(hi[i]*dim);
        }
        return cellx;
    }

    FaceVtx getFaceVtxDeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = deformed.segment<dim>(nodal_indices[i]*dim);
        }
        return cellx;
    }

    FaceVtx getFaceVtxUndeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = undeformed.segment<dim>(nodal_indices[i]*dim);
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
    void iterateHingeSerial(const OP& f)
    {
        for (int i = 0; i < hinges.rows(); i++)
        {
            const Vector<int, 4> nodes = hinges.row(i);
            f(nodes);   
        }
    }

    void buildHingeStructure();
    void computeRestShape();
    int nFaces () { return faces.rows() / 3; }
    T computeTotalVolume() { return rest_area.sum() * thickness[0]; }
    void addShellEnergy(T& energy);
    void addShellForceEntries(VectorXT& residual);
    void addShellHessianEntries(std::vector<Entry>& entries,bool project_PD);
    double E_2 = 1e6;

    void addUnilateralQubicPenaltyEnergy(T w, T& energy);
    void addUnilateralQubicPenaltyForceEntries(T w, VectorXT& residuals);
    void addUnilateralQubicPenaltyHessianEntries(T w, std::vector<Entry>& entries);
    T y_bar;
    T bar_param;

    // Fast IMLS Implementation
    void addFastIMLSEnergy(T& energy);
    void addFastIMLSForceEntries(VectorXT& residual);
    void addFastIMLSHessianEntries(std::vector<Entry>& entries,bool project_PD);
    void addFastIMLSSameSideEnergy(T& energy);
    void addFastIMLSSameSideForceEntries(VectorXT& residual);
    void addFastIMLSSameSideHessianEntries(std::vector<Entry>& entries,bool project_PD);
    void addFastIMLS12Energy(T& energy);
    void addFastIMLS12ForceEntries(VectorXT& residual);
    void addFastIMLS12HessianEntries(std::vector<Entry>& entries,bool project_PD);
    std::vector<std::vector<int>> current_indices;
    Tree accelerator;
    void BuildAcceleratorTree(bool is_master = true);
    AScalar radius = 0.6;
    AScalar IMLS_pen_scale = 1.;

    std::vector<std::vector<int>> master_nodes_adjacency;
    std::vector<std::vector<int>> slave_nodes_adjacency;
    std::vector<int> close_slave_nodes;
    std::vector<int> close_master_nodes;
    void FASTIMLSTestHessian();
    std::vector<std::pair<int,double>> dist_info;
    int slave_nodes_shift = 0;

    // Correct Projection force
    void addL2CorrectEnergy(T& energy);
    void addL2CorrectForceEntries(VectorXT& residual);
    void addL2CorrectHessianEntries(std::vector<Entry>& entries,bool project_PD);
    int additional_dof = 0;

    //IMLS Both Side
    void addFastIMLSAllEnergy(T& energy, int surface_id = 0, int node_id = 1);
    void addFastIMLSAllForceEntries(VectorXT& residual, int surface_id = 0, int node_id = 1);
    void addFastIMLSAllHessianEntries(std::vector<Entry>& entries,bool project_PD, int surface_id = 0, int node_id = 1);
    int num_cnt = 0;

    //Self Contact
    Eigen::SparseMatrix<double> geodist_close_matrix;
    std::vector<int> mesh_begin_vertices;
    std::vector<bool> is_surface_vertex;
    std::vector<std::vector<int>> nodes_adjacency;

    void BuildAcceleratorTreeSC();
    void addFastIMLSSCEnergy(T& energy, bool test = false);
    void addFastIMLSSCForceEntries(VectorXT& residual);
    void addFastIMLSSCHessianEntries(std::vector<Entry>& entries,bool project_PD);

    void IMLS_local_gradient_to_global_gradient(
    VectorXa& local_grad,
    Eigen::VectorXi ids,
    int dim1,
    Eigen::VectorXd& grad);

    void IMLS_local_hessian_to_global_triplets(
    VectorXa& local_hessian,
    Eigen::VectorXi ids,
    int dim1,
    std::vector<Eigen::Triplet<double>>& triplets);

    void IMLS_local_hessian_matrix_to_global_triplets(
    MatrixXa& local_hessian,
    Eigen::VectorXi ids,
    int dim1,
    std::vector<Eigen::Triplet<double>>& triplets);

    void IMLS_vector_muliplication_to_triplets(
    VectorXa& v1,
    VectorXa& v2,
    AScalar scale,
    std::vector<Eigen::Triplet<double>>& triplets);

    AScalar sigma_r=3.0, sigma_n= 1.0;
    void BuildAcceleratorTreeSCTest();
    void addFastRIMLSSCTestEnergy(T& energy);
    void addFastRIMLSSCEnergy(T& energy);
    void addFastRIMLSSCForceEntries(VectorXT& residual);
    void addFastRIMLSSCHessianEntries(std::vector<Entry>& entries,bool project_PD);

    double evaluateUnsignedDistanceSq(int i, double t, const VectorXa& du, double ipc_stepsize, double& distance);
    double evaluateUnsignedDistanceSqGradient(int i, double t,const VectorXa& du, double ipc_stepsize);
    double evaluateUnsignedDistanceSqRIMLS(int i, double t, const VectorXa& du, double ipc_stepsize, double& distance);
    double evaluateUnsignedDistanceSqGradientRIMLS(int i, double t,const VectorXa& du, double ipc_stepsize);
    T computeCollisionFreeStepsizeUnsigned(const VectorXT& _u, const VectorXT& du);

    void IMLS_local_gradient_to_global_gradient_and_multiply(
    VectorXa& local_grad,
    Eigen::VectorXi ids,
    int dim1,
    AScalar scale,
    VectorXa& global_vector,
    std::vector<Eigen::Triplet<double>>& triplets);

    VectorXa divisionGradient(AScalar& fx, AScalar& gx, VectorXa& dfxdx, VectorXa& dgxdx);
    void divisionHessian(AScalar& fx, AScalar& gx, VectorXa& dfxdx, VectorXa& dgxdx, std::vector<Eigen::Triplet<double>>& d2fxdx2, std::vector<Eigen::Triplet<double>>& d2gxdx2, std::vector<Eigen::Triplet<double>>& out);

    bool use_sherman_morrison = false;
    MatrixXa UV;

    ipc::CollisionMesh collisionMesh;

    VectorXa prev_contact_force;
    void addFrictionEnergy(T& energy);
    void addFrictionForceEntries(VectorXT& residual);
    void addFrictionHessianEntries(std::vector<Entry>& entries,bool project_PD);
    AScalar friction_epsilon = 1e-4;
    
    FEMSolver() {
        if(USE_SHELL)
        {

            gravity[1] = 0;
            updateLameParameters();
        }
    }
    ~FEMSolver() {}

};

template<int dim>
class Rosenbrock : public pallas::GradientCostFunction {
public:
    virtual ~Rosenbrock() {}
    virtual bool Evaluate(const double* parameters,
                          double* cost,
                          double* gradient) const;
    virtual int NumParameters() const { return 1; }

    FEMSolver<dim>* solver;
    int index;
    VectorXa du;
    double ipc_stepsize;
};

template<int dim>
class MyOptimization: public bayesopt::ContinuousModel
{
 public:
    MyOptimization(bayesopt::Parameters param):
    bayesopt::ContinuousModel(1,param) 
    {
       
    }
    double evaluateSample( const boost::numeric::ublas::vector<double> &query );
    bool checkReachability( const boost::numeric::ublas::vector<double> &query )
    { 
        return true;
    };

    FEMSolver<dim>* solver;
    int index;
    VectorXa du;
};

#endif
