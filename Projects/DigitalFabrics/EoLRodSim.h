#ifndef EOL_ROD_SIM_H
#define EOL_ROD_SIM_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include "VecMatDef.h"

#include "UnitPatch.h"
#include "CurvatureFunction.h"

#define WARP 0
#define WEFT 1


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


template<class T, int dim>
class EoLRodSim
{
public:
    using Simulation = EoLRodSim<T, dim>;
    
    using TV = Vector<T, dim>;

    using TV2 = Vector<T, 2>;
    using TV3 = Vector<T, 3>;
    using TVDOF = Vector<T, dim+2>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    

    using TM = Matrix<T, dim, dim>;
    using TM3 = Matrix<T, 3, 3>;
    using TMDOF = Matrix<T, dim + 2, dim + 2>;

    using TV3Stack = Matrix<T, 3, Eigen::Dynamic>;
    using IV3Stack = Matrix<int, 3, Eigen::Dynamic>;
    using IV4Stack = Matrix<int, 4, Eigen::Dynamic>;
    using DOFStack = Matrix<T, dim + 2, Eigen::Dynamic>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;

    using IV2 = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;
    using IV5 = Vector<int, 5>;
    
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

    // takes a Eulerian Coord and returns the curvature at rest state
    int N_PBC_BENDING_ELE = 5;

    int dof = dim + 2;
    
    DOFStack q, q0;
    IV3Stack rods;
    IV4Stack connections;
    TV3Stack normal;
    int n_nodes;
    int n_dof;
    int n_rods;
    int n_pb_cons;
    int n_non_interp_nodes;
    IV2 n_rod_uv;
    
    const static int grid_range = 3;
    int final_dim;

    T dt = 1;
    T newton_tol = 1e-4;
    T E = 3.5e9; //PLA
    T R = 0.0002;

    T unit = 1;

    T rho = 1;
    T ks = 1.0;  // stretching term
    T kc = 1e2;  //constraint term
    T kn = 1e-3; //
    T kb = 1.0;
    T kb_penalty = 1.0;
    T km = 1e-3; //mass term
    T kx = 1.0; // shearing term
    T k_pbc = 1.0; // perodic BC term
    T k_strain = 1.0;
    T L = 1;
    T ke = 1e-2; // Eulerian DoF penalty
    T kr = 1e3;
    T k_yc = 1.0;

    std::vector<int> dof_offsets;    

    float theta = 0.f;

    T tunnel_u = R * 4.0;
    T tunnel_v = R * 4.0;

    TV gravity = TV::Zero();
    bool verbose = false;

    bool add_stretching = true;
    bool add_bending = true;
    bool add_shearing = true;
    bool add_penalty = false;
    bool add_rotation_penalty = true;
    bool add_regularizor = false;
    bool add_pbc = true;
    bool add_pbc_bending = true;
    bool add_eularian_reg = true;
    bool disable_sliding = true;
    bool subdivide = false;
    bool print_force_mag = false;
    bool add_contact_penalty = true;
    bool use_alm = true;
    bool run_diff_test = false;
    bool use_discrete_rest_bending = true;

    TVDOF fix_all, fix_eulerian, fix_lagrangian, fix_u, fix_v;
    std::unordered_map<int, std::pair<TVDOF, TVDOF>> dirichlet_data;
    std::vector<std::vector<int>> pbc_bending_pairs;
    std::vector<std::vector<int>> pbc_bending_bn_pairs;
    std::vector<std::vector<int>> yarns;
    std::unordered_map<int, int> yarn_map;
    std::unordered_map<IV2, int, VectorHash<2>> pbc_pairs;
    // pbc_ref[direction] = (node_i, node_j)
    std::vector<std::pair<int, IV2>> pbc_ref;

    std::vector<int> sliding_nodes;
    IV2 slide_over_n_rods = IV2::Zero();

    std::vector<IV2> pbc_ref_unique;

    std::vector<std::pair<IV2, std::pair<TV, T>>> pbc_strain_data;

    std::vector<std::vector<int>> yarn_group;
    std::vector<bool> is_end_nodes;

    std::vector<CurvatureFunction<T, dim>*> curvature_functions;

    StiffnessMatrix W;

public:

    EoLRodSim()
    {
        gravity[1] = -9.8;
        fix_eulerian.setOnes();
        fix_lagrangian.setOnes();
        fix_all.setOnes();
        fix_u.setZero();
        fix_v.setZero();
        fix_v[dof-1] = 1.0;
        fix_u[dof-2] = 1.0;
        fix_lagrangian.template segment<2>(dim).setZero();
        fix_eulerian.template segment<dim>(0).setZero();

        config();
    }
    ~EoLRodSim() {}

    void config()
    {
        T area = M_PI * R * R;
        ks = E * area;
        kb = E * area * R * R * 0.25;
        kx = E/T(2)/(1.0 + 0.42) * area;

        std::cout << "ks: " << ks << " kb: " << kb << " kx: " << kx << std::endl;
    }
    
    // TODO: use ... operator
    void cout5Nodes(int n0, int n1, int n2, int n3, int n4)
    {
        std::cout << n0 << " " << n1 << " " << n2 << " " << n3 << " " << n4 << std::endl;
    }

    void cout4Nodes(int n0, int n1, int n2, int n3)
    {
        std::cout << n0 << " " << n1 << " " << n2 << " " << n3 << std::endl;
    }

    void cout3Nodes(int n0, int n1, int n2)
    {
        std::cout << n0 << " " << n1 << " " << n2 << std::endl;
    } 

    template <class OP>
    void iteratePBCBendingPairs(const OP& f) {
        for (auto pair : pbc_bending_pairs){
            std::vector<int> node_ids;
            for(int i = 0; i < pair.size() - 1; i++)
                node_ids.push_back(pair[i]);
            f(node_ids, pair[pair.size() - 1]);
        } 
    }

    template <class OP>
    void iterateSlidingNodes(const OP& f) {
        for (int idx : sliding_nodes){
            f(idx);
        } 
    }

    
    template <class OP>
    void iteratePBCBoundaryPairs(const OP& f) {
        for (auto pair : pbc_bending_bn_pairs){
            std::vector<int> node_ids;
            if(std::find(pair.begin(), pair.end(), -1) != pair.end())
                std::cout << "[EoLRodSim.h] -1 in PBC bending pairs" << std::endl;
            for(int i = 0; i < pair.size() - 1; i++)
                node_ids.push_back(pair[i]);
            f(node_ids, pair[pair.size() - 1]);
        } 
    }

    template <class OP>
    void iterateDirichletData(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second.first, dirichlet.second.second);
        } 
    }

    template <class OP>
    void iteratePBCReferencePairs(const OP& f) {
        for (auto data : pbc_ref){
            f(data.first, data.second(0), data.second(1));
        } 
    }

    template <class OP>
    void iteratePBCStrainData(const OP& f) {
        for (auto data : pbc_strain_data){
            f(data.first(0), data.first(1), data.second.first, data.second.second);
        } 
    }

    template <class OP>
    void iterateYarnCrossingsSerial(const OP& f) {
        for (int i = 0; i < n_nodes; i++)
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
            // f(i, connections(2, i), connections(3, i), connections(0, i), connections(1, i));
    }

    template <class OP>
    void iterateYarnCrossingsParallel(const OP& f) {
        tbb::parallel_for(0, n_nodes, [&](int i)
        {
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
        }); 
    }

    
    // EoLSim.cpp
    T computeTotalEnergy(Eigen::Ref<const VectorXT> dq, 
        Eigen::Ref<const DOFStack> lambdas, T kappa, 
        bool verbose = false);

    T computeResidual(Eigen::Ref<VectorXT> residual, Eigen::Ref<const VectorXT> dq,
         Eigen::Ref<const DOFStack> lambdas, T kappa);
    
    void addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K);
    bool projectDirichletEntrySystemMatrix(StiffnessMatrix& A);
    void addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K,
         Eigen::Ref<const VectorXT> dq, T kappa);
    void addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K);
    void buildSystemMatrix(
         Eigen::Ref<const VectorXT> dq, StiffnessMatrix& K, T kappa);
    
    bool linearSolve(StiffnessMatrix& K, 
        Eigen::Ref<const VectorXT> residual, Eigen::Ref<VectorXT> ddq);
    T newtonLineSearch(Eigen::Ref<VectorXT> dq, 
        Eigen::Ref<const VectorXT> residual, 
        Eigen::Ref<const DOFStack> lambdas, T kappa,
        int line_search_max = 100);
    void implicitUpdate(Eigen::Ref<VectorXT> dq);

    void advanceOneStep();

    void setVerbose(bool v) { verbose = v; }
    void resetScene() { q = q0; }
    void fixEulerian();
    void freeEulerian();
    
private:

    void toMapleNodesVector(std::vector<Vector<T, dim + 1>>& x, Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes, int yarn_type);
    void convertxXforMaple(std::vector<TV>& x, 
        const std::vector<TV>& X,
        Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes);
    void getMaterialPositions(Eigen::Ref<const DOFStack> q_temp, 
        const std::vector<int>& nodes, std::vector<TV>& X, int uv_offset,
        std::vector<TV>& dXdu, std::vector<TV>& d2Xdu2, bool g, bool h);

    
public:
    // Elasticity.cpp
    void setUniaxialStrain(T theta, T s, TV& strain_dir, TV& ortho_dir);
    void setBiaxialStrain(T theta1, T s1, T theta2, T s2, TV& strain_dir, TV& ortho_dir);
    void setBiaxialStrainWeighted(T theta1, T s1, T theta2, T s2, T w);
    // void computeMacroStress(TM& sigma, TV strain_dir);
    void computeDeformationGradientUnitCell();
    void fitDeformationGradientUnitCell();
    

    // Scene.cpp 
    void buildSceneFromUnitPatch(int patch_id);

    void checkConnections();
    void build5NodeTestScene();
    void buildLongRodForBendingTest();
    void buildShearingTest();
    void buildPlanePeriodicBCScene3x3();
    void buildPlanePeriodicBCScene3x3Subnodes(int sub_div = 1);
    
    void subdivideRods(int sub_div);
    void subdivideRodIntoWeightMatrix(int div);
    void buildPeriodicNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    
    //Visualization.cpp
    void getColorPerYarn(Eigen::MatrixXd& C, int n_rod_per_yarn = 4);
    void getEulerianDisplacement(Eigen::MatrixXd& X, Eigen::MatrixXd& x);
    void getColorFromStretching(Eigen::MatrixXd& C);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::Ref<const DOFStack> q_display, Eigen::Ref<const IV3Stack> rods_display,
        Eigen::Ref<const TV3Stack> normal_tile);
    void markSlidingRange(int idx, int dir, int depth, std::vector<bool>& can_slide, int root);
    
    // DerivativeTest.cpp
    void runDerivativeTest();
    void checkGradientSecondOrderTerm(Eigen::Ref<VectorXT> dq);
    void checkHessianHigherOrderTerm(Eigen::Ref<VectorXT> dq);
    void checkGradient(Eigen::Ref<VectorXT> dq);
    void checkHessian(Eigen::Ref<VectorXT> dq);

    void checkMaterialPositionDerivatives();


    // ======================== Energy Forces and Hessian Entries ========================
    //                                             so -df/dx
    // Bending.cpp
    
    void addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addBendingEnergy(Eigen::Ref<const DOFStack> q_temp);

    // Stretching.cpp
    void addStretchingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addStretchingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addStretchingEnergy(Eigen::Ref<const DOFStack> q_temp);

    // Shearing.cpp
    void addShearingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, bool top_right);  
    void addShearingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual, bool top_right);
    T addShearingEnergy(Eigen::Ref<const DOFStack> q_temp, bool top_right);
    void toMapleNodesVector(std::vector<Vector<T, dim>>& x, Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes);

    //PeriodicBC.cpp
    T addPBCEnergy(Eigen::Ref<const DOFStack> q_temp);
    T addPBCEnergyALM(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<const DOFStack> lambdas, T kappa);
    void addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    void addPBCForceALM(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual,
         Eigen::Ref<const DOFStack> lambdas, T kappa);
    void addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addPBCKALM(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, T kappa);  
    void buildMapleRotationPenaltyData(Eigen::Ref<const DOFStack> q_temp, 
        std::vector<TV>& data, std::vector<int>& nodes);

    // EulerianConstraints.cpp
    T addEulerianRegEnergy(Eigen::Ref<const DOFStack> q_temp);
    void addEulerianRegForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    void addEulerianRegK(std::vector<Eigen::Triplet<T>>& entry_K);  

    // ParallelContact.cpp
    T addParallelContactEnergy(Eigen::Ref<const DOFStack> q_temp);
    void addParallelContactForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    void addParallelContactK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);
};

#endif