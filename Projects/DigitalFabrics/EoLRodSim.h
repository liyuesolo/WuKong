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

    int dof = dim + 2;
    
    DOFStack q, q0;
    IV3Stack rods;
    IV4Stack connections;
    TV3Stack normal;
    int n_nodes;
    int n_rods;
    IV2 n_rod_uv;
    
    const static int grid_range = 3;
    int final_dim;

    T dt = 1;
    T newton_tol = 1e-6;
    T E = 1e7;
    T R = 0.01;

    T rho = 1;
    T ks = 1.0;  // stretching term
    T kc = 1e2;  //constraint term
    T kn = 1e-3; //
    T kb = 1.0;
    T kb_penalty = 1.0;
    T km = 1e-3; //mass term
    T kx = 1.0; // shearing term
    T k_pbc = 1.0; // perodic BC term
    T L = 1;
    T ke = 1e-2; // Eulerian DoF penalty
    T kr = 1e3;
    

    TV gravity = TV::Zero();

    bool add_stretching = true;
    bool add_bending = true;
    bool add_shearing = true;
    bool add_penalty = false;
    bool add_regularizor = false;
    bool add_pbc = true;
    bool add_eularian_reg = true;
    bool disable_sliding = true;

    TVDOF fix_all, fix_eulerian, fix_lagrangian, fix_u, fix_v;
    std::unordered_map<int, std::pair<TVDOF, TVDOF>> dirichlet_data;
    std::vector<std::vector<int>> pbc_bending_pairs;
    std::vector<std::vector<int>> yarns;
    std::unordered_map<int, int> yarn_map;
    std::unordered_map<IV2, int, VectorHash<2>> pbc_pairs;
    // pbc_ref[direction] = (node_i, node_j)
    std::vector<std::pair<int, IV2>> pbc_ref;

    std::vector<IV2> pbc_ref_unique;

    std::vector<std::pair<IV2, std::pair<TV, T>>> pbc_strain_data;

    std::vector<std::vector<int>> yarn_group;
    std::vector<bool> is_end_nodes;

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
    }
    ~EoLRodSim() {}
    
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
    }

    template <class OP>
    void iterateYarnCrossingsParallel(const OP& f) {
        tbb::parallel_for(0, n_nodes, [&](int i)
        {
            f(i, connections(0, i), connections(1, i), connections(2, i), connections(3, i));
        }); 
    }


    void initializeSystemMatrix(Eigen::Ref<const TVStack> dq, 
        std::vector<int> &entryCol, 
        std::vector<TMDOF> &entryVal)
    {
        final_dim = std::pow(grid_range, dof);
        int n_rows = n_nodes;
        entryCol.resize(n_rows * final_dim);
        entryVal.resize(n_rows * final_dim);
        tbb::parallel_for(0, n_rows, [&](int g) {
            for (int i = 0; i < final_dim; ++i) {
                entryCol[g * final_dim + i] = -1;
                entryVal[g * final_dim + i] = TMDOF::Zero();
            }
        });
    }

    void fixEulerian()
    {
        for(int i = 0; i < n_nodes; i++)
            dirichlet_data[i] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
    }
    void freeEulerian()
    {
        dirichlet_data.clear();
        dirichlet_data[2] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[9] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[16] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[3] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[10] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[17] = std::make_pair(TVDOF::Zero(), fix_eulerian);

        dirichlet_data[1] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[8] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[15] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[0] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[7] = std::make_pair(TVDOF::Zero(), fix_eulerian);
        dirichlet_data[14] = std::make_pair(TVDOF::Zero(), fix_eulerian);

        dirichlet_data[12] = std::make_pair(TVDOF::Zero(), fix_lagrangian);
    }

    // EoLSim.cpp
    T computeTotalEnergy(Eigen::Ref<const DOFStack> dq);
    T computeResidual(Eigen::Ref<DOFStack> residual, Eigen::Ref<const DOFStack> dq);
    void addMassMatrix(std::vector<Eigen::Triplet<T>>& entry_K);
    bool projectDirichletEntrySystemMatrix(StiffnessMatrix& A);
    void addStiffnessMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq);
    void addConstraintMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq);
    void buildSystemMatrix(std::vector<Eigen::Triplet<T>>& entry_K, Eigen::Ref<const DOFStack> dq);
    bool linearSolve(const std::vector<Eigen::Triplet<T>>& entry_K, 
        Eigen::Ref<const DOFStack> residual, Eigen::Ref<DOFStack> ddq);
    T newtonLineSearch(Eigen::Ref<DOFStack> dq, 
        Eigen::Ref<const DOFStack> residual, int line_search_max = 10000);
    void implicitUpdate(Eigen::Ref<DOFStack> dq);
    void advanceOneStep();

    void resetScene() { q = q0; }
    
public:
    // Elasticity.cpp
    void setUniaxialStrain(T theta, T s, TV& strain_dir);
    void computeMacroStress(TM& sigma, TV strain_dir);
    void computeDeformationGradientUnitCell();
    void fitDeformationGradientUnitCell();
    

    // Scene.cpp 
    void checkConnections();
    void build5NodeTestScene();
    void buildLongRodForBendingTest();
    void buildShearingTest();
    void buildPlanePeriodicBCScene3x3();
    void buildPlanePeriodicBCScene3x3Subnodes(int sub_div = 1);
    void subdivideRods(int sub_div);
    void buildPlanePeriodicBCScene1x1();
    void buildRodNetwork(int width, int height);
    void buildPeriodicNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    
    //Visualization.cpp
    void getColorPerYarn(Eigen::MatrixXd& C, int n_rod_per_yarn = 4);
    void getEulerianDisplacement(Eigen::MatrixXd& X, Eigen::MatrixXd& x);
    void getColorFromStretching(Eigen::MatrixXd& C);
    void buildMeshFromRodNetwork(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        Eigen::Ref<const DOFStack> q_display, Eigen::Ref<const IV3Stack> rods_display,
        Eigen::Ref<const TV3Stack> normal_tile);

    // BoundaryCondtion.cpp
    void addBCStretchingTest();
    void addBCShearingTest();
    
    // DerivativeTest.cpp
    void runDerivativeTest();
    void checkGradient(Eigen::Ref<DOFStack> dq);
    void checkHessian(Eigen::Ref<DOFStack> dq);


    // ======================== Energy Forces and Hessian Entries ========================
    //                                             so -df/dx
    // Bending.cpp
    void toMapleNodesVector(std::vector<Vector<T, dim + 1>>& x, Eigen::Ref<const DOFStack> q_temp,
        std::vector<int>& nodes, int yarn_type);
    void entryHelperBending(Eigen::Ref<const DOFStack> q_temp, 
        std::vector<Eigen::Triplet<T>>& entry_K, int n0, int n1, int n2, int uv_offset);
    void addBendingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addBendingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addBendingEnergy(Eigen::Ref<const DOFStack> q_temp);
    T bendingEnergySingleDirection(Eigen::Ref<const DOFStack> q_temp, int n0, int n1, int n2, int uv_offset);
    void addBendingForceSingleDirection(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual,
        int n0, int n1, int n2, int uv_offset);

    // Stretching.cpp
    void addStretchingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void addStretchingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    T addStretchingEnergy(Eigen::Ref<const DOFStack> q_temp);

    // Shearing.cpp
    void addShearingK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K, bool top_right);  
    void addShearingForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual, bool top_right);
    T addShearingEnergy(Eigen::Ref<const DOFStack> q_temp, bool top_right);

    //PeriodicBC.cpp
    T addPBCEnergy(Eigen::Ref<const DOFStack> q_temp);
    void addPBCForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    void addPBCK(Eigen::Ref<const DOFStack> q_temp, std::vector<Eigen::Triplet<T>>& entry_K);  
    void buildMapleRotationPenaltyData(Eigen::Ref<const DOFStack> q_temp, 
        std::vector<TV>& data, std::vector<int>& nodes);

    // EulerianConstraints.cpp
    T addEulerianRegEnergy(Eigen::Ref<const DOFStack> q_temp);
    void addEulerianRegForce(Eigen::Ref<const DOFStack> q_temp, Eigen::Ref<DOFStack> residual);
    void addEulerianRegK(std::vector<Eigen::Triplet<T>>& entry_K);  

    // ParallelContact.cpp
    T addParallelContactEnergy(Eigen::Ref<const DOFStack> q_temp);
};

#endif