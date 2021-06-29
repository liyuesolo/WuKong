#ifndef UNIT_PATCH_H
#define UNIT_PATCH_H

#include "EoLRodSim.h"

template<class T, int dim>
class EoLRodSim;

template<class T, int dim>
class UnitPatch
{
public:
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

private:
    EoLRodSim<T, dim>& sim;

    DOFStack& q = sim.q;
    IV3Stack& rods = sim.rods;
    IV4Stack& connections = sim.connections;


public:
    UnitPatch(EoLRodSim<T, dim>& eol_sim) : sim(eol_sim) {}
    ~UnitPatch() {}

    void buildScene(int patch_type);

    void build3x3StraightRod();
    void buildPlanePeriodicBCScene3x3();

    void buildStraightAndSineScene(int sub_div);
    void buildStraightAndHemiCircleScene(int sub_div);
    void buildTwoRodsScene(int sub_div);
    void buildSlidingTestScene(int sub_div);
    void buildStraightYarnScene(int sub_div);
    void buildUnitFromC2Curves(int sub_div);

    void buildZigZagScene(int sub_div);

    void buildStraightYarn3x1(int sub_div);


private:
    T LDis(int i, int j) { return (q.col(i).template segment<dim>(0) - q.col(j).template segment<dim>(0)).norm(); }
    void add4Nodes(int front, int end, int yarn_id, int rod_id)
    {
        if (rods(0, rod_id) == front)
        {
            sim.pbc_bending_bn_pairs[yarn_id][0] = front;
            sim.pbc_bending_bn_pairs[yarn_id][1] = rods(1, rod_id);
            sim.pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
        }
        if (rods(1, rod_id) == end)
        {
            sim.pbc_bending_bn_pairs[yarn_id][3] = end;
            sim.pbc_bending_bn_pairs[yarn_id][2] = rods(0, rod_id);
            sim.pbc_bending_bn_pairs[yarn_id][4] = rods(2, rod_id);
        }
    };
    void set_left_right(int idx, int left)
    {
        connections(0, idx) = left;
        connections(1, left) = idx;
    }
    void set_top_bottom(int idx, int top)
    {
        connections(3, idx) = top;
        connections(2, top) = idx;
    }

    void setLagPos(int idx, const TV& lpos)
    {
        if(idx >= q.cols())
            std::cout << "[UnitPatch.h] invalid idx --- exceeding matrix size " << std::endl;
        q.col(idx).template segment<dim>(0) = lpos;
    }

    void setEulPos(int idx, const TV2& epos)
    {
        if(idx >= q.cols())
            std::cout << "[UnitPatch.h] invalid idx --- exceeding matrix size " << std::endl;
        q.col(idx).template segment<2>(dim) = epos;
    }

    void setPos(int idx, const TV& lpos, const TV2& epos)
    {
        if(idx >= q.cols())
            std::cout << "[UnitPatch.h] invalid idx --- exceeding matrix size " << std::endl;
        q.col(idx).template segment<dim>(0) = lpos;
        q.col(idx).template segment<2>(dim) = epos;
    }

    void fixEulerian(int idx)
    {
        sim.dirichlet_data[idx] = std::make_pair(TVDOF::Zero(), sim.fix_eulerian);
    }

    void clearSimData();
    
    void addRods(std::vector<int>& nodes, int yarn_type, int& cnt, int yarn_idx = 0);

    void addStraightYarnCrossNPoints(const TV& from, const TV& to,
        const std::vector<TV>& passing_points, 
        const std::vector<int>& passing_points_id, 
        int sub_div,
        std::vector<TV>& sub_points, std::vector<int>& node_idx,
        std::vector<int>& key_points_location,
        int start, bool pbc = false);

    void subdivideStraightYarns(int sub_div);

    void markDoFSingleStrand(
        const std::vector<int>& nodes_on_strand, 
        const std::vector<int>& key_points_location_on_strand, 
        const std::vector<int>& key_points_location_global, 
        const std::vector<int>& key_points_location_dof, 
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt, int yarn_type);

    void addKeyPointsDoF(const std::vector<int>& key_points, 
        std::vector<Eigen::Triplet<T>>& w_entry,
        int& dof_cnt);
};

#endif