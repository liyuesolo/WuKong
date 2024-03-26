#ifndef MORTAR_METHOD_DIFF_H
#define MORTAR_METHOD_DIFF_H
#include <eigen3/Eigen/Core>
#include "VecMatDef.h"
#include <algorithm>
#include <cppad/cppad.hpp>

struct all_info
{
    double return_value;
    Eigen::VectorXd return_value_grad;
    Eigen::MatrixXd return_value_hess;
};

class MortarMethodDiff
{
public:
    //Input Values
    Eigen::MatrixXd V;
    std::vector<int> slave_indices;
    std::vector<int> master_indices;
    std::unordered_map<int,std::pair<int,int>> segments;

    //Mutiple Input data
    // std::vector<Eigen::MatrixXd> Vs;
    // std::vector<std::vector<int>> multiple_slave_indices;
    // std::vector<std::vector<int>> multiple_master_indices;

    bool deformed = true;

    //Output Values
    std::unordered_map<int,all_info> normals;
    std::vector<int> slave_nodes;
    std::vector<int> master_nodes;

    struct segment_info_diff
    {
        int master_id;
        std::vector<all_info> xs;
    };
    std::unordered_map<int, std::vector<segment_info_diff>> seg_info;
    
    std::vector<all_info> gap_functions;
    all_info gap_energy;

    //IMLS Data
    bool use_mortar_IMLS = false;
    int mc_samples = 50;
    std::vector<double> SamplePoints;

public:
    double randfrom(double min, double max) 
    {
        double range = (max - min); 
        double div = RAND_MAX / range;
        return min + (rand() / div);
    }
    void initialization(Eigen::MatrixXd& Vi, std::vector<int>& slave_indices_in, std::vector<int>& master_indices_in, 
                        std::unordered_map<int,std::pair<int,int>>& segments_in)
    {
        std::cout<<V<<std::endl;
        V = Vi;
        slave_indices = slave_indices_in;
        master_indices = master_indices_in;
        segments = segments_in;

        // SamplePoints.resize(mc_samples);
        // for(int i=0; i<mc_samples; ++i)
        // {
        //     SamplePoints[i] = randfrom(-1,1);
        //     std::cout<<SamplePoints[i]<<" ";
        // }
        // std::cout<<std::endl;
    }

    void updateVertices(Eigen::MatrixXd& Vi)
    {
        V = Vi;
    }

    void testcase();
    void calculateNormals(bool update = false);
    void calculateSegments(bool update = false);
    all_info projectSlaveToMaster(int slave, int normal, int master_1, int master_2, bool update = false);
    all_info projectMasterToSlave(int master, int n1, int n2, int slave_1, int slave_2, bool update = false);
    void calculateMortarGapEnergy(int slave_segment_id, std::vector<all_info>& gap_erengy_per_element, bool update = false);
    void MortarMethod(bool update = false);
    void findContactMaster(std::vector<double>& master_contact_nodes);

    MortarMethodDiff(){}
    ~MortarMethodDiff(){}
};


#endif