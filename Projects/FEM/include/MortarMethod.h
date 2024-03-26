#ifndef MORTAR_METHOD_H
#define MORTAR_METHOD_H
#include <eigen3/Eigen/Core>
#include "VecMatDef.h"
#include <algorithm>

class Mortar
{
public:
    //Input Values
    Eigen::MatrixXd V;
    Eigen::MatrixXd displacement;
    std::vector<int> slave_indices;
    std::vector<int> master_indices;
    std::unordered_map<int,std::pair<int,int>> segments;
    bool deformed = true;
    double max_distance = 2;

    //Output Values
    std::unordered_map<int,Eigen::Vector2d> normals;
    std::unordered_map<int,Eigen::Vector2d> tangents;
    std::vector<int> slave_nodes;
    std::vector<int> master_nodes;

    struct segment_info
    {
        int master_id;
        Eigen::Vector2d xs;
    };
    std::unordered_map<int, std::vector<segment_info>> seg_info;
    Eigen::MatrixXd D;
    Eigen::MatrixXd M;
    Eigen::VectorXd G;

    struct contact_info_per_slave_node
    {
        int master_id;
        Eigen::Vector2d xs;
        double l;
    };
    

public:
    void initialization(Eigen::MatrixXd& Vi, Eigen::MatrixXd& u, std::vector<int>& slave_indices_in, std::vector<int>& master_indices_in, 
                        std::unordered_map<int,std::pair<int,int>>& segments_in)
    {
        V = Vi;
        displacement = u;
        slave_indices = slave_indices_in;
        master_indices = master_indices_in;
        segments = segments_in;
    }

    void calculateSegments();
    void calculateNormals();
    double projectSlaveToMaster(Eigen::Vector2d slave, Eigen::Vector2d& n, Eigen::Vector2d master_1, Eigen::Vector2d master_2);
    double projectMasterToSlave(Eigen::Vector2d master, Eigen::Vector2d& n1, Eigen::Vector2d& n2, Eigen::Vector2d slave_1, Eigen::Vector2d slave_2);
    void calculateMortarMatrices(int slave_segment_id, Eigen::MatrixXd& De, std::unordered_map<int,Eigen::MatrixXd>& Me);
    void MortarMethod();
    void testcase();
    void calculateContactSegments(int slave_index,std::vector<contact_info_per_slave_node>& result);
    void MortarContactMethod();
    void calculateMortarMatriceGap(int slave_segment_id, Eigen::MatrixXd& De, std::unordered_map<int,Eigen::MatrixXd>& Me, Eigen::VectorXd& ge);

    Mortar(){};
    ~Mortar(){};

};

#endif