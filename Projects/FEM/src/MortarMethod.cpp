#include "../include/MortarMethod.h"
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

const std::vector<std::pair<double,double>> IntegrationPoints = {
    std::pair<double,double>(5.0/9.0, -sqrt(3.0/5.0)),
    std::pair<double,double>(8.0/9.0, 0.0),
    std::pair<double,double>(5.0/9.0, +sqrt(3.0/5.0))
};

void Mortar::testcase()
{
    // Test calculateMortarMatrices
    // V.setZero(4,2);
    // V(0,0) = 1.0; V(0,1) = 2.0;
    // V(1,0) = 3.0; V(1,1) = 2.0;
    // V(2,0) = 2.0; V(2,1) = 2.0;
    // V(3,0) = 0.0; V(3,1) = 2.0;

    // segments.clear();
    // segments[0] = std::pair<int,int>(0,1);
    // segments[1] = std::pair<int,int>(2,3);

    // normals.clear();
    // normals[0] = Eigen::Vector2d(0.0,-1.0);
    // normals[1] = Eigen::Vector2d(0.0,-1.0);

    // seg_info.clear();
    // seg_info[0].push_back({1,Eigen::Vector2d(-1.0,0.0)});
    // Eigen::MatrixXd De;
    // std::unordered_map<int,Eigen::MatrixXd> Me;

    // calculateMortarMatrices(0,De,Me);

    // std::cout<<De<<std::endl;
    // std::cout<<Me(0)<<std::endl;

    // Test calculate normals
    // V.setZero(3,2);
    // V(0,0) = 7.0; V(0,1) = 7.0;
    // V(1,0) = 4.0; V(1,1) = 3.0;
    // V(2,0) = 0.0; V(2,1) = 0.0;

    // segments.clear();
    // segments[0] = std::pair<int,int>(0,1);
    // segments(0) = std::pair<int,int>(1,2);

    // calculateNormals();

    // for(int i=0; i<3; ++i)
    // {
    //     std::cout<<normals[i].transpose()<<std::endl;
    // }

    // Test Projections
    // Eigen::Vector2d master_1(7.0,2.0);
    // Eigen::Vector2d master_2(4.0,-2.0);
    // Eigen::Vector2d slave(0.0, 0.0);
    // Eigen::Vector2d n(3.0/5.0, -4.0/5.0);

    // double xi2 = projectSlaveToMaster(slave,n,master_1,master_2);
    // std::cout<<xi2<<std::endl;

    // Eigen::Vector2d master(4.0,-2.0);
    // Eigen::Vector2d slave_1(0.0,0.0);
    // Eigen::Vector2d slave_2(4.0,3.0);
    // Eigen::Vector2d n1(3.0/5.0, -4.0/5.0);
    // Eigen::Vector2d n2(sqrt(2.)/2.,-sqrt(2.)/2.);

    // double xi1 = projectMasterToSlave(master,n1,n2,slave_1,slave_2);
    // std::cout<<xi1<<std::endl;

    //Test Compute Segments
    // V.setZero(4,2);
    // V(0,0) = 1.0; V(0,1) = 2.0;
    // V(1,0) = 3.0; V(1,1) = 2.0;
    // V(2,0) = 0.0; V(2,1) = 2.0;
    // V(3,0) = 2.0; V(3,1) = 2.0;    

    // segments.clear();
    // segments[0] = std::pair<int,int>(0,1);
    // segments[1] = std::pair<int,int>(2,3);

    // normals.clear();
    // normals[0] = Eigen::Vector2d(0.0,-1.0);
    // normals[1] = Eigen::Vector2d(0.0,-1.0);

    // slave_indices.clear();
    // slave_indices.push_back(0);
    // master_indices.clear();
    // master_indices.push_back(1);

    // calculateSegments();
    // //std::cout<<seg_info[0].size()<<std::endl;
    // std::cout<<seg_info[0][0].master_id<<" "<<seg_info[0][0].xs.transpose()<<std::endl;

    //Test Mortar Method
    // V.setZero(6,2);
    // V(0,0) = 0.0; V(0,1) = 2.0;
    // V(1,0) = 5.0/4.0; V(1,1) = 2.0;
    // V(2,0) = 2.0; V(2,1) = 2.0;
    // V(3,0) = 0.0; V(3,1) = 1.0;    
    // V(4,0) = 1.0; V(4,1) = 1.0;    
    // V(5,0) = 2.0; V(5,1) = 1.0;

    // displacement.setZero(6,2);

    // segments.clear();
    // segments[0] = std::pair<int,int>(0,1);
    // segments[1] = std::pair<int,int>(1,2);
    // segments[2] = std::pair<int,int>(3,4);
    // segments[3] = std::pair<int,int>(4,5);

    // slave_indices.clear();
    // slave_indices.push_back(0);
    // slave_indices.push_back(1);
    // master_indices.clear();
    // master_indices.push_back(2);
    // master_indices.push_back(3);

    // MortarContactMethod();
    
    // std::cout<<"Mortar Matrix D: "<<std::endl;
    // std::cout<<D<<std::endl;
    // std::cout<<std::endl;

    // std::cout<<"Mortar Matrix M: "<<std::endl;
    // std::cout<<M<<std::endl;
    // std::cout<<std::endl;
    // //std::cout<<G.transpose()<<std::endl;

    // Eigen::VectorXd um(3);
    // um<<1,2,3;
    // Eigen::VectorXd b = M*um;
    // Eigen::LDLT<Eigen::MatrixXd> ldlt;
    // ldlt.compute(D);
    // Eigen::VectorXd x = ldlt.solve(b);

    // std::cout<<"master displacement: "<<std::endl;
    // std::cout<<um.transpose()<<std::endl;

    // std::cout<<"slave displacement: "<<std::endl;
    // std::cout<<x.transpose()<<std::endl;

    //Test Contact segment
    V.setZero(6,2);
    V(0,0) = 0.0; V(0,1) = 0.0;
    V(1,0) = 1.0; V(1,1) = 0.0;
    V(2,0) = 2.0; V(2,1) = 0.0;
    V(3,0) = 0.0; V(3,1) = -0.2; 
    V(4,0) = 1.0; V(4,1) = 1.8; 
    V(5,0) = 2.0; V(5,1) = 2.8; 

    displacement.setZero(6,2);   
    //V(3,0) = 0.0; V(3,1) = 0.0; 

    segments.clear();
    segments[0] = std::pair<int,int>(0,1);
    segments[1] = std::pair<int,int>(1,2);
    segments[2] = std::pair<int,int>(3,4);
    segments[3] = std::pair<int,int>(4,5);

    slave_indices.clear();
    slave_indices.push_back(0);
    slave_indices.push_back(1);
    master_indices.clear();
    master_indices.push_back(2);
    master_indices.push_back(3);

    // calculateNormals();
    // std::vector<contact_info_per_slave_node> result;

    // calculateContactSegments(0,result);
    // std::cout<<result.size()<<std::endl;
    MortarContactMethod();
    std::cout<<G.transpose()<<std::endl;

    // V.setZero(11,2);
    // V(0,0) = 8.0; V(0,1) = 10.0;
    // V(1,0) = 7.0; V(1,1) = 7.0;
    // V(2,0) = 4.0; V(2,1) = 3.0;
    // V(3,0) = 0.0; V(3,1) = 0.0;    
    // V(4,0) = -3.0; V(4,1) = 0.0;    
    // V(5,0) = 12.0; V(5,1) = 10.0;
    // V(6,0) = 10.0; V(6,1) = 4.0;
    // V(7,0) = 7.0; V(7,1) = 2.0;
    // V(8,0) = 4.0; V(8,1) = -2.0;
    // V(9,0) = 0.0; V(9,1) = -3.0;
    // V(10,0) = -4.0; V(10,1) = -3.0;

    // displacement.setZero(11,2);
    // deformed = false;

    // segments.clear();
    // segments[0] = std::pair<int,int>(0,1);
    // segments[1] = std::pair<int,int>(1,2);
    // segments[2] = std::pair<int,int>(2,3);
    // segments[3] = std::pair<int,int>(3,4);
    // segments[4] = std::pair<int,int>(5,6);
    // segments[5] = std::pair<int,int>(6,7);
    // segments[6] = std::pair<int,int>(7,8);
    // segments[7] = std::pair<int,int>(8,9);
    // segments[8] = std::pair<int,int>(9,10);

    // slave_indices.clear();
    // slave_indices.push_back(0);
    // slave_indices.push_back(1);
    // slave_indices.push_back(2);
    // slave_indices.push_back(3);
    // master_indices.clear();
    // master_indices.push_back(4);
    // master_indices.push_back(5);
    // master_indices.push_back(6);
    // master_indices.push_back(7);
    // master_indices.push_back(8);

    // MortarMethod();
    
    // std::cout<<"Mortar Matrix D: "<<std::endl;
    // std::cout<<D<<std::endl;
    // std::cout<<std::endl;

    // std::cout<<"Mortar Matrix M: "<<std::endl;
    // std::cout<<M<<std::endl;
    // std::cout<<std::endl;
    //std::cout<<G.transpose()<<std::endl;

    // Eigen::VectorXd um(3);
    // um<<1,2,3;
    // Eigen::VectorXd b = M*um;
    // Eigen::LDLT<Eigen::MatrixXd> ldlt;
    // ldlt.compute(D);
    // Eigen::VectorXd x = ldlt.solve(b);

    // std::cout<<"master displacement: "<<std::endl;
    // std::cout<<um.transpose()<<std::endl;

    // std::cout<<"slave displacement: "<<std::endl;
    // std::cout<<x.transpose()<<std::endl;

}

void Mortar::calculateNormals()
{
    normals.clear();
    tangents.clear();

    for(auto it=segments.begin(); it!=segments.end(); ++it)
    {
        int i1 = it->second.first;
        int i2 = it->second.second;

        double dx = V(i2,0)-V(i1,0);
        double dy = V(i2,1)-V(i1,1);

        Eigen::Vector2d n(-dy,dx);
        n.normalize();
        
        for(int j=0; j<2; ++j)
        {
            double index = i1;
            if(j == 1) index = i2;
            auto it = normals.find(index);
            if(it == normals.end())
            {
                normals[index] = Eigen::Vector2d(0,0);
            }
            normals[index] += n;
        }
    }
    for(auto it = normals.begin(); it != normals.end(); it++)
    {
        it->second.normalize();
        std::cout<<"normals: "<<it->first<<" "<<it->second.transpose()<<std::endl;
        tangents[it->first] = Eigen::Vector2d(it->second(1),-it->second(0));
    }
}

double Mortar::projectSlaveToMaster(Eigen::Vector2d slave, Eigen::Vector2d& n, Eigen::Vector2d master_1, Eigen::Vector2d master_2)
{
    double upper = n(0)*(master_1(1) + master_2(1) - 2*slave(1)) - n(1)*(master_1(0) + master_2(0) - 2*slave(0));
    double lower = n(0)*(master_1(1) - master_2(1)) + n(1)*(master_2(0) - master_1(0));

    return upper/lower;
}

double Mortar::projectMasterToSlave(Eigen::Vector2d master, Eigen::Vector2d& n1, Eigen::Vector2d& n2, Eigen::Vector2d slave_1, Eigen::Vector2d slave_2)
{
    if ((n1-n2).norm() < 1e-5)
    {
        double upper = -2*n1(0)*master(1) + n1(0)*slave_1(1) + n1(0)*slave_2(1) + 2*n1(1)*master(0) - n1(1)*slave_1(0) - n1(1)*slave_2(0);
        double lower = n1(0)*slave_1(1) - n1(0)*slave_2(1) - n1(1)*slave_1(0) + n1(1)*slave_2(0);
        //std::cout<<upper<<" "<<lower<<std::endl;
        return upper/lower;
    }

    double b_upper = 2*n1(0)*master(1) - 2*n1(0)*slave_1(1) - 2*n1(1)*master(0) + 2*n1(1)*slave_1(0) - 2*n2(0)*master(1) + 2*n2(0)*slave_2(1) + 2*n2(1)*master(0) - 2*n2(1)*slave_2(0);
    double b_lower = n1(0)*slave_1(1) - n1(0)*slave_2(1) - n1(1)*slave_1(0) + n1(1)*slave_2(0) - n2(0)*slave_1(1) + n2(0)*slave_2(1) + n2(1)*slave_1(0) - n2(1)*slave_2(0);
    double c_upper = -2*n1(0)*master(1) + n1(0)*slave_1(1) + n1(0)*slave_2(1) + 2*n1(1)*master(0) - n1(1)*slave_1(0) - n1(1)*slave_2(0) - 2*n2(0)*master(1) + n2(0)*slave_1(1) + n2(0)*slave_2(1) + 2*n2(1)*master(0) - n2(1)*slave_1(0) - n2(1)*slave_2(0);
    double c_lower = n1(0)*slave_1(1) - n1(0)*slave_2(1) - n1(1)*slave_1(0) + n1(1)*slave_2(0) - n2(0)*slave_1(1) + n2(0)*slave_2(1) + n2(1)*slave_1(0) - n2(1)*slave_2(0);

    double a = 1.0;
    double b = b_upper / b_lower;
    double c = c_upper / c_lower;
    double d = b*b - 4*a*c;
    if (d < 0)
    {
        std::cerr<<"Mortar Method Error: negative discriminant "<<d<<std::endl;
        return -1;
    }
    double res1 = (-b + sqrt(d))/(2.0*a);
    double res2 = (-b - sqrt(d))/(2.0*a);

    //std::cout<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<res1<<" "<<res2<<"\n";

    if(fabs(res1) < fabs(res2)) return res1;
    else return res2;
}

void Mortar::calculateSegments()
{
    seg_info.clear();
    for(int i=0; i<slave_indices.size(); ++i)
    {
        auto it = seg_info.find(slave_indices[i]);
        if(it == seg_info.end())
        {
            seg_info[slave_indices[i]] = std::vector<segment_info>(0);
        }

        Eigen::Vector2d slave_1 = V.row(segments[slave_indices[i]].first);
        Eigen::Vector2d slave_2 = V.row(segments[slave_indices[i]].second);

        Eigen::Vector2d n1 = normals[segments[slave_indices[i]].first];
        Eigen::Vector2d n2 = normals[segments[slave_indices[i]].second];

        for(int j=0; j<master_indices.size(); ++j)
        {
            Eigen::Vector2d master_1 = V.row(segments[master_indices[j]].first);
            Eigen::Vector2d master_2 = V.row(segments[master_indices[j]].second);

            double stm1 = projectSlaveToMaster(slave_1,n1,master_1,master_2);
            double stm2 = projectSlaveToMaster(slave_2,n2,master_1,master_2);

            if(stm1 > 1.0 && stm2 > 1.0) continue;
            if(stm1 < -1.0 && stm2 < -1.0) continue;

            double mts1 = projectMasterToSlave(master_1,n1,n2,slave_1,slave_2);
            double mts2 = projectMasterToSlave(master_2,n1,n2,slave_1,slave_2);


            Eigen::Vector2d res = {std::clamp(mts1,-1.0,1.0),std::clamp(mts2,-1.0,1.0)};
            // std::cout<<slave_indices[i]<<" "<<master_indices[j]<<" "<<res.transpose()<<std::endl;
            // std::cout<<mts1<<" "<<mts2<<std::endl;
            if(fabs(res(1)-res(0)) < 1e-5) continue;
            seg_info[slave_indices[i]].push_back({master_indices[j],res});
        }
    }
}

void Mortar::calculateMortarMatrices(int slave_segment_id, Eigen::MatrixXd& De, std::unordered_map<int,Eigen::MatrixXd>& Me)
{
    De.setZero(2,2);
    Me.clear();
    
    Eigen::Vector2d slave_1 = V.row(segments[slave_segment_id].first);
    Eigen::Vector2d slave_2 = V.row(segments[slave_segment_id].second);

    Eigen::Vector2d n1 = normals[segments[slave_segment_id].first];
    Eigen::Vector2d n2 = normals[segments[slave_segment_id].second];

    double J = (slave_2-slave_1).norm()/2.0;

    std::vector<segment_info> slave_projections = seg_info[slave_segment_id];

    for(int i=0; i<slave_projections.size(); ++i)
    {
        int master_id = slave_projections[i].master_id;
        double coor1 = slave_projections[i].xs(0);
        double coor2 = slave_projections[i].xs(1);

        double s = fabs(coor2-coor1)/2.0;

        for(int j=0; j<IntegrationPoints.size(); ++j)
        {
            double w = IntegrationPoints[j].first;
            double pos = IntegrationPoints[j].second;

            double coor_gauss = (1-pos)/2.*coor1 + (1+pos)/2.*coor2;
            Eigen::Vector2d N((1-coor_gauss)/2., (1+coor_gauss)/2.);
            De += w*N*N.transpose()*s*J;

        }
    }

    for(int i=0; i<slave_projections.size(); ++i)
    {
        int master_segment_id = slave_projections[i].master_id;
        Eigen::Vector2d master_1 = V.row(segments[master_segment_id].first);
        Eigen::Vector2d master_2 = V.row(segments[master_segment_id].second);

        double coor1 = slave_projections[i].xs(0);
        double coor2 = slave_projections[i].xs(1);
        double s = fabs(coor2-coor1)/2.0;

        Me[master_segment_id] = Eigen::MatrixXd(2,2);
        Me[master_segment_id].setZero();


        for(int j=0; j<IntegrationPoints.size(); ++j)
        {
            double w = IntegrationPoints[j].first;
            double pos = IntegrationPoints[j].second;

            double coor_gauss = (1-pos)/2*coor1 + (1+pos)/2*coor2;
            Eigen::Vector2d N((1-coor_gauss)/2, (1+coor_gauss)/2);
            
            Eigen::Vector2d n_gauss = N(0)*n1 + N(1)*n2;
            Eigen::Vector2d slave_gauss = N(0)*slave_1 + N(1)*slave_2;

            double slave_coor_gauss = projectSlaveToMaster(slave_gauss, n_gauss, master_1, master_2);
            Eigen::Vector2d N2((1-slave_coor_gauss)/2, (1+slave_coor_gauss)/2);
            Me[master_segment_id] += w*N*N2.transpose()*s*J;
        }
    }
}

void Mortar::MortarMethod()
{
    for(int i=0; i<slave_indices.size(); ++i)
    {
        slave_nodes.push_back(segments[slave_indices[i]].first);
        slave_nodes.push_back(segments[slave_indices[i]].second);
    }

    for(int i=0; i<master_indices.size(); ++i)
    {
        master_nodes.push_back(segments[master_indices[i]].first);
        master_nodes.push_back(segments[master_indices[i]].second);
    }

    std::sort(slave_nodes.begin(), slave_nodes.end());
    std::sort(master_nodes.begin(), master_nodes.end());

    auto last = std::unique(slave_nodes.begin(), slave_nodes.end());
    slave_nodes.erase(last, slave_nodes.end());
    last = std::unique(master_nodes.begin(), master_nodes.end());
    master_nodes.erase(last, master_nodes.end());



    calculateNormals();
    calculateSegments();

    int size = V.rows();
    Eigen::MatrixXd D_temp, M_temp;

    D_temp.setZero(size,size);
    M_temp.setZero(size,size);

    for(int i=0; i<slave_indices.size(); ++i)
    {
        std::vector<int> master_segment_ids;
        for(int j=0; j<seg_info[slave_indices[i]].size(); ++j)
        {
            master_segment_ids.push_back(seg_info[slave_indices[i]][j].master_id);
        }

        Eigen::MatrixXd De;
        std::unordered_map<int,Eigen::MatrixXd> Me;
        calculateMortarMatrices(slave_indices[i],De,Me);

        int i1 = segments[slave_indices[i]].first;
        int i2 = segments[slave_indices[i]].second;

        D_temp(i1,i1) += De(0,0);D_temp(i1,i2) += De(0,1);
        D_temp(i2,i1) += De(1,0);D_temp(i2,i2) += De(1,1);

        for(int j=0; j<master_segment_ids.size(); j++)
        {
            int master_segment_index = master_segment_ids[j];
            int m1 = segments[master_segment_index].first;
            int m2 = segments[master_segment_index].second;

            M_temp(i1,m1) += Me[master_segment_index](0,0);M_temp(i1,m2) += Me[master_segment_index](0,1);
            M_temp(i2,m1) += Me[master_segment_index](1,0);M_temp(i2,m2) += Me[master_segment_index](1,1);

        }
    }

    D.setZero(slave_nodes.size(),slave_nodes.size());
    M.setZero(master_nodes.size(),master_nodes.size());

    // for(int i=0; i<slave_nodes.size(); ++i)
    // {
    //     std::cout<<slave_nodes[i]<<" ";
    // }
    // std::cout<<std::endl;

    for(int i=0; i<slave_nodes.size(); ++i)
    {
        for(int j=0; j<slave_nodes.size(); ++j)
        {
            D(i,j) = D_temp(slave_nodes[i],slave_nodes[j]);
        }
    }

    for(int i=0; i<slave_nodes.size(); ++i)
    {
        for(int j=0; j<master_nodes.size(); ++j)
        {
            M(i,j) = M_temp(slave_nodes[i],master_nodes[j]);
        }
    }
}

void Mortar::calculateContactSegments(int slave_index, std::vector<contact_info_per_slave_node>& result)
{
    result.clear();

    int slave_node_index_1 = segments[slave_indices[slave_index]].first;
    int slave_node_index_2 = segments[slave_indices[slave_index]].second;
    Eigen::Vector2d slave_node_1 = V.row(slave_node_index_1);
    Eigen::Vector2d slave_node_2 = V.row(slave_node_index_2);

    if(deformed)
    {
        slave_node_1 += displacement.row(slave_node_index_1);
        slave_node_2 += displacement.row(slave_node_index_2);
    }

    Eigen::Vector2d slave_midpoint = 0.5*(slave_node_1+slave_node_2);
    double slave_length = (slave_node_2-slave_node_1).norm();

    Eigen::Vector2d n1 = normals[segments[slave_indices[slave_index]].first];
    Eigen::Vector2d n2 = normals[segments[slave_indices[slave_index]].second];

    for(int i=0; i<master_indices.size(); ++i)
    {
        int master_node_index_1 = segments[master_indices[i]].first;
        int master_node_index_2 = segments[master_indices[i]].second;

        Eigen::Vector2d master_node_1 = V.row(master_node_index_1);
        Eigen::Vector2d master_node_2 = V.row(master_node_index_2);

        if(deformed)
        {
            master_node_1 += displacement.row(master_node_index_1);
            master_node_2 += displacement.row(master_node_index_2);
        }

        Eigen::Vector2d master_midpoint = 0.5*(master_node_1+master_node_2);
        double master_length = (master_node_2-master_node_1).norm();

        double dist = (master_midpoint-slave_midpoint).norm();
        double cl = dist/std::max(slave_length,master_length);

        std::cout<<"cl: "<<cl<<std::endl;

        if(cl>max_distance) continue;

        double mts1 = projectMasterToSlave(master_node_1,n1,n2,slave_node_1,slave_node_2);
        double mts2 = projectMasterToSlave(master_node_2,n1,n2,slave_node_1,slave_node_2);


        Eigen::Vector2d res = {std::clamp(mts1,-1.0,1.0),std::clamp(mts2,-1.0,1.0)};
        std::cout<<res.transpose()<<std::endl;
        if(fabs(res(1)-res(0)) < 1e-5) continue;
        double l = 0.5*fabs(res(1)-res(0));

        result.push_back({master_indices[i], res, l});
    }
}

void Mortar::MortarContactMethod()
{
    for(int i=0; i<slave_indices.size(); ++i)
    {
        slave_nodes.push_back(segments[slave_indices[i]].first);
        slave_nodes.push_back(segments[slave_indices[i]].second);
    }

    for(int i=0; i<master_indices.size(); ++i)
    {
        master_nodes.push_back(segments[master_indices[i]].first);
        master_nodes.push_back(segments[master_indices[i]].second);
    }

    std::sort(slave_nodes.begin(), slave_nodes.end());
    std::sort(master_nodes.begin(), master_nodes.end());

    auto last = std::unique(slave_nodes.begin(), slave_nodes.end());
    slave_nodes.erase(last, slave_nodes.end());
    last = std::unique(master_nodes.begin(), master_nodes.end());
    master_nodes.erase(last, master_nodes.end());



    calculateNormals();
    calculateSegments();

    int size = V.rows();
    Eigen::MatrixXd D_temp, M_temp;
    Eigen::VectorXd G_temp;

    D_temp.setZero(size,size);
    M_temp.setZero(size,size);
    G_temp.setZero(size);

    for(int i=0; i<slave_indices.size(); ++i)
    {
        std::vector<int> master_segment_ids;
        for(int j=0; j<seg_info[slave_indices[i]].size(); ++j)
        {
            master_segment_ids.push_back(seg_info[slave_indices[i]][j].master_id);
        }

        Eigen::MatrixXd De;
        std::unordered_map<int,Eigen::MatrixXd> Me;
        Eigen::VectorXd ge;
        calculateMortarMatriceGap(slave_indices[i],De,Me,ge);

        int i1 = segments[slave_indices[i]].first;
        int i2 = segments[slave_indices[i]].second;

        D_temp(i1,i1) += De(0,0);D_temp(i1,i2) += De(0,1);
        D_temp(i2,i1) += De(1,0);D_temp(i2,i2) += De(1,1);
        G_temp(i1) += ge(0);
        G_temp(i2) += ge(1);

        for(int j=0; j<master_segment_ids.size(); j++)
        {
            int master_segment_index = master_segment_ids[j];
            int m1 = segments[master_segment_index].first;
            int m2 = segments[master_segment_index].second;

            M_temp(i1,m1) += Me[master_segment_index](0,0);M_temp(i1,m2) += Me[master_segment_index](0,1);
            M_temp(i2,m1) += Me[master_segment_index](1,0);M_temp(i2,m2) += Me[master_segment_index](1,1);

        }
    }

    D.setZero(slave_nodes.size(),slave_nodes.size());
    G.setZero(slave_nodes.size());
    M.setZero(master_nodes.size(),master_nodes.size());

    // for(int i=0; i<slave_nodes.size(); ++i)
    // {
    //     std::cout<<slave_nodes[i]<<" ";
    // }
    // std::cout<<std::endl;

    for(int i=0; i<slave_nodes.size(); ++i)
    {
        for(int j=0; j<slave_nodes.size(); ++j)
        {
            D(i,j) = D_temp(slave_nodes[i],slave_nodes[j]);
        }
        G(i) = G_temp(slave_nodes[i]);
    }

    for(int i=0; i<slave_nodes.size(); ++i)
    {
        for(int j=0; j<master_nodes.size(); ++j)
        {
            M(i,j) = M_temp(slave_nodes[i],master_nodes[j]);
        }
    }
}

void Mortar::calculateMortarMatriceGap(int slave_segment_id, Eigen::MatrixXd& De, std::unordered_map<int,Eigen::MatrixXd>& Me, Eigen::VectorXd& ge)
{
    De.setZero(2,2);
    Me.clear();
    ge.setZero(2);
    
    Eigen::Vector2d slave_1 = V.row(segments[slave_segment_id].first);
    Eigen::Vector2d slave_2 = V.row(segments[slave_segment_id].second);
    Eigen::Vector2d slave_1_deformed = V.row(segments[slave_segment_id].first);
    Eigen::Vector2d slave_2_deformed = V.row(segments[slave_segment_id].second);
    Eigen::Vector2d us1; us1.setZero();
    Eigen::Vector2d us2; us2.setZero();

    if(deformed)
    {
        us1 = displacement.row(segments[slave_segment_id].first);
        us2 = displacement.row(segments[slave_segment_id].second);
    }

    slave_1_deformed+=us1;
    slave_2_deformed+=us2;

    Eigen::Vector2d n1 = normals[segments[slave_segment_id].first];
    Eigen::Vector2d n2 = normals[segments[slave_segment_id].second];

    double J = (slave_2-slave_1).norm()/2.0;

    std::vector<segment_info> slave_projections = seg_info[slave_segment_id];

    for(int i=0; i<slave_projections.size(); ++i)
    {
        int master_id = slave_projections[i].master_id;
        double coor1 = slave_projections[i].xs(0);
        double coor2 = slave_projections[i].xs(1);

        double s = fabs(coor2-coor1)/2.0;

        for(int j=0; j<IntegrationPoints.size(); ++j)
        {
            double w = IntegrationPoints[j].first;
            double pos = IntegrationPoints[j].second;

            double coor_gauss = (1-pos)/2.*coor1 + (1+pos)/2.*coor2;
            Eigen::Vector2d N((1-coor_gauss)/2., (1+coor_gauss)/2.);
            De += w*N*N.transpose()*s*J;

        }
    }

    for(int i=0; i<slave_projections.size(); ++i)
    {
        int master_segment_id = slave_projections[i].master_id;
        Eigen::Vector2d master_1 = V.row(segments[master_segment_id].first);
        Eigen::Vector2d master_2 = V.row(segments[master_segment_id].second);
        Eigen::Vector2d master_1_deformed = V.row(segments[master_segment_id].first);
        Eigen::Vector2d master_2_deformed = V.row(segments[master_segment_id].second);

        Eigen::Vector2d um1; um1.setZero();
        Eigen::Vector2d um2; um2.setZero();

        if(deformed)
        {
            um1 = displacement.row(segments[master_segment_id].first);
            um2 = displacement.row(segments[master_segment_id].second);
        }

        master_1_deformed+=um1;
        master_2_deformed+=um2;

        double coor1 = slave_projections[i].xs(0);
        double coor2 = slave_projections[i].xs(1);
        double s = fabs(coor2-coor1)/2.0;

        Me[master_segment_id] = Eigen::MatrixXd(2,2);
        Me[master_segment_id].setZero();


        for(int j=0; j<IntegrationPoints.size(); ++j)
        {
            double w = IntegrationPoints[j].first;
            double pos = IntegrationPoints[j].second;

            double coor_gauss = (1-pos)/2*coor1 + (1+pos)/2*coor2;
            Eigen::Vector2d N((1-coor_gauss)/2, (1+coor_gauss)/2);
            
            Eigen::Vector2d n_gauss = N(0)*n1 + N(1)*n2;
            Eigen::Vector2d slave_gauss = N(0)*slave_1 + N(1)*slave_2;
            Eigen::Vector2d master_gauss = N(0)*master_1 + N(1)*master_2;
            Eigen::Vector2d us_gauss = N(0)*us1 + N(1)*us2;
            Eigen::Vector2d um_gauss = N(0)*um1 + N(1)*um2;

            Eigen::Vector2d slave_deformed_gauss = slave_gauss+us_gauss;
            Eigen::Vector2d master_deformed_gauss = master_gauss+um_gauss;

            double slave_coor_gauss = projectSlaveToMaster(slave_gauss, n_gauss, master_1, master_2);
            Eigen::Vector2d N2((1-slave_coor_gauss)/2, (1+slave_coor_gauss)/2);
            Me[master_segment_id] += w*N*N2.transpose()*s*J;

            // std::cout<<"n_gauss "<<n_gauss.transpose()<<std::endl;
            // std::cout<<"difference_gauss "<<(master_deformed_gauss-slave_deformed_gauss).transpose()<<std::endl;
            ge += w*N*((master_deformed_gauss-slave_deformed_gauss).dot(n_gauss))*s*J;
        }
    }
}