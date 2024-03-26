#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/remove_duplicate_vertices.h>
#include "../include/Contact2D.h"


#include <random>
#include <cmath>
#include <fstream>

std::random_device rd;
std::mt19937 gen( rd() );
std::uniform_real_distribution<> dis( 0.0, 1.0 );
std::uniform_real_distribution<> dis2( -1.0, 1.0 );
// float DISPLAYSMENT;


static double zeta()
{
	return dis(gen);
}

static double zeta2()
{
	return dis2(gen);
}

void Contact2D::initializeSimulationData()
{
    bool load_mesh = true;
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    Eigen::MatrixXd VS;
    Eigen::MatrixXi FS, FS_Quad;
    Eigen::MatrixXi HS;
    
    std::vector<int> dirichlet_vertices;
    solver.boundary_info.clear();
    solver.CauchyStressTensor.clear();
    solver.master_nodes.clear();
    solver.slave_nodes.clear();
    solver.master_segments.clear();
    solver.slave_segments.clear();
    solver.slave_ele_indices.clear();

    solver.dirichlet_data.clear();
    solver.penalty_pairs.clear();

    solver.boundary_segments.clear();
    solver.sample_points.clear();
    solver.use_multiple_pairs = false;

    if (load_mesh)
    {   
        //igl::readOBJ("/home/parallels/Desktop/WuKong/Projects/FEM/data/beam.obj", V, F);
        
        if(SLIDING_TEST)
        {
            int num_pts = solver.sliding_res+1;
            double dtheta = M_PI/(2*solver.sliding_res);

            VS.resize(2*num_pts+1,2);
            FS.resize(2*solver.sliding_res,3);
            FS_Quad.resize(solver.sliding_res,4);

            solver.slave_nodes.push_back(2*num_pts);
            for(int i=0; i<num_pts; ++i)
            {
                double theta = (num_pts-1-i)*dtheta;
                VS(i,0) = solver.r1*(cos(theta)); VS(i,1) = -solver.r1*(sin(theta));
                VS(i+num_pts,0) = solver.r2*(cos(theta)); VS(i+num_pts,1) = -solver.r2*(sin(theta));
                solver.master_nodes.push_back(i);

                if(i!=num_pts-1)
                {
                    FS(2*i,0) = i; FS(2*i,1) = i+num_pts; FS(2*i,2) = i+1;
                    FS(2*i+1,0) = i+1; FS(2*i+1,1) = i+num_pts; FS(2*i+1,2) = i+num_pts+1;
                    FS_Quad(i,0) = i; FS_Quad(i,1) = i+num_pts; FS_Quad(i,2) = i+num_pts+1; FS_Quad(i,3) = i+1;
                    if(USE_IMLS)
                        solver.master_segments.push_back(std::pair<int,int>(i,i+1));
                    else
                        solver.master_segments.push_back(std::pair<int,int>(i+1,i));
                }
            }
            if(USE_TRUE_IPC_2D){
                VS(2*num_pts,0) = (solver.r1+GAP)*(cos(solver.theta1)); VS(2*num_pts,1) = -(solver.r1+GAP)*(sin(solver.theta1));
            }else{
                VS(2*num_pts,0) = (solver.r1-GAP)*(cos(solver.theta1)); VS(2*num_pts,1) = -(solver.r1-GAP)*(sin(solver.theta1));
            }
            if(USE_IMLS) std::reverse(solver.master_nodes.begin(),solver.master_nodes.end());
        }
        else if(TEST)
        {
            solver.left_boundaries.clear();
            solver.right_boundaries.clear();
            solver.left_boundaries_master.clear();
            solver.right_boundaries_master.clear();

            int num_x = 5*RES+1, num_y = 3*RES+1;
            int num_nodes_1 = num_x*num_y;

            VS.resize(20,2);
            VS(0,0) = 0; VS(0,1) = 0;
            VS(1,0) = 3; VS(1,1) = 0;
            VS(2,0) = 6; VS(2,1) = 0;
            VS(3,0) = 10; VS(3,1) = 0;
            VS(4,0) = 12; VS(4,1) = 0;

            VS(5,0) = 0; VS(5,1) = 2;
            VS(6,0) = 3; VS(6,1) = 2;
            VS(7,0) = 6; VS(7,1) = 2;
            VS(8,0) = 10; VS(8,1) = 2;
            VS(9,0) = 12; VS(9,1) = 2;

            VS(10,0) = 0; VS(10,1) = 2+GAP;
            VS(11,0) = 2; VS(11,1) = 2+GAP;
            VS(12,0) = 4; VS(12,1) = 2+GAP;
            VS(13,0) = 8; VS(13,1) = 2+GAP;
            VS(14,0) = 12; VS(14,1) = 2+GAP;

            VS(15,0) = 0; VS(15,1) = 4+GAP;
            VS(16,0) = 2; VS(16,1) = 4+GAP;
            VS(17,0) = 4; VS(17,1) = 4+GAP;
            VS(18,0) = 8; VS(18,1) = 4+GAP;
            VS(19,0) = 12; VS(19,1) = 4+GAP;

            if(USE_IMLS)
            {
                for(int i=9; i>=5; --i)
                {
                    solver.master_nodes.push_back(i);
                    if(i != 9) solver.master_segments.push_back(std::pair<int,int>(i,i+1));
                }
                // for(int i=0; i<5; ++i)
                // {
                //     solver.master_nodes.push_back(i);
                //     if(i != 5) solver.master_segments.push_back(std::pair<int,int>(i,i+1));
                // }

                // for(int i=15; i>=10; --i)
                // {
                //     //solver.slave_nodes.push_back(i);
                //     if(i != 15) solver.slave_segments.push_back(std::pair<int,int>(i,i+1));
                // }
                for(int i=10; i<15; ++i)
                {
                    solver.slave_nodes.push_back(i);
                    if(i != 15) solver.slave_segments.push_back(std::pair<int,int>(i,i+1));
                }
            }
            else
            {
                for(int i=5; i<10; ++i)
                {
                    solver.master_nodes.push_back(i);
                    if(i != 9) solver.master_segments.push_back(std::pair<int,int>(i+1,i));
                }

                for(int i=10; i<15; ++i)
                {
                    solver.slave_nodes.push_back(i);
                    if(i != 14) solver.slave_segments.push_back(std::pair<int,int>(i+1,i));
                }
            }

            

            FS_Quad.resize(8,4);
            FS_Quad(0,0) = 0; FS_Quad(0,1) = 1; FS_Quad(0,2) = 6; FS_Quad(0,3) = 5;
            FS_Quad(1,0) = 1; FS_Quad(1,1) = 2; FS_Quad(1,2) = 7; FS_Quad(1,3) = 6;
            FS_Quad(2,0) = 2; FS_Quad(2,1) = 3; FS_Quad(2,2) = 8; FS_Quad(2,3) = 7;
            FS_Quad(3,0) = 3; FS_Quad(3,1) = 4; FS_Quad(3,2) = 9; FS_Quad(3,3) = 8;

            FS_Quad(4,0) = 10; FS_Quad(4,1) = 11; FS_Quad(4,2) = 16; FS_Quad(4,3) = 15;
            FS_Quad(5,0) = 11; FS_Quad(5,1) = 12; FS_Quad(5,2) = 17; FS_Quad(5,3) = 16;
            FS_Quad(6,0) = 12; FS_Quad(6,1) = 13; FS_Quad(6,2) = 18; FS_Quad(6,3) = 17;
            FS_Quad(7,0) = 13; FS_Quad(7,1) = 14; FS_Quad(7,2) = 19; FS_Quad(7,3) = 18;

            for(int i=4; i<8; ++i)
            {
                solver.slave_ele_indices.push_back(i);
            }

            FS.resize(16,3);
            FS(0,0) = 0; FS(0,1) = 1; FS(0,2) = 6; 
            FS(1,0) = 0; FS(1,1) = 6; FS(1,2) = 5; 
            FS(2,0) = 1; FS(2,1) = 2; FS(2,2) = 7; 
            FS(3,0) = 1; FS(3,1) = 7; FS(3,2) = 6; 
            FS(4,0) = 2; FS(4,1) = 3; FS(4,2) = 8; 
            FS(5,0) = 2; FS(5,1) = 8; FS(5,2) = 7; 
            FS(6,0) = 3; FS(6,1) = 4; FS(6,2) = 9; 
            FS(7,0) = 3; FS(7,1) = 9; FS(7,2) = 8; 

            FS(8,0) = 10; FS(8,1) = 11; FS(8,2) = 16; 
            FS(9,0) = 10; FS(9,1) = 16; FS(9,2) = 15; 
            FS(10,0) = 11; FS(10,1) = 12; FS(10,2) = 17; 
            FS(11,0) = 11; FS(11,1) = 17; FS(11,2) = 16; 
            FS(12,0) = 12; FS(12,1) = 13; FS(12,2) = 18; 
            FS(13,0) = 12; FS(13,1) = 18; FS(13,2) = 17; 
            FS(14,0) = 13; FS(14,1) = 14; FS(14,2) = 19; 
            FS(15,0) = 13; FS(15,1) = 19; FS(15,2) = 18; 

            solver.left_boundaries.push_back(10); solver.left_boundaries.push_back(15);
            solver.right_boundaries.push_back(14); solver.right_boundaries.push_back(19);
            solver.left_boundaries_master.push_back(0); solver.left_boundaries_master.push_back(5);
            solver.right_boundaries_master.push_back(4); solver.right_boundaries_master.push_back(9);
            


            // VS.resize(num_nodes_1,2);
            // for(int j=0; j<num_y; ++j){
            //     for(int i=0; i<num_x; ++i){
            //         VS(i+j*num_x,0) = (i)/double(num_x-1)*5.;
            //         VS(i+j*num_x,1) = (j)/double(num_y-1)*3.;

            //         if(j == 0)
            //         {
            //             solver.slave_nodes.push_back(i+j*num_x);
            //         }
            //     }
            // }
            
            // FS_Quad.resize((num_x-1)*(num_y-1),4);
            // for(int i=0; i<num_x-1; ++i){
            //     for(int j=0; j<num_y-1; ++j){
            //         int i1 = (i*(num_y-1)+j);
            //         FS_Quad(i1,0) = i+j*num_x;
            //         FS_Quad(i1,1) = (i+1)+j*num_x;
            //         FS_Quad(i1,2) = (i+1)+(j+1)*num_x;
            //         FS_Quad(i1,3) = (i)+(j+1)*num_x;

            //         if(j == 0)
            //         {
            //             solver.slave_segments.push_back(std::pair<int,int>((i+1)+(j)*num_x ,(i)+(j)*num_x));
            //             solver.slave_ele_indices.push_back(i1);
            //         }
            //     }
            // }

            // FS.resize(2*(num_x-1)*(num_y-1),3);
            // for(int i=0; i<num_x-1; ++i){
            //     for(int j=0; j<num_y-1; ++j){
            //         int i1 = 2*(i*(num_y-1)+j);
            //         FS(i1,0) = i+j*num_x;
            //         FS(i1,1) = (i+1)+j*num_x;
            //         FS(i1,2) = (i+1)+(j+1)*num_x;

            //         int i2 = 2*(i*(num_y-1)+j)+1;
            //         FS(i2,0) = i+j*num_x;
            //         FS(i2,1) = (i+1)+(j+1)*num_x;
            //         FS(i2,2) = (i)+(j+1)*num_x;
            //         if(j == 0 && !USE_QUAD_ELEMENT)
            //         {
            //             solver.slave_segments.push_back(std::pair<int,int>((i+1)+(j)*num_x ,(i)+(j)*num_x));
            //         }
            //     }
            // }
        }
        else
        {
            std::cout<<"Displacement: "<<DISPLAYSMENT<<std::endl;
            int num_x = WIDTH_1*RES+1, num_y = HEIGHT_1*RES+1;
            int num_x_2 = WIDTH_2*SCALAR*RES_2+1, num_y_2 = HEIGHT_2*SCALAR*RES_2+1;

            // find displacement: Insert two new cols
            bool addone = false;
            bool addtwo = false;
            int i1 = -1;
            int i2 = -1;

            std::vector<double> x_pools1, x_pools2;
            std::vector<double> y_pools1, y_pools2;

            if(USE_NONUNIFORM_MESH)
            {
                x_pools1.push_back(0);
                for(int i=0 ; i<num_x-2; ++i)
                    x_pools1.push_back(double(i+1)/double(num_x)*WIDTH_1 + zeta2()*double(0.5)/double(num_x)*WIDTH_1);
                x_pools1.push_back(WIDTH_1);
                x_pools2.push_back(DISPLAYSMENT);
                for(int i=0 ; i<num_x_2-2; ++i)
                    x_pools2.push_back(double(i+1)/double(num_x_2)*WIDTH_2 + zeta2()*double(0.5)/double(num_x_2)*WIDTH_2+DISPLAYSMENT);
                x_pools2.push_back(DISPLAYSMENT+WIDTH_2);

                y_pools1.push_back(0);
                for(int i=0 ; i<num_y-2; ++i)
                    y_pools1.push_back(double(i+1)/double(num_y)*HEIGHT_1 + zeta2()*double(0.5)/double(num_y)*HEIGHT_1);
                y_pools1.push_back(HEIGHT_1);
                y_pools2.push_back(HEIGHT_1 + GAP);
                for(int i=0 ; i<num_y_2-2; ++i)
                    y_pools2.push_back(double(i+1)/double(num_y_2)*HEIGHT_2 + zeta2()*double(0.5)/double(num_y_2)*HEIGHT_2+ GAP+HEIGHT_1);
                y_pools2.push_back(HEIGHT_1+GAP+HEIGHT_2);

                sort(x_pools1.begin(), x_pools1.end());   
                sort(x_pools2.begin(), x_pools2.end());
                sort(y_pools1.begin(), y_pools1.end());   
                sort(y_pools2.begin(), y_pools2.end());      
            }
        
        
            if(MODE == 2)
            {
                double seg_len = double(1)/double(RES);
                if(fmod(DISPLAYSMENT, seg_len)*seg_len != 0)
                {
                    i1 = DISPLAYSMENT/seg_len+1;
                    num_x++;
                    addone = true;
                }
                if(fmod(WIDTH_2+DISPLAYSMENT, seg_len)*seg_len != 0)
                {
                    i2 = (DISPLAYSMENT+WIDTH_2)/seg_len+1;
                    if(addone) i2++;
                    num_x++;
                    addtwo = true;
                }
            }
            

            std::cout<<i1<<" "<<i2<<" "<<num_x<<std::endl;
            
            int num_nodes_1 = num_x*num_y;
            int num_nodes_2 = num_x_2*num_y_2;

            solver.master_segments.clear();
            solver.slave_nodes.clear();
            solver.Object_indices.clear();
            solver.Object_indices_rot.clear();
            solver.pbc_pairs.clear();
            solver.left_boundaries.clear();
            solver.right_boundaries.clear();
            solver.left_boundaries_master.clear();
            solver.right_boundaries_master.clear();

            Eigen::VectorXd upper(num_nodes_1);
            Eigen::VectorXd lower(num_nodes_2);

            VS.resize(num_nodes_1+num_nodes_2,2);
            for(int j=0; j<num_y; ++j){
                for(int i=0; i<num_x; ++i){
                    int cur_i = i;
                    int x_num = num_x;

                    if(i == 0) solver.left_boundaries_master.push_back(i+j*num_x);
                    if(i == num_x-1) solver.right_boundaries_master.push_back(i+j*num_x);

                    if(addone && i == i1)
                    {
                        VS(i+j*num_x,0) = DISPLAYSMENT;
                    }
                    else if(addtwo && i == i2)
                    {
                        VS(i+j*num_x,0) = DISPLAYSMENT+WIDTH_2;
                    }
                    else
                    {
                        if(addone && i >= i1){
                            cur_i--;
                        } 
                        if(addtwo && i >= i2){
                            cur_i--;
                        } 
                        if(addone) x_num--;
                        if(addtwo) x_num--;
                        
                        if(USE_NONUNIFORM_MESH)
                            VS(i+j*num_x,0) = x_pools1[cur_i];
                        else
                            VS(i+j*num_x,0) = (cur_i)/double(x_num-1)*WIDTH_1;

                        //std::cout<<cur_i<<" "<<x_num<<" "<<VS(i+j*num_x,0)<<std::endl;
                    }

                    if(USE_NONUNIFORM_MESH)
                        VS(i+j*num_x,1) = y_pools1[j];
                    else
                        VS(i+j*num_x,1) = (j)/double(num_y-1)*HEIGHT_1;
                    upper(i+j*num_x) = i+j*num_x;
                    if(j == num_y-1)
                    {
                        solver.master_nodes.push_back(i+j*num_x);
                    }
                    
                    std::cout<<VS(i+j*num_x,0)<<" ";
                }
                std::cout<<std::endl;
            }

            for(int j=0; j<num_y_2; ++j){
                for(int i=0; i<num_x_2; ++i){
                    if(USE_NONUNIFORM_MESH)
                        VS(i+j*num_x_2+num_nodes_1,0) = x_pools2[i];
                    else
                        VS(i+j*num_x_2+num_nodes_1,0) = (i)/double(num_x_2-1)*WIDTH_2+DISPLAYSMENT;
                    if(USE_NONUNIFORM_MESH)
                        VS(i+j*num_x_2+num_nodes_1,1) = y_pools2[j];
                    else
                        VS(i+j*num_x_2+num_nodes_1,1) = (j)/double(num_y_2-1)*HEIGHT_2+HEIGHT_1 + GAP;
                    lower(i+j*num_x_2) = i+j*num_x_2+num_nodes_1;
                    if(i == 0)
                    {
                        solver.left_boundaries.push_back(i+j*num_x_2+num_nodes_1);
                    }
                    if(i == num_x_2-1)
                    {
                        solver.right_boundaries.push_back(i+j*num_x_2+num_nodes_1);
                    }
                    if(j == 0)
                    {
                        solver.slave_nodes.push_back(i+j*num_x_2+num_nodes_1);
                    }
                }
            } 
            //if(!solver.use_virtual_spring)
                solver.Object_indices.push_back(upper);
            solver.Object_indices.push_back(lower);
            solver.Object_indices_rot.push_back(lower);

            if(solver.use_PBC_penalty)
            {
                for(int j=0; j<num_y; ++j)
                {
                    int i1 = 0+j*num_x;
                    int i2 = num_x-1+j*num_x;
                    solver.pbc_pairs.push_back(std::pair<int,int>(i1,i2));
                    std::cout<<"PBC Nodes "<<j<<" "<<i1<<" "<<i2<<std::endl;
                }
            }

            FS_Quad.resize((num_x-1)*(num_y-1) + (num_x_2-1)*(num_y_2-1),4);
            for(int i=0; i<num_x-1; ++i){
                for(int j=0; j<num_y-1; ++j){
                    int i1 = (i*(num_y-1)+j);
                    FS_Quad(i1,0) = i+j*num_x;
                    FS_Quad(i1,1) = (i+1)+j*num_x;
                    FS_Quad(i1,2) = (i+1)+(j+1)*num_x;
                    FS_Quad(i1,3) = (i)+(j+1)*num_x;

                    if(j == num_y-2)
                    {
                        solver.master_segments.push_back(std::pair<int,int>((i+1)+(j+1)*num_x,(i)+(j+1)*num_x));
                    }
                }
            }

            for(int i=0; i<num_x_2-1; ++i){
                for(int j=0; j<num_y_2-1; ++j){
                    int i1 = (i*(num_y_2-1)+j) + (num_x-1)*(num_y-1);
                    FS_Quad(i1,0) = i+j*num_x_2+num_nodes_1;
                    FS_Quad(i1,1) = (i+1)+j*num_x_2+num_nodes_1;
                    FS_Quad(i1,2) = (i+1)+(j+1)*num_x_2+num_nodes_1;
                    FS_Quad(i1,3) = (i)+(j+1)*num_x_2+num_nodes_1;
                    if(j == 0)
                    {
                        solver.slave_segments.push_back(std::pair<int,int>((i+1)+(j)*num_x_2 + num_nodes_1,(i)+(j)*num_x_2 + num_nodes_1));
                        solver.slave_ele_indices.push_back(i1);
                    }
                }
            }

            FS.resize(2*(num_x-1)*(num_y-1) + 2*(num_x_2-1)*(num_y_2-1),3);
            for(int i=0; i<num_x-1; ++i){
                for(int j=0; j<num_y-1; ++j){
                    int i1 = 2*(i*(num_y-1)+j);
                    FS(i1,0) = i+j*num_x;
                    FS(i1,1) = (i+1)+j*num_x;
                    FS(i1,2) = (i+1)+(j+1)*num_x;

                    int i2 = 2*(i*(num_y-1)+j)+1;
                    FS(i2,0) = i+j*num_x;
                    FS(i2,1) = (i+1)+(j+1)*num_x;
                    FS(i2,2) = (i)+(j+1)*num_x;

                    if(j == num_y-2 && !USE_QUAD_ELEMENT)
                    {
                        solver.master_segments.push_back(std::pair<int,int>((i+1)+(j+1)*num_x,(i)+(j+1)*num_x));
                    }
                }
            }

            for(int i=0; i<num_x_2-1; ++i){
                for(int j=0; j<num_y_2-1; ++j){
                    int i1 = 2*(i*(num_y_2-1)+j) + 2*(num_x-1)*(num_y-1);
                    FS(i1,0) = i+j*num_x_2+num_nodes_1;
                    FS(i1,1) = (i+1)+j*num_x_2+num_nodes_1;
                    FS(i1,2) = (i+1)+(j+1)*num_x_2+num_nodes_1;

                    int i2 = 2*(i*(num_y_2-1)+j)+1 + 2*(num_x-1)*(num_y-1);
                    FS(i2,0) = i+j*num_x_2+num_nodes_1;
                    FS(i2,1) = (i+1)+(j+1)*num_x_2+num_nodes_1;
                    FS(i2,2) = (i)+(j+1)*num_x_2+num_nodes_1;

                    if(j == 0 && !USE_QUAD_ELEMENT)
                    {
                        solver.slave_segments.push_back(std::pair<int,int>((i+1)+(j)*num_x_2 + num_nodes_1,(i)+(j)*num_x_2 + num_nodes_1));
                    }
                    
                }
            }
            //VS(12,1) += 0.05;
        }

        // VS << 0,0, 10,0, 10,1, 0,1;
        // FS << 0,1,1,2,2,3,3,0;

        // V = VS;
        // F = FS;

        //igl::triangle::triangulate(VS,FS,HS,"a0.05q",V,F);

        solver.V_all = VS;
        solver.F_all = FS;
        solver.F_all_Quad = FS_Quad;


        // Eigen::MatrixXd V1(num_nodes_1,3);
        // V1.setZero();
        // V1.block(0,0,num_nodes_1,2) = VS.block(0,0,num_nodes_1,2);
        // Eigen::MatrixXd V2(num_nodes_2,3);
        // V2.setZero();
        // V2.block(0,0,num_nodes_2,2) = VS.block(num_nodes_1,0,num_nodes_2,2);

        // Eigen::MatrixXi F1 = FS_Quad.block(0,0,(num_x-1)*(num_y-1),4);
        // Eigen::MatrixXi shift((num_x_2-1)*(num_y_2-1),4);
        // for(int i=0; i<shift.size(); ++i) shift(i) = num_nodes_1;
        // Eigen::MatrixXi F2 = FS_Quad.block((num_x-1)*(num_y-1),0,(num_x_2-1)*(num_y_2-1),4)- shift;

        // igl::writeOBJ("lower.obj",V1,F1);
        // igl::writeOBJ("upper.obj",V2,F2);
    }

    // int xs = 4; int ys = 3;
    // double scales = 1;
    // VS.resize((xs+1)*(ys+1),2);
    // FS.resize(2*(xs)*(ys),3);
    // FS_Quad.resize((xs)*(ys),4);

    // for(int i=0; i<=xs; ++i)
    // {
    //     for(int j=0; j<=ys; ++j)
    //     {
    //         VS(i*(ys+1)+j,0) = i*scales;
    //         VS(i*(ys+1)+j,1) = j*scales;
    //     }
    // }

    // for(int i=0; i<xs; ++i)
    // {
    //     for(int j=0; j<ys; ++j)
    //     {
    //        FS(2*(i*ys+j),0) = i*(ys+1)+j;
    //        FS(2*(i*ys+j),1) = (i+1)*(ys+1)+j;
    //        FS(2*(i*ys+j),2) = (i+1)*(ys+1)+j+1;

    //        FS(2*(i*ys+j)+1,0) = i*(ys+1)+j;
    //        FS(2*(i*ys+j)+1,1) = (i+1)*(ys+1)+j+1;
    //        FS(2*(i*ys+j)+1,2) = i*(ys+1)+j+1;

    //        FS_Quad(i*ys+j,0) = i*(ys+1)+j;
    //        FS_Quad(i*ys+j,1) = (i+1)*(ys+1)+j;
    //        FS_Quad(i*ys+j,2) = (i+1)*(ys+1)+j+1;
    //        FS_Quad(i*ys+j,3) = i*(ys+1)+j+1;
    //     }
    // }
    // solver.V_all = VS;
    // solver.F_all = FS;
    // solver.F_all_Quad = FS_Quad;


    solver.initializeElementData(VS, FS, FS,FS_Quad, FS_Quad);
    //solver.initializeBoundaryInfo();
}

void Contact2D::initializeSimulationDataFromSTL()
{
    bool load_mesh = true;
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    Eigen::MatrixXd VS;
    Eigen::MatrixXi FS,TS,FS_Quad;
    Eigen::MatrixXi HS;
    
    std::vector<int> dirichlet_vertices;
    solver.boundary_info.clear();
    solver.CauchyStressTensor.clear();
    solver.master_nodes.clear();
    solver.slave_nodes.clear();
    solver.master_segments.clear();
    solver.slave_segments.clear();
    solver.slave_ele_indices.clear();

    solver.dirichlet_data.clear();
    solver.penalty_pairs.clear();

    solver.boundary_segments.clear();
    solver.sample_points.clear();

    solver.Object_indices.clear();
    solver.Object_indices_rot.clear();
    solver.pbc_pairs.clear();
    solver.left_boundaries.clear();
    solver.right_boundaries.clear();
    solver.left_boundaries_master.clear();
    solver.right_boundaries_master.clear();
    if(PULL_TEST) TEST_CASE = 5;

    if (load_mesh)
    {
        Eigen::MatrixXd V_upper, V_lower, SV;
        Eigen::MatrixXi F_upper, F_lower;
        Eigen::MatrixXi SF, SVI, SVJ;

        std::string filename_upper;
        std::string filename_lower;

        if(PULL_TEST)
        {
            // filename_upper = "pulltest1.obj";
            // filename_lower = "pulltest2.obj";
            filename_upper = "fan.obj";
            filename_lower = "hole.obj";
            // filename_upper = "inside_low_res_2.obj";
            // filename_lower = "outside_low_res_2.obj";
            // filename_upper = "inside_low_res_1.obj";
            // filename_lower = "outside_low_res_1.obj";
        }
        else if(TEST_CASE == 0)
        {
            filename_upper = "circular_mesh_quad.off";
            filename_lower = "rectangle_mesh_quad.off";
        }
        else if(TEST_CASE == 1)
        {
            // filename_upper = "circular_mesh_upper_adaptive_lowres3.off";
            // filename_lower = "circular_mesh_upper_adaptive_lowres3.off";
            filename_upper = "circular_mesh_upper_hires2.off";
            filename_lower = "circular_mesh_upper_hires2.off";
        }
        else if(TEST_CASE == 2)
        {
            filename_upper = "circular_mesh_upper_hires.off";
            filename_lower = "circular_mesh_upper_hires.off";
        }
        else if(TEST_CASE == 3)
        {
            filename_upper = "full_circle_low.off";
            filename_lower = "half_circle_low.off";
        }
        
        double eps = 1e-1;
        if(PULL_TEST) eps = 1e-7;
        else if(TEST_CASE == 1) eps = 0;
        else if(TEST_CASE == 2) eps = 1e-2;
        igl::readOFF(filename_lower,SV,SF);
        if(PULL_TEST)
            igl::readOBJ(filename_upper,SV,SF);

        Eigen::MatrixXd SV_2(SV.rows()-1,3);
        Eigen::MatrixXi SF_2(SF.rows(),SF.cols());
        if(TEST_CASE == 1 || TEST_CASE == 2 || TEST_CASE == 3)
        {
            for(int i=1; i<SV.rows(); ++i)
            {
                for(int j=0; j<3; ++j)
                {
                    SV_2(i-1,j) = SV(i,j);
                }
                
            }
            for(int i=0; i<SF.rows(); ++i)
            {
                for(int j=0; j<SF.cols(); ++j)
                {
                    SF_2(i,j) = SF(i,j)-1;
                }
            }
            
            
        }

        // double min_dist = 1000;
        // for(int i=0; i<SV_2.rows(); ++i)
        // {
        //     for(int j=i+1; j<SV_2.rows(); ++j)
        //     {
        //         double dist = (SV_2.row(i)-SV_2.row(j)).norm();
        //         min_dist = std::min(min_dist,dist);
        //     }
        // }
        // std::cout<<"min_dist: "<<min_dist<<std::endl;
        
        int num_vertices = SV_2.rows();
        if(TEST_CASE == 1)
        {
            V_upper = SV_2;
            F_upper = SF_2;
        }
        else if(TEST_CASE == 2)
        {
            V_upper = SV_2;
            F_upper = SF_2;
        }
        else if(TEST_CASE == 3)
        {
            V_upper = SV_2;
            F_upper = SF_2;
        }
        // else
        //     igl::remove_duplicate_vertices(SV,SF,eps,V_upper,SVI,SVJ,F_upper);
       
        if(TEST_CASE == 2) eps = 1e-1;
        
        igl::readOFF(filename_lower,SV,SF);
        if(PULL_TEST) igl::readOBJ(filename_lower,SV,SF);
        SV_2.setZero(SV.rows()-1,3);
        SF_2.setZero(SF.rows(),4);
        if(TEST_CASE == 1 || TEST_CASE == 2 || TEST_CASE == 3)
        {
            for(int i=1; i<SV.rows(); ++i)
            {
               for(int j=0; j<3; ++j)
                {
                    SV_2(i-1,j) = -SV(i,j);
                }
            }
            for(int i=0; i<SF.rows(); ++i)
            {
                for(int j=0; j<4; ++j)
                {
                    SF_2(i,j) = SF(i,j)-1;
                }
            }
        }
        num_vertices = SV.rows();

        if(TEST_CASE == 1)
        {
            V_lower = SV_2;
            F_lower = SF_2;
        }
        else if(TEST_CASE == 2)
        {
            V_lower = SV_2;
            F_lower = SF_2;
        }
        else if(TEST_CASE == 3)
        {
            int num_nodes = SV_2.rows();
            int num_faces = SF_2.rows();
            V_lower.setZero(2*num_nodes,3);
            F_lower.setZero(2*num_faces,4);
            
            for(int i=0; i<num_nodes; ++i)
            {
                V_lower.row(i) = SV_2.row(i);
                V_lower(i,1) += 0.1;
                V_lower.row(i+num_nodes) = -SV_2.row(i);
                V_lower(i+num_nodes,1) += 1.9;
            }

            for(int i=0; i<num_faces; ++i)
            {
                F_lower.row(i) = SF_2.row(i);
                F_lower.row(i+num_faces) = SF_2.row(i);
                for(int j=0; j<4; ++j)
                    F_lower(i+num_faces,j) += num_nodes;
            }
        }
        // else
        //     igl::remove_duplicate_vertices(SV,SF,eps,V_lower,SVI,SVJ,F_lower);
        
        // V_lower = SV;
        // F_lower = SF;

        // std::cout<<V_upper.rows()<<std::endl;
        // for(int i=0; i<F_upper.rows(); ++i)
        // {
        //     std::cout<<F_upper.row(i)<<std::endl;
        // }

        int num_upper_v = V_upper.rows();
        int num_upper_f = F_upper.rows();
        int num_lower_v = V_lower.rows();
        int num_lower_f = F_lower.rows();

        double upper_shift_y = GAP;
        if(TEST_CASE == 0) upper_shift_y += 2;
        if(PULL_TEST) upper_shift_y = 0;

        VS.resize(num_upper_v+num_lower_v,2);
        FS.resize((num_upper_f+num_lower_f)*2,3);
        if(PULL_TEST) FS.resize((num_upper_f+num_lower_f),3);
        FS_Quad.resize((num_upper_f+num_lower_f),4);

        solver.Object_indices.resize(2);
        solver.Object_indices[0].setZero(num_lower_v);
        solver.Object_indices[1].setZero(num_upper_v);
        solver.Object_indices_rot.resize(1);
        solver.Object_indices_rot[0].setZero(num_upper_v);
        solver.face_indices.resize(2);
        solver.face_indices[0].setZero(num_lower_f);
        solver.face_indices[1].setZero(num_upper_f);
        solver.Vs.resize(2);
        solver.Fs.resize(2);

        solver.Vs[0].setZero(V_lower.rows(),3);
        solver.Vs[1].setZero(V_upper.rows(),3);
        if(!PULL_TEST)
        {
            solver.Fs[0].setZero(2*F_lower.rows(),3);
            solver.Fs[1].setZero(2*F_upper.rows(),3);
        }else
        {
            solver.Fs[0].setZero(F_lower.rows(),3);
            solver.Fs[1].setZero(F_upper.rows(),3);
        }
        


        for(int i=0; i<num_lower_v; ++i)
        {
            VS(i,0) = V_lower(i,0);
            VS(i,1) = V_lower(i,1);

            solver.Object_indices[0](i) = i;
        }
        for(int i=0; i<num_upper_v; ++i)
        {
            VS(i+num_lower_v,0) = V_upper(i,0) + DISPLAYSMENT;
            VS(i+num_lower_v,1) = V_upper(i,1) + upper_shift_y;
            solver.Object_indices[1](i) = i+num_lower_v;
            solver.Object_indices_rot[0](i) = i+num_lower_v;
        }

        for(int i=0; i<num_lower_f; ++i)
        {
            if(!PULL_TEST)
            {
                FS_Quad.row(i) = F_lower.row(i);
                FS(2*i,0) = F_lower(i,0);
                FS(2*i,1) = F_lower(i,1);
                FS(2*i,2) = F_lower(i,2);

                FS(2*i+1,0) = F_lower(i,3);
                FS(2*i+1,1) = F_lower(i,0);
                FS(2*i+1,2) = F_lower(i,2); 

                solver.Fs[0].row(2*i) = FS.row(2*i);
                solver.Fs[0].row(2*i+1) = FS.row(2*i+1);
            }
            else
            {
                FS.row(i) = F_lower.row(i);
                solver.Fs[0].row(i) = FS.row(i);
            }
            
            solver.face_indices[0](i) = i;
        }
        Eigen::VectorXi shift(4);
        shift<<num_lower_v,num_lower_v,num_lower_v,num_lower_v;
        for(int i=0; i<num_upper_f; ++i)
        {
            if(!PULL_TEST)
            {
                for(int j=0;j<4;++j)
                    FS_Quad(i+num_lower_f, j) = F_upper(i,j)+num_lower_v;
                FS(2*(i+num_lower_f),0) = F_upper(i,0)+num_lower_v;
                FS(2*(i+num_lower_f),1) = F_upper(i,1)+num_lower_v;
                FS(2*(i+num_lower_f),2) = F_upper(i,2)+num_lower_v;

                FS(2*(i+num_lower_f)+1,0) = F_upper(i,3)+num_lower_v;
                FS(2*(i+num_lower_f)+1,1) = F_upper(i,0)+num_lower_v;
                FS(2*(i+num_lower_f)+1,2) = F_upper(i,2)+num_lower_v;

                for(int j=0; j<3; ++j)
                {
                    solver.Fs[1](2*i,j) = FS(2*(i+num_lower_f),j)-num_lower_v; 
                    solver.Fs[1](2*i+1,j) = FS(2*(i+num_lower_f)+1,j) - num_lower_v;
                }
            }else
            {
               for(int j=0; j<3; ++j)
                {
                    FS(i+num_lower_f,j) = F_upper(i,j)+num_lower_v;
                }
                solver.Fs[1].row(i) = F_upper.row(i);
            }
            
            solver.face_indices[1](i) = i+num_lower_f;
        }

        Eigen::VectorXd lower_boundary;
        if(!PULL_TEST)
            igl::boundary_loop(FS.block(0,0,2*(num_lower_f),3),lower_boundary);
        else
            igl::boundary_loop(FS.block(0,0,(num_lower_f),3),lower_boundary);
        Eigen::VectorXd upper_boundary;
        if(!PULL_TEST)
            igl::boundary_loop(FS.block(2*(num_lower_f),0,2*(num_upper_f),3),upper_boundary);
        else
            igl::boundary_loop(FS.block((num_lower_f),0,(num_upper_f),3),upper_boundary);

        for(int i=0; i<lower_boundary.size(); ++i)
        {
            std::cout<<"lower boundary: "<<i<<" "<<lower_boundary(i)<<std::endl;
        }

        for(int i=0; i<upper_boundary.size(); ++i)
        {
            std::cout<<"upper boundary: "<<i<<" "<<upper_boundary(i)<<std::endl;
        }

        std::vector<int> break_points_slave;
        std::vector<int> break_points_master;

        if(true)
        {   
            if(TEST_CASE == 0)
            {
                solver.slave_nodes = {136,139,143,149,157,162,166};
                if(!USE_IMLS)
                {
                    std::reverse(solver.slave_nodes.begin(),solver.slave_nodes.end());
                }
                for(int j=6; j<133; j+=7)
                {
                    if(j == 62) solver.master_nodes.push_back(61);
                    else solver.master_nodes.push_back(j);
                }
                if(USE_IMLS)
                {
                    std::reverse(solver.master_nodes.begin(),solver.master_nodes.end());
                }
            }
            //else if(TEST_CASE == 1 || TEST_CASE == 2)
            else if(TEST_CASE == 1 || TEST_CASE == 2)
            {
                // solver.slave_nodes = {213,222,232,243,254,265,276,287,298,309,318};
                // solver.master_nodes = {41,51,62,73,84,95,106,117,128,137,146};
                // solver.slave_nodes = {44,48,54,62,67};
                // solver.master_nodes = {8,13,21,27,31};
                // solver.slave_nodes = {105,110,117,122,125,130,134,140,149};
                // solver.master_nodes = {22,31,37,41,46,49,54,61,66};
                solver.slave_nodes.clear();
                solver.master_nodes.clear();
                solver.force_nodes.clear();
                // int start_2 = 30;
                // int end_2 = 8;
                // int start_1 = 158;
                // int end_1 = 175;

                int start_2 = 37;
                int end_2 = 29;
                int start_1 = 643;
                int end_1 = 632;

                int start_force = 603;
                int end_force = 663;

                if(TEST_CASE == 1)
                {
                    start_2 = 95;
                    end_2 = 14;
                    start_1 = 783;
                    end_1 = 698;

                    start_force = 689;
                    end_force = 791;
                }
                else if(TEST_CASE == 5)
                {
                    start_2 = 95;
                    end_2 = 14;
                    start_1 = 1098;
                    end_1 = 1099;

                    start_force = 1575;
                    end_force = 1608;
                }

                int start = -1;
                for(int i=0; i<upper_boundary.size(); ++i)
                {
                    if(upper_boundary[i] == start_1|| upper_boundary[i] == end_1)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                        solver.slave_nodes.push_back(upper_boundary(i));
                }
                // if(!USE_IMLS)
                // {
                //     std::reverse(solver.slave_nodes.begin(),solver.slave_nodes.end());
                // }

                start = -1;
                for(int i=0; i<upper_boundary.size(); ++i)
                {
                    if(upper_boundary[i] == start_force|| upper_boundary[i] == end_force)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                        solver.force_nodes.push_back(upper_boundary(i));
                }
                std::reverse(solver.force_nodes.begin(),solver.force_nodes.end());

                start = -1;
                for(int i=0; i<lower_boundary.size(); ++i)
                {
                    if(lower_boundary[i] == start_2|| lower_boundary[i] == end_2)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                        solver.master_nodes.push_back(lower_boundary(i));
                }
                // if(!USE_IMLS)
                // {
                //     std::reverse(solver.master_nodes.begin(),solver.master_nodes.end());
                // }
                

                for(int i=0; i<solver.slave_nodes.size(); ++i)
                {
                    std::cout<<solver.slave_nodes[i]<<" ";
                }
                std::cout<<std::endl;
                for(int i=0; i<solver.master_nodes.size(); ++i)
                {
                    std::cout<<solver.master_nodes[i]<<" ";
                }
                std::cout<<std::endl;
                std::cout<<"Force Nodes"<<std::endl;
                for(int i=0; i<solver.force_nodes.size(); ++i)
                {
                    std::cout<<solver.force_nodes[i]<<" ";
                }
                std::cout<<std::endl;
                // solver.slave_nodes = {171,173,170,172,169,151,166,168,165,167,164};
                // solver.master_nodes = {4,17,15,18,16,1,19,22,20,23,21};
                // solver.slave_nodes = {169,151,166};
                // solver.master_nodes = {16,1,19};
            }
            else if(TEST_CASE == 2)
            {
                solver.slave_nodes = {281,288,297,302,308,315,324,331,339,352,360,367,377,383,390,398,404};
                solver.master_nodes = {52,61,75,83,88,97,109,119,127,139,144,153,160,169,179,191,194};
                // if(!USE_IMLS)
                // {
                //     std::reverse(solver.slave_nodes.begin(),solver.slave_nodes.end());
                // }
                if(USE_IMLS)
                {
                    std::reverse(solver.master_nodes.begin(),solver.master_nodes.end());
                    std::reverse(solver.slave_nodes.begin(),solver.slave_nodes.end());
                    
                }
            }
            else if(TEST_CASE == 3 || TEST_CASE == 5)
            {
                Eigen::VectorXd master_boundary_1;
                Eigen::VectorXd master_boundary_2;
                Eigen::VectorXd slave_boundary;

                if(PULL_TEST)
                {
                    igl::boundary_loop(FS.block(0,0,(num_lower_f),3),master_boundary_1);
                    igl::boundary_loop(FS.block(0,0,(num_lower_f),3),master_boundary_2);
                    igl::boundary_loop(FS.block((num_lower_f),0,(num_upper_f),3),slave_boundary);
                }else
                {
                    igl::boundary_loop(FS.block((num_lower_f),0,(num_lower_f),3),master_boundary_2);
                    igl::boundary_loop(FS.block(2*(num_lower_f),0,2*(num_upper_f),3),slave_boundary);
                    igl::boundary_loop(FS.block(0,0,(num_lower_f),3),master_boundary_1);
                }

                solver.use_multiple_pairs = true;
                solver.multiple_slave_nodes.clear();
                solver.multiple_master_nodes.clear();
                solver.multiple_slave_nodes.resize(2);
                solver.multiple_master_nodes.resize(2);
                int a1 = 155, a2 = 140, a3 = 6, a4 = 29;
                if(PULL_TEST)
                {
                    // a2 = 755; a1 = 1115;
                    // a4 = 812; a3 = 1114;
                    a2 = 684; a1 = 384;
                    a3 = 396; a4 = 678;

                    a2 = 2385; a1 = 1294;
                    a4 = 2370; a3 = 1311;

                    // a2 = 189; a1 = 80;
                    // a4 = 185; a3 = 73;

                    // a2 = 663; a1 = 349;
                    // a4 = 658; a3 = 348;
                }

                int start = -1;
                for(int i=0; i<master_boundary_2.size(); ++i)
                {
                    //std::cout<<master_boundary_2[i]<<std::endl;
                    if(!PULL_TEST)
                        solver.multiple_master_nodes[1].push_back(master_boundary_1(i));
                    if(master_boundary_2[i] == a1 || master_boundary_2[i] == a2)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                    {
                        solver.master_nodes.push_back(master_boundary_2(i));
                        if(PULL_TEST)
                        {
                            solver.multiple_master_nodes[1].push_back(master_boundary_2(i));
                        }
                        
                    }   
                        
                }
                if(!USE_IMLS)
                {
                    std::reverse(solver.master_nodes.begin(),solver.master_nodes.end());
                }
                if(solver.master_nodes.size()>0)
                    break_points_master.push_back(solver.master_nodes.back());
                
                int size_1 = solver.master_nodes.size();
                start = -1;

                for(int i=0; i<master_boundary_1.size(); ++i)
                {
                    if(!PULL_TEST)
                        solver.multiple_master_nodes[0].push_back(master_boundary_1(i));
                    if(master_boundary_1[i] == a3 || master_boundary_1[i] == a4)
                    {
                        start*=-1;
                    }
                    if(start == 1){
                        solver.master_nodes.push_back(master_boundary_1(i));
                        if(PULL_TEST)
                        {
                            solver.multiple_master_nodes[0].push_back(master_boundary_2(i));
                        }
                    }
                        
                }
                if(!USE_IMLS)
                {
                    std::reverse(solver.master_nodes.begin()+size_1,solver.master_nodes.end());
                    std::reverse(solver.multiple_master_nodes[1].begin(),solver.multiple_master_nodes[1].end());
                }

                // Compute slave nodes and segments
                int b1 = 283, b2 = 260, b3 = 300, b4 = 311;
                if(PULL_TEST)
                {   //
                    b1 = 700; b2 = 1112;
                    b3 = 1097; b4 = 699;

                    b4 = 3538; b3 = 2683;
                    b2 = 2800; b1 = 3546;

                    // b2 = 308; b1 = 210;
                    // b4 = 204; b3 = 309;

                    // b4 = 1022; b3 = 713;
                    // b2 = 739; b1 = 1027;
                    
                }
                
                start = -1;
                for(int i=0; i<slave_boundary.size(); ++i)
                {
                    if(slave_boundary[i] == b1 || slave_boundary[i] == b2)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                    {
                        solver.slave_nodes.push_back(slave_boundary(i));
                        solver.multiple_slave_nodes[0].push_back(slave_boundary(i));
                    }
                        
                }
                // if(USE_IMLS)
                // {
                //     std::reverse(solver.slave_nodes.begin(),solver.slave_nodes.end());
                //     std::reverse(solver.multiple_slave_nodes[0].begin(),solver.multiple_slave_nodes[0].end());
                // }
                if(solver.slave_nodes.size()>0)
                    break_points_master.push_back(solver.slave_nodes.back());
                //break_points_master.push_back(solver.slave_nodes.back());
                
                int size_2 = solver.slave_nodes.size();

                start = -1;
                for(int i=0; i<slave_boundary.size(); ++i)
                {
                    if(slave_boundary[i] == b3 || slave_boundary[i] == b4)
                    {
                        start*=-1;
                    }
                    if(start == 1)
                    {
                        solver.slave_nodes.push_back(slave_boundary(i));
                        solver.multiple_slave_nodes[1].push_back(slave_boundary(i));
                    }
                        
                }
                // if(USE_IMLS)
                // {
                //     std::reverse(solver.slave_nodes.begin()+size_2,solver.slave_nodes.end());
                //     std::reverse(solver.multiple_slave_nodes[1].begin(),solver.multiple_slave_nodes[1].end());
                // }

                for(int i=0; i<solver.multiple_master_nodes[1].size(); ++i)
                {
                    std::cout<<solver.multiple_master_nodes[1][i]<<" ";
                }
                std::cout<<std::endl;

                for(int i=0; i<solver.multiple_master_nodes[0].size(); ++i)
                {
                    std::cout<<solver.multiple_master_nodes[0][i]<<" ";
                }
                std::cout<<std::endl;
                
                for(int i=0; i<solver.multiple_slave_nodes[1].size(); ++i)
                {
                    std::cout<<solver.multiple_slave_nodes[1][i]<<" ";
                }
                std::cout<<std::endl;

                for(int i=0; i<solver.multiple_slave_nodes[0].size(); ++i)
                {
                    std::cout<<solver.multiple_slave_nodes[0][i]<<" ";
                }
                std::cout<<std::endl;

                // for(int i=0; i<master_boundary_1.size(); ++i)
                // {
                //     std::cout<<"master_boundary_1: "<<i<<" "<<master_boundary_1(i)<<std::endl;
                // }
                // for(int i=0; i<master_boundary_2.size(); ++i)
                // {
                //     std::cout<<"master_boundary_2: "<<i<<" "<<master_boundary_2(i)<<std::endl;
                // }

                // for(int i=0; i<slave_boundary.size(); ++i)
                // {
                //     std::cout<<"slave_boundary: "<<i<<" "<<slave_boundary(i)<<std::endl;
                // }
            }
            
            if(solver.slave_nodes.size()>1)
            {
                for(int i=0; i<solver.slave_nodes.size()-1; ++i)
                {
                    int i1 = solver.slave_nodes[i];
                    int i2 = solver.slave_nodes[(i+1)%solver.slave_nodes.size()];
                    for(int j=0; j<break_points_slave.size(); j++)
                        if(solver.slave_nodes[i] == break_points_slave[j]) continue;
                    solver.slave_segments.push_back(std::pair<int,int>(i2,i1));
                }
            }
            
            if(solver.master_nodes.size()>1)
            {
                for(int i=0; i<solver.master_nodes.size()-1; ++i)
                {
                    int i1 = solver.master_nodes[i];
                    int i2 = solver.master_nodes[(i+1)%solver.master_nodes.size()];
                    for(int j=0; j<break_points_master.size(); j++)
                        if(solver.master_nodes[i] == break_points_master[j]) continue;
                    solver.master_segments.push_back(std::pair<int,int>(i2,i1));
                }
            }
        }
        else
        {
            std::vector<std::vector<int>> boundaries;
            igl::boundary_loop(FS,boundaries);
            solver.slave_nodes = boundaries[1];
            solver.master_nodes = boundaries[0];
            for(int i=0; i<solver.slave_nodes.size(); ++i)
            {
                int i1 = solver.slave_nodes[i];
                int i2 = solver.slave_nodes[(i+1)%solver.slave_nodes.size()];
                solver.slave_segments.push_back(std::pair<int,int>(i1,i2));
            }

            for(int i=0; i<solver.master_nodes.size(); ++i)
            {
                int i1 = solver.master_nodes[i];
                int i2 = solver.master_nodes[(i+1)%solver.master_nodes.size()];
                solver.master_segments.push_back(std::pair<int,int>(i1,i2));
            }
        }

        // for(int i=0; i<solver.slave_nodes.size(); ++i)
        // {
        //     std::cout<<solver.slave_nodes[i]<<std::endl;
        // }

        // if(success)
        // {
        //     // Construct Non Quad faces
        //     int num_faces = FS_Quad.rows();
        //     FS.resize(2*num_faces,3);
        //     for(int i=0; i<num_faces; ++i)
        //     {
        //         FS(2*i,0) = FS_Quad(i,0);
        //         FS(2*i,1) = FS_Quad(i,1);
        //         FS(2*i,2) = FS_Quad(i,2);

        //         FS(2*i+1,0) = FS_Quad(i,2);
        //         FS(2*i+1,1) = FS_Quad(i,3);
        //         FS(2*i+1,2) = FS_Quad(i,0);
        //     }
        // }
    }
    
    solver.V_all = VS;
    solver.F_all = FS;
    solver.F_all_Quad = FS_Quad;

    solver.initializeElementData(VS, FS, FS,FS_Quad, FS_Quad);
}