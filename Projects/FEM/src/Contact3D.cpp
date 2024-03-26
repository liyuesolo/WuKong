#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/doublearea.h>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readMSH.h>
#include <igl/readMESH.h>
#include <igl/writeOBJ.h>
#include <igl/triangle/triangulate.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <igl/massmatrix.h>
#include <igl/adjacency_list.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/remove_unreferenced.h>
#include "../include/Contact3D.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"


#include <random>
#include <cmath>
#include <fstream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

std::random_device rd3D;
std::mt19937 gen3D( rd3D() );
std::uniform_real_distribution<> dis3D( 0.0, 1.0 );
std::uniform_real_distribution<> dis23D( -1.0, 1.0 );
// float DISPLAYSMENT;


static double zeta3D()
{
	return dis3D(gen3D);
}

static double zeta23D()
{
	return dis23D(gen3D);
}

void Contact3D::initializeSimulationDataFromSTL()
{
    bool load_mesh = true;
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    Eigen::MatrixXi TT;
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


    solver.slave_nodes_3d.clear();
    solver.slave_surfaces_3d.clear();
    solver.slave_nodes_area_3d.clear();
    solver.master_nodes_3d.clear();
    solver.master_surfaces_3d.clear();
    solver.master_nodes_area_3d.clear();
    solver.force_nodes_3d.clear();
    solver.force_surfaces_3d.clear();
    solver.force_nodes_area_3d.clear();
    solver.slave_surfaces_global_index.clear();
    solver.master_surfaces_global_index.clear();

    solver.boundary_info_start_3d.clear();

    if (load_mesh)
    {
        Eigen::MatrixXd V_upper, V_lower, SV;
        Eigen::MatrixXi F_upper, F_lower;
        Eigen::MatrixXi T_upper, T_lower;
        Eigen::MatrixXi SF, SVI, SVJ;
        Eigen::MatrixXi ST;
        Eigen::VectorXi TriTag, TetTag;

        std::string filename_upper;
        std::string filename_lower;

        if(TEST_CASE == 0 || TEST_CASE == 1)
        {
            filename_upper = "3d_mesh.msh";
            filename_lower = "3d_mesh.msh";

            if(TEST_CASE == 1)
            {
                filename_upper = "3d_mesh2.msh";
                filename_lower = "3d_mesh2.msh";
            }

            igl::readMSH(filename_lower,SV,SF,ST,TriTag,TetTag);
            V_lower = SV;
            F_lower = SF;
            // std::cout<<"HHHHHHHH: "<<F_lower.rows()<<std::endl;
            // Eigen::MatrixXi Fa;
            // Eigen::VectorXi Ja;
            // Eigen::VectorXi Ka;
            // igl::boundary_facets(ST,Fa,Ja,Ka);
            // std::cout<<"HHHHHHHH: "<<Fa.rows()<<std::endl;
            T_lower = ST;

            std::cout<<TriTag.transpose()<<std::endl;

            std::unordered_map<int,int> hm;
            std::vector<Eigen::VectorXi> surface;

            solver.master_nodes_area_3d.resize(1);
            solver.slave_nodes_area_3d.resize(1);
            solver.force_nodes_area_3d.resize(1);
            solver.slave_surfaces_global_index.resize(1);
            solver.master_surfaces_global_index.resize(1);


            for(int i=0; i<SF.rows(); ++i)
            {
                if((TEST_CASE == 0 && TriTag(i) == 1) || (TEST_CASE == 1 && (TriTag(i) == 14 || TriTag(i) == 18|| TriTag(i) == 20 || TriTag(i) == 22)))
                {
                    for(int j=0; j<3; ++j)
                    {
                        int node_index = SF(i,j);
                        if(hm.find(node_index+SV.rows()) == hm.end())
                        {
                            hm[node_index+SV.rows()] = hm.size();
                        }
                    }
                    
                    surface.push_back(SF.row(i).array() += SV.rows());
                    solver.master_surfaces_global_index[0].push_back(i+SF.rows());
                }
            }

            solver.master_nodes_3d.push_back(hm);
            solver.master_surfaces_3d.push_back(surface);

            Eigen::MatrixXd V_temp(hm.size(),3);
            Eigen::MatrixXi F_temp(surface.size(),3);

            for(auto it = hm.begin(); it != hm.end(); ++it)
            {
                V_temp.row(it->second) = SV.row(it->first-SV.rows());
            }
            for(int i=0; i<surface.size(); ++i)
            {
                for(int j=0; j<3; ++j)
                    F_temp(i,j) = hm[surface[i](j)];
            }
            Eigen::SparseMatrix<double> mass;
            igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);
            for(int i=0; i<hm.size(); ++i)
            {
                solver.master_nodes_area_3d[0].push_back(mass.coeff(i,i));
            }



            igl::readMSH(filename_upper,SV,SF,ST,TriTag,TetTag);
            V_upper = SV;
            Eigen::VectorXd ones(3);
            ones<<DISPLAYSMENT,GAP+2,0;
            for(int i=0; i<V_upper.rows(); ++i)
            {
                V_upper(i,0) = SV(i,0);
                V_upper(i,1) = -SV(i,1);
                V_upper(i,2) = -SV(i,2);
                V_upper.row(i) += ones.transpose();
            }
            F_upper = SF;
            T_upper = ST;

            hm.clear();
            surface.clear();

            for(int i=0; i<SF.rows(); ++i)
            {
                if((TEST_CASE == 0 && TriTag(i) == 1) || (TEST_CASE == 1 && (TriTag(i) == 14 || TriTag(i) == 18|| TriTag(i) == 20 || TriTag(i) == 22)))
                {
                   for(int j=0; j<3; ++j)
                    {
                        int node_index = SF(i,j);
                        if(hm.find(node_index) == hm.end())
                        {
                            hm[node_index] = hm.size();
                        }
                    }
                    surface.push_back(SF.row(i));
                    solver.slave_surfaces_global_index[0].push_back(i);
                }
            }

            solver.slave_nodes_3d.push_back(hm);
            solver.slave_surfaces_3d.push_back(surface);

            V_temp.setZero(hm.size(),3);
            F_temp.setZero(surface.size(),3);

            for(auto it = hm.begin(); it != hm.end(); ++it)
            {
                V_temp.row(it->second) = SV.row(it->first);
            }
            for(int i=0; i<surface.size(); ++i)
            {
                for(int j=0; j<3; ++j)
                    F_temp(i,j) = hm[surface[i](j)];
            }

            igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);
            for(int i=0; i<hm.size(); ++i)
            {
                solver.slave_nodes_area_3d[0].push_back(mass.coeff(i,i));
            }

            hm.clear();
            surface.clear();

            for(int i=0; i<SF.rows(); ++i)
            {
                if((TEST_CASE == 0 && (TriTag(i) == 2 || TriTag(i) == 3)) || (TEST_CASE == 1 && TriTag(i) == 16))
                {
                    for(int j=0; j<3; ++j)
                    {
                        int node_index = SF(i,j);
                        if(hm.find(node_index) == hm.end())
                        {
                            hm[node_index] = hm.size();
                        }
                    }
                    surface.push_back(SF.row(i));
                }
            }

            solver.force_nodes_3d.push_back(hm);
            solver.force_surfaces_3d.push_back(surface);

            V_temp.setZero(hm.size(),3);
            F_temp.setZero(surface.size(),3);

            for(auto it = hm.begin(); it != hm.end(); ++it)
            {
                V_temp.row(it->second) = V_upper.row(it->first);
            }
            for(int i=0; i<surface.size(); ++i)
            {
                for(int j=0; j<3; ++j)
                    F_temp(i,j) = hm[surface[i](j)];
            }

            std::cout<<V_temp<<std::endl;
            std::cout<<"-----------------------------------------------"<<std::endl;
            std::cout<<F_temp<<std::endl;

            solver.force_nodes_area_3d[0].resize(hm.size());
            // igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_VORONOI,mass);
            // igl::writeOBJ("test.obj",V_temp,F_temp);
            // std::cout<<mass<<std::endl;
            for(auto it=hm.begin(); it!=hm.end(); ++it)
            {
                solver.force_nodes_area_3d[0][it->second] = mass.coeff(it->second,it->second);
            }
        }
        
        V.setZero((V_upper.rows()+V_lower.rows()),3);
        for(int i=0; i<V_upper.rows(); ++i)
        {
            V.row(i) = V_upper.row(i);
        }
        for(int i=0; i<V_lower.rows(); ++i)
        {
            V.row(i+V_upper.rows()) = V_lower.row(i);
        }

        
        F.setZero((F_upper.rows()+F_lower.rows()),3);
        for(int i=0; i<F_upper.rows(); ++i)
        {
            F.row(i) = F_upper.row(i);
        }
        for(int i=0; i<F_lower.rows(); ++i)
        {
            F.row(i+F_upper.rows()) = F_lower.row(i);
            F.row(i+F_upper.rows()).array() += V_upper.rows();
        }

        TT.setZero((T_upper.rows()+T_lower.rows()),4);
        for(int i=0; i<T_upper.rows(); ++i)
        {
            TT.row(i) = T_upper.row(i);
        }
        for(int i=0; i<T_lower.rows(); ++i)
        {
            TT.row(i+T_upper.rows()) = T_lower.row(i);
            TT.row(i+T_upper.rows()).array() += V_upper.rows();
        }

        std::vector<std::vector<int>> dump;
        igl::vertex_triangle_adjacency(V,F,solver.vertex_triangle_indices,dump);

        // for(int i=0; i<solver.vertex_triangle_indices[108].size(); ++i)
        // {
        //     std::cout<<F.row(solver.vertex_triangle_indices[108][i])<<std::endl;
        // }

        std::cout<<"force nodes: ";
        for(auto it = solver.force_nodes_3d[0].begin(); it!= solver.force_nodes_3d[0].end(); ++it)
        {
            std::cout<<it->first<<" ";
        }
        std::cout<<std::endl;
        
        igl::doublearea(V,F,solver.doublearea);
    }
    
    solver.V_all = VS;
    solver.F_all = FS;

    solver.initializeElementData(V, F, TT, FS_Quad,FS_Quad);
}

void Contact3D::initializeSimulationDataBunnyFunnel()
{
    bool load_mesh = true;
    Eigen::MatrixXd V, C;
    Eigen::MatrixXi F;
    Eigen::MatrixXi TT;
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


    solver.slave_nodes_3d.clear();
    solver.slave_surfaces_3d.clear();
    solver.slave_nodes_area_3d.clear();
    solver.master_nodes_3d.clear();
    solver.master_surfaces_3d.clear();
    solver.master_nodes_area_3d.clear();
    solver.force_nodes_3d.clear();
    solver.force_surfaces_3d.clear();
    solver.force_nodes_area_3d.clear();
    solver.slave_surfaces_global_index.clear();
    solver.master_surfaces_global_index.clear();

    solver.boundary_info_start_3d.clear();

    if (load_mesh)
    {
        Eigen::MatrixXd V_upper, V_lower, SV;
        Eigen::MatrixXi F_upper, F_lower;
        Eigen::MatrixXi T_upper, T_lower;
        Eigen::MatrixXi SF, SVI, SVJ;
        Eigen::MatrixXi ST;
        Eigen::VectorXi TriTag, TetTag;

        std::string filename_upper;
        std::string filename_lower;

        //filename_upper =  "bunny.obj";
        //filename_lower =  "torus3.obj";
        filename_upper =  "Armadillo13K.mesh";
        //filename_upper =  "sphere.mesh";
        //filename_lower =  "funnel_3_low.obj";
        filename_lower =  "square_32_32.obj";

        std::unordered_map<int,int> hm;
        std::vector<Eigen::VectorXi> surface;

        solver.master_nodes_area_3d.resize(1);
        solver.slave_nodes_area_3d.resize(1);
        solver.force_nodes_area_3d.resize(1);
        solver.slave_surfaces_global_index.resize(1);
        solver.master_surfaces_global_index.resize(1);

        igl::readOBJ(filename_lower,SV,SF);
       /*  for(int i=0; i<SF.rows(); ++i)
        {
            Eigen::Vector3i f = SF.row(i);
            SF(i,1) = f(2);
            SF(i,2) = f(1);
        } */
         /* for(int i=0; i<SV.rows(); ++i)
         {
             Eigen::Vector3d f = SV.row(i);
             SV(i,1) = f(2);
             SV(i,2) = f(1);
         } */

        //igl::readMESH(filename_lower,SV,ST,SF);
        // Eigen::Matrix3d rot;
        // rot<< 0.,-1.,0.,1.,0.,0.,0.,0.,1.;
        // for(int i=0; i<SV.rows(); ++i)
        // {
        //     Eigen::Vector3d row = SV.row(i);
        //     SV.row(i) = rot*row;
        // }
        Eigen::VectorXi SJ;
        Eigen::VectorXi SK;
        // igl::boundary_facets(ST,SF,SJ,SK);
        // igl::writeOBJ("dolphin.obj",SV,SF);
        // V_upper = SV/25.;
        // F_upper = SF;
        // T_upper = ST;

        V_upper = SV;
        F_upper = SF;
        //Tetrahedralize
        //igl::copyleft::tetgen::tetrahedralize(V_upper,F_upper, "pq1.414Y", V_upper,T_upper,F_upper);
        /* for(int i=0; i<F_upper.rows(); ++i)
        {
            Eigen::Vector3i f = F_upper.row(i);
            F_upper(i,1) = f(2);
            F_upper(i,2) = f(1);
        } */
        igl::adjacency_list(F_upper,solver.master_nodes_adjacency,true);
        for(int i=0; i<solver.master_nodes_adjacency.size(); ++i)
        {
            if(solver.master_nodes_adjacency[i].size()>10)
                std::cout<<i<<" "<<solver.master_nodes_adjacency[i].size()<<std::endl;
        }
        std::cout<<V_upper.rows()<<std::endl;
        std::cout<<F_upper.rows()<<std::endl;
        //igl::writeOBJ("test.obj",V_upper,F_upper);

        hm.clear();
        surface.clear();

        for(int i=0; i<SF.rows(); ++i)
        {
            bool count =true;
            if(true)
            {
                for(int j=0; j<3; ++j)
                {
                    int node_index = SF(i,j);
                    if(SV(node_index,1) > 0.3)
                    {
                        //count = true;
                        //continue;
                    }
                    if(hm.find(node_index) == hm.end())
                    {
                        hm[node_index] = hm.size();
                    }
                }
                if(count)
                {
                    surface.push_back(SF.row(i));
                    solver.master_surfaces_global_index[0].push_back(i);
                }
                
            }
        }

        solver.master_nodes_3d.push_back(hm);
        solver.master_surfaces_3d.push_back(surface);

        Eigen::MatrixXd V_temp(hm.size(),3);
        Eigen::MatrixXi F_temp(surface.size(),3);

        V_temp.setZero(hm.size(),3);
        F_temp.setZero(surface.size(),3);
        Eigen::SparseMatrix<double> mass;

        for(auto it = hm.begin(); it != hm.end(); ++it)
        {
            V_temp.row(it->second) = SV.row(it->first);
        }
        for(int i=0; i<surface.size(); ++i)
        {
            for(int j=0; j<3; ++j)
                F_temp(i,j) = hm[surface[i](j)];
        }

        igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);
        for(int i=0; i<hm.size(); ++i)
        {
            solver.master_nodes_area_3d[0].push_back(mass.coeff(i,i));
        }
        

        //igl::readMSH(filename_lower,SV,SF,ST,TriTag,TetTag);
        igl::readMESH(filename_upper,SV,ST,SF);
        Eigen::Matrix3d rot;
        rot<< 1.,0.,0.,0.,0.,-1.,0.,1.,0.;
        for(int i=0; i<SV.rows(); ++i)
        {
            Eigen::Vector3d row = SV.row(i);
            SV.row(i) = rot*row;
        }


        // Eigen::VectorXi SJ;
        // Eigen::VectorXi SK;
        igl::boundary_facets(ST,SF,SJ,SK);
        V_lower = SV/250.;
        F_lower = SF;
        T_lower = ST;

       // std::vector<std::vector<int>> adj_list;
        //igl::adjacency_list(SF,adj_list,true);
        // for(int i=0; i<adj_list.size(); ++i)
        // {
        //     if(adj_list[i].size()>6)
        //         std::cout<<i<<" "<<adj_list[i].size()<<std::endl;
        //     //std::reverse(solver.master_nodes_adjacency[i].begin(),solver.master_nodes_adjacency[i].end());
        // }

        //igl::readOBJ(filename_upper,SV,SF);
        /* for(int i=0; i<SF.rows(); ++i)
        {
            Eigen::Vector3i f = SF.row(i);
            SF(i,1) = f(2);
            SF(i,2) = f(1);
        } */
        // for(int i=0; i<SV.rows(); ++i)
        // {
        //     Eigen::Vector3d f = SV.row(i);
        //     SV(i,1) = f(2);
        //     SV(i,2) = f(1);
        // }

        //igl::readMESH(filename_lower,SV,ST,SF);
        /*  Eigen::Matrix3d rot;
         rot<< 0.,-1.,0.,1.,0.,0.,0.,0.,1.;
         for(int i=0; i<SV.rows(); ++i)
        {
            Eigen::Vector3d row = SV.row(i);
             SV.row(i) = rot*row;
        } */
        // igl::boundary_facets(ST,SF,SJ,SK);
        // igl::writeOBJ("dolphin.obj",SV,SF);
        // V_upper = SV/25.;
        // F_upper = SF;
        // T_upper = ST;
        //Tetrahedralize
        //igl::copyleft::tetgen::tetrahedralize(V_lower,F_lower, "pq1.414Y", V_lower,T_lower,F_lower);
       /*  for(int i=0; i<F_lower.rows(); ++i)
        {
            Eigen::Vector3i f = F_lower.row(i);
            F_lower(i,1) = f(2);
            F_lower(i,2) = f(1);
        } */
        igl::adjacency_list(F_upper,solver.slave_nodes_adjacency,true);

        Eigen::Vector3d disp(0.0,DISPLAYSMENT,0.0);
        for(int i=0; i<V_lower.rows(); ++i)
            V_lower.row(i) += disp;
        for(int i=0; i<F_lower.rows(); ++i)
        {
            Eigen::Vector3i temp = F_lower.row(i);
            F_lower(i,0) = temp(0);
            F_lower(i,1) = temp(2);
            F_lower(i,2) = temp(1);
        }

        solver.slave_nodes_shift = V_upper.rows();        
        igl::adjacency_list(F_lower,solver.slave_nodes_adjacency,true);

        for(int i=0; i<solver.slave_nodes_adjacency.size(); ++i)
        {
            if(solver.slave_nodes_adjacency[i].size()>11)
                std::cout<<i<<" "<<solver.slave_nodes_adjacency[i].size()<<std::endl;
            for(int j=0; j<solver.slave_nodes_adjacency[i].size(); ++j)
                solver.slave_nodes_adjacency[i][j] += V_upper.rows();
            //std::reverse(solver.master_nodes_adjacency[i].begin(),solver.master_nodes_adjacency[i].end());
        }
        
        hm.clear();
        surface.clear();

        // for(int i=0; i<SV.rows(); ++i)
        // {
        //     if(i%2 != 0) continue;
        //     int node_index = i;
        //     if(hm.find(node_index+V_upper.rows()) == hm.end())
        //     {
        //         hm[node_index+V_upper.rows()] = hm.size();
        //     }
        // }
        //int cnt = 0;
        std::unordered_map<int,int> hm2;
        int count_2 = 0;
        for(int i=0; i<SF.rows(); ++i)
        {
            if(true)
            {
                for(int j=0; j<3; ++j)
                {
                    int node_index = SF(i,j);
                    if(hm2.find(node_index+V_upper.rows()) == hm2.end())
                    {
                        //if(hm2.size()%2 == 0)
                            //hm[node_index+V_upper.rows()] = hm.size();
                        hm2[node_index+V_upper.rows()] = count_2+1;
                        count_2++;
                        //std::cout<<node_index+V_upper.rows()<<" "<<hm2.size()<<std::endl;
                    }
                }
                
                surface.push_back(SF.row(i).array() += V_upper.rows());
                solver.slave_surfaces_global_index[0].push_back(i+F_upper.rows());
            }
        }

        V_temp.setZero(hm2.size(),3);
        F_temp.setZero(surface.size(),3);

        solver.slave_nodes_3d.push_back(hm2);
        solver.slave_surfaces_3d.push_back(surface);

        // for(auto it = hm2.begin(); it != hm2.end(); ++it)
        // {
        //     V_temp.row(it->second-1) = SV.row(it->first-V_upper.rows());
        // }
        // for(int i=0; i<surface.size(); ++i)
        // {
        //     for(int j=0; j<3; ++j)
        //         F_temp(i,j) = hm2[surface[i](j)]-1;
        // }
        
        // igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_BARYCENTRIC,mass);
        // for(int i=0; i<hm2.size(); ++i)
        // {
        //     solver.slave_nodes_area_3d[0].push_back(mass.coeff(i,i));
        // }

        hm2.clear();
        surface.clear();

        // for(int i=0; i<SF.rows(); ++i)
        // {
        //     if((TEST_CASE == 0 && (TriTag(i) == 2 || TriTag(i) == 3)) || (TEST_CASE == 1 && TriTag(i) == 16))
        //     {
        //         for(int j=0; j<3; ++j)
        //         {
        //             int node_index = SF(i,j);
        //             if(hm.find(node_index) == hm.end())
        //             {
        //                 hm[node_index] = hm.size();
        //             }
        //         }
        //         surface.push_back(SF.row(i));
        //     }
        // }

        solver.force_nodes_3d.push_back(hm);
        solver.force_surfaces_3d.push_back(surface);

        V_temp.setZero(hm.size(),3);
        F_temp.setZero(surface.size(),3);

        for(auto it = hm.begin(); it != hm.end(); ++it)
        {
            V_temp.row(it->second) = V_upper.row(it->first);
        }
        for(int i=0; i<surface.size(); ++i)
        {
            for(int j=0; j<3; ++j)
                F_temp(i,j) = hm[surface[i](j)];
        }

        std::cout<<V_temp<<std::endl;
        std::cout<<"-----------------------------------------------"<<std::endl;
        std::cout<<F_temp<<std::endl;

        solver.force_nodes_area_3d[0].resize(hm.size());
        // igl::massmatrix(V_temp,F_temp,igl::MASSMATRIX_TYPE_VORONOI,mass);
        // igl::writeOBJ("test.obj",V_temp,F_temp);
        // std::cout<<mass<<std::endl;
        for(auto it=hm.begin(); it!=hm.end(); ++it)
        {
            solver.force_nodes_area_3d[0][it->second] = mass.coeff(it->second,it->second);
        }
        
        V.setZero((V_upper.rows()+V_lower.rows()),3);
        for(int i=0; i<V_upper.rows(); ++i)
        {
            V.row(i) = V_upper.row(i);
        }
        for(int i=0; i<V_lower.rows(); ++i)
        {
            V.row(i+V_upper.rows()) = V_lower.row(i);
        }

        
        F.setZero((F_upper.rows()+F_lower.rows()),3);
        for(int i=0; i<F_upper.rows(); ++i)
        {
            F.row(i) = F_upper.row(i);
        }
        for(int i=0; i<F_lower.rows(); ++i)
        {
            F.row(i+F_upper.rows()) = F_lower.row(i);
            F.row(i+F_upper.rows()).array() += V_upper.rows();
        }

        TT.setZero(T_upper.rows()+T_lower.rows(),4);
        for(int i=0; i<T_upper.rows(); ++i)
        {
            TT.row(i) = T_upper.row(i);
        }
        for(int i=0; i<T_lower.rows(); ++i)
        {
            TT.row(i+T_upper.rows()) = T_lower.row(i);
            TT.row(i+T_upper.rows()).array() += V_upper.rows();
        }

        std::vector<std::vector<int>> dump;
        igl::vertex_triangle_adjacency(V,F,solver.vertex_triangle_indices,dump);

        // for(int i=0; i<solver.vertex_triangle_indices[108].size(); ++i)
        // {
        //     std::cout<<F.row(solver.vertex_triangle_indices[108][i])<<std::endl;
        // }

        std::cout<<"force nodes: ";
        for(auto it = solver.force_nodes_3d[0].begin(); it!= solver.force_nodes_3d[0].end(); ++it)
        {
            std::cout<<it->first<<" ";
        }
        std::cout<<std::endl;
        
        igl::doublearea(V,F,solver.doublearea);

        //solver.shell_start_index = F_upper.rows();
         solver.faces = Eigen::VectorXi(3*F_upper.rows());

         for(int i=0;i<F_upper.rows(); ++i)
         {
             solver.faces.segment<3>(3*i) = F_upper.row(i);
        }
        
    }
    
    solver.V_all = VS;
    solver.F_all = FS;
    

    solver.initializeElementData(V, F, TT, FS_Quad,FS_Quad);
}

void Contact3D::initializeSimulationSelfContact()
{
    // Read Meshes, compute the adjancy nodes for normal computation
    // Compute the geodist close matrix
    Eigen::MatrixXi FS_Quad;

    std::vector<Eigen::Vector3d> nodes_data;
    std::vector<Eigen::Vector3i> faces_data;
    std::vector<Eigen::Vector3i> surfaces_data;
    std::vector<Eigen::Vector4i> tetrahedras_data;
    std::vector<Eigen::Triplet<double>> tripletList;
    solver.nodes_adjacency.clear();
    solver.is_surface_vertex.clear();

    Eigen::VectorXi SJ;
    Eigen::VectorXi SK;
    Eigen::MatrixXi SFF;

    int node_start = 0;
    int face_start = 0;
    for(int i=0; i<mesh_names.size(); ++i)
    {
        std::string file_name = mesh_names[i].first;
        int mesh_dim = mesh_names[i].second;
        Eigen::MatrixXd V,NV;
        Eigen::MatrixXi F,NF;
        Eigen::MatrixXi T;

        if(mesh_dim == 2)
        {
            igl::readOBJ(file_name,V,F);

            V/=2.;
            
            for(int j=0; j<F.rows(); ++j)
            {
                Eigen::Vector3i surface = F.row(j);
                surface(0) += node_start; surface(1) += node_start; surface(2) += node_start;
                surfaces_data.push_back(surface);
            }
            if(i == 0)
            {
                // Eigen::Vector3d disp(0.25, 0.0, 0.25);
                // for(int j=0; j<V.rows(); ++j)
                //     V.row(j) += disp;
                Eigen::Vector3d disp(0.0,0,0.0);
                for(int j=0; j<V.rows(); ++j)
                    V.row(j) += disp;
                
                 Eigen::Matrix3d rot;
                rot<< 1.,0.,0.,0.,0.,-1.,0.,1.,0.;
                for(int i=0; i<V.rows(); ++i)
                {
                    Eigen::Vector3d row = V.row(i);
                    V.row(i) = rot*row;
                }
                
                for(int j=0; j<F.rows(); ++j)
                {
                    Eigen::Vector3i temp = F.row(j);
                    F(j,0) = temp(0);
                    F(j,1) = temp(2);
                    F(j,2) = temp(1);
                }
            }

            if(i == 1)
            {
                igl::readOBJ(file_name,V,F);
                //igl::copyleft::tetgen::tetrahedralize(V,F, "pq1.414Y", V,T,F);
                //V/=10.;
                // Eigen::Vector3d disp(0.0,DISPLAYSMENT,0.0);
                // for(int j=0; j<V.rows(); ++j)
                //     V.row(j) += disp;
                Eigen::Vector3d disp(0.0,0.0,0.0);
                for(int j=0; j<V.rows(); ++j)
                    V.row(j) += disp;
                // for(int j=0; j<F.rows(); ++j)
                // {
                //     Eigen::Vector3i temp = F.row(j);
                //     F(j,0) = temp(0);
                //     F(j,1) = temp(2);
                //     F(j,2) = temp(1);
                // }
            }

            
            
        }
        else if(mesh_dim == 3)
        {
            Eigen::MatrixXd SV;
            Eigen::MatrixXi SF;
            Eigen::MatrixXi ST;
            //igl::readMESH(file_name,SV,ST,SF);
            // SV/=20.;
            // Eigen::Vector3d disp(0.5, 0.5, 0.005);
            // for(int j=0; j<SV.rows(); ++j)
            //      SV.row(j) += disp;
            //Eigen::Matrix3d rot;
            // rot<< 1.,0.,0.,0.,0.,-1.,0.,1.,0.;
            // for(int i=0; i<SV.rows(); ++i)
            // {
            //     Eigen::Vector3d row = SV.row(i);
            //     SV.row(i) = rot*row;
            // }

            
            // igl::boundary_facets(ST,SF,SJ,SK);
            // V = SV;
            // F = SF;
            // T = ST;

            igl::readOBJ(file_name,SV,SF);
            // SV/=2;
            igl::copyleft::tetgen::tetrahedralize(SV,SF, "pq1.414Y", V,T,F);
            V/=4.;
             Eigen::Vector3d disp(0, 0, -0.5);
             for(int j=0; j<V.rows(); ++j)
                 V.row(j) += disp;


            // if(i == 0)
            // {
            //     for(int j=0; j<V.rows(); ++j)
            //         V.row(j) += Eigen::Vector3d(0,-0.1,-0.1);
            // }
            // if(i == 1)
            // {
            //     for(int j=0; j<V.rows(); ++j)
            //         V.row(j) += Eigen::Vector3d(0,-0.1,0.1);
            // }
            // if(i == 2)
            // {
            //     for(int j=0; j<V.rows(); ++j)
            //         V.row(j) += Eigen::Vector3d(0,0.1,-0.1);
            // }
            // if(i == 3)
            // {
            //     for(int j=0; j<V.rows(); ++j)
            //         V.row(j) += Eigen::Vector3d(0,0.1,0.1);
            // }

            for(int j=0; j<F.rows(); ++j)
            {
                Eigen::Vector3i temp = F.row(j);
                F(j,0) = temp(0);
                F(j,1) = temp(2);
                F(j,2) = temp(1);
            }
            
        }
        std::cout<<"Num of nodes: "<<V.rows()<<std::endl;
        Eigen::VectorXi I,J;
        igl::remove_unreferenced(V,F,NV,NF,I,J);
        igl::writeOBJ("temp.obj",NV,NF);
        Eigen::MatrixXd N;
        igl::per_vertex_normals(NV,NF,N);
        std::unique_ptr<SurfaceMesh> mesh;
        std::unique_ptr<VertexPositionGeometry> geometry;
        std::tie(mesh, geometry) = readSurfaceMesh("temp.obj");

        HeatMethodDistanceSolver heatSolver(*geometry);

        for(int i=0; i<V.rows(); ++i)
        {
            solver.nodes_adjacency.push_back(std::vector<int>());
            solver.is_surface_vertex.push_back(0);
        }
            

        int num = 0;
        for(Vertex v:mesh->vertices())
        {
            // VertexData<double> distToSource = heatSolver.computeDistance(v);
            solver.is_surface_vertex[J(num)+node_start] = 1;
            // for(int j=0; j<distToSource.size(); ++j)
            // {
            //     if(distToSource[j] <= search_radius)
            //     {
            //         //std::cout<<v.getIndex()+node_start<<" "<<j+node_start<<std::endl;
            //         //if(mesh_dim == 2)
            //         tripletList.push_back(Eigen::Triplet<double>(J(v.getIndex())+node_start,J(j)+node_start,1));
            //         // else if(mesh_dim == 3)
            //         //     tripletList.push_back(Eigen::Triplet<double>(v.getIndex()+node_start,j+node_start,1));
            //     }
            //     // else if((V.row(J(num))-V.row(J(j))).norm() <= solver.radius && N.row(j).dot(V.row(J(num))-V.row(J(j))) <= 0)
            //     // {
            //     //     //std::cout<<v.getIndex()+node_start<<" "<<j+node_start<<std::endl;
            //     //     //if(mesh_dim == 2)
            //     //     tripletList.push_back(Eigen::Triplet<double>(J(v.getIndex())+node_start,J(j)+node_start,1));
            //     //     // else if(mesh_dim == 3)
            //     //     //     tripletList.push_back(Eigen::Triplet<double>(v.getIndex()+node_start,j+node_start,1));
            //     // }
            // }
            // if(num == 0){
            //     std::cout<<distToSource.size()<<std::endl;
            // } 
            num++;
        }
        std::cout<<num<<std::endl;

        // for(int i=0; i<V.rows(); ++i)
        // {
        //     if(fabs(V(i,0)-0)< 1e-5 || fabs(V(i,0)-2)< 1e-5)
        //     {
        //         solver.is_surface_vertex[i+node_start] = 0;
        //     } 
        // }
         


        solver.mesh_begin_vertices.push_back(node_start);
        std::vector<std::vector<int>> current_node_adjancy;
        if(mesh_dim == 2)
            igl::adjacency_list(F,current_node_adjancy,true);
        else if(mesh_dim == 3)
            igl::adjacency_list(NF,current_node_adjancy,true);
        //igl::adjacency_list(F,current_node_adjancy,true);
        for(int j=0; j<current_node_adjancy.size(); ++j)
        {
            for(int k=0; k<current_node_adjancy[j].size(); ++k)
            {
                solver.nodes_adjacency[J(j)+node_start].push_back(J(current_node_adjancy[j][k]) + node_start);
            }
            if(solver.nodes_adjacency[J(j)+node_start].size()>11) std::cout<<"Too Much!!!"<<std::endl;
        }
        for(int j=0; j<V.rows(); ++j)
        {
            nodes_data.push_back(V.row(j));
        }
        for(int j=0; j<F.rows(); ++j)
        {
            Eigen::Vector3i surface = F.row(j);
            surface(0) += node_start; surface(1) += node_start; surface(2) += node_start;
            faces_data.push_back(surface);
        }
        for(int j=0; j<T.rows(); ++j)
        {
            Eigen::Vector4i tet = T.row(j);
            tet(0) += node_start; tet(1) += node_start; tet(2) += node_start; tet(3) += node_start;
            tetrahedras_data.push_back(tet);
        }
        node_start += V.rows();
        face_start += F.rows();
    }


    // Assembly the close matrix and V,F,T
    solver.faces = Eigen::VectorXi(3*surfaces_data.size());

    for(int i=0;i<surfaces_data.size(); ++i)
    {
        solver.faces.segment<3>(3*i) = surfaces_data[i];
    }

    Eigen::MatrixXd V_all(nodes_data.size(),3);
    Eigen::MatrixXi F_all(faces_data.size(),3);
    Eigen::MatrixXi T_all(tetrahedras_data.size(),4);

    for(int i=0;i<nodes_data.size(); ++i)
    {
        V_all.row(i) = nodes_data[i];
    }
    for(int i=0;i<faces_data.size(); ++i)
    {
        F_all.row(i) = faces_data[i];
    }
    for(int i=0;i<tetrahedras_data.size(); ++i)
    {
        T_all.row(i) = tetrahedras_data[i];
    }
    solver.geodist_close_matrix.resize(V_all.rows(),V_all.rows());
    solver.geodist_close_matrix.setFromTriplets(tripletList.begin(),tripletList.end());
    solver.V_all = V_all;
    solver.F_all = F_all;

    std::cout<<"Number of tet: "<<tetrahedras_data.size()<<std::endl;
    

    solver.initializeElementData(V_all, F_all, T_all, FS_Quad,FS_Quad);
}