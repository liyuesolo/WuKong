#include "../include/vtk_io.h"
#include <fstream>

void WriteVTK(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<std::string>& attr_names, std::vector<VTK_ATTRIBUTE_TYPE>& attr_types, std::vector<Eigen::VectorXd>& attr_values, std::vector<std::vector<std::pair<int, int>>>& paths)
{
    std::ofstream writer(filename);
    writer << "# vtk DataFile Version 3.0" << std::endl;
    writer << "vtk output" << std::endl;
    writer << "ASCII" << std::endl;
    writer << "DATASET UNSTRUCTURED_GRID" << std::endl;
    writer << "POINTS " << V.rows() << " float" << std::endl;

    for(int i=0; i<V.rows(); ++i)
        writer << V(i,0) << " " << V(i,1) << " " << V(i,2) << std::endl;

    int total_edges = 0;
    for(int i=0; i<paths.size(); ++i)
        total_edges += paths[i].size();

    writer << "CELLS " << F.rows() +total_edges << " " << F.rows()*4 + total_edges*3 << std::endl;

    for(int i=0; i<F.rows(); ++i)
        writer << "3 " << F(i,0) << " " << F(i,1) << " " << F(i,2) << std::endl;

    //writer << "LINES " << total_edges << " " << total_edges*3 << std::endl;
    for(int i=0; i<paths.size(); ++i)
        for(int j=0; j<paths[i].size(); ++j)
            writer << "2 " << paths[i][j].first << " " << paths[i][j].second << std::endl;

    writer << "CELL_TYPES " << F.rows() +total_edges << std::endl;
    for(int i=0; i<F.rows(); ++i)
        writer << "5" << std::endl;
    for(int i=0; i<total_edges; ++i)
        writer << "3" << std::endl;

    int n_va = 0;
    int n_ca = 0;

    int n_vv = 0;
    int n_cv = 0;

    for(int i=0; i<attr_names.size(); ++i)
    {
        if(attr_types[i]==VERTEX_SCALAR)
            ++n_va;
        else if(attr_types[i]==FACE_SCALAR)
            ++n_ca;
        else if(attr_types[i]==VERTEX_VECTOR)
            ++n_vv;
        else if(attr_types[i]==FACE_VECTOR)
            ++n_cv;
    }

    if(n_va>0)
    {
        writer << "POINT_DATA " << V.rows() << std::endl;
        for(int i=0; i<attr_names.size(); ++i)
        {
            if(attr_types[i]==VERTEX_SCALAR)
            {
                writer << "SCALARS " << attr_names[i] << " float 1" << std::endl;
                writer << "LOOKUP_TABLE default" << std::endl;

                for(int j=0; j<attr_values[i].rows(); ++j)
                    writer << attr_values[i][j] << std::endl;
            }
        }
    }

    if(n_vv>0)
    {
        if(n_va==0)
            writer << "POINT_DATA " << V.rows()  << std::endl;
        for(int i=0; i<attr_names.size(); ++i)
        {
            if(attr_types[i]==VERTEX_VECTOR)
            {
                writer << "VECTORS " << attr_names[i] << " float" << std::endl;

                for(int j=0; j<attr_values[i].rows()/3; ++j)
                    writer << attr_values[i].segment(j*3,3).transpose() << std::endl;
            }
        }
    }

    if(n_ca > 0)
    {
        writer << "CELL_DATA " << F.rows() + total_edges << std::endl;
        for(int i=0; i<attr_names.size(); ++i)
        {
            if(attr_types[i]==FACE_SCALAR)
            {
                writer << "SCALARS " << attr_names[i] << " float 1" << std::endl;
                writer << "LOOKUP_TABLE default" << std::endl;

                for(int j=0; j<attr_values[i].rows(); ++j)
                    writer << attr_values[i][j] << std::endl;
            }
        }
    }

    if(n_cv>0)
    {
        if(n_ca==0)
            writer << "CELL_DATA " << F.rows() + total_edges << std::endl;
        for(int i=0; i<attr_names.size(); ++i)
        {
            if(attr_types[i]==FACE_VECTOR)
            {
                writer << "VECTORS " << attr_names[i] << " float" << std::endl;

                for(int j=0; j<attr_values[i].rows()/3; ++j)
                    writer << attr_values[i].segment(j*3,3).transpose() << std::endl;
            }
        }
    }

    writer.close();
}