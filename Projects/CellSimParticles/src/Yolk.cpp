#include <igl/readOBJ.h>
#include <igl/edges.h>
#include "../include/CellSim.h"
#include "../autodiff/EdgeEnergy.h"
#include "../autodiff/AreaEnergy.h"
#include "../autodiff/VolumeEnergy.h"

template <int dim>
void CellSim<dim>::constructYolkMesh3D()
{
    if constexpr (dim == 3)
    {
        std::string filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSimParticles/data/yolk3D.obj";
        MatrixXT yolk_obj_vertices; MatrixXi yolk_obj_faces;
        igl::readOBJ(filename, yolk_obj_vertices, yolk_obj_faces);

        membrane_vtx = yolk_obj_vertices * 1.1;
        membrane_faces = yolk_obj_faces;

        yolk_obj_vertices *= 0.96;
        int n_pt_yolk = yolk_obj_vertices.rows();
        std::cout<< n_pt_yolk << std::endl;
        undeformed.conservativeResize(num_cells * dim + n_pt_yolk * dim);
        for (int i = 0; i < n_pt_yolk; i++)
            undeformed.segment<dim>(num_cells * dim + i * dim) = yolk_obj_vertices.row(i);
        num_nodes = num_cells + n_pt_yolk;
        yolk_cell_starts = num_cells;
        rest_edge_length.resize(n_pt_yolk);
        MatrixXi offset(yolk_obj_faces.rows(), 3); offset.setConstant(num_cells);
        yolk_obj_faces += offset;
        ipc_faces = yolk_obj_faces;
        yolk_faces = ipc_faces;
        igl::edges(yolk_obj_faces, ipc_edges);
        int n_edges_yolk = ipc_edges.rows();
        yolk_edges.resize(n_edges_yolk);
        rest_edge_length.resize(n_edges_yolk);
        
        tbb::parallel_for(0, n_edges_yolk, [&](int i)
        {
            Edge e = ipc_edges.row(i);
            yolk_edges[i] = e;
            TV xi = undeformed.segment<dim>(e[0] * dim);
            TV xj = undeformed.segment<dim>(e[1] * dim);
            rest_edge_length[i] = (xi - xj).norm();
        });
        ipc_vertices.resize(num_nodes, 3);
        for (int i = 0; i < num_nodes; i++)
        {
            ipc_vertices.row(i) = undeformed.segment<dim>(i * dim);
        }
        
    }
}

template <int dim>
void CellSim<dim>::constructYolkMesh2D()
{
    if constexpr (dim == 2)
    {
        int n_pt_yolk = 100;
        T dtheta = 2.0 * M_PI / T(n_pt_yolk);
        yolk_cell_starts = num_cells;
        undeformed.conservativeResize(num_cells * dim + n_pt_yolk * dim);
        for (int i = 0; i < n_pt_yolk; i++)
        {
            undeformed[num_cells * dim + i * dim + 0] = centroid[0] + (0.5 - 1. * radius) * std::cos(T(i) * dtheta);
            undeformed[num_cells * dim + i * dim + 1] = centroid[1] + (0.5 - 1. * radius) * std::sin(T(i) * dtheta);
        }
        num_nodes = num_cells + n_pt_yolk;
        yolk_edges.resize(n_pt_yolk);
        rest_edge_length.resize(n_pt_yolk);
        tbb::parallel_for(yolk_cell_starts, num_nodes, [&](int i)
        {
            int j;
            if (i == num_nodes - 1)j = yolk_cell_starts;
            else j = i + 1;
            yolk_edges[i - yolk_cell_starts] = Edge(i, j);
            TV xi = undeformed.segment<dim>(i * 2);
            TV xj = undeformed.segment<dim>(j * 2);
            rest_edge_length[i - yolk_cell_starts] = (xi - xj).norm();
        });
    }
    
}

template <int dim>
T CellSim<dim>::computeYolkArea()
{
    T total_area = 0.0;
    if constexpr (dim == 2)
    {
        for (int i = yolk_cell_starts; i < num_nodes; i++)
        {
            int j = 0;
            if (i == num_nodes - 1)
                j = yolk_cell_starts;
            else
                j = i + 1;

            TV vi = deformed.segment<dim>(i * dim);
            TV vj = deformed.segment<dim>(j * dim);

            T area;
            computeSignedTriangleArea(vi, vj, centroid, area);
            total_area += area;
        }
    }
    else
    {
        for (int i = 0; i < yolk_faces.rows(); i++)
        {
            TV vi, vj, vk;
            vi = deformed.segment<dim>(yolk_faces(i, 0) * dim);
            vj = deformed.segment<dim>(yolk_faces(i, 2) * dim);
            vk = deformed.segment<dim>(yolk_faces(i, 1) * dim);
            T volume;
            computeTetVolume(vi, vj, vk, centroid, volume);
            total_area += volume;
        }
        
    }
    return total_area;
}

template <int dim>
void CellSim<dim>::addYolkPreservationEnergy(T& energy)
{
    T total_area = computeYolkArea();
    energy += 0.5 * w_yolk * (total_area - yolk_area_rest) * (total_area - yolk_area_rest);
}

template <int dim>
void CellSim<dim>::addYolkPreservationForceEntries(VectorXT& residual)
{
    T total_area = computeYolkArea();
    T de_dsum = w_yolk * (total_area - yolk_area_rest);
    if constexpr (dim == 2)
    {    
        for (int i = yolk_cell_starts; i < num_nodes; i++)
        {
            int j = 0;
            if (i == num_nodes - 1)
                j = yolk_cell_starts;
            else
                j = i + 1;

            TV vi = deformed.segment<dim>(i * dim);
            TV vj = deformed.segment<dim>(j * dim);

            Vector<T, 4> dedx;
            computeSignedTriangleAreaGradient(vi, vj, centroid, dedx);
            addForceEntry<4>(residual, {i, j}, -de_dsum * dedx);
        }
    }
    else
    {
        for (int i = 0; i < yolk_faces.rows(); i++)
        {
            TV vi, vj, vk;
            vi = deformed.segment<dim>(yolk_faces(i, 0) * dim);
            vj = deformed.segment<dim>(yolk_faces(i, 2) * dim);
            vk = deformed.segment<dim>(yolk_faces(i, 1) * dim);
            Vector<T, 9> dedx;
            computeTetVolumeGradient(vi, vj, vk, centroid, dedx);
            addForceEntry<9>(residual, {yolk_faces(i, 0), 
                                        yolk_faces(i, 2),
                                        yolk_faces(i, 1)}, -de_dsum * dedx);
        }
    }
}

template <int dim>
void CellSim<dim>::addYolkPreservationHessianEntries(std::vector<Entry>& entries,
    MatrixXT& WoodBuryMatrix)
{
    T total_area = computeYolkArea();
    T de_dsum = w_yolk * (total_area - yolk_area_rest);
    VectorXT dAdx_full = VectorXT::Zero(deformed.rows());

    if constexpr (dim == 2)
    {
        
        for (int i = yolk_cell_starts; i < num_nodes; i++)
        {
            int j = 0;
            if (i == num_nodes - 1)
                j = yolk_cell_starts;
            else
                j = i + 1;

            TV vi = deformed.segment<dim>(i * dim);
            TV vj = deformed.segment<dim>(j * dim);

            Vector<T, 4> dedx;
            computeSignedTriangleAreaGradient(vi, vj, centroid, dedx);
            
            addForceEntry<4>(dAdx_full, {i, j}, dedx);

            Matrix<T, 4, 4> d2edx2;
            computeSignedTriangleAreaHessian(vi, vj, centroid, d2edx2);

            Matrix<T, 4, 4> hessian = de_dsum * d2edx2;
            addHessianEntry(entries, {i, j}, hessian);
        }
    }
    else
    {
        for (int i = 0; i < yolk_faces.rows(); i++)
        {
            TV vi, vj, vk;
            vi = deformed.segment<dim>(yolk_faces(i, 0) * dim);
            vj = deformed.segment<dim>(yolk_faces(i, 2) * dim);
            vk = deformed.segment<dim>(yolk_faces(i, 1) * dim);
            Vector<T, 9> dedx;
            computeTetVolumeGradient(vi, vj, vk, centroid, dedx);
            addForceEntry<9>(dAdx_full, {yolk_faces(i, 0), 
                                        yolk_faces(i, 2),
                                        yolk_faces(i, 1)}, dedx);

            Matrix<T, 9, 9> d2edx2;
            computeTetVolumeHessian(vi, vj, vk, centroid, d2edx2);

            Matrix<T, 9, 9> hessian = de_dsum * d2edx2;
            addHessianEntry(entries, {yolk_faces(i, 0), 
                                        yolk_faces(i, 2),
                                        yolk_faces(i, 1)}, hessian);

        }
    }

    if (woodbury)
    {
        dAdx_full *= std::sqrt(w_yolk);
        if (!run_diff_test)
        {
            iterateDirichletDoF([&](int offset, T target)
            {
                dAdx_full[offset] = 0.0;
            });
        }
        int n_row = num_nodes * 3, n_col = WoodBuryMatrix.cols();
        WoodBuryMatrix.conservativeResize(n_row, n_col + 1);
        WoodBuryMatrix.col(n_col) = dAdx_full;
    }
    else
    {
        for (int i = yolk_cell_starts; i < num_nodes; i++)
        {
            for (int j = yolk_cell_starts; j < num_nodes; j++)
            {
                Vector<T, dim * 2> dAdx;
                getSubVector<dim * 2>(dAdx_full, {i, j}, dAdx);
                TV dAdxi = dAdx.template segment<dim>(0);
                TV dAdxj = dAdx.template segment<dim>(dim);
                TM hessian_partial = w_yolk * dAdxi * dAdxj.transpose();
                // if (hessian_partial.nonZeros() > 0)
                addHessianBlock<dim>(entries, {i, j}, hessian_partial);
            }    
        }
    }

}

template <int dim>
void CellSim<dim>::addYolkEdgeRegEnergy(T& energy)
{
    int cnt = 0;
    for(auto e : yolk_edges)
    {
        TV vi = deformed.segment<dim>(e[0] * dim);
        TV vj = deformed.segment<dim>(e[1] * dim);
        T ei;
        computeEdgeSpringEnergy<dim>(vi, vj, rest_edge_length[cnt++], ei);
        energy += w_reg_edge * ei;
    }   
}

template <int dim>
void CellSim<dim>::addYolkEdgeRegForceEntries(VectorXT& residual)
{
    int cnt = 0;
    for(auto e : yolk_edges)
    {
        TV vi = deformed.segment<dim>(e[0] * dim);
        TV vj = deformed.segment<dim>(e[1] * dim);
        Vector<T, dim * 2> dedx;
        computeEdgeSpringEnergyGradient<dim>(vi, vj, rest_edge_length[cnt++], dedx);
        addForceEntry<dim * 2>(residual, {e[0], e[1]}, -w_reg_edge * dedx);
        
    }  
}

template <int dim>
void CellSim<dim>::addYolkEdgeRegHessianEntries(std::vector<Entry>& entries)
{
    int cnt = 0;
    for(auto e : yolk_edges)
    {
        TV vi = deformed.segment<dim>(e[0] * dim);
        TV vj = deformed.segment<dim>(e[1] * dim);
        Matrix<T, dim * 2, dim * 2> hessian;
        computeEdgeSpringEnergyHessian<dim>(vi, vj, rest_edge_length[cnt++], hessian);
        addHessianEntry<dim * 2>(entries, {e[0], e[1]}, w_reg_edge * hessian);
    }  
}

template <int dim>
void CellSim<dim>::addYolkCollisionEnergy(T& energy)
{

}

template <int dim>
void CellSim<dim>::addYolkCollisionForceEntries(VectorXT& residual)
{

}

template <int dim>
void CellSim<dim>::addYolkCollisionHessianEntries(std::vector<Entry>& entries)
{

}

template class CellSim<2>;
template class CellSim<3>;