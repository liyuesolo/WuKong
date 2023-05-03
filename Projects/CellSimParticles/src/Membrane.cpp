#include <igl/per_vertex_normals.h>
#include <igl/edges.h>
#include <igl/readOBJ.h>
#include "../include/CellSim.h"

template <int dim>
void CellSim<dim>::checkMembranePenetration()
{
    for (int i = 0; i < target_trajectories.cols(); i++)
    {
        target_positions = target_trajectories.col(i);
        int n_pts = target_positions.rows() / dim;
        VectorXT invalid = VectorXT::Zero(n_pts);
        tbb::parallel_for(0, n_pts, [&](int j){
            TV xi = target_positions.segment<dim>(j * dim);
            if (!sdf.inside(xi))
                invalid[j] = 1.0;
        });

        if (invalid.sum() > 1e-6)
        {
            std::cout << "penetration" << std::endl; 
            break;
        }
    }
    std::cout << "No penetration from data points" << std::endl;
}

template <int dim>
void CellSim<dim>::constructMembraneLevelset()
{
    std::string filename = "/home/yueli/Documents/ETH/WuKong/Projects/CellSimParticles/data/yolk3D.obj";
    
    igl::readOBJ(filename, membrane_vtx, membrane_faces);

    membrane_vtx *= 1.2;
    
    MatrixXT vtx_normals;
    igl::per_vertex_normals(membrane_vtx, membrane_faces, vtx_normals);
    VectorXT vtx_normals_vector(vtx_normals.rows() * dim);
    VectorXT vtx_vector(membrane_vtx.rows() * dim);
    for (int i = 0; i < vtx_normals.rows(); i++)
    {
        vtx_normals_vector.segment<dim>(i*dim) = vtx_normals.row(i);
        vtx_vector.segment<dim>(i*dim) = membrane_vtx.row(i);
    }
    
    MatrixXi membrane_edges;
    igl::edges(membrane_faces, membrane_edges);
    T ref_spacing = 0.0;
    for (int i = 0; i < membrane_edges.rows(); i++)
    {
        ref_spacing +=(membrane_vtx.row(membrane_edges(i, 0)) - membrane_vtx.row(membrane_edges(i, 1))).norm();
    }
    ref_spacing /= T(membrane_edges.rows());
    sdf.setRefDis(ref_spacing);
    sdf.initializedMeshData(
        vtx_vector, 
        VectorXi(),
        vtx_normals_vector, 0.0);
    int cnt = 0;
    std::ofstream out("outside.obj");
    for (int i = 0; i < num_nodes; i++)
    {
        TV vi = deformed.segment<dim>(i * dim);
        if (!sdf.inside(vi))
        {
            // std::cout << i << "/" << num_cells << " " << sdf.value(vi) << std::endl;
            // out << "v " << vi.transpose() << std::endl;
            cnt++;
        }
    }
    out.close();
    std::cout << cnt << "/" << num_nodes << " are outside the sdf " << std::endl;
}

template <int dim>
void CellSim<dim>::addMembraneEnergy(T& energy)
{
    VectorXT energies(num_nodes);
    energies.setZero();
    iterateNodeParallel([&](int i)
    {
        TV xi = deformed.segment<dim>(i * dim);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        energies[i] = bound_coeff * std::pow(sdf.value(xi), 3);
    });
    energy += energies.sum();
}

template <int dim>
void CellSim<dim>::addMembraneForceEntries(VectorXT& residual)
{
    iterateNodeParallel([&](int i)
    {
        TV xi = deformed.segment<dim>(i * dim);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        Vector<T, dim> dedx;
        sdf.gradient(xi, dedx);
        T value = sdf.value(xi);
        addForceEntry<dim>(residual, {i}, -3.0 * bound_coeff * std::pow(value, 2) * dedx);
    });
}

template <int dim>
void CellSim<dim>::addMembraneHessianEntries(std::vector<Entry>& entries, bool projectPD)
{
    std::vector<TM> sub_hessian(num_nodes, TM::Zero());
    iterateNodeParallel([&](int i)
    {
        TV xi = deformed.segment<dim>(i * dim);
        if (sdf.inside(xi) && !run_diff_test)
            return;
        TM d2phidx2;
        sdf.hessian(xi, d2phidx2);
        TV dphidx;
        sdf.gradient(xi, dphidx);
        T value = sdf.value(xi);
        TM hessian = bound_coeff * (6.0 * value * dphidx * dphidx.transpose() + 3.0 * value * value * d2phidx2);
        
        if (projectPD) 
            projectBlockPD<dim>(hessian);
        sub_hessian[i] = hessian;
    });
    for (int i = 0; i < num_nodes; i++)
    {
        addHessianEntry<dim>(entries, {i}, sub_hessian[i]);    
    }
}

template class CellSim<2>;
template class CellSim<3>;