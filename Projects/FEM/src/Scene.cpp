#include <igl/readOBJ.h>
#include "../include/FEMSolver.h"
#include <fstream>

template <int dim>
void FEMSolver<dim>::initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    num_nodes = V.rows();
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = V.row(i);
    });
    deformed = undeformed;
    u = VectorXT::Zero(num_nodes * dim);
    f = VectorXT::Zero(num_nodes * dim);

    num_ele = 0;
    
    num_surface_faces = F.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(F(i, 0), F(i, 1), F(i, 2));
    });
}
template <int dim>
void FEMSolver<dim>::initializeElementData(Eigen::MatrixXd& TV, 
    const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT)
{
    num_nodes = TV.rows();
    
    undeformed.resize(num_nodes * dim);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.segment<3>(i * dim) = TV.row(i);
    });
    deformed = undeformed;    
    u = VectorXT::Zero(num_nodes * dim);
    f = VectorXT::Zero(num_nodes * dim);

    num_ele = TT.rows();
    indices.resize(num_ele * 4);
    tbb::parallel_for(0, num_ele, [&](int i)
    {
        indices.segment<4>(i * 4) = TT.row(i);
        // indices[i * 4 + 0] = TT(i, 1);
        // indices[i * 4 + 1] = TT(i, 2);
        // indices[i * 4 + 2] = TT(i, 3);
        // indices[i * 4 + 3] = TT(i, 0);
    });

    num_surface_faces = TF.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(TF(i, 1), TF(i, 0), TF(i, 2));
    });

    computeBoundingBox();
    center = 0.5 * (max_corner + min_corner);
    // std::cout << max_corner.transpose() << " " << min_corner.transpose() << std::endl;
    // std::getchar();

    // use_ipc = true;
    if (use_ipc)
    {
        add_friction = false;
        barrier_distance = 1e-3;
        barrier_weight = 1e10;
        computeIPCRestData();
    }

    // E = 1e4;
    E = 2.6 * 1e5;
    nu = 0.48;
    
    penalty_weight = 1e8;
    use_penalty = false;
}

template <int dim>
void FEMSolver<dim>::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    V.resize(num_nodes, 3);
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V.row(i) = deformed.segment<3>(i * dim);
    });

    F.resize(num_surface_faces, 3);
    C.resize(num_surface_faces, 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        C.row(i) = TV(0, 0.3, 1.0);
    });

}

template <int dim>
void FEMSolver<dim>::computeBoundingBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], deformed[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], deformed[i * 3 + d]);
        }
    }
}


// template class FEMSolver<2>;
template class FEMSolver<3>;