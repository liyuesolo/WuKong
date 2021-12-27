#include "../include/FEMSolver.h"


void FEMSolver::initializeElementData(const Eigen::MatrixXd& TV, 
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
    });

    num_surface_faces = TF.rows();
    surface_indices.resize(num_surface_faces * 3);
    tbb::parallel_for(0, num_surface_faces,  [&](int i)
    {
        surface_indices.segment<3>(i * 3) = Face(TF(i, 1), TF(i, 0), TF(i, 2));
    });

    computeBoundingBox();
    center = 0.5 * (max_corner + min_corner);
}

void FEMSolver::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
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

void FEMSolver::computeBoundingBox()
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

void FEMSolver::appendCylinder(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const TV& _center, const TV& direction, T R)
{
    T visual_R = R;
    int n_div = 10;
    T theta = 2.0 * EIGEN_PI / T(n_div);
    VectorXT points = VectorXT::Zero(n_div * 3);
    for(int i = 0; i < n_div; i++)
        points.segment<3>(i * 3) = TV(visual_R * std::cos(theta * T(i)), 
        0.0, visual_R*std::sin(theta*T(i)));
    
    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    V.conservativeResize(n_row_V + rod_offset_v, 3);
    F.conservativeResize(n_row_F + rod_offset_f, 3);
    C.conservativeResize(n_row_F + rod_offset_f, 3);

    TV vtx_from = _center - direction * 2.0 * R;
    TV vtx_to = _center + direction * 2.0 * R;

    TV axis_world = vtx_to - vtx_from;
    TV axis_local(0, axis_world.norm(), 0);

    Matrix<T, 3, 3> rotation_matrix = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
    
    for(int i = 0; i < n_div; i++)
    {
        for(int d = 0; d < 3; d++)
        {
            V(n_row_V + i, d) = points[i * 3 + d];
            V(n_row_V + i+n_div, d) = points[i * 3 + d];
            if (d == 1)
                V(n_row_V + i+n_div, d) += axis_world.norm();
        }

        // central vertex of the top and bottom face
        V.row(n_row_V + i) = (V.row(n_row_V + i) * rotation_matrix).transpose() + vtx_from;
        V.row(n_row_V + i + n_div) = (V.row(n_row_V + i + n_div) * rotation_matrix).transpose() + vtx_from;

        F.row(n_row_F + i*2 ) = IV(n_row_V + i, n_row_V + i+n_div, n_row_V + (i+1)%(n_div));
        F.row(n_row_F + i*2 + 1) = IV(n_row_V + (i+1)%(n_div), n_row_V + i+n_div, n_row_V + (i+1)%(n_div) + n_div);

        C.row(n_row_F + i*2 ) = TV(1.0, 0.0, 0.0);
        C.row(n_row_F + i*2 + 1) = TV(1.0, 0.0, 0.0);
    }
}