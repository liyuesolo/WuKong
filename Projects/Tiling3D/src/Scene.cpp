#include <igl/readOBJ.h>
#include "../include/FEMSolver.h"
#include <fstream>

void FEMSolver::initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
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

void FEMSolver::initializeElementData(Eigen::MatrixXd& TV, 
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

    use_ipc = true;
    if (use_ipc)
    {
        add_friction = false;
        barrier_distance = 1e-3;
        barrier_weight = 1e6;
        computeIPCRestData();
    }

    // E = 1e4;
    E = 2.6 * 1e7;
    E_steel = 2 * 10e11;
    nu = 0.48;
    
    penalty_weight = 1e2;
    use_penalty = false;
    bending_direction = 90.0 / 180.0 * M_PI;
    curvature = 1;
    verbose = true;
    // max_newton_iter = 10000;
    project_block_PD = false;

    cylinder_tet_start = num_ele;
    cylinder_face_start = num_surface_faces;
    cylinder_vtx_start = num_nodes;
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

    if (three_point_bending_with_cylinder)
    {
        MatrixXd cylinder_color(cylinder_faces.rows(), 3);
        cylinder_color.col(0).setConstant(0); cylinder_color.col(1).setConstant(1); cylinder_color.col(2).setConstant(0);
        appendMesh(V, F, C, cylinder_vertices, cylinder_faces, cylinder_color);
    }
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

void FEMSolver::appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, T scale, const TV& center)
{
    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere.obj", v_sphere, f_sphere);
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);

    v_sphere = v_sphere * scale;

    tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
        v_sphere.row(row_idx) += center;
    });

    int n_vtx_prev = V.rows();
    int n_face_prev = F.rows();

    tbb::parallel_for(0, (int)f_sphere.rows(), [&](int row_idx){
        f_sphere.row(row_idx) += Eigen::Vector3i(n_vtx_prev, n_vtx_prev, n_vtx_prev);
    });

    V.conservativeResize(V.rows() + v_sphere.rows(), 3);
    F.conservativeResize(F.rows() + f_sphere.rows(), 3);

    V.block(n_vtx_prev, 0, v_sphere.rows(), 3) = v_sphere;
    F.block(n_face_prev, 0, f_sphere.rows(), 3) = f_sphere;
}

void FEMSolver::appendCylinder(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const TV& _center, const TV& direction, T R, T length)
{
    T visual_R = R;
    int n_div = 30;
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

    TV vtx_from = _center - direction * 0.5 * length;
    TV vtx_to = _center + direction * 0.5 * length;

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

void FEMSolver::appendCylinderMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
    const TV& _center, const TV& direction, T R, T length, int sub_div_R, int sub_div_L)
{
    // std::ofstream out("cylinder_closed.obj");
    T theta = 2.0 * EIGEN_PI / T(sub_div_R);
    VectorXT points = VectorXT::Zero(sub_div_R * 3);
    for(int i = 0; i < sub_div_R; i++)
        points.segment<3>(i * 3) = TV(R * std::cos(theta * T(i)), 
        0.0, R * std::sin(theta*T(i)));
    
    int offset_v = sub_div_R * (1 + sub_div_L) + 2;
    int offset_f = sub_div_R * sub_div_L * 2 + sub_div_R * 2;

    int n_row_V = V.rows();
    int n_row_F = F.rows();

    V.conservativeResize(n_row_V + offset_v, 3);
    F.conservativeResize(n_row_F + offset_f, 3);

    TV vtx_from = _center - direction * 0.5 * length;
    TV vtx_to = _center + direction * 0.5 * length;

    TV axis_world = vtx_to - vtx_from;
    TV axis_local(0, axis_world.norm(), 0);

    Matrix<T, 3, 3> rotation_matrix = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
    
    for (int j = 0; j < sub_div_L + 1; j++)
    {
        for(int i = 0; i < sub_div_R; i++)
        {
            for(int d = 0; d < dim; d++)
            {
                V(n_row_V + i + sub_div_R * j, d) = points[i * 3 + d];
                if (d == 1)
                    V(n_row_V + i + sub_div_R * j, d) += axis_world.norm() / T(sub_div_L) * j;
            }
            V.row(n_row_V + j * sub_div_R + i) = (V.row(n_row_V + j * sub_div_R + i) * rotation_matrix).transpose() + vtx_from;
            if (j < sub_div_L)
            {
                F.row(n_row_F + j * sub_div_R * 2 + i*2 ) = 
                    IV(n_row_V + i + sub_div_R * j, 
                        n_row_V + i + sub_div_R * (j + 1), 
                        n_row_V + (i+1)%(sub_div_R) + sub_div_R * j);
                F.row(n_row_F + j * sub_div_R * 2 + i*2 + 1) = 
                    IV(n_row_V + (i+1)%(sub_div_R) + sub_div_R * j,
                     n_row_V + i+ sub_div_R * (j + 1), 
                     n_row_V + (i+1)%(sub_div_R) + sub_div_R * (j + 1));
            }
        }
    }
    V.row(n_row_V + offset_v - 2) = vtx_from;
    V.row(n_row_V + offset_v - 1) = vtx_to;
    for(int i = 0; i < sub_div_R; i++)
    {
        F.row(n_row_F + offset_f - 2 * sub_div_R + i) = IV(n_row_V + offset_v - 2, n_row_V + i + sub_div_L * R, n_row_V + (i + 1) % sub_div_R);
        F.row(n_row_F + offset_f - 1 * sub_div_R + i) = 
            IV(n_row_V + offset_v - 1, 
                n_row_V + (i + 1) % sub_div_R + sub_div_L * sub_div_R,
                n_row_V + i + sub_div_L * sub_div_R); 
    }

    // for (int i = 0; i < V.rows(); i++)
    //     out << "v " << V.row(i) << std::endl;
    // for (int i =0 ;i < F.rows(); i++)
    //     out << "f " << F.row(i) + IV::Ones().transpose() << std::endl;
    // out.close();
    // std::exit(0);
}