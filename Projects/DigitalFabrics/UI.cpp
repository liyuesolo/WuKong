#include "UI.h"

void appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, double scale, Vector<double, 3> shift)
{
    Eigen::MatrixXd v_sphere;
    Eigen::MatrixXi f_sphere;

    // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere.obj", v_sphere, f_sphere);
    igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);

    v_sphere = v_sphere * scale;

    tbb::parallel_for(0, (int)v_sphere.rows(), [&](int row_idx){
        v_sphere.row(row_idx) += shift;
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

void removeSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    if(V.rows() > 0)
    {
        Eigen::MatrixXd v_sphere;
        Eigen::MatrixXi f_sphere;

        // igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere.obj", v_sphere, f_sphere);
        igl::readOBJ("/home/yueli/Documents/ETH/WuKong/Projects/DigitalFabrics/Data/sphere162.obj", v_sphere, f_sphere);

        int n_vtx_prev = V.rows();
        int n_face_prev = F.rows();

        V.conservativeResize(V.rows() - v_sphere.rows(), 3);
        F.conservativeResize(F.rows() - f_sphere.rows(), 3);
    }
}

void appendCylinderMesh(igl::opengl::glfw::Viewer& viewer,
    Eigen::MatrixXd& V, Eigen::MatrixXi& F,
    std::vector<Vector<double, 2>>& points_on_curve, 
    bool backward, int n_div)
{
    using T = double;
    using TV2 = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using IV3 = Vector<int, 3>;
    using TM3 = Matrix<double, 3, 3>;

    Eigen::Vector4f eye_n = (viewer.core().view).inverse().col(3);
    
    if (points_on_curve.size() < 2)
    {
        if(backward)
        {
            V.setZero();
            F.setZero();
            return;
        }
        else
            return;
    }
        
    

    int rod_offset_v = n_div * 2;
    int rod_offset_f = n_div * 2;

    int n_rods = points_on_curve.size() - 1;
    

    V.resize(n_rods * rod_offset_v, 3);
    V.setZero();
    F.resize(n_rods * rod_offset_f, 3);
    F.setZero();

    T theta = 2.0 * M_PI / T(n_div);

    T visual_R = 0.01;
    Eigen::MatrixXd points(n_div, 3);
    // bottom face vertices
    for(int i = 0; i < n_div; i++)
        points.row(i) = Eigen::Vector3d(visual_R * std::cos(theta * T(i)), 0.0, visual_R*std::sin(theta*T(i)));
    
    tbb::parallel_for(0, n_rods, [&](int rod_cnt){
        int rov = rod_cnt * rod_offset_v;
        int rof = rod_cnt * rod_offset_f;

        TV2 from2d = points_on_curve[rod_cnt];
        TV2 to2d = points_on_curve[rod_cnt+1];

        TV3 from3d, to3d;
        
        igl::unproject_on_plane(from2d, viewer.core().proj*viewer.core().view, 
            viewer.core().viewport, eye_n, from3d);
        igl::unproject_on_plane(to2d, viewer.core().proj*viewer.core().view, 
            viewer.core().viewport, eye_n, to3d);

        TV3 axis_world = to3d - from3d;
        TV3 axis_local(0, axis_world.norm(), 0);
        
        TM3 R = Eigen::Quaternion<T>().setFromTwoVectors(axis_world, axis_local).toRotationMatrix();
        for(int i = 0; i < n_div; i++)
        {
            for(int d = 0; d < 3; d++)
            {
                V(rov + i, d) = points(i, d);
                V(rov + i+n_div, d) = points(i, d);
                if (d == 1)
                    V(rov + i+n_div, d) += axis_world.norm();
            }

            V.row(rov + i) = (V.row(rov + i) * R).transpose() + from3d;
            V.row(rov + i + n_div) = (V.row(rov + i + n_div) * R).transpose() + from3d;

            F.row(rof + i*2 ) = IV3(rov + i, rov + i+n_div, rov + (i+1)%(n_div));
            F.row(rof + i*2 + 1) = IV3(rov + (i+1)%(n_div), rov + i+n_div, rov + (i+1)%(n_div) + n_div);
            
        }
    });
}