#include <igl/copyleft/tetgen/tetrahedralize.h>
#include "../include/FEMSolver.h"

void FEMSolver::imposeCylindricalBending()
{   
    T curvature = 0.2;
    
    T theta = M_PI * 0.5;

    TV K1_dir(std::cos(theta), std::sin(theta), 0.0);

    TV K2_dir = K1_dir.cross(TV(0, 0, 1)).normalized();

    T radius = 1.0 / curvature;

    TV cylinder_center = center - TV(0, 0, radius);
    
    iterateDirichletVertices([&](const TV& vtx, int idx)
    {
        TV d = vtx - center;
        T distance_along_cylinder_dir = d.dot(K1_dir);
        T distance_along_unwrapped_plane = d.dot(K2_dir);
        // unwrap cylinder to xy plane
        T arc_central_angle = distance_along_unwrapped_plane / radius;

        TV pt_projected = cylinder_center + distance_along_cylinder_dir * K1_dir + 
            radius * (std::sin(arc_central_angle) * K2_dir + std::cos(arc_central_angle) * TV(0, 0, 1));
                
        for (int d = 0; d < dim; d++)
            dirichlet_data[idx * dim + d] = (pt_projected[d] - vtx[d]);
    });
}

void FEMSolver::computeCylindricalBendingBCPenaltyPairs()
{
    
    TV K1_dir(std::cos(bending_direction), std::sin(bending_direction), 0.0);

    TV K2_dir = K1_dir.cross(TV(0, 0, 1)).normalized();

    T radius = 1.0 / curvature;

    TV cylinder_center = center - TV(0, 0, radius);

    iterateDirichletVertices([&](const TV& vtx, int idx)
    {
        TV d = vtx - center;
        T distance_along_cylinder_dir = d.dot(K1_dir);
        T distance_along_unwrapped_plane = d.dot(K2_dir);
        // unwrap cylinder to xy plane
        T arc_central_angle = distance_along_unwrapped_plane / radius;

        TV pt_projected = cylinder_center + distance_along_cylinder_dir * K1_dir + 
            radius * (std::sin(arc_central_angle) * K2_dir + std::cos(arc_central_angle) * TV(0, 0, 1));
                
        for (int d = 0; d < dim; d++)
            penalty_pairs.push_back(std::make_pair(idx * dim + d, pt_projected[d]));
    });
    use_penalty = true;
}

void FEMSolver::addBackSurfaceToDirichletVertices()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[2] < min_corner[2] + 1e-6)
            dirichlet_vertices.push_back(i);
    }
}

void FEMSolver::addBackSurfaceBoundaryToDirichletVertices()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        bool back_face = x[2] < min_corner[2] + 1e-6;
        bool h_border = x[0] < min_corner[0] + 1e-6 || x[0] > max_corner[0] - 1e-6;
        bool v_border = x[1] < min_corner[1] + 1e-6 || x[1] > max_corner[1] - 1e-6;
        if (back_face && (h_border || v_border))
        // if (back_face)
            dirichlet_vertices.push_back(i);
    }
}

void FEMSolver::fixEndPointsX()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[0] < min_corner[0] + 1e-6 || x[0] > max_corner[0] - 1e-6)
        {
            for (int d = 0; d < dim; d++)
            {
                dirichlet_data[i * dim + d] = 0.0;
            }
        }
    }
}

void FEMSolver::dragMiddle()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if ((x[0] > center[0] - 0.1 && x[0] < center[0] + 0.1) 
            && (x[2] < min_corner[2] + 1e-6))
        {
            dirichlet_data[i * dim + 2] = -1;
            // f[i * dim + 2] = -100.0;
        }
    }
}

void FEMSolver::applyForceTopBottom()
{
    T dy = max_corner[1] - min_corner[1];

    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[1] > max_corner[1] - 0.1 * dy)
            f[i * dim + 1] = -0.1;
        else if (x[1] < min_corner[1] + dy * 0.1)
            // f[i * dim + 1] = 1.0;
            dirichlet_data[i * dim + 1] = 0.0;
    }
}

void FEMSolver::applyForceLeftRight()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[0] > max_corner[0] - 1e-6)
            f[i * dim + 0] = -0.5;
        else if (x[0] < min_corner[0] + 1e-6)
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = 0.0;
        }
    }
}

void FEMSolver::ThreePointBendingTest()
{
    for (int i = 0; i < num_nodes; i++)
    {
        // TV x = undeformed.segment<3>(i * dim);
        // bool x_border = x[0] < min_corner[0] + 1e-6 || x[0] > max_corner[0] - 1e-6;
        // bool y_bottom = x[1] < min_corner[1] + 1e-6;
        // if (x_border && y_bottom)
        // {
        //     for (int d = 0; d < dim; d++)
        //         dirichlet_data[i * dim + d] = 0.0;
        // }
        // if (x[0] > center[0] - 1e-3 && x[0] < center[0] + 1e-3 && x[1] > max_corner[1] - 1e-6)
        //     f[i * dim + 1] = -0.01;

        TV x = undeformed.segment<3>(i * dim);
        bool x_border = x[0] < min_corner[0] + 1e-3 || x[0] > max_corner[0] - 1e-3;
        bool y_bottom = x[1] < min_corner[1] + 1e-3;
        if (x_border && y_bottom)
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = 0.0;
        }
        if (x[0] > center[0] - 1e-2 && x[0] < center[0] + 1e-2 && x[1] > max_corner[1] - 1e-1)
            f[i * dim + 1] = -0.1;
    }
    // cylinder_tet_start = num_ele;
    // cylinder_face_start = num_surface_faces;
    // cylinder_vtx_start = num_nodes;
    std::cout << "force norm: " << f.norm() << std::endl;
}

void FEMSolver::fixNodes(const std::vector<int>& node_indices)
{
    
}

void FEMSolver::ThreePointBendingTestWithCylinder()
{
    

    three_point_bending_with_cylinder = true;


    
    T dx = max_corner[1] - min_corner[1];
    T radius = 0.3 * dx;
    MatrixXd cylinder_color;
    T epsilon = 1e-4;

    MatrixXd solid_cylinder_v;
    MatrixXi solid_cylinder_f;
    appendCylinderMesh(solid_cylinder_v, solid_cylinder_f, 
        TV(center[0], max_corner[1] + radius + epsilon, center[2]), TV(0, 0, 1), radius, 1.2 * (max_corner[2] - min_corner[2]), 30, 10);
    Eigen::MatrixXd tet_v;
    Eigen::MatrixXi tet_f;
    Eigen::MatrixXi tet_ele;
    igl::copyleft::tetgen::tetrahedralize(solid_cylinder_v, solid_cylinder_f, "pq1.414Y", tet_v, tet_ele, tet_f);
    
    int nv_sc = tet_v.rows(), nf_sc = tet_f.rows(), nele_sc = tet_ele.rows();
    int nv_curr = num_nodes;
    cylinder_tet_start = num_ele;
    cylinder_vtx_start = num_nodes;
    cylinder_face_start = num_surface_faces;

    num_nodes += nv_sc;
    num_ele += nele_sc;
    num_surface_faces += nf_sc;

    undeformed.conservativeResize(num_nodes * dim);
    surface_indices.conservativeResize(num_surface_faces * 3);
    indices.conservativeResize(num_ele * 4);

    tbb::parallel_for(0, nv_sc, [&](int i){
        undeformed.segment<3>((num_nodes - nv_sc + i) * dim) = tet_v.row(i);
    });
    tbb::parallel_for(0, nf_sc, [&](int i){
        for (int j = 0; j < 3; j++)
        {
            surface_indices[(num_surface_faces - nf_sc + i) * 3 + j] = tet_f(i, 2 - j) + nv_curr;
        }

    });
    tbb::parallel_for(0, nele_sc, [&](int i){
        
        for (int j = 0; j < 4; j++)
        {
            indices[(num_ele - nele_sc + i) * 4 + j] = tet_ele(i, j) + nv_curr;
        }
    });
    deformed = undeformed;
    u = deformed; f = deformed;
    u.setZero(); f.setZero();
    computeIPCRestData();
    
    appendCylinderMesh(cylinder_vertices, cylinder_faces, 
        TV(min_corner[0] + 0.3 * dx, min_corner[1] - radius - epsilon, center[2]), TV(0, 0, 1), radius, 1.2 * (max_corner[2] - min_corner[2]), 20, 10);

    appendCylinderMesh(cylinder_vertices, cylinder_faces, 
        TV(max_corner[0] - 0.3 * dx, min_corner[1] - radius - epsilon, center[2]), TV(0, 0, 1), radius, 1.2 * (max_corner[2] - min_corner[2]), 20, 10);

    std::vector<Edge> edges;
    for (int i = 0; i < cylinder_faces.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int k = (j + 1) % 3;
            Edge ei(cylinder_faces(i, j), cylinder_faces(i, k));
            auto find_iter = std::find_if(edges.begin(), edges.end(), 
                [&ei](const Edge e)->bool {return (ei[0] == e[0] && ei[1] == e[1] ) 
                    || (ei[0] == e[1] && ei[1] == e[0]); });
            if (find_iter == edges.end())
            {
                edges.push_back(ei);
            }
        }
    }
    int num_edges_curr = ipc_edges.rows();
    int nv = ipc_vertices.rows();
    ipc_edges.conservativeResize(num_edges_curr + edges.size(), 2);
    for (int i = 0; i < edges.size(); i++)
        ipc_edges.row(i + num_edges_curr) = Edge(edges[i][0] + nv, edges[i][1] + nv);  
    
    appendMesh(ipc_vertices, ipc_faces, cylinder_vertices, cylinder_faces);
    // std::cout << ipc_faces.row(num_surface_faces) << std::endl;
    // std::cout << num_nodes << std::endl;
    num_ipc_vtx = ipc_vertices.rows();
    
    for (int i = cylinder_vtx_start; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        // if (x[0] > center[0] - 1e-1 && x[0] < center[0] + 1e-1 && x[1] > max_corner[1] - 1e-2)
            f[i * dim + 1] = -10;
    }
    
    std::cout << (max_corner - min_corner).transpose() << std::endl;
    std::cout << "total external force: " << f.norm() << std::endl;
    std::cout << "#ele in the structure " << cylinder_tet_start << " #ele in the cylinder " << num_ele - cylinder_tet_start << std::endl;
}