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

void FEMSolver::addBackSurfaceToDirichletVertices()
{
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[2] < min_corner[2] + 1e-6)
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
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * dim);
        if (x[1] > max_corner[1] - 1e-6)
            f[i * dim + 1] = -1.0;
        else if (x[1] < min_corner[1] + 1e-6)
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
            f[i * dim + 0] = -50.0;
        else if (x[0] < min_corner[0] + 1e-6)
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = 0.0;
        }
    }
}