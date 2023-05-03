#include "../include/FEMSolver.h"
#include <gmsh.h>
void loadQuadraticTriangleMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, 
    Eigen::MatrixXi& F, Eigen::MatrixXi& V_quad)
{
    using TV3 = Vector<T, 3>;
    using IV3 = Vector<int, 3>;
    using IV6 = Vector<int, 6>;

    std::ifstream in(filename);
    std::string token;

	while(token != "POINTS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    
    int n_points;
	in >> n_points;

	in >> token; //double

	V.resize(n_points, 3);

	for(int i=0; i<n_points; i++)
        for (int  j = 0; j < 3; j++)
            in >> V(i, j);
        
    while(token != "CELLS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    int n_cells, n_entries;
	in >> n_cells;
	in >> n_entries;
    
    int cell_type;
	std::vector<IV3> faces;
    std::vector<IV6> quad_node_indices;
    // std::cout << n_cells << std::endl;
	for(int i=0; i<n_cells; ++i)
	{
		in >> cell_type;

		if(cell_type == 3)
		{
            IV3 face;
			for(int j = 0; j < 3; j++)
				in >> face[j];
            faces.push_back(face);
		}
        else if (cell_type == 6)
        {
            IV6 quad_nodes;
            for (int j = 0; j < 6; j++)
            {
                in >> quad_nodes[j];
            }
            quad_node_indices.push_back(quad_nodes);
        }
		else
		{
			// type 1 2
			for(int j = 0; j < cell_type; j++)
				in >> token;
		}
	}
    int n_faces = faces.size();
    F.resize(n_faces, 3);
    tbb::parallel_for(0, n_faces, [&](int i)
    {
        TV3 ei = V.row(faces[i][0]) - V.row(faces[i][1]);
        TV3 ej = V.row(faces[i][2]) - V.row(faces[i][1]);
        if (ej.cross(ei).dot(TV3(0, 0, 1)) < 0)
            F.row(i) = IV3(faces[i][1], faces[i][0], faces[i][2]);
        else
            F.row(i) = faces[i];
    });
    int n_quad_nodes = quad_node_indices.size();
    V_quad.resize(n_quad_nodes, 6);
    tbb::parallel_for(0, n_quad_nodes, [&](int i)
    {
        V_quad.row(i) = quad_node_indices[i];
    });
    in.close();
}

void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    using TV3 = Vector<T, 3>;
    using IV3 = Vector<int, 3>;

    

    std::ifstream in(filename);
    std::string token;

	while(token != "POINTS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    
    int n_points;
	in >> n_points;

	in >> token; //double

	V.resize(n_points, 3);

	for(int i=0; i<n_points; i++)
        for (int  j = 0; j < 3; j++)
            in >> V(i, j);
        
    while(token != "CELLS")
    {
		in >> token;
        if (in.eof())
            break;
    }
    int n_cells, n_entries;
	in >> n_cells;
	in >> n_entries;
    
    int cell_type;
	std::vector<Vector<int, 3>> faces;
	for(int i=0; i<n_cells; ++i)
	{
		in >> cell_type;

		if(cell_type == 3)
		{
            IV3 face;
			for(int j = 0; j < 3; j++)
				in >> face[j];
            faces.push_back(face);
		}
		else
		{
			// type 1 2
			for(int j = 0; j < cell_type; j++)
				in >> token;
		}
	}
    int n_faces = faces.size();
    F.resize(n_faces, 3);
    tbb::parallel_for(0, n_faces, [&](int i)
    {
        TV3 ei = V.row(faces[i][0]) - V.row(faces[i][1]);
        TV3 ej = V.row(faces[i][2]) - V.row(faces[i][1]);
        if (ej.cross(ei).dot(TV3(0, 0, 1)) < 0)
            F.row(i) = IV3(faces[i][1], faces[i][0], faces[i][2]);
        else
            F.row(i) = faces[i];
    });
    in.close();
}

template <int dim>
void FEMSolver<dim>::generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C)
{
    int num_nodes = deformed.rows() / dim;
    
    int num_ele = surface_indices.rows() / 3;
    V.resize(num_nodes, 3); V.setZero();
    F.resize(num_ele, 3); C.resize(num_ele, 3);
    
    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        V.row(i).head<dim>() = deformed.template segment<dim>(i * dim);
    });

    tbb::parallel_for(0, num_ele, [&](int i)
    {
        F.row(i) = surface_indices.segment<3>(i * 3);
        C.row(i) = TV3(0.0, 0.3, 1.0);
    });
}

template <int dim>
void FEMSolver<dim>::generateBeamScene(int resolution, const TV& min_corner, const TV& max_corner)
{
    E = 2.6 * 1e5;
    quadratic = false;
    mass_lumping = true;
    T eps = 1e-5;
    gmsh::initialize();

    gmsh::model::add("tiling");
    gmsh::logger::start();

    gmsh::option::setNumber("Geometry.ToleranceBoolean", eps);
    gmsh::option::setNumber("Geometry.Tolerance", eps);
    gmsh::option::setNumber("Mesh.ElementOrder", quadratic ? 2 : 1);

    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

    T bb_diag = (max_corner - min_corner).norm();

    if constexpr (dim == 2)
    {
        int acc = 1;
        std::vector<TV> corners = {min_corner, TV(max_corner[0], min_corner[1]), max_corner, TV(min_corner[0], max_corner[1])};
        for (int i = 0; i < 4; ++i)
            gmsh::model::occ::addPoint(corners[i][0], corners[i][1], 0, 2, acc++);

        //Lines
        acc = 1;
        int starting_vtx = 1;

        // add clipping box
        gmsh::model::occ::addLine(1, 2, acc++); 
        gmsh::model::occ::addLine(2, 3, acc++); 
        gmsh::model::occ::addLine(3, 4, acc++); 
        gmsh::model::occ::addLine(4, 1, acc++);

        starting_vtx = 5;

        acc = 1;
        int acc_loop = 1;

        gmsh::model::occ::addCurveLoop({1, 2, 3, 4}, acc++);
    

        gmsh::model::occ::addPlaneSurface({1});

        gmsh::model::occ::synchronize();

        gmsh::model::mesh::field::add("Distance", 1);

        gmsh::model::mesh::field::add("Threshold", 2);
        gmsh::model::mesh::field::setNumber(2, "InField", 1);
        if (resolution == 0)
        {
            gmsh::model::mesh::field::setNumber(2, "SizeMin", bb_diag * 0.05);
            gmsh::model::mesh::field::setNumber(2, "SizeMax", bb_diag * 0.05);
        }
        else if (resolution == 1)
        {
            gmsh::model::mesh::field::setNumber(2, "SizeMin", bb_diag * 0.01);
            gmsh::model::mesh::field::setNumber(2, "SizeMax", bb_diag * 0.01);
        }
        
        
    }
    gmsh::model::mesh::field::setAsBackgroundMesh(dim);
    gmsh::model::occ::synchronize();

    gmsh::model::mesh::generate(dim);

    
    gmsh::write("beam.vtk");
    gmsh::finalize();
    Eigen::MatrixXd V; Eigen::MatrixXi F, V_quad;
    if (quadratic)
    {
        loadQuadraticTriangleMeshFromVTKFile("beam.vtk", V, F, V_quad);
        F.resize(V_quad.rows(), 3);
        F.col(0) = V_quad.col(0); F.col(1) = V_quad.col(1); F.col(2) = V_quad.col(2);
        TV3 e0(V.row(F(0, 1)) - V.row(F(0, 0)));
        TV3 e1(V.row(F(0, 2)) - V.row(F(0, 0)));
        if (e1.cross(e0).dot(TV3(0, 0, 1)) > 0)
        {
            F.col(0) = V_quad.col(0); F.col(1) = V_quad.col(2); F.col(2) = V_quad.col(1);
            Eigen::MatrixXi V_quad_backup = V_quad;
            V_quad.col(1) = V_quad_backup.col(2); V_quad.col(2) = V_quad_backup.col(1);
            V_quad.col(5) = V_quad_backup.col(4); V_quad.col(4) = V_quad_backup.col(5);
        }
    }
    else
    {
        loadMeshFromVTKFile("beam.vtk", V, F);
    }

    num_nodes = V.rows(), num_ele = F.rows();
    undeformed.resize(num_nodes * dim);
    deformed.resize(num_nodes * dim);
    u.resize(num_nodes * dim); u.setZero();
    f.resize(num_nodes * dim); f.setZero();
    surface_indices.resize(num_ele * 3);
    if (quadratic)
        indices.resize(num_ele * 6);
    else
        indices.resize(num_ele * 3);

    tbb::parallel_for(0, num_nodes, [&](int i)
    {
        undeformed.template segment<dim>(i * dim) = V.row(i).head<dim>();
        deformed.template segment<dim>(i * dim) = V.row(i).head<dim>();
    });

    tbb::parallel_for(0, num_ele, [&](int i)
    {
        surface_indices.segment<3>(i * 3) = F.row(i);
        if (quadratic)
            indices.segment<6>(i * 6) = V_quad.row(i);
        else
            indices.segment<3>(i * 3) = F.row(i);
    });

    for (int i = 0; i < num_nodes; i++)
    {
        if (undeformed[i * dim + 0] < 1e-6)
        {
            for (int d = 0; d < dim; d++)
                dirichlet_data[i * dim + d] = 0;
        }
    }
    // std::cout << dirichlet_data.size() << std::endl;
    for (int i = 0; i < num_nodes; i++)
    {
        f[i * dim + 1] = -9.8 * 0.1;
    }
    if (mass_lumping)
        computeMassScaledForceVector(f);
    verbose = true;
    project_block_PD = false;
    max_newton_iter = 500;
}

template class FEMSolver<2>;
template class FEMSolver<3>;