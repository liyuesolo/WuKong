#include "../include/Util.h"

void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    std::ifstream in(filename);
    std::string token;

	while(token != "POINTS")
		in >> token;
    
    int n_points;
	in >> n_points;

	in >> token; //double

	V.resize(n_points, 3);

	for(int i=0; i<n_points; i++)
        for (int  j = 0; j < 3; j++)
            in >> V(i, j);
        
    while(token != "CELLS")
		in >> token;
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
            Vector<int, 3> face;
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
        F.row(i) = faces[i];
    });
    in.close();
}