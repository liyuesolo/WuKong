#include "../include/Util.h"

void loadMeshFromVTKFile(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
    using TV3 = Vector<T, 3>;
    using IV3 = Vector<int, 3>;

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


void loadPBCDataFromMSHFile(const std::string& filename, 
    std::vector<std::vector<Vector<int ,2>>>& pbc_pairs)
    // std::vector<Vector<int, 3>>& pbc_pairs)
{
    pbc_pairs.clear();
    pbc_pairs.resize(2, std::vector<Vector<int ,2>>());
    std::ifstream in(filename);
    
    std::string token;
    
	while(token != "$Periodic")
    {
        in >> token;
        if (in.eof())
            break;
    }

    int n_pairs;
    in >> n_pairs;
    std::cout << "n_pairs " << n_pairs << std::endl;
    int dir, node0, node1;
    for (int i = 0; i < n_pairs; i++)
    {
        in >> dir >> node0 >> node1;
        pbc_pairs[dir].push_back(Vector<int, 2>(node0 - 1, node1 - 1));
        // pbc_pairs.push_back(Vector<int, 3>(dir, node0, node1));
        int n_entry;
        in >> n_entry;
        for (int j = 0; j < n_entry; j++)
            in >> token;
        in >> n_entry;
        for (int j = 0; j < n_entry * 2; j++)
            in >> token;
    }
    in.close();
}