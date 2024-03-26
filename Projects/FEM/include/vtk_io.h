#ifndef VTK_IO
#define VTK_IO
#include <Eigen/Core>
#include <fstream>

enum VTK_ATTRIBUTE_TYPE
{
	VERTEX_SCALAR,
	VERTEX_VECTOR,
	FACE_SCALAR,
	FACE_VECTOR
};

void WriteVTK(const std::string& filename, Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<std::string>& attr_names, std::vector<VTK_ATTRIBUTE_TYPE>& attr_types, std::vector<Eigen::VectorXd>& attr_values, std::vector<std::vector<std::pair<int, int>>>& paths);

#endif