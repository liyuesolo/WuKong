#ifndef CS2D_TRI_H
#define CS2D_TRI_H
#include <Eigen/Dense>
#include <utility>


using Eigen::Matrix, Eigen::Dynamic, Eigen::Vector;

template <int dim>
using VertexMatrix = Matrix<double, Dynamic, dim>;

using TriangleMatrix = Matrix<int, Dynamic, 3>;

template <int dim>
std::pair<VertexMatrix<dim>, TriangleMatrix> centroidTriangulate(const VertexMatrix<dim>& vertices) {
	const int n_verts = vertices.rows();
	Vector<double, dim> centroid = vertices.colwise().sum()/n_verts;
	VertexMatrix<dim> tri_verts(n_verts+1, dim);
	tri_verts << vertices, centroid.transpose();
	TriangleMatrix tri_faces(n_verts, 3);
	for (int i = 0; i < n_verts; i++) {
		if (i < n_verts-1)
			tri_faces.row(i) << n_verts, i, i+1;
		else
			tri_faces.row(i) << n_verts, i, 0;
	}
	return {tri_verts, tri_faces};

}
#endif
