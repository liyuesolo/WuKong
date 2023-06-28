#include "../include/Util.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/tuple.h>
#include <boost/lexical_cast.hpp>

//CGAL 
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3  Point_3;
typedef std::array<std::size_t,3> Facet;

void triangulatePointCloud(const Eigen::VectorXd& points, Eigen::VectorXi& triangle_indices)
{
    int n_pt = points.rows() / 3;

    std::vector<Point_3> pointsCGAL;
    std::vector<Facet> facets;

    for (int i = 0; i < n_pt; i++)
        pointsCGAL.push_back(Point_3(points[i * 3 + 0],
        points[i * 3 + 1],
        points[i * 3 + 2]));
    
    
    CGAL::advancing_front_surface_reconstruction(pointsCGAL.begin(),
                                                pointsCGAL.end(),
                                                std::back_inserter(facets));
    
    triangle_indices.resize(facets.size() * 3);
    for (int i = 0; i < facets.size(); i++)
        triangle_indices.segment<3>(i * 3) = Eigen::Vector3i(facets[i][2], facets[i][1], facets[i][0]);
}