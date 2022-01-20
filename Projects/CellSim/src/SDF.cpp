#include "../include/SDF.h"

void SDF::initialize(const VectorXT& _data_points, 
    const VectorXT& _data_point_normals, T _radius, T _search_radius)
{
    data_points = _data_points;
    data_point_normals = _data_point_normals;
    radius = _radius;
    search_radius = _search_radius;
    n_points = data_points.rows();
}  

void SDF::valueIMLS(const TV& test_point, T& value)
{
    T weights_sum = 0.0;
    for (int i = 0; i < n_points; i++)
    {
        
    }
    
}

void SDF::gradientIMLS(const TV& test_point, TV& dphidx)
{

}

void SDF::hessianIMLS(const TV& test_point, TM& d2phidx2)
{

}