#include "../include/Tiling3D.h"
#include "../include/Util.h"
#include <fstream>

void Tiling3D::cropTranslationalUnitByparallelogram(const std::vector<PointLoops>& input_points,
    std::vector<TV2>& output_points, const TV2& top_left, const TV2& top_right,
    const TV2& bottom_right, const TV2& bottom_left, std::vector<Vector<int, 2>>& edge_pairs)
{
    std::vector<TV2> parallogram = {top_left, top_right, bottom_right, bottom_left};

    using Edge = std::pair<TV2, TV2>;

    std::vector<Edge> edges;

    for (auto one_tile : input_points)
    {
        // -2 because the tile vertices loop back to the first one
        for (int i = 0; i < one_tile.size() - 2; i++)
        {
            const TV2 xi = one_tile[i];
            const TV2 xj = one_tile[i+ 1];
            
            bool xi_inside = insidePolygon(parallogram, xi);
            bool xj_inside = insidePolygon(parallogram, xj);

            // both points are inside the parallelogram
            if (xi_inside && xj_inside)
            {
                
                Edge xij = std::make_pair(xi, xj);

                auto find_edge_iter = std::find_if(edges.begin(), edges.end(), [&xij](Edge e)
                    {   
                        return (((e.first - xij.first).norm() < 1e-6) && ((e.second - xij.second).norm() < 1e-6)) || 
                            (((e.first - xij.second).norm() < 1e-6) && ((e.second - xij.first).norm() < 1e-6));
                    }
                );

                bool new_edge = find_edge_iter == edges.end();
                
                if(new_edge)
                {
                    edges.push_back(std::make_pair(xi, xj));
                    // std::cout << xi.transpose() << " " << xj.transpose() << std::endl;
                    auto find_xi_iter = std::find_if(output_points.begin(), output_points.end(), 
                        [&xi](const TV2 x)->bool
                            { return (x - xi).norm() < 1e-6; }
                            );

                    int xi_idx = -1, xj_idx = -1;
                
                    if (find_xi_iter == output_points.end())
                    {
                        // xi is a new vtx
                        output_points.push_back(xi);
                        xi_idx = int(output_points.size()) - 1;
                        //pre push this edge
                        
                    }
                    else
                    {
                        int index = std::distance(output_points.begin(), find_xi_iter);
                        
                        xi_idx = index;
                    }

                    auto find_xj_iter = std::find_if(output_points.begin(), output_points.end(), 
                            [&xj](const TV2 x)->bool
                            { return (x - xj).norm() < 1e-6; }
                    );
                    if (find_xj_iter == output_points.end())
                    {
                        output_points.push_back(xj);
                        xj_idx = int(output_points.size()) - 1;
                        
                    }
                    else
                    {
                        int index = std::distance(output_points.begin(), find_xj_iter);
                        
                        xj_idx = index;
                    }
                    
                    edge_pairs.push_back(Vector<int,2>(xi_idx, xj_idx));
                }
            }
            else if(!xi_inside && xj_inside)
            {
                
                // std::cout << "One is inside" << std::endl;
                Edge xij = std::make_pair(xi, xj);
                auto find_edge_iter = std::find_if(edges.begin(), edges.end(), [&xij](Edge e)
                    {   
                        return (((e.first - xij.first).norm() < 1e-6) && ((e.second - xij.second).norm() < 1e-6)) || 
                            (((e.first - xij.second).norm() < 1e-6) && ((e.second - xij.first).norm() < 1e-6));
                    }
                );

                bool new_edge = find_edge_iter == edges.end();

                if(new_edge)
                {
                    edges.push_back(std::make_pair(xi, xj));
                    TV2 intersection;
                    int xj_idx = -1;
                    bool intersected = false;
                    int intersecting_edge = -1;
                    if (lineSegementsIntersect2D(xi, xj, parallogram[0], parallogram[1], intersection))
                    {
                        intersected = true;
                        intersecting_edge = 0;
                    }
                    else if(lineSegementsIntersect2D(xi, xj, parallogram[1], parallogram[2], intersection))
                    {
                        intersected = true;
                        intersecting_edge = 1;
                    }
                    else if(lineSegementsIntersect2D(xi, xj, parallogram[2], parallogram[3], intersection))
                    {
                        intersected = true;
                        intersecting_edge = 2;
                    }
                    else if (lineSegementsIntersect2D(xi, xj, parallogram[3], parallogram[0], intersection))
                    {
                        intersected = true;
                        intersecting_edge = 3;
                    }
                    if (intersected)
                    {
                        
                        output_points.push_back(intersection);
                        int xi_idx = output_points.size() - 1;

                        auto find_xj_iter = std::find_if(output_points.begin(), output_points.end(), [&xj](const TV2 x)->bool
                            { return (x - xj).norm() < 1e-6; }
                            );
                        if (find_xj_iter == output_points.end())
                        {
                            output_points.push_back(xj);
                            xj_idx = int(output_points.size()) - 1;
                            
                        }
                        else
                        {
                            int index = std::distance(output_points.begin(), find_xj_iter);
                            
                            xj_idx = index;
                        }

                        edge_pairs.push_back(Vector<int,2>(xj_idx, xi_idx));
                    }
                }
                
            }
            else if(!xj_inside && xi_inside)
            {
                // ignored due to duplicated edges
            }
            else
            {
                // continue;
            }
        }
        // return;
    }  
}

void Tiling3D::fetchTilingCropped(int IH, T* params, 
    std::vector<TV2>& valid_points, 
    std::vector<Edge>& edge_pairs,
    T square_width)
{
    std::vector<PointLoops> raw_points;
    fetchOneFamilyFillRegion(IH, params, raw_points, 15, 30);
    TV2 min_corner = TV2(1e6, 1e6), max_corner = TV2(-1e6, -1e6);
    for (const PointLoops& points_loop : raw_points)
    {
        for (const TV2 & pt : points_loop)
        {
            for (int d = 0; d < 2; d++)
            {
                min_corner[d] = std::min(min_corner[d], pt[d]);
                max_corner[d] = std::max(max_corner[d], pt[d]);
            }
        }
    }

    for (PointLoops& points_loop : raw_points)
    {
        for (TV2 & pt : points_loop)
        {
            pt = (pt - 0.5 * (max_corner + min_corner)) / (max_corner - min_corner).norm() * 5.0;
        }
    }
    cropTranslationalUnitByparallelogram(raw_points, valid_points, 
        TV2(-square_width, square_width),TV2(square_width, square_width), TV2(square_width, -square_width), 
        TV2(-square_width, -square_width), edge_pairs);
}

void Tiling3D::clapBottomLayerWithSquare(int IH, T* params, 
    PointLoops& point_loop_unit,
    std::vector<TV2>& valid_points, 
    std::vector<Vector<int, 2>>& edge_pairs,
    T square_width)
{
    TV2 T1, T2;

    fetchOneFamily(IH, params, T1, T2, point_loop_unit, 10, 10);

    TV2 min_corner = TV2(1e6, 1e6), max_corner = TV2(-1e6, -1e6);

    std::vector<PointLoops> all_points;

    int n_tile_T1 = 20, n_tile_T2 = 20;

    for (int tile_T1 = 0; tile_T1 < n_tile_T1; tile_T1++)
    {
        for (int tile_T2 = 0; tile_T2 < n_tile_T2; tile_T2++)
        {
            std::vector<TV2> shifted_points = point_loop_unit;
            for (TV2& pt : shifted_points)
            {
                pt += T(tile_T1) * T1 + T(tile_T2) * T2;
                for (int d = 0; d < 2; d++)
                {
                    max_corner[d] = std::max(pt[d], max_corner[d]);
                    min_corner[d] = std::min(pt[d], min_corner[d]);
                }
            }
            all_points.push_back(shifted_points);
        }
    }

    for (PointLoops& points_loop : all_points)
    {
        for (TV2 & pt : points_loop)
        {
            pt = (pt - 0.5 * (max_corner + min_corner)) / (max_corner - min_corner).norm() * 5.0;
        }
    }

    

    cropTranslationalUnitByparallelogram(all_points, valid_points, 
        TV2(-square_width, square_width),TV2(square_width, square_width), TV2(square_width, -square_width), 
        TV2(-square_width, -square_width), edge_pairs);
    // int n_inner_vtx = valid_points.size();

    // valid_points.push_back(TV2(-width, width));
    // valid_points.push_back(TV2(width, width));
    // valid_points.push_back(TV2(width, -width));
    // valid_points.push_back(TV2(-width, -width));

    // for (TV2 & pt : valid_points)
    //     pt *= square_width;

    // edge_pairs.push_back(Vector<int, 2>(n_inner_vtx + 0, n_inner_vtx + 1));
    // edge_pairs.push_back(Vector<int, 2>(n_inner_vtx + 1, n_inner_vtx + 2));
    // edge_pairs.push_back(Vector<int, 2>(n_inner_vtx + 2, n_inner_vtx + 3));
    // edge_pairs.push_back(Vector<int, 2>(n_inner_vtx + 3, n_inner_vtx + 0));
}


void Tiling3D::thickenLines(const std::vector<TV2>& valid_points, 
    const std::vector<Vector<int, 2>>& edge_pairs,
    std::vector<TV>& vertices, std::vector<Face>& faces, T thickness,
    std::vector<IdList>& boundary_indices)
{
    TM2 R90 = TM2::Zero();

    R90.row(0) = TV2(0, -1);
    R90.row(1) = TV2(1, 0);

    auto getEndPoint = [&](const Vector<int, 2>& edge_pair)
    {
        TV2 from = valid_points[edge_pair[0]];
        TV2 to = valid_points[edge_pair[1]];
        return std::make_pair(TV(from[0], from[1], 0.0), TV(to[0], to[1], 0));
    };
    
    // for (int i = 0; i < edge_pairs.size(); i++)
    // {
    //     IdList id_list0, id_list1;
    //     int sub_division = 5;
    //     auto from_to = getEndPoint(edge_pairs[i]);
    //     TV from = from_to.first, to = from_to.second;
    //     TV extend = 0.01 * (to - from);
        
    //     int idx_from = edge_pairs[i][0];
    //     int idx_to = edge_pairs[i][1];

    //     TV2 ortho_dir_2D = (R90 * (to.head<2>() - from.head<2>())).normalized();
    //     TV ortho_dir(ortho_dir_2D[0], ortho_dir_2D[1], 0.0);
    //     TV v0, v1;
    //     int idx0, idx1;
    //     for (int j = 0; j < sub_division + 1; j++)
    //     {
    //         TV sub_point = from + (to - from) * T(j) / T(sub_division);
    //         TV v2 = sub_point + 0.5 * thickness * ortho_dir;
    //         TV v3 = sub_point - 0.5 * thickness * ortho_dir;
    //         vertices.push_back(v2);
    //         int idx2 = vertices.size() - 1;
    //         vertices.push_back(v3);
    //         int idx3 = vertices.size() - 1;
    //         id_list0.push_back(idx2);
    //         id_list1.push_back(idx3);

    //         if ( j != 0 )
    //         {
    //             faces.push_back(Face(idx0, idx2, idx1));
    //             faces.push_back(Face(idx3, idx1, idx2));
    //         }
    //         v0 = v2; v1 = v3;
    //         idx0 = idx2; idx1 = idx3;
    //     }
    //     boundary_indices.push_back(id_list0);
    //     boundary_indices.push_back(id_list1);
    // }

    for (int i = 0; i < valid_points.size(); i++)
    {
        vertices.push_back(TV(valid_points[i][0], valid_points[i][1], 0.0));
    }
    

    for (int i = 0; i < edge_pairs.size(); i++)
    {
        IdList id_list0, id_list1;
        int sub_division = 200;
        auto from_to = getEndPoint(edge_pairs[i]);
        TV from = from_to.first, to = from_to.second;
        TV extend = 0.01 * (to - from);
        
        int idx_from = edge_pairs[i][0];
        int idx_to = edge_pairs[i][1];

        TV2 ortho_dir_2D = (R90 * (to.head<2>() - from.head<2>())).normalized();
        TV ortho_dir(ortho_dir_2D[0], ortho_dir_2D[1], 0.0);
        TV v0, v1;
        int idx0, idx1;
        id_list0.push_back(edge_pairs[i][0]);
        id_list1.push_back(edge_pairs[i][0]);
        for (int j = 1; j < sub_division; j++)
        {
            TV sub_point = from + (to - from) * T(j) / T(sub_division);
            TV v2 = sub_point + 0.5 * thickness * ortho_dir;
            TV v3 = sub_point - 0.5 * thickness * ortho_dir;
            vertices.push_back(v2);
            int idx2 = vertices.size() - 1;
            vertices.push_back(v3);
            int idx3 = vertices.size() - 1;
            if (j == 1)
            {
                faces.push_back(Face(edge_pairs[i][0], idx2, idx3));
            }
            else if (j == sub_division - 1)
            {
                faces.push_back(Face(idx0, idx2, idx1));
                faces.push_back(Face(idx3, idx1, idx2));
                faces.push_back(Face(idx2, edge_pairs[i][1], idx3));
            }
            else
            {
                faces.push_back(Face(idx0, idx2, idx1));
                faces.push_back(Face(idx3, idx1, idx2));
            }
            id_list0.push_back(idx2);
            id_list1.push_back(idx3);
            v0 = v2; v1 = v3;
            idx0 = idx2; idx1 = idx3;
        }
        id_list0.push_back(edge_pairs[i][1]);
        id_list1.push_back(edge_pairs[i][1]);
        boundary_indices.push_back(id_list0);
        boundary_indices.push_back(id_list1);
    }
    
}


void Tiling3D::extrudeInZ(std::vector<TV>& vertices, std::vector<Face>& faces, 
    T height, std::vector<IdList>& boundary_indices)
{
    std::vector<TV> extruded_vertices;
    std::vector<Face> extruded_faces;

    for(int i = 0; i < vertices.size(); i++)
        extruded_vertices.push_back(vertices[i]);

    for(int i = 0; i < vertices.size(); i++)
        extruded_vertices.push_back(TV(vertices[i][0], vertices[i][1], height));

    int nv = vertices.size();
    int nf = faces.size();
    
    extruded_faces = faces;
    for (int i = 0; i < nf; i++)
    {
        extruded_faces.push_back(Face(
          faces[i][2] + nv, faces[i][1] + nv, faces[i][0] + nv
        ));
    }

    for (IdList id_list : boundary_indices)
    {
        for (int i = 0; i < id_list.size() - 1; i++)
        {
            extruded_faces.push_back(Face(
                id_list[i], id_list[i] + nv, id_list[i + 1]
            ));
            extruded_faces.push_back(Face(
                id_list[i + 1], id_list[i] + nv, id_list[i + 1] + nv
            ));
        }
        
    }
    faces = extruded_faces;
    vertices = extruded_vertices;
}


void Tiling3D::getMeshForPrintingWithLines(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    PointLoops point_loop_unit;

    
    //IH 10 work
    int IH = 9;
    T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5

    // int IH = 29;
    // T params[] = {0}; //Isohedral 29

    // int IH = 0;
    // T params[] = {0.1224, 0.6434, 0.207, 0.8131}; //Isohedral 0

    // unit is mm here to be used for Prusa directly

    T square_width = 50;
    T width = 0.5, height = 5;
    
    std::vector<TV2> valid_points;
    std::vector<Vector<int, 2>> edge_pairs;

    clapBottomLayerWithSquare(IH, params, point_loop_unit, valid_points, edge_pairs, square_width);

    std::vector<TV> vertices;
    std::vector<Face> faces;

    std::vector<IdList> boundary_indices;

    thickenLines(valid_points, edge_pairs, vertices, faces, width, boundary_indices);

    extrudeInZ(vertices, faces, height, boundary_indices);

    std::string name = "IH_" + std::to_string(IH) + ".obj";

    std::ofstream out(name);
    for (TV pt : vertices)
        out << "v " << pt.transpose() << std::endl;
    
    for (Face face : faces)
        out << "f " << face[0] + 1 << " " << face[1] + 1 << " " << face[2] + 1 << std::endl;
    out.close();

    // std::ofstream out("line_mesh.obj");
    // for (TV2 pt : valid_points)
    //     out << "v " << pt.transpose() << " 0" << std::endl;
    
    // for (Vector<int, 2> line : edge_pairs)
    //     out << "l " << line[0] + 1 << " " << line[1] + 1 << std::endl;
    // out.close();
}

void Tiling3D::getMeshForPrinting(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    std::vector<PointLoops> raw_points;
    // int IH = 5;
    // T params[] = {0.1224, 0.4979, 0.0252, 0.4131, 0.4979}; //Isohedral 5

    // int IH = 0;
    // T params[] = {0.1161, 0.5464, 0.4313, 0.5464}; //Isohedral 0

    // int IH = 13;
    // T params[] = {0.1, 0.2}; //Isohedral 7

    // int IH = 29;
    // T params[] = {0}; //Isohedral 29

    int IH = 6;
    T params[] = {0.5, 0.5, 0.5, 0.5, 0.5}; //Isohedral 06

    // int IH = 1;
    // T params[] = {0.207, 0.7403, 0.304, 1.2373}; //Isohedral 01

    // int IH = 2;
	// T params[] = {0.3767, 0.5949, 0, 0}; //Isohedral 02

    TV2 T1, T2;

    fetchOneFamilyFillRegion(IH, params, raw_points, 10, 10);

    T height = 1.0;

    std::vector<TV2> unique_points;
    std::vector<IdList> polygon_ids;

    for (const PointLoops& pl : raw_points)
    {
        TV2 center = TV2::Zero();
        IdList id_list;
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 pt = pl[i];
            center += pt;
            auto find_iter = std::find_if(unique_points.begin(), unique_points.end(), 
                            [&pt](const TV2 x)->bool
                               { return (x - pt).norm() < 1e-6; }
                             );
            if (find_iter == unique_points.end())
            {
                unique_points.push_back(pt);
                id_list.push_back(unique_points.size() - 1);
            }
            else
                id_list.push_back(std::distance(unique_points.begin(), find_iter));
        }

        polygon_ids.push_back(id_list);
        center /= T(pl.size() - 1);
        id_list.clear();
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 inner = center + (pl[i] - center) * 0.85;
            unique_points.push_back(inner);
            id_list.push_back(unique_points.size() - 1);
        }
        polygon_ids.push_back(id_list);
    }

    std::vector<Face> faces;
    for (int i = 0; i < polygon_ids.size() / 2; i++)
    {
        auto outer = polygon_ids[i * 2];
        auto inner = polygon_ids[i * 2 + 1];

        for (int j = 0; j < outer.size(); j++)
        {
            faces.push_back(Face(outer[j] + 1, inner[j] + 1, inner[(j + 1) % outer.size()] + 1));
            faces.push_back(Face(outer[j] + 1, inner[(j + 1) % outer.size()] + 1, outer[(j + 1) % outer.size()] + 1));
        }
    }

    std::vector<TV> vertices;
    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], 0));

    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], height));

    int nv = unique_points.size();
    int nf = faces.size();

    for (int i = 0; i < nf; i++)
    {
        faces.push_back(Face(
          faces[i][2] + nv, faces[i][1] + nv, faces[i][0] + nv
        ));
    }
    
    for (IdList id_list : polygon_ids)
    {
        for (int i = 0; i < id_list.size(); i++)
        {
            faces.push_back(Face(
                id_list[i] + 1, id_list[(i + 1)%id_list.size()] + 1, id_list[i] + nv + 1
            ));
            faces.push_back(Face(
                id_list[(i + 1)%id_list.size()] + 1, id_list[(i + 1)%id_list.size()] + nv + 1, id_list[i] + nv + 1 
            ));
        }
    }

    V.resize(vertices.size(), 3);
    tbb::parallel_for(0, (int)vertices.size(), [&](int i){
        V.row(i) = vertices[i];
    });

    F.resize(faces.size(), 3);
    C.resize(faces.size(), 3);

    tbb::parallel_for(0, (int)faces.size(), [&](int i){
        F.row(i) = faces[i];
        C.row(i) = Eigen::Vector3d(0, 0.3, 1.0);
    });
    
    std::ofstream out("test_mesh.obj");
    for (const TV& pt : vertices)
        out << "v " << pt.transpose() * 10.0 << std::endl;
    for (auto face : faces)
        out << "f " << face.transpose() << std::endl;
    out.close();

}

void Tiling3D::extrudeToMesh(const std::vector<PointLoops>& raw_points, 
        T width, T height, std::string filename)
{
    std::vector<TV2> unique_points;
    std::vector<IdList> polygon_ids;

    for (const PointLoops& pl : raw_points)
    {
        TV2 center = TV2::Zero();
        IdList id_list;
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 pt = pl[i];
            center += pt;
            auto find_iter = std::find_if(unique_points.begin(), unique_points.end(), 
                            [&pt](const TV2 x)->bool
                               { return (x - pt).norm() < 1e-6; }
                             );
            if (find_iter == unique_points.end())
            {
                unique_points.push_back(pt);
                id_list.push_back(unique_points.size() - 1);
            }
            else
                id_list.push_back(std::distance(unique_points.begin(), find_iter));
        }

        polygon_ids.push_back(id_list);
        center /= T(pl.size() - 1);
        id_list.clear();
        for (int i = 0; i < pl.size() - 1; i++)
        {
            TV2 inner = center + (pl[i] - center) * 0.9;
            unique_points.push_back(inner);
            id_list.push_back(unique_points.size() - 1);
        }
        polygon_ids.push_back(id_list);
    }

    std::vector<Face> faces;
    for (int i = 0; i < polygon_ids.size() / 2; i++)
    {
        auto outer = polygon_ids[i * 2];
        auto inner = polygon_ids[i * 2 + 1];

        for (int j = 0; j < outer.size(); j++)
        {
            faces.push_back(Face(outer[j] + 1, inner[j] + 1, inner[(j + 1) % outer.size()] + 1));
            faces.push_back(Face(outer[j] + 1, inner[(j + 1) % outer.size()] + 1, outer[(j + 1) % outer.size()] + 1));
        }
    }

    std::vector<TV> vertices;
    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], 0));

    for(int i = 0; i < unique_points.size(); i++)
        vertices.push_back(TV(unique_points[i][0], unique_points[i][1], height));

    int nv = unique_points.size();
    int nf = faces.size();

    for (int i = 0; i < nf; i++)
    {
        faces.push_back(Face(
          faces[i][2] + nv, faces[i][1] + nv, faces[i][0] + nv
        ));
    }
    
    for (IdList id_list : polygon_ids)
    {
        for (int i = 0; i < id_list.size(); i++)
        {
            faces.push_back(Face(
                id_list[i] + 1, id_list[(i + 1)%id_list.size()] + 1, id_list[i] + nv + 1
            ));
            faces.push_back(Face(
                id_list[(i + 1)%id_list.size()] + 1, id_list[(i + 1)%id_list.size()] + nv + 1, id_list[i] + nv + 1 
            ));
        }
    }
    
    
    std::ofstream out(filename);
    for (const TV& pt : vertices)
        out << "v " << pt.transpose() * 10.0 << std::endl;
    for (auto face : faces)
        out << "f " << face.transpose() << std::endl;
    out.close();
}
