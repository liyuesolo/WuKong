//#include "../include/Apollonius.h"
//#include "../include/CodeGen.h"
//
//#include <CGAL/Simple_cartesian.h>
//
//typedef CGAL::Simple_cartesian<double> Kernel;
//
//#include <CGAL/Apollonius_graph_2.h>
//#include <CGAL/Triangulation_data_structure_2.h>
//#include <CGAL/Apollonius_graph_vertex_base_2.h>
//#include <CGAL/Triangulation_face_base_2.h>
//#include <CGAL/Apollonius_graph_filtered_traits_2.h>
//
//// typedef for the traits; the filtered traits class is used
//typedef CGAL::Apollonius_graph_filtered_traits_2<Kernel> Traits;
//// typedefs for the algorithm
//// With the second template argument in the vertex base class being
//// false, we indicate that there is no need to store the hidden sites.
//// One case where this is indeed not needed is when we only do
//// insertions, like in the main program below.
//typedef CGAL::Apollonius_graph_vertex_base_2<Traits, true> Vb;
//typedef CGAL::Triangulation_face_base_2<Traits> Fb;
//typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Agds;
//typedef CGAL::Apollonius_graph_2<Traits, Agds> Apollonius_graph;
//
//VectorXi Apollonius::getDualGraph(const VectorXT &vertices) {
//    int n_vtx = vertices.rows() / 2;
//
//    Apollonius_graph ag;
//
//    for (int i = 0; i < n_vtx; i++) {
//        double x = vertices(i * 2 + 0);
//        double y = vertices(i * 2 + 1);
//        double w = 0;
//        if (i == 0) {
//            w = 0.02;
//        }
//        Apollonius_graph::Site_2 site(CGAL::Point_2<Kernel>(x, y), w);
//        ag.insert(site);
//    }
//
//    // TODO: Hack - I don't know why I need this, but otherwise some faces involving the last sites are missing...
//    Apollonius_graph::Site_2 site(CGAL::Point_2<Kernel>(100, 0), 0);
//    ag.insert(site);
//    Apollonius_graph::Site_2 site2(CGAL::Point_2<Kernel>(100, 1), 0);
//    ag.insert(site2);
//
//    int n_faces = ag.tds().number_of_faces();
//
//    VectorXi tri;
//    tri.resize(118 * 3);
//
//    std::cout << "num verts " << n_vtx << " " << ag.number_of_vertices() << std::endl;
//
//    for (int i = 0; i < n_vtx; i++) {
//        std::cout << ag.tds().vertices()[i].site().x() << " " << vertices(i * 2 + 0) << std::endl;
//    }
//    std::cout << ag.tds().vertices()[80].site().x() << std::endl;
//
//    int j = 0;
//    for (int i = 0; i < n_faces; i++) {
//        if (ag.tds().faces().is_used(i)) {
//            int v0 = ag.tds().vertices().index(ag.tds().faces()[i].vertex(0));
//            int v1 = ag.tds().vertices().index(ag.tds().faces()[i].vertex(1));
//            int v2 = ag.tds().vertices().index(ag.tds().faces()[i].vertex(2));
//
//            int q0 = -1, q1 = -1, q2 = -1;
//
//            int vmax = ag.tds().number_of_vertices() - 2;
//            if (v0 > 0 && v1 > 0 && v2 > 0 && v0 < vmax && v1 < vmax && v2 < vmax) {
//                while (ag.tds().vertices()[v0].site().x() != vertices((v0 + q0) * 2 + 0)) {
//                    q0++;
//                }
//                while (ag.tds().vertices()[v1].site().x() != vertices((v1 + q1) * 2 + 0)) {
//                    q1++;
//                }
//                while (ag.tds().vertices()[v2].site().x() != vertices((v2 + q2) * 2 + 0)) {
//                    q2++;
//                }
//                tri(j * 3 + 0) = v0 + q0;
//                tri(j * 3 + 1) = v1 + q1;
//                tri(j * 3 + 2) = v2 + q2;
////                std::cout << tri(j * 3 + 0) << " " << tri(j * 3 + 1) << " " << tri(j * 3 + 2) << std::endl;
//                j++;
//            }
//        }
//    }
//
//    return tri;
//}
//
//VectorXT Apollonius::getNodes(const VectorXT &vertices, const VectorXi &dual) {
//    int n_vtx = vertices.rows() / 2;
//    int n_faces = dual.rows() / 3;
//
//    VectorXT weights;
//    weights.resize(n_vtx);
//    weights.setZero();
//    weights(0) = 0.02;
//
//    VectorXT x;
//    x.resize(n_faces * 2);
//
//    for (int i = 0; i < n_faces; i++) {
//        int v1 = dual(i * 3 + 0);
//        int v2 = dual(i * 3 + 1);
//        int v3 = dual(i * 3 + 2);
//
//        double x1 = vertices(v1 * 2 + 0);
//        double y1 = vertices(v1 * 2 + 1);
//        double x2 = vertices(v2 * 2 + 0);
//        double y2 = vertices(v2 * 2 + 1);
//        double x3 = vertices(v3 * 2 + 0);
//        double y3 = vertices(v3 * 2 + 1);
//
//        double r1 = weights(v1);
//        double r2 = weights(v2);
//        double r3 = weights(v3);
//
//        double a2 = 2 * (x1 - x2);
//        double b2 = 2 * (y1 - y2);
//        double c2 = 2 * (r1 - r2);
//        double d2 = (x1 * x1 + y1 * y1 - r1 * r1) - (x2 * x2 + y2 * y2 - r2 * r2);
//
//        double a3 = 2 * (x1 - x3);
//        double b3 = 2 * (y1 - y3);
//        double c3 = 2 * (r1 - r3);
//        double d3 = (x1 * x1 + y1 * y1 - r1 * r1) - (x3 * x3 + y3 * y3 - r3 * r3);
//
//        double q = a2 * b3 - a3 * b2;
//        double fx = (-b3 * c2 + b2 * c3) / q;
//        double gx = (b3 * d2 - b2 * d3) / q;
//        double fy = (a3 * c2 - a2 * c3) / q;
//        double gy = (-a3 * d2 + a2 * d3) / q;
//
//        double a4 = fx * fx + fy * fy - 1;
//        double b4 = 2 * fx * (gx - x1) + 2 * fy * (gy - y1) - 2 * r1;
//        double c4 = (gx - x1) * (gx - x1) + (gy - y1) * (gy - y1) - r1 * r1;
//
//        double r4 = (-b4 - sqrt(b4 * b4 - 4 * a4 * c4)) / 2 * a4;
//        double xc = fx * r4 + gx;
//        double yc = fy * r4 + gy;
//
//        x.segment<2>(i * 2) = TV(xc, yc);
//    }
//
//    return x;
//}
//
//// TODO: Not implemented, this is incorrect
//Eigen::SparseMatrix<double> Apollonius::getNodesGradient(const VectorXT &vertices, const VectorXi &dual) {
//    return evaluate_dxdc_voronoi(vertices, dual);
//}
//
//
