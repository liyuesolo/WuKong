#include <igl/triangle/triangulate.h>
// libigl library must be included first
#include "../../include/Tessellation/Power.h"
#include "Projects/Foam3D/include/Energy/PerTriangleFunction.h"
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Regular_triangulation_3<K> Regular_triangulation;

int idxClosest(const TV3 &p, const VectorXT &c) {
    int n_vtx = c.rows() / 4;

    int closest = -1;
    double dmin = 1000;

    for (int i = 0; i < n_vtx; i++) {
        TV3 p2 = c.segment<3>(i * 4);
        double d = (p2 - p).squaredNorm();
        if (d < dmin) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void Power::getDualGraph() {
    int n_vtx = c.rows() / 4;

    std::vector<Regular_triangulation::Weighted_point> wpoints;
    for (int i = 0; i < n_vtx; i++) {
        VectorXT v = c.segment<4>(i * 4);
//        std::cout << "Dual " << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << std::endl;
        Regular_triangulation::Weighted_point wp({v(0), v(1), v(2)}, wmul * v(3));
        wpoints.push_back(wp);
    }
    Regular_triangulation rt(wpoints.begin(), wpoints.end());
    auto inf_vtx = rt.infinite_vertex();

    std::map<Regular_triangulation::Cell_handle, Node> dual;
    for (auto it = rt.all_cells_begin(); it != rt.all_cells_end(); it++) {
        if (it->has_vertex(inf_vtx)) continue;

        auto v0 = it->vertex(0)->point();
        TV3 V0 = {v0.x(), v0.y(), v0.z()};
        auto v1 = it->vertex(1)->point();
        TV3 V1 = {v1.x(), v1.y(), v1.z()};
        auto v2 = it->vertex(2)->point();
        TV3 V2 = {v2.x(), v2.y(), v2.z()};
        auto v3 = it->vertex(3)->point();
        TV3 V3 = {v3.x(), v3.y(), v3.z()};

        Node node;
        node.type = NodeType::STANDARD;
        node.gen[0] = idxClosest(V0, c);
        node.gen[1] = idxClosest(V1, c);
        node.gen[2] = idxClosest(V2, c);
        node.gen[3] = idxClosest(V3, c);
        std::sort(std::begin(node.gen), std::end(node.gen));
        dual[it] = node;
    }

    for (auto it = rt.all_edges_begin(); it != rt.all_edges_end(); it++) {
        auto dualCell0 = it->first;
        if (dualCell0->has_vertex(inf_vtx)) continue;

        auto vh0 = dualCell0->vertex(it->second);
        auto v0 = vh0->point();
        TV3 V0 = {v0.x(), v0.y(), v0.z()};

        auto vh1 = dualCell0->vertex(it->third);
        auto v1 = vh1->point();
        TV3 V1 = {v1.x(), v1.y(), v1.z()};

        Face dualEdge;
        dualEdge.site0 = idxClosest(V0, c);
        dualEdge.site1 = idxClosest(V1, c);

        bool bad = false;
        auto dualCell = dualCell0;
        do {
            if (dualCell->has_vertex(inf_vtx)) {
                bad = true;
                break;
            }
            dualEdge.nodes.push_back(dual.at(dualCell));
            dualCell = dualCell->neighbor(
                    Regular_triangulation::next_around_edge(dualCell->index(vh0), dualCell->index(vh1)));
        } while (dualCell != dualCell0);

        if (!bad) {
            faces.push_back(dualEdge);
        }
    }
}

VectorXT Power::getDefaultVertexParams(const VectorXT &vertices) {
    int n_vtx = vertices.rows() / 3;

    VectorXT w = VectorXT::Zero(n_vtx);

    return w;
}
