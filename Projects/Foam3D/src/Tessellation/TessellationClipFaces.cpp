#include <igl/copyleft/cgal/remesh_self_intersections.h>
#include <igl/remove_unreferenced.h>
#include <igl/winding_number.h>
#include <igl/extract_manifold_patches.h>

#include "../../include/Tessellation/Tessellation.h"
#include "Projects/Foam3D/include/Energy/PerTriangleFunction.h"
#include <set>
#include <chrono>

bool operator<(const Node &a, const Node &b) {
    if (a.type != b.type) return a.type < b.type;
    if (a.gen[0] != b.gen[0]) return a.gen[0] < b.gen[0];
    if (a.gen[1] != b.gen[1]) return a.gen[1] < b.gen[1];
    if (a.gen[2] != b.gen[2]) return a.gen[2] < b.gen[2];
    if (a.gen[3] != b.gen[3]) return a.gen[3] < b.gen[3];
    return false;
}

// TODO: This function is horrendous.
void Tessellation::clipFaces() {
    int n_cells = c.rows() / getDims();

    std::vector<Face> unclippedFaces(faces.size());
    std::copy(faces.begin(), faces.end(), unclippedFaces.begin());
    faces.clear();

    std::map<Node, int> nodeIndices;
    {
        int iCool = 0;
        for (auto n: nodes) {
            nodeIndices[n.first] = iCool;
            iCool++;
        }
    }

    MatrixXT V(nodes.size() + boundary->v.size(), 3);
    std::vector<Node> nodeVector(V.rows());
    {
        int iCool = 0;
        for (auto n: nodes) {
            V.row(iCool) = n.second.pos;
            nodeVector[iCool] = n.first;
            iCool++;
        }
        for (auto v: boundary->v) {
            V.row(iCool) = v.pos;
            nodeVector[iCool].type = NodeType::B_VERTEX;
            nodeVector[iCool].gen[0] = iCool - nodes.size();
            nodeVector[iCool].gen[1] = -1;
            nodeVector[iCool].gen[2] = -1;
            nodeVector[iCool].gen[3] = -1;
            iCool++;
        }
    }

    int ntri_unclipped = 0;
    for (Face face: unclippedFaces) {
        ntri_unclipped += face.nodes.size() - 2;
    }

    MatrixXi F(ntri_unclipped + boundary->f.size(), 3);
    VectorXi Fsource(F.rows());
    {
        int iCool = 0;
        int source = 0;
        for (Face face: unclippedFaces) {
            for (int j = 1; j < face.nodes.size() - 1; j++) {
                F.row(iCool) = IV3(nodeIndices.at(face.nodes[0]),
                                   nodeIndices.at(face.nodes[j]),
                                   nodeIndices.at(face.nodes[j + 1]));
                Fsource(iCool) = source;
                iCool++;
            }
            source++;
        }
        for (auto face: boundary->f) {
            F.row(iCool) = face.vertices + IV3::Constant(nodes.size());
            Fsource(iCool) = source;
            source++;
            iCool++;
        }
    }

    igl::copyleft::cgal::RemeshSelfIntersectionsParam param;

    MatrixXT VV, SV;
    MatrixXi FF, SF;
    MatrixXi IF;
    VectorXi J, SJ;
    VectorXi IM, SIM;

    // Resolve intersections
    igl::copyleft::cgal::remesh_self_intersections(V, F, param, VV, FF, IF, J, IM);

    // Identify edge and face which generated each new vertex
    MatrixXi preClipFaces = MatrixXi::Constant(VV.rows(), 2, -1);
    for (int i = 0; i < FF.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            int v = FF(i, j);
            if (preClipFaces(v, 0) == -1) {
                preClipFaces(v, 0) = J(i);
            } else if (preClipFaces(v, 0) != J(i)) {
                preClipFaces(v, 1) = J(i);
            }
        }
    }
    MatrixXi originFaces = MatrixXi::Constant(VV.rows(), 3, -1); // Row is (edge face 0, edge face 1, intersected face).
    for (int i = V.rows(); i < VV.rows(); i++) {
        int j = IM(i);
        if (preClipFaces(i, 1) == -1) {
            originFaces(j, 2) = preClipFaces(i, 0);
        } else {
            originFaces.block<1, 2>(j, 0) = preClipFaces.row(i);
        }
    }

    // Merge duplicate vertices
    std::for_each(FF.data(), FF.data() + FF.size(), [&IM](int &a) { a = IM(a); });
    igl::remove_unreferenced(VV, FF, SV, SF, SIM, SJ);

    // Eliminate out-of-bounds triangles using winding number
    MatrixXT Q(SF.rows(), 3);
    for (int i = 0; i < SF.rows(); i++) {
        // TODO: FP errors because these points are right on the surface...
        TV3 v0 = SV.row(SF(i, 0));
        TV3 v1 = SV.row(SF(i, 1));
        TV3 v2 = SV.row(SF(i, 2));
        Q.row(i) = (v0 + v1 + v2) / 3.0 - 1e-10 * (v1 - v0).cross(v2 - v0).normalized();
    }
    VectorXT W;
    MatrixXT WV = V.block(nodes.size(), 0, boundary->v.size(), 3);
    MatrixXi WF =
            F.block(ntri_unclipped, 0, boundary->f.size(), 3) - MatrixXi::Constant(boundary->f.size(), 3, nodes.size());
    igl::winding_number(WV, WF, Q, W);
    std::vector<int> rk;
    for (int i = 0; i < W.rows(); i++) {
        if (fabs(W(i)) > 0.5) rk.push_back(i);
    }
    // Remove rows from SF -> FF2
    MatrixXi FF2(rk.size(), 3);
    VectorXi J2(rk.size());
    for (int i = 0; i < rk.size(); i++) {
        FF2.row(i) = SF.row(rk[i]);
        J2(i) = J(rk[i]);
    }

    Eigen::MatrixXi E, uE;
    Eigen::VectorXi EMAP;
    std::vector<std::vector<size_t>> uE2E;
    igl::unique_edge_map(FF2, E, uE, EMAP, uE2E);
    auto edge_index_to_face_index = [&](size_t ei) { return ei % FF2.rows(); };
    auto face_and_corner_index_to_edge_index = [&](size_t fi, size_t ci) {
        return ci * FF2.rows() + fi;
    };
    auto is_non_manifold_or_bedge = [&](size_t fi, size_t ci) -> bool {
        const size_t ei = face_and_corner_index_to_edge_index(fi, ci);

        auto matching_edges = uE2E[EMAP(ei)];
        if (matching_edges.size() != 2) return true;

        int source = Fsource(J2(fi));
//        std::cout << "bedge" << std::endl;
        for (auto e: matching_edges) {
//            std::cout << ei << " " << e << " " << source << " " << Fsource(J2(edge_index_to_face_index(e)))
//                      << std::endl;
            if (Fsource(J2(edge_index_to_face_index(e))) != source) return true;
        }

        return false;
    };

    VectorXi P;
    igl::extract_manifold_patches(FF2, EMAP, uE2E, P);
    std::map<std::tuple<int, int, int>, std::tuple<int, int, int>> coolMap; // Maps (vertex, patch, source face) to (next vertex, face index, pre clip face)
    for (int i = 0; i < FF2.rows(); i++) {
        IV3 tri = FF2.row(i);

        if (is_non_manifold_or_bedge(i, 2)) {
            coolMap[{tri(0), P(i), Fsource(J2(i))}] = {tri(1), i, J2(i)};
        }
        if (is_non_manifold_or_bedge(i, 0)) {
            coolMap[{tri(1), P(i), Fsource(J2(i))}] = {tri(2), i, J2(i)};
        }
        if (is_non_manifold_or_bedge(i, 1)) {
            coolMap[{tri(2), P(i), Fsource(J2(i))}] = {tri(0), i, J2(i)};
        }
//            std::cout << "wow " << tri(0) << " " << tri(1) << " " << tri(2) << std::endl;
//            std::cout << is_non_manifold_or_bedge(i, 0) << " " << is_non_manifold_or_bedge(i, 1) << " "
//                      << is_non_manifold_or_bedge(i, 2)
//                      << std::endl;
//            std::cout << Fsource(J2(i)) << " " << J2(i) << std::endl;
//        }
    }

    std::vector<std::vector<std::tuple<int, int, int>>> tuples;
    std::set<int> badVerts;
    while (!coolMap.empty()) {
        auto p = coolMap.begin();

        int vstart = std::get<0>(p->first);
        int patch = std::get<1>(p->first);
        int source = std::get<2>(p->first);

        std::vector<std::tuple<int, int, int>> thisTuples;
        int v = vstart;
        do {
            int vprev = v;
            auto mapped = coolMap.at({v, patch, source});
            v = std::get<0>(mapped);
            thisTuples.push_back(mapped);
            coolMap.erase({vprev, patch, source});
        } while (v != vstart);

        for (int i = 0; i < thisTuples.size(); i++) {
            int tupSource0 = std::get<2>(thisTuples[i]);
            int tupSource1 = std::get<2>(thisTuples[(i + 1) % thisTuples.size()]);
            int tupV = std::get<0>(thisTuples[i]);
            if (tupSource0 != tupSource1 && SJ(tupV) >= nodes.size() + boundary->v.size()) {
                badVerts.insert(tupV);
            }
        }

        tuples.push_back(thisTuples);
    }

    for (std::vector<std::tuple<int, int, int>> thisTuples: tuples) {
        Face thisFaceReal;

        TV3 avgPos = TV3::Zero();
        for (int i = 0; i < thisTuples.size(); i++) {
            int tupV = std::get<0>(thisTuples[i]);
            if (badVerts.find(tupV) == badVerts.end()) {
                IV3 sources = originFaces.row(SJ(tupV));

                Node node;
                if (SJ(tupV) < nodeVector.size()) {
                    node = nodeVector[SJ(tupV)];
                } else {
                    for (int j = 0; j < 3; j++) {
                        sources(j) = Fsource(sources(j));
                    }
                    if (sources(0) < unclippedFaces.size()) {
                        node.type = NodeType::B_FACE; // gen order (bf, c0, c1, c2)
                        node.gen[0] = sources(2) - unclippedFaces.size();
                        node.gen[1] = unclippedFaces[sources(0)].site0;
                        node.gen[2] = unclippedFaces[sources(0)].site1;
                        if (unclippedFaces[sources(1)].site0 == node.gen[1] ||
                            unclippedFaces[sources(1)].site0 == node.gen[2]) {
                            node.gen[3] = unclippedFaces[sources(1)].site1;
                        } else {
                            node.gen[3] = unclippedFaces[sources(1)].site0;
                        }
                        std::sort(std::begin(node.gen) + 1, std::end(node.gen));
                    } else {
                        node.type = NodeType::B_EDGE; // gen order (b0, b1, c0, c1)
                        IV3 bf0 = boundary->f[sources(0) - unclippedFaces.size()].vertices;
                        IV3 bf1 = boundary->f[sources(1) - unclippedFaces.size()].vertices;
                        int icv = 0;
                        for (int j = 0; j < 3; j++) {
                            int iv = bf0(j);
                            if (iv == bf1(0) || iv == bf1(1) || iv == bf1(2)) {
                                node.gen[icv] = iv;
                                icv++;
                            }
                        }
                        node.gen[2] = unclippedFaces[sources(2)].site0;
                        node.gen[3] = unclippedFaces[sources(2)].site1;
                        std::sort(std::begin(node.gen), std::begin(node.gen) + 2);
                        std::sort(std::begin(node.gen) + 2, std::end(node.gen));
                    }
                }

                thisFaceReal.nodes.push_back(node);

//                NodePosition nodePos;
//                nodePos.pos = SV.row(tupV);
//                nodes[node] = nodePos;

                avgPos += SV.row(tupV);
            }
        }
        avgPos /= thisFaceReal.nodes.size();

        int refFace = Fsource(std::get<2>(thisTuples[0]));
        if (refFace < unclippedFaces.size()) {
            thisFaceReal.site0 = unclippedFaces[refFace].site0;
            thisFaceReal.site1 = unclippedFaces[refFace].site1;
        } else {
//            for (Node node: thisFaceReal.nodes) {
//                if (node.type == NodeType::B_FACE || node.type == NodeType::B_EDGE) {
//                    double dmin = 1e10;
//                    int jmin = -1;
//                    for (int j = (node.type == NodeType::B_FACE ? 1 : 2); j < 4; j++) {
//                        double dcurr = (c.segment<3>(node.gen[j] * 4) - avgPos).squaredNorm();
//                        if (dcurr < dmin) {
//                            dmin = dcurr;
//                            jmin = j;
//                        }
//                    }
//                    thisFaceReal.site0 = node.gen[jmin];
//                    thisFaceReal.site1 = -1;
//
//                    break;
//                }
//            }

            // TODO: The method commented above failed for unintersected boundary faces (i.e. tris with 3 boundary verts)
            double dmin = 1e10;
            int jmin = -1;
            for (int j = 0; j < n_cells; j++) {
                // TODO: This only works for Power with wmul == 1.0
                double dcurr = (c.segment<3>(j * 4) - avgPos).squaredNorm() - 1.0 * c(j * 4 + 3);
                if (dcurr < dmin) {
                    dmin = dcurr;
                    jmin = j;
                }
            }
            thisFaceReal.site0 = jmin;
            thisFaceReal.site1 = -1;
        }

        faces.push_back(thisFaceReal);
    }
}
