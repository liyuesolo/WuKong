#include "../include/VoronoiMetrics.h"

double Site::getMetricDistance(const TV &p) const {
    TV d = p - position;
    double theta = atan2(d.y(), d.x());
    if (theta < 0) theta += 2 * M_PI;

    MetricPoint p0, p1;

    auto lowerbnd = std::lower_bound(metric.begin(), metric.end(), theta,
                                     [](const MetricPoint &a, const double &b) { return a.theta < b; });
    if (lowerbnd == metric.begin() || lowerbnd == metric.end()) {
        p0 = metric[metric.size() - 1];
        p1 = metric[0];
    } else {
        p1 = *lowerbnd;
        p0 = *(--lowerbnd);
    }

    double x0 = p0.a * cos(p0.theta);
    double y0 = p0.a * sin(p0.theta);
    double x1 = p1.a * cos(p1.theta);
    double y1 = p1.a * sin(p1.theta);

    double A = y1 - y0;
    double B = -(x1 - x0);
    double C = A * x0 + B * y0;

    double a = C / (A * cos(theta) + B * sin(theta));

//    std::cout << std::endl;
//    std::cout << position.x() << " " << position.y() << std::endl;;
//    std::cout << p.x() << " " << p.y() << std::endl;;
//    std::cout << p0.theta << " " << theta << " " << p1.theta << std::endl;;
//    std::cout << a << " " << (p - position).norm() / a << std::endl;

    return (p - position).norm() / a;
}

VoronoiMetrics::VoronoiMetrics() {

}

int VoronoiMetrics::getClosestSiteByMetric(const TV &p) const {
    int closest = -1;
    double dmin = 1e10;

    for (int i = 0; i < sites.size(); i++) {
        double d = sites[i].getMetricDistance(p);
        if (d < dmin) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

int VoronoiMetrics::getClosestSiteToSelect(const TV &p, double threshold) const {
    int closest = -1;
    double dmin = 1e10;

    for (int i = 0; i < sites.size(); i++) {
        double d = (p - sites[i].position).norm();
        if (d < dmin && d < threshold) {
            closest = i;
            dmin = d;
        }
    }

    return closest;
}

void VoronoiMetrics::createSite(const TV &p, const VectorXT &metricX, const VectorXT &metricY) {
    Site newSite;
    newSite.position = p;

    int n = metricX.rows();
    for (int i = 0; i < n; i++) {
        double theta = atan2(metricY(i), metricX(i));
        if (theta < 0) theta += 2 * M_PI;
        double a = sqrt(pow(metricX(i), 2) + pow(metricY(i), 2));

        MetricPoint mp;
        mp.theta = theta;
        mp.a = a;

        newSite.metric.push_back(mp);
    }
    std::sort(newSite.metric.begin(), newSite.metric.end(),
              [](const MetricPoint &a, const MetricPoint &b) { return a.theta < b.theta; });

    sites.push_back(newSite);
}

void VoronoiMetrics::deleteSite(const int index) {
    sites.erase(sites.begin() + index);
}

void VoronoiMetrics::moveSite(const int index, const TV &p) {
    sites[index].position = p;
}

void VoronoiMetrics::setMetric(const VectorXT &metricX, const VectorXT &metricY) {
    std::vector<MetricPoint> metricPoints;

    int n = metricX.rows();
    for (int i = 0; i < n; i++) {
        double theta = atan2(metricY(i), metricX(i));
        if (theta < 0) theta += 2 * M_PI;
        double a = sqrt(pow(metricX(i), 2) + pow(metricY(i), 2));

        MetricPoint mp;
        mp.theta = theta;
        mp.a = a;

        metricPoints.push_back(mp);
    }
    std::sort(metricPoints.begin(), metricPoints.end(),
              [](const MetricPoint &a, const MetricPoint &b) { return a.theta < b.theta; });

    for (int i = 0; i < sites.size(); i++) {
        sites[i].metric = metricPoints;
    }
}

void VoronoiMetrics::computeVoronoiEdges(std::vector<TV> &nodes, std::vector<IV> &edges) const {
    std::vector<std::vector<std::pair<int, int>>> unclippedNodes0Idx;
    std::vector<std::vector<std::pair<int, int>>> unclippedNodes1Idx;
    std::vector<TV> unclippedNodes0;
    std::vector<TV> unclippedNodes1;
    for (int i = 0; i < sites.size(); i++) {
        Site site0 = sites[i];
        for (int j = i + 1; j < sites.size(); j++) {
            Site site1 = sites[j];
            MetricPoint p00, p01, p10, p11;
            for (int ii = 0; ii < site0.metric.size(); ii++) {
                p00 = site0.metric[ii];
                p01 = site0.metric[(ii + 1) % site0.metric.size()];
                for (int jj = 0; jj < site1.metric.size(); jj++) {
                    p10 = site1.metric[jj];
                    p11 = site1.metric[(jj + 1) % site1.metric.size()];

                    double A0 = p01.y() - p00.y();
                    double B0 = -(p01.x() - p00.x());
                    double C0 = A0 * p00.x() + B0 * p00.y();
                    double d0 = fabs(C0) / sqrt(A0 * A0 + B0 * B0);
                    TV n0 = TV(A0, B0).normalized() / d0;

                    double A1 = p11.y() - p10.y();
                    double B1 = -(p11.x() - p10.x());
                    double C1 = A1 * p10.x() + B1 * p10.y();
                    double d1 = fabs(C1) / sqrt(A1 * A1 + B1 * B1);
                    TV n1 = TV(A1, B1).normalized() / d1;

                    if ((n1 - n0).norm() < 1e-10) continue;

                    TV D0 = TV(-B0, A0);
                    TV D1 = TV(-B1, A1);
                    double a0 = 1;
                    double a1;
                    if (fabs(D1.dot(n0)) < 1e-10) {
                        a1 = 0;
                    } else {
                        a1 = a0 * D0.dot(n1) / D1.dot(n0);
                    }
                    TV line = a0 * D0 + a1 * D1;

                    // Points p on bisector satisfy (p - p0).dot(n0) = (p - p1).dot(n1)
                    // Line vector satisfies v.dot(n0) = v.dot(n1)
                    double xint = (site1.position.dot(n1) - site0.position.dot(n0)) / (n1.x() - n0.x());
                    double yint = (site1.position.dot(n1) - site0.position.dot(n0)) / (n1.y() - n0.y());
                    TV start(xint, 0);
                    if (std::isnan(xint) || std::isinf(xint) || fabs(xint) > 1e6) {
                        start = TV(0, yint);
                    }

                    double dmin = 1e10;
                    double dmax = -1e10;
                    IV imin(sites.size(), 0), imax(sites.size(), 0);
                    double x1 = start.x(), y1 = start.y(), x2 = (start + line).x(), y2 = (start + line).y();

                    std::vector<IV> points_idx = {{i, (ii - 1 + site0.metric.size()) % site0.metric.size()},
                                                  {i, (ii + 1) % site0.metric.size()},
                                                  {j, (jj - 1 + site1.metric.size()) % site1.metric.size()},
                                                  {j, (jj + 1) % site1.metric.size()}};
                    std::vector<TV> points3 = {site0.position, site0.position + TV(p01.x(), p01.y()), site1.position,
                                               site1.position + TV(p11.x(), p11.y())};
                    std::vector<TV> points4 = {site0.position + TV(p00.x(), p00.y()), site0.position,
                                               site1.position + TV(p10.x(), p10.y()), site1.position};

                    for (int k = 0; k < points3.size(); k++) {
                        if (dmin < dmax) break;
                        double x3 = points3[k].x(),
                                y3 = points3[k].y(),
                                x4 = points4[k].x(),
                                y4 = points4[k].y();

                        double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
                        bool bad = TV(-(y4 - y3), (x4 - x3)).dot(start - points3[k]) < 0;
                        if (fabs(denom) < 1e-10 && bad && k < 4) {
                            dmin = 1e-10;
                            dmax = 1e10;
                        } else if (fabs(denom) < 1e-10) {
                            continue;
                        }

                        double intx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
                        double inty = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;

                        TV intersect(intx, inty);
                        double d = intersect.dot(line);

                        bool ccw = TV(-(y4 - y3), (x4 - x3)).dot(line) > 0;
                        if (!ccw && d < dmin) {
                            dmin = d;
                            imin = points_idx[k];
                        } else if (ccw && d > dmax) {
                            dmax = d;
                            imax = points_idx[k];
                        }
                    }

                    if (dmax < dmin) {
                        dmax = std::max(dmax, -10 * line.norm());
                        dmin = std::min(dmin, 10 * line.norm());

                        std::vector<std::pair<int, int>> nodeIndices0;
                        nodeIndices0.push_back({i, ii});
                        nodeIndices0.push_back({j, jj});
                        nodeIndices0.push_back({imin(0), imin(1)});

                        std::vector<std::pair<int, int>> nodeIndices1;
                        nodeIndices1.push_back({i, ii});
                        nodeIndices1.push_back({j, jj});
                        nodeIndices1.push_back({imax(0), imax(1)});

                        unclippedNodes0Idx.push_back(nodeIndices0);
                        unclippedNodes0.emplace_back(start + (dmin - start.dot(line)) * line / line.squaredNorm());
                        unclippedNodes1Idx.push_back(nodeIndices1);
                        unclippedNodes1.emplace_back(start + (dmax - start.dot(line)) * line / line.squaredNorm());
                    }
                }
            }
        }
    }

    using NodeIdx = std::tuple<std::pair<int, int>, std::pair<int, int>, std::pair<int, int>>;
    std::map<NodeIdx, TV> map;
    std::vector<std::pair<NodeIdx, NodeIdx>> edgePairs;

    for (int i = 0; i < unclippedNodes0.size(); i++) {
        TV p0 = unclippedNodes0[i];
        TV p1 = unclippedNodes1[i];
        std::pair<int, int> idx0 = unclippedNodes0Idx[i][0];
        std::pair<int, int> idx1 = unclippedNodes0Idx[i][1];

        std::vector<std::pair<double, std::vector<std::pair<int, int>>>> intersects_t;
        intersects_t.push_back({0, unclippedNodes0Idx[i]});
        for (int j = 0; j < unclippedNodes0.size(); j++) {
            std::pair<int, int> idx2 = unclippedNodes0Idx[j][0];
            std::pair<int, int> idx3 = unclippedNodes0Idx[j][1];

            if (idx0 == idx2 && idx1 == idx3) continue;
            if (idx0 != idx2 && idx1 != idx2 && idx0 != idx3 && idx1 != idx3) continue;
            if (idx2 == idx0 || idx2 == idx1) idx2 = idx3;

            TV p2 = unclippedNodes0[j];
            TV p3 = unclippedNodes1[j];

            if (std::min(p0.x(), p1.x()) > std::max(p2.x(), p3.x()) ||
                std::max(p0.x(), p1.x()) < std::min(p2.x(), p3.x()) ||
                std::min(p0.y(), p1.y()) > std::max(p2.y(), p3.y()) ||
                std::max(p0.y(), p1.y()) < std::min(p2.y(), p3.y())) {
                continue;
            }

            double s1_x, s1_y, s2_x, s2_y;
            s1_x = p1.x() - p0.x();
            s1_y = p1.y() - p0.y();
            s2_x = p3.x() - p2.x();
            s2_y = p3.y() - p2.y();

            double s, t;
            s = (-s1_y * (p0.x() - p2.x()) + s1_x * (p0.y() - p2.y())) / (-s2_x * s1_y + s1_x * s2_y);
            t = (s2_x * (p0.y() - p2.y()) - s2_y * (p0.x() - p2.x())) / (-s2_x * s1_y + s1_x * s2_y);

            if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
                // Collision detected
                intersects_t.push_back({t, {idx0, idx1, idx2}});
            }
        }
        intersects_t.push_back({1, unclippedNodes1Idx[i]});
        std::sort(intersects_t.begin(), intersects_t.end());

        for (int k = 0; k < intersects_t.size(); k++) {
            std::sort(intersects_t[k].second.begin(), intersects_t[k].second.end());
        }
        for (int k = 0; k < intersects_t.size() - 1; k++) {
            int mm = getClosestSiteByMetric(
                    p0 + (intersects_t[k].first + intersects_t[k + 1].first) / 2 * (p1 - p0));
            if (mm == idx0.first || mm == idx1.first) {
                NodeIdx tuple0 = {intersects_t[k].second[0], intersects_t[k].second[1], intersects_t[k].second[2]};
                NodeIdx tuple1 = {intersects_t[k + 1].second[0], intersects_t[k + 1].second[1],
                                  intersects_t[k + 1].second[2]};

                map.insert({tuple0, p0 + intersects_t[k].first * (p1 - p0)});
                map.insert({tuple1, p0 + intersects_t[k + 1].first * (p1 - p0)});
                edgePairs.push_back({tuple0, tuple1});
            }
        }
    }

    std::map<NodeIdx, int> mapidx;
    int mapidxcount = 0;
    for (auto it = map.begin(); it != map.end(); it++) {
        nodes.push_back(it->second);
        mapidx.insert({it->first, mapidxcount});
        mapidxcount++;
    }
    for (std::pair<NodeIdx, NodeIdx> edgePair: edgePairs) {
        edges.push_back({mapidx.at(edgePair.first), mapidx.at(edgePair.second)});
    }
}
