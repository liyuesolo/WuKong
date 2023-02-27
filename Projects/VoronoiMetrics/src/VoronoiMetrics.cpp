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
//    std::cout << a << std::endl;

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

void VoronoiMetrics::createSite(const TV &p) {
    Site newSite;
    newSite.position = p;

    int n = 8;
    for (int i = 0; i < n; i++) {
        double theta = i * 2 * M_PI / n;

        MetricPoint mp;
        mp.theta = theta;
        mp.a = 0.5 * sin(2 * theta) * sin(2 * theta) + 0.3 + 0.2 * sin(theta);
//        mp.a = sqrt(2 * sin(theta) * sin(theta) + cos(theta) * cos(theta));

        newSite.metric.push_back(mp);
    }

    sites.push_back(newSite);
}

void VoronoiMetrics::deleteSite(const int index) {
    sites.erase(sites.begin() + index);
}

void VoronoiMetrics::moveSite(const int index, const TV &p) {
    sites[index].position = p;
}
