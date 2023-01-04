#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>

#include "../include/CellSim.h"

using std::vector, std::pair;

vector<pair<int, int>> interval_sweepline(const vector<pair<double, double>>& points) {
	vector<pair<int, int>> collisions;
	vector<pair<double, int>> edges;

	std::unordered_set<int> active_intervals;

	const int n_points = points.size();
	for (int i = 0; i < n_points; ++i) {
		auto [begin, end] = points[i];
		edges.emplace_back(begin, i);
		edges.emplace_back(end, i);
	}
	std::sort(edges.begin(), edges.end());

	for (auto [value, i]: edges) {
		if (active_intervals.erase(i)) {
			for (int j: active_intervals)
				collisions.emplace_back(std::min(i, j), std::max(i, j));
		} else
			active_intervals.insert(i);
	}
	return collisions;
}

vector<pair<int, int>> cs2d::bb_collisions(const vector<pair<double, double>>& bbs_x, const vector<pair<double, double>>& bbs_y) {
	auto coll_x = interval_sweepline(bbs_x);
	auto coll_y = interval_sweepline(bbs_y);
	std::vector<pair<int, int>> collisions;
	std::sort(coll_x.begin(), coll_x.end());
	std::sort(coll_y.begin(), coll_y.end());
	auto x_it = coll_x.begin(), y_it = coll_y.begin();
	while (x_it != coll_x.end() && y_it != coll_y.end()) {
		if (*x_it < *y_it) {
			++x_it;
			continue;
		}
		if (*x_it > *y_it) {
			++y_it;
			continue;
		}
		collisions.push_back(*x_it);
		++x_it;
		++y_it;
	}
	return collisions;
}
