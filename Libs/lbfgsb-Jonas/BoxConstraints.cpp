#include "BoxConstraints.h"

#include <iostream>

// #include <LBFGS.h>

namespace BoxConstraints
{

int computeBoundType(double lb, double ub)
{
	bool boundedBelow = lb > -std::numeric_limits<double>::max();
	bool boundedAbove = ub < std::numeric_limits<double>::max();
	if (boundedBelow)
	{
		if (boundedAbove)
		{
			return 2;
		}
		else
		{
			return 1;
		}
	}
	else
	{
		if (boundedAbove)
		{
			return 3;
		}
		else
		{
			return 0;
		}
	}
}

void computeBoundTypes(const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	std::vector<int> &boundTypes)
{
	boundTypes.resize(lowerBounds.size());
	for (int i = 0; i < lowerBounds.size(); i++)
	{
		boundTypes[i] = computeBoundType(lowerBounds[i], upperBounds[i]);
	}
}

bool computeIsBound(
	double epsilon,
	double lowerBound,
	double upperBound,
	int boundType,
	double x,
	double gradient)
{
	if (gradient < 0.0)
	{
		if (isBoundAbove(boundType))
		{
			return upperBound - x <= epsilon;
		}
	}
	else
	{
		if (isBoundBelow(boundType))
		{
			return x - lowerBound <= epsilon;
		}
	}
	return false;
}

void computeIsBound(double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	std::vector<bool> &isBound)
{
	int n = (int)lowerBounds.size();
	assert(n == upperBounds.size());
	assert(n == boundTypes.size());
	assert(n == x.size());
	assert(n == gradient.size());

	isBound.resize(n);
	for (int i = 0; i < n; i++)
	{
		isBound[i] = computeIsBound(
			epsilon,
			lowerBounds[i],
			upperBounds[i],
			boundTypes[i],
			x[i],
			gradient[i]);
	}
}
void computeIsBound(double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	std::vector<int> &isBound)
{
	int n = (int)lowerBounds.size();
	assert(n == upperBounds.size());
	assert(n == boundTypes.size());
	assert(n == x.size());
	assert(n == gradient.size());

	isBound.resize(n);
	for (int i = 0; i < n; i++)
	{
		isBound[i] = 0;
		if (gradient[i] < 0.0)
		{
			if (isBoundAbove(boundTypes[i])
				&& upperBounds[i] - x[i] <= epsilon)
			{
				isBound[i] = 2;
			}
		}
		else
		{
			if (isBoundBelow(boundTypes[i])
				&& x[i] - lowerBounds[i] <= epsilon)
			{
				isBound[i] = 1;
			}
		}
	}
}

void projectX(
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	Eigen::VectorXd &x)
{
	int n = (int)lowerBounds.size();
	assert(n == upperBounds.size());
	assert(n == boundTypes.size());
	assert(n == x.size());

	for (int i = 0; i < n; i++)
	{
		if (isBoundAbove(boundTypes[i]))
		{
			x[i] = std::min(upperBounds[i], x[i]);
		}
		if (isBoundBelow(boundTypes[i]))
		{
			x[i] = std::max(lowerBounds[i], x[i]);
		}
	}
}

void projectGradient(
	double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const std::vector<int> &boundTypes,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	Eigen::VectorXd &projGradient)
{
	int n = (int)lowerBounds.size();
	assert(n == upperBounds.size());
	assert(n == boundTypes.size());
	assert(n == x.size());
	assert(n == gradient.size());

	projGradient.resize(n);

	for (int i = 0; i < n; i++)
	{
		double gi = gradient[i];
		bool isBound = computeIsBound(
			epsilon,
			lowerBounds[i],
			upperBounds[i],
			boundTypes[i],
			x[i],
			gi);
		if (isBound) projGradient[i] = 0.0;
		else projGradient[i] = gi;
	}
}
void projectGradient(
	double epsilon,
	const Eigen::VectorXd &lowerBounds,
	const Eigen::VectorXd &upperBounds,
	const Eigen::VectorXd &x,
	const Eigen::VectorXd &gradient,
	Eigen::VectorXd &projGradient)
{
	int n = (int)lowerBounds.size();
	assert(n == upperBounds.size());
	assert(n == x.size());
	assert(n == gradient.size());

	projGradient.resize(n);

	for (int i = 0; i < n; i++)
	{
		double gi = gradient[i];
		int boundType = computeBoundType(lowerBounds[i], upperBounds[i]);
		bool isBound = computeIsBound(
			epsilon,
			lowerBounds[i],
			upperBounds[i],
			boundType,
			x[i],
			gi);
		if (isBound) projGradient[i] = 0.0;
		else projGradient[i] = gi;
	}
}



}
