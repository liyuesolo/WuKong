#include "LBFGSB.h"

#include "BoxConstraints.h"

#include "lbfgsb/LBFGSB-fable.h"

LBFGSB::LBFGSB()
{
	m_historySize = 20;
	m_printLevel = 10;
	m_state = INITIAL;
	m_accuracyFactor = 1e5; // 1.d + 12 for low accuracy; 1.d+7 for moderate accuracy; 1.d+1 for extremely  high accuracy.
	m_projectedGradientTolerance = 1e-6; // stopping criterion on the maximum entry of the projected gradient
}
void LBFGSB::setX(const Eigen::VectorXd &x)
{
	m_x = x;
}
void LBFGSB::setBounds(const Eigen::VectorXd &lowerBounds, const Eigen::VectorXd &upperBounds)
{
	m_lowerBounds = lowerBounds;
	m_upperBounds = upperBounds;
	BoxConstraints::computeBoundTypes(lowerBounds, upperBounds, m_boundTypes);
}
void LBFGSB::solve()
{
	while (m_state == INITIAL || m_state == STARTED)
	{
		takeStep();
	}
}

void LBFGSB::takeStep()
{
	fem::common cmn;
	int n = (int)m_x.size();
	int m = m_historySize;

	if (m_state == INITIAL)
	{
		if (m_lowerBounds.size() == 0 && n != 0)
		{
			Eigen::VectorXd lowerBounds = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
			Eigen::VectorXd upperBounds = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
			setBounds(lowerBounds, upperBounds);
		}

		m_gradient.resize(n);

		wa.resize((2 * m + 5)*n + 11 * m * m + 8 * m);
		iwa.resize(3 * n);

		m_task.assign(60, ' ');

		m_task[0] = 'S';
		m_task[1] = 'T';
		m_task[2] = 'A';
		m_task[3] = 'R';
		m_task[4] = 'T';

		fem::str_ref strRefTask(&m_task[0], (int)m_task.size()); //need to construct this after resizing m_task....

		csave.resize(60);

		isave.resize(44);
		dsave.resize(29);

		lbfgsb::setulb(
			cmn,
			n,
			m,
			*m_x.data(),
			*m_lowerBounds.data(),
			*m_upperBounds.data(),
			*m_boundTypes.data(),
			m_objectiveValue,
			*m_gradient.data(),
			m_accuracyFactor,
			m_projectedGradientTolerance,
			*wa.data(),
			*iwa.data(),
			strRefTask,
			m_printLevel,
			fem::str_ref(csave.data(), (int)csave.size()),
			*lsave,
			*isave.data(),
			*dsave.data());

		m_state = STARTED;
	}
	else
	{
		fem::str_ref strRefTask(&m_task[0], (int)m_task.size()); //need to construct this after resizing m_task....

		lbfgsb::setulb(
			cmn,
			n,
			m,
			*m_x.data(),
			*m_lowerBounds.data(),
			*m_upperBounds.data(),
			*m_boundTypes.data(),
			m_objectiveValue,
			*m_gradient.data(),
			m_accuracyFactor,
			m_projectedGradientTolerance,
			*wa.data(),
			*iwa.data(),
			strRefTask,
			m_printLevel,
			fem::str_ref(csave.data(), (int)csave.size()),
			*lsave,
			*isave.data(),
			*dsave.data());
	}

	fem::str_ref strRefTask(&m_task[0], (int)m_task.size()); //need to construct this after resizing m_task....

	while(strRefTask(1, 2) == "FG")
	{
		if(m_printLevel > 0) std::cout << " new task " << m_task << std::endl;
		
		m_objectiveValue = m_objectiveFunctor(m_x, m_gradient);
		lbfgsb::setulb(
			cmn,
			n,
			m,
			*m_x.data(),
			*m_lowerBounds.data(),
			*m_upperBounds.data(),
			*m_boundTypes.data(),
			m_objectiveValue,
			*m_gradient.data(),
			m_accuracyFactor,
			m_projectedGradientTolerance,
			*wa.data(),
			*iwa.data(),
			fem::str_ref(&m_task[0], (int)m_task.size()),
			m_printLevel,
			fem::str_ref(csave.data(), (int)csave.size()),
			*lsave,
			*isave.data(),
			*dsave.data());
	}

	if (strRefTask == "NEW_X")
	{
	}
	else if (strRefTask(1, 4) == "CONV")
	{
		m_state = CONVERGED;
	}
	else if (strRefTask(1, 4) == "WARN")
	{
		m_state = FAILED;
	}
	else if (strRefTask(1, 5) == "ERROR")
	{
		m_state = FAILED;
	}
	else if (strRefTask(1, 30) == "ABNORMAL_TERMINATION_IN_LNSRCH")
	{
		m_state = FAILED;
	}
	else
	{
		std::cout << " unknown new task " << m_task << std::endl;
		throw std::logic_error(" unknown new task " + m_task);
	}
}
Eigen::VectorXd LBFGSB::computeProjectedGradient()
{
	int n = (int)m_x.size();

	Eigen::VectorXd projg(n);
	m_objectiveValue = m_objectiveFunctor(m_x, m_gradient);

	BoxConstraints::projectGradient(1e-12, m_lowerBounds, m_upperBounds, m_boundTypes, m_x, m_gradient, projg);

	return projg;
}


