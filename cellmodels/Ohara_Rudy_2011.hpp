#ifndef OHARA_RUDY_2011_HPP
#define OHARA_RUDY_2011_HPP

#include "enums/enum_Ohara_Rudy_2011.hpp"

  	// void initConsts();
	// void initConsts(double type);
	// void initConsts(bool is_dutta);
	void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, bool is_dutta, int offset);
	void computeRates(double TIME, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC, int offset);
	void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset );
	static double set_time_step(double TIME,double time_point,double max_time_step,
  double* CONSTANTS,
  double* RATES,
  double* STATES,
  double* ALGEBRAIC,
  int offset);
    void ___applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset);
	void ___applyDutta(double *CONSTANTS, int offset);
	void ___initConsts(double *CONSTANTS, double *STATES, double type, int offset);

#endif