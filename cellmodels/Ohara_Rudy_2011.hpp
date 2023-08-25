#ifndef OHARA_RUDY_2011_HPP
#define OHARA_RUDY_2011_HPP

#include "cellmodel.hpp"
#include "enums/enum_Ohara_Rudy_2011.hpp"

class Ohara_Rudy_2011 : public Cellmodel
{
public:
  Ohara_Rudy_2011();
  ~Ohara_Rudy_2011();
  	void initConsts();
	void initConsts(double type);
	void initConsts(bool is_dutta);
	void initConsts(double type, double conc, double *ic50, bool is_dutta );
	void computeRates(double TIME, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC);
	void solveAnalytical( double dt );
	static double set_time_step(double TIME,double time_point,double max_time_step,
  double* CONSTANTS,
  double* RATES,
  double* STATES,
  double* ALGEBRAIC);
private:
	void ___applyDrugEffect(double conc, double *ic50, double epsilon);
	void ___applyDutta();
	void ___initConsts(double type);

};


#endif

