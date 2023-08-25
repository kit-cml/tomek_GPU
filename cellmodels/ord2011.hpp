#ifndef ORD2011_HPP
#define ORD2011_HPP

#include "cellmodel.hpp"
//#include "enums/enum_ord2011.hpp"
#include "enums/enum_Ohara_Rudy_2011.hpp"

#define NA_CHANNEL
#define NAL_CHANNEL
#define TO_CHANNEL
#define KR_CHANNEL
#define KS_CHANNEL
#define K1_CHANNEL
#define CAL_CHANNEL
class ord2011 : public Cellmodel{
public:
	ord2011();
	~ord2011();
	void initConsts();
	void initConsts(double type);
	void initConsts(double type, double conc, double *ic50 );
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
