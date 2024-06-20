#ifndef OHARA_RUDY_CIPA_V1_2017_HPP
#define OHARA_RUDY_CIPA_V1_2017_HPP

#include "cellmodel.hpp"
#include "enums/enum_ohara_rudy_cipa_v1_2017.hpp"
#include <cuda_runtime.h>

// #define EULER

// class ohara_rudy_cipa_v1_2017 : public Cellmodel
// {
// public:
// 	ohara_rudy_cipa_v1_2017();
// 	~ohara_rudy_cipa_v1_2017();
// 	void initConsts ();
// 	void initConsts(double type);
// 	void initConsts(double type, double conc, const double *hill, const double *herg );
// 	void computeRates( double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC );
// 	void solveAnalytical( double dt );
// 	void solveRK4(double TIME,double dt);
// 	double set_time_step(double TIME,double time_point, 
// 		double min_time_step, double max_time_step, 
// 		double min_dV, double max_dV);
// private:
// 	void ___applyDrugEffect(double conc, const double *hill);
// 	void ___applyHERGBinding(double conc, const double *herg);
// 	void ___initConsts(double type);
// 	void ___gaussElimination(double *A, double *b, double *x, int N);
// };

	__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, double *cvar, bool is_dutta, bool is_cvar, double bcl, int offset);
	__device__ void computeRates(double TIME, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC, int offset);
	__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset);
	__device__ double set_time_step(double TIME, double time_point, double max_time_step, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC, int offset);
    __device__ void applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset);
	__device__ void ___applyDutta(double *CONSTANTS, int offset);
	__device__ void ___applyCvar(double *CONSTANTS, double *cvar, int offset);
	__device__ void ___initConsts(double *CONSTANTS, double *STATES, double type, double bcl, int offset);
	__device__ void ___applyHERGBinding(double conc, const double *herg);
	__device__ double ___gaussElimination(double *A, double *b, double *x, int N);


#endif

