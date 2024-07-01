#ifndef CELL_HPP
#define CELL_HPP

class Cellmodel
{
protected:
  Cellmodel(){}
public:
  short algebraic_size;
  short constants_size;
  short states_size;
  short gates_size;
  short current_size;
  short concs_size;
  double ALGEBRAIC[255];
  double CONSTANTS[255];
  double RATES[255];
  double STATES[255];
  char gates_header[255];
  short gates_indices[255];
  char current_header[255];
  short current_indices[255];
  char concs_header[255];
  short concs_indices[255];
  virtual ~Cellmodel() {}
  virtual void initConsts() = 0;
  virtual void initConsts(double type){}
  virtual void initConsts(double type, double conc, const double *hill){}
  virtual void initConsts(double type, double conc, const double *hill, bool is_dutta){}
  virtual void initConsts(double type, double conc, const double *hill, const double* herg){}
  virtual void computeRates(double TIME, double *CONSTANTS, double *RATES, double *STATES, double *ALGEBRAIC) = 0;
  virtual void solveAnalytical(double dt) {};
  virtual void solveRK4(double TIME,double dt) {};
  virtual double set_time_step(double TIME,double time_point, 
		double min_time_step, double max_time_step, 
		double min_dV, double max_dV) {return 0.005;};
};


#endif