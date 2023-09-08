#ifndef GPU_CUH
#define GPU_CUH

__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned int sample_size);

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned short sample_id, double *tcurr, 
                                       double *dt, unsigned int sample_size);


#endif