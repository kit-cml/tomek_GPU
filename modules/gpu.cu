// #include "cellmodels/enums/enum_Ohara_Rudy_2011.hpp"
#include "../cellmodels/Ohara_Rudy_2011.hpp"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "glob_funct.hpp"
#include "glob_type.hpp"
#include "gpu_glob_type.cuh"

/*
all kernel function has been moved. Unlike the previous GPU code, now we seperate everything into each modules.
all modules here are optimised for GPU and slightly different than the original component based code
differences are related to GPU offset calculations
*/

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned short sample_id, double *tcurr, 
                                       double *dt, unsigned int sample_size)
    {
    
    unsigned int input_counter = 0;
    unsigned short cnt;

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    // int num_of_rates = 41;

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = 0.005;
    double tmax;
    double max_time_step = 1.0, time_point = 25.0;
    double dt_set;

    bool writen = false;

    // files for storing results
    // time-series result
    // FILE *fp_vm, *fp_inet, *fp_gate;

    // features
    // double inet, qnet;

    // looping counter
    // unsigned short idx;
  
    // simulation parameters
    // double dtw = 2.0;
    // const char *drug_name = "bepridil";
    const double bcl = 2000; // bcl is basic cycle length
    
    // const double inet_vm_threshold = -88.0;
    // const unsigned short pace_max = 300;
    // const unsigned short pace_max = 1000;
    const unsigned short pace_max = 10;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    // const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    // double conc = 243.0; //mmol
    double conc = 0.0;
    double type = 0.;
    bool dutta = false;
    double epsilon = 10E-14;


    // printf("Core %d:\n",sample_id);
    initConsts(d_CONSTANTS, d_STATES, type, conc, d_ic50, dutta, sample_id);

    applyDrugEffect(d_CONSTANTS, conc, d_ic50, epsilon, sample_id);

    d_CONSTANTS[BCL + (sample_id * num_of_constants)] = bcl;

    // generate file for time-series output

    tmax = pace_max * bcl;
    int pace_count = 0;
    // printf("%lf", d_ic50[0]);
  
    // printf("%d,%lf,%lf,%lf,%lf\n", sample_id, dt[sample_id], tcurr[sample_id], d_STATES[V + (sample_id * num_of_states)],d_RATES[V + (sample_id * num_of_rates)]);

    while (tcurr[sample_id]<tmax){
        dt_set = set_time_step( tcurr[sample_id], time_point, max_time_step, 
        d_CONSTANTS, 
        d_RATES, 
        d_STATES, 
        d_ALGEBRAIC, 
        sample_id); 
        
        computeRates(tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, sample_id); 
        
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
        }
        else {
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];
          pace_count++;
          writen = false;
          // printf("core %d, pace_count: %d, tcurr: %lf\n", sample_id, pace_count, tcurr);
          // printf("timestep corrected in core %d \n", sample_id);
        }
        if(sample_id==0 && pace_count%10==0 && pace_count>99 && !writen){
        // printf("Calculating... watching core 0: %.2lf %% done\n",(tcurr[sample_id]/tmax)*100.0);
        printf("[");
        for (cnt=0; cnt<pace_count/10;cnt++){
          printf("=");
        }
        for (cnt=pace_count/10; cnt<pace_max/10;cnt++){
          printf("_");
        }
        printf("] %.2lf %% \n",(tcurr[sample_id]/tmax)*100.0);
        //mvaddch(0,pace_count,'=');
        //refresh();
        //system("clear");
        writen = true;
        }
        solveAnalytical(d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, dt[sample_id], sample_id);
        tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];
       
        if (pace_count > pace_max-2){
        time[input_counter + sample_id] = tcurr[sample_id];
        out_dt[input_counter + sample_id] = dt[sample_id];
        states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)];
        inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)];
        input_counter = input_counter + sample_size;
        //printf("counter: %d core: %d\n",input_counter,sample_id);
        }
    }
}



__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, 
                                       double *d_ALGEBRAIC, double *time, double *out_dt, double *states,
                                       double *ical, double *inal, unsigned int sample_size)
  {
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    double time_for_each_sample[2000];
    double dt_for_each_sample[2000];
    
    // printf("Calculating %d\n",thread_id);
    kernel_DoDrugSim(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                          time, out_dt, states, ical, inal, thread_id, 
                          time_for_each_sample, dt_for_each_sample, sample_size);
                          // __syncthreads();
    // printf("Calculation for core %d done\n",sample_id);
  }