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

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                       double *time, double *states, double *out_dt,  double *cai_result, 
                                       double *ina, double *inal,
                                       double *ical, double *ito,
                                       double *ikr, double *iks, 
                                       double *ik1,
                                       double *tcurr, double *dt, unsigned short sample_id, unsigned int sample_size,
                                       param_t *p_param)
    {
    
    unsigned int input_counter = 0;
    unsigned short cnt;

    int num_of_constants = 146;
    int num_of_states = 41;
    int num_of_algebraic = 199;
    int num_of_rates = 41;

    tcurr[sample_id] = 0.000001;
    dt[sample_id] = p_param->dt;
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
    // const double bcl = 2000; // bcl is basic cycle length
    const double bcl = p_param->bcl;
    
    const double inet_vm_threshold = p_param->inet_vm_threshold;
    // const unsigned short pace_max = 300;
    // const unsigned short pace_max = 1000;
    const unsigned short pace_max = 10;
    // const unsigned short celltype = 0.;
    // const unsigned short last_pace_print = 3;
    // const unsigned short last_drug_check_pace = 250;
    // const unsigned int print_freq = (1./dt) * dtw;
    // unsigned short pace_count = 0;
    // unsigned short pace_steepest = 0;
    // double conc = 99.0; //mmol
    double conc = 0.0;
    double type = p_param->celltype;
    bool dutta = p_param->is_dutta;
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
        // printf("Core: %d, Pace %d\n",sample_id,pace_count);
        computeRates(tcurr[sample_id], d_CONSTANTS, d_RATES, d_STATES, d_ALGEBRAIC, sample_id); 
        
        if (floor((tcurr[sample_id] + dt_set) / bcl) == floor(tcurr[sample_id] / bcl)) { 
          dt[sample_id] = dt_set;
        }
        else {
          dt[sample_id] = (floor(tcurr[sample_id] / bcl) + 1) * bcl - tcurr[sample_id];
          pace_count++;
          writen = false;
          // printf("core %d, pace_count: %d, dt: %lf\n", sample_id, pace_count, dt[sample_id]);
          // printf("timestep corrected in core %d \n", sample_id);
        }

        //// progress bar starts ////
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
        // //// progress bar ends ////

        solveAnalytical(d_CONSTANTS, d_STATES, d_ALGEBRAIC, d_RATES, dt[sample_id], sample_id);
        tcurr[sample_id] = tcurr[sample_id] + dt[sample_id];

        if (pace_count > pace_max-2){
        time[input_counter + sample_id] = tcurr[sample_id];
        states[input_counter + sample_id] = d_STATES[V + (sample_id * num_of_states)];
        out_dt[input_counter + sample_id] = d_RATES[V + (sample_id * num_of_rates)];
        cai_result[input_counter + sample_id] = d_ALGEBRAIC[cai + (sample_id * num_of_algebraic)];

        ina[input_counter + sample_id] = d_ALGEBRAIC[INa + (sample_id * num_of_algebraic)] ;
        inal[input_counter + sample_id] = d_ALGEBRAIC[INaL + (sample_id * num_of_algebraic)] ;

        ical[input_counter + sample_id] = d_ALGEBRAIC[ICaL + (sample_id * num_of_algebraic)] ;
        ito[input_counter + sample_id] = d_ALGEBRAIC[Ito + (sample_id * num_of_algebraic)] ;

        ikr[input_counter + sample_id] = d_ALGEBRAIC[IKr + (sample_id * num_of_algebraic)] ;
        iks[input_counter + sample_id] = d_ALGEBRAIC[IKs + (sample_id * num_of_algebraic)] ;

        ik1[input_counter + sample_id] = d_ALGEBRAIC[IK1 + (sample_id * num_of_algebraic)] ;

        input_counter = input_counter + sample_size;
        //printf("counter: %d core: %d\n",input_counter,sample_id);
        }
    }
    // __syncthreads();
}



__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                      double *time, double *states, double *out_dt,  double *cai_result, 
                                      double *ina, double *inal, 
                                      double *ical, double *ito,
                                      double *ikr, double *iks,
                                      double *ik1,
                                      unsigned int sample_size,
                                      param_t *p_param)
  {
    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    double time_for_each_sample[2000];
    double dt_for_each_sample[2000];
    
    // printf("Calculating %d\n",thread_id);
    kernel_DoDrugSim(d_ic50, d_CONSTANTS, d_STATES, d_RATES, d_ALGEBRAIC, 
                          time, states, out_dt, cai_result,
                          ina, inal, 
                          ical, ito,
                          ikr, iks, 
                          ik1,
                          time_for_each_sample, dt_for_each_sample, thread_id, sample_size,
                          p_param);
                          // __syncthreads();
    // printf("Calculation for core %d done\n",sample_id);
  }